//! Executable pipeline parallelism for decoder-only language models.
//!
//! A [`crate::architectures::distributed::pipeline::PipelineModel`] owns one dependency-safe, balanced contiguous
//! decoder-layer range and the boundary modules required by its explicit stage
//! role. [`crate::architectures::distributed::pipeline::PipelineInferenceScheduler`]
//! keeps independent request caches and
//! drains bounded, round-robin microbatch queues so different requests can
//! occupy different pipeline stages concurrently. Communication groups are
//! borrowed for each operation and are never retained by model state.
//! Multimodal families place their semantic encoder roots on stage zero and
//! transport only adapter-declared decoder ingress tensors to later stages;
//! tower and decoder units share the same rank-local residency plan.

use std::{
    collections::{BTreeMap, HashMap},
    ops::Range,
    path::{Path, PathBuf},
    sync::Arc,
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
    api::{
        common::{attention::AttentionInput, linear, linear::project_logits_maybe_quantized},
        deepseek_v3, gemma4, gpt_oss, inkling, kimi_linear, lfm2, llama, nemotron_h, qwen3_vl,
        ModelKind, ModelLoadOptions,
    },
    architectures::{
        gemma4::layerwise::{Gemma4Layer, Gemma4LayerwiseAdapter},
        inkling::layerwise::{InklingLayer, InklingLayerwiseAdapter},
        kimi_linear::layerwise::KimiLinearLayerwiseAdapter,
        lfm2::layerwise::Lfm2LayerwiseAdapter,
        nemotron_h::layerwise::NemotronHLayerwiseAdapter,
        qwen::{
            dense as dense_qwen,
            hybrid::{
                layerwise::{QwenHybridLayer, QwenHybridLayerwiseAdapter},
                qwen3_5 as qwen_hybrid,
            },
            vl::layerwise::{Qwen3VlLayer, Qwen3VlLayerwiseAdapter, Qwen3VlPipelinePrepared},
        },
    },
    error::Error,
    nn::{
        parallel::{
            planned_kv_head_layout, planned_optional_kv_head_layout,
            planned_optional_partition_widths,
        },
        tensor::create_causal_mask,
    },
    runtime::cache::residency::{
        load_prompt_cache_state_tensors, open_prompt_cache, validate_prompt_cache_model_identity,
        CacheRankIdentity, CacheResidencyManager, CacheResidencyPolicy, CacheResidencyReport,
        PagedCacheOptions, PromptCacheDescriptor, PromptCacheManifest, PromptCacheModelIdentity,
        PromptCacheOptions, PromptCacheStateArray, PromptCacheTopology, StateTensorOwner,
        StateTensorPolicy, StateTensorRole,
    },
    runtime::cache::{
        CompressedLatentCache, ConcatKeyValueCache, KeyValueCache, PagedKeyValueCache,
    },
    runtime::checkpoint::binding::{
        binding_bytes, materialize_module_bindings, populate_module_from_arrays_excluding,
        populate_module_from_dense_arrays_quantized, populate_module_from_lease,
    },
    runtime::checkpoint::quantization::{quantize_tensor, WeightQuantization},
    runtime::checkpoint::store::{GgufWeightStore, WeightStore, WeightStoreDiagnostics},
    runtime::distributed::expert::{
        dispatch_local_with, dispatch_replicated_with, ExpertAssignment, RoutingStatistics,
    },
    runtime::distributed::parallel::{
        sample_and_synchronize, ParallelBuildContext, ParallelExecutionContext, ShardingPolicy,
        SynchronizedToken,
    },
    runtime::distributed::topology::{ParallelCoordinates, ParallelTopology},
    runtime::execution::inspection::ActivationObserver,
    runtime::execution::layerwise::{
        open_safetensors_weight_store, shard_layer_bindings, ArchitectureAdapter,
        DenseDiskStreamReport, DenseStreamController, LayerWeightResidency, LayerwiseLoadOptions,
        SharedWeightStore, StaticUnitBindings,
    },
    runtime::generation::sampler::Sampler,
    runtime::media::{PreparedModelInput, PreparedModelInputIdentity},
    runtime::residency::expert_cache::{
        ExpertCache, ExpertCacheLoadOptions, ExpertCacheReport, ExpertPass,
    },
    runtime::residency::manager::{
        OffloadUnit, ResidencyManager, ResidencyReport, ResidentLayerGroup,
    },
    runtime::residency::policy::{
        MemoryTier, OffloadConfig, OffloadPlan, OffloadUnitId, OffloadUnitSpec, ResidencyPolicy,
    },
    runtime::scheduler::{
        FairScheduler, RequestId, RequestStatus, SchedulerLimits, SchedulerReport, WorkDescriptor,
        WorkId,
    },
};

#[cfg(test)]
use crate::runtime::execution::layerwise::WeightResidency;

use safemlx::ops::indexing::TryIndexOp;

/// Immutable, inspectable description of the local pipeline stage.
#[derive(Debug, Clone)]
pub struct PipelineStageInfo {
    /// Complete Cartesian topology and local TP/PP/EP coordinates.
    pub topology: ParallelTopology,
    /// Rank in the distributed group.
    pub global_rank: usize,
    /// Zero-based pipeline coordinate.
    pub pipeline_stage: usize,
    /// Number of pipeline stages.
    pub pipeline_stages: usize,
    /// Whether this stage performs token embedding.
    pub is_first: bool,
    /// Whether this stage performs final normalization and projection.
    pub is_last: bool,
    /// Global decoder-layer indices owned by this stage.
    pub global_layer_range: Range<usize>,
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
    /// Decoder hidden width.
    pub hidden_size: i32,
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

    /// Returns the validated batch dimension.
    pub const fn batch_size(self) -> i32 {
        self.batch_size
    }

    /// Returns the validated sequence dimension.
    pub const fn sequence_length(self) -> i32 {
        self.sequence_length
    }

    fn activation_shape(self, hidden_size: i32) -> [i32; 3] {
        [self.batch_size, self.sequence_length, hidden_size]
    }
}

/// The cache transition performed by one scheduled microbatch.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum PipelineInferencePhase {
    /// Add one prompt chunk to a request cache.
    Prefill,
    /// Add one autoregressive token to a request cache.
    Decode,
}

impl PipelineInferencePhase {
    const fn wire_tag(self) -> u32 {
        match self {
            Self::Prefill => 0,
            Self::Decode => 1,
        }
    }
}

/// Rank-local input for one scheduled pipeline cache transition.
///
/// For token ingress, stage zero supplies token ids and later stages retain an
/// empty token ingress. For typed multimodal ingress, every first-stage TP/EP
/// coordinate owns a [`PreparedModelInput`], while later stages retain its
/// payload-free [`PreparedModelInputIdentity`]. An architecture-specific
/// additive mask, when supported, is supplied independently on every rank.
#[derive(Debug, Clone)]
pub struct PipelineMicrobatchInput {
    request: RequestId,
    phase: PipelineInferencePhase,
    step: PipelineStep,
    ingress: ScheduledPipelineIngress,
    mask: Option<Array>,
}

#[derive(Debug, Clone)]
enum ScheduledPipelineIngress {
    Tokens(Option<Array>),
    Prepared(PreparedPipelineIngress),
}

#[derive(Debug, Clone)]
enum PreparedPipelineIngress {
    Payload(PreparedModelInput),
    Identity(PreparedModelInputIdentity),
}

impl ScheduledPipelineIngress {
    fn prepared_identity(&self) -> Option<PreparedModelInputIdentity> {
        match self {
            Self::Prepared(PreparedPipelineIngress::Payload(input)) => Some(input.identity()),
            Self::Prepared(PreparedPipelineIngress::Identity(identity)) => Some(identity.clone()),
            Self::Tokens(_) => None,
        }
    }

    fn encode_descriptor(&self, output: &mut Vec<u32>) -> Result<(), Error> {
        match self {
            Self::Tokens(_) => output.push(0),
            Self::Prepared(PreparedPipelineIngress::Payload(input)) => {
                output.push(1);
                input.identity().encode_descriptor(output)?;
            }
            Self::Prepared(PreparedPipelineIngress::Identity(identity)) => {
                output.push(1);
                identity.encode_descriptor(output)?;
            }
        }
        Ok(())
    }
}

impl PipelineMicrobatchInput {
    /// Creates a microbatch without rank-local token or mask tensors.
    pub const fn new(
        request: RequestId,
        phase: PipelineInferencePhase,
        step: PipelineStep,
    ) -> Self {
        Self {
            request,
            phase,
            step,
            ingress: ScheduledPipelineIngress::Tokens(None),
            mask: None,
        }
    }

    /// Supplies stage-zero token ids.
    pub fn with_tokens(mut self, tokens: Array) -> Self {
        self.ingress = ScheduledPipelineIngress::Tokens(Some(tokens));
        self
    }

    /// Supplies an owned prepared input on a first-stage TP/EP coordinate.
    ///
    /// Build the matching identity with [`PreparedModelInput::identity`] before
    /// moving the payload, then submit it with
    /// [`Self::with_prepared_input_identity`] on every later stage.
    pub fn with_prepared_input(mut self, input: PreparedModelInput) -> Self {
        self.ingress = ScheduledPipelineIngress::Prepared(PreparedPipelineIngress::Payload(input));
        self
    }

    /// Supplies only the matching prepared-input identity on a later stage.
    pub fn with_prepared_input_identity(mut self, identity: PreparedModelInputIdentity) -> Self {
        self.ingress =
            ScheduledPipelineIngress::Prepared(PreparedPipelineIngress::Identity(identity));
        self
    }

    /// Supplies an architecture-supported additive attention mask.
    pub fn with_mask(mut self, mask: Array) -> Self {
        self.mask = Some(mask);
        self
    }

    /// Returns the target request.
    pub const fn request(&self) -> RequestId {
        self.request
    }

    /// Returns the cache transition phase.
    pub const fn phase(&self) -> PipelineInferencePhase {
        self.phase
    }

    /// Returns the validated batch and sequence geometry.
    pub const fn step(&self) -> PipelineStep {
        self.step
    }

    /// Returns stage-zero token ids when present on this rank.
    pub const fn tokens(&self) -> Option<&Array> {
        match &self.ingress {
            ScheduledPipelineIngress::Tokens(tokens) => tokens.as_ref(),
            ScheduledPipelineIngress::Prepared(_) => None,
        }
    }

    /// Returns the collective identity for typed prepared ingress, when used.
    pub fn prepared_input_identity(&self) -> Option<PreparedModelInputIdentity> {
        self.ingress.prepared_identity()
    }

    /// Returns the rank-local additive mask when present.
    pub const fn mask(&self) -> Option<&Array> {
        self.mask.as_ref()
    }
}

/// Result metadata for one completed microbatch.
#[derive(Debug)]
pub struct PipelineMicrobatchOutput {
    work: WorkId,
    phase: PipelineInferencePhase,
    step: PipelineStep,
    logits: Option<Array>,
}

impl PipelineMicrobatchOutput {
    /// Returns the globally agreed work identity.
    pub const fn work(&self) -> WorkId {
        self.work
    }

    /// Returns the completed cache transition phase.
    pub const fn phase(&self) -> PipelineInferencePhase {
        self.phase
    }

    /// Returns the completed batch and sequence geometry.
    pub const fn step(&self) -> PipelineStep {
        self.step
    }

    /// Returns vocabulary logits on the final rank and `None` elsewhere.
    pub const fn logits(&self) -> Option<&Array> {
        self.logits.as_ref()
    }

    /// Consumes the record and returns final-rank logits when present.
    pub fn into_logits(self) -> Option<Array> {
        self.logits
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
    ModelInput(crate::api::input::ModelInput<'a>),
}

/// Result of one stage-local forward operation.
#[derive(Debug)]
pub enum PipelineStageOutput {
    /// Hidden activations to transfer to the next stage.
    Hidden(PipelinePayload),
    /// Vocabulary logits produced only by the final stage.
    Logits(Array),
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

    /// Returns the authoritative semantic tensor descriptor.
    pub const fn policy(&self) -> &StateTensorPolicy {
        &self.policy
    }

    /// Returns the materialized recurrent or convolution tensor, if initialized.
    pub const fn value(&self) -> Option<&Array> {
        self.value.as_ref()
    }

    /// Returns the number of input positions incorporated into this slot.
    pub const fn offset(&self) -> i32 {
        self.offset
    }

    fn clear(&mut self) {
        self.value = None;
        self.offset = 0;
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
}

/// Architecture-checked stage-local inference cache.
#[derive(Debug, Clone)]
pub struct PipelineCache {
    model_kind: ModelKind,
    layers: Vec<PipelineLayerCache>,
    residency_manager: Option<CacheResidencyManager>,
}

impl PipelineCache {
    /// Creates a cache from an explicit architecture identity and ordered layer entries.
    pub(crate) fn new(model_kind: ModelKind, layers: Vec<PipelineLayerCache>) -> Self {
        Self {
            model_kind,
            layers,
            residency_manager: None,
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
        }
    }

    /// Returns the architecture identity checked by stage execution.
    pub const fn model_kind(&self) -> ModelKind {
        self.model_kind
    }

    /// Returns the ordered local cache entries.
    pub fn layers(&self) -> &[PipelineLayerCache] {
        &self.layers
    }

    /// Returns the global decoder-layer ids represented locally.
    pub fn global_layers(&self) -> Vec<usize> {
        self.layers
            .iter()
            .map(|layer| match layer {
                PipelineLayerCache::StateSlots { global_layer, .. }
                | PipelineLayerCache::KeyValue { global_layer, .. }
                | PipelineLayerCache::CompressedLatent { global_layer, .. } => *global_layer,
            })
            .collect()
    }

    /// Clears retained state without changing local layer ownership.
    pub fn reset(&mut self) -> Result<(), Error> {
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
                    cache.clear()?;
                    slots.iter_mut().for_each(PipelineStateSlot::clear);
                }
                PipelineLayerCache::CompressedLatent { cache, slots, .. } => {
                    cache.clear()?;
                    slots.iter_mut().for_each(PipelineStateSlot::clear);
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
struct PipelineRequestState {
    cache: PipelineCache,
    batch_size: Option<i32>,
    last_phase: Option<PipelineInferencePhase>,
}

#[derive(Debug, Clone)]
struct ScheduledPipelineMicrobatch {
    phase: PipelineInferencePhase,
    step: PipelineStep,
    ingress: ScheduledPipelineIngress,
    mask: Option<Array>,
}

impl WorkDescriptor for ScheduledPipelineMicrobatch {
    fn encode_descriptor(&self, output: &mut Vec<u32>) -> Result<(), Error> {
        output.extend_from_slice(&[
            self.phase.wire_tag(),
            self.step.batch_size() as u32,
            self.step.sequence_length() as u32,
        ]);
        self.ingress.encode_descriptor(output)?;
        match &self.mask {
            Some(mask) => {
                output.extend_from_slice(&[1, mask.dtype() as u32, mask.ndim() as u32]);
                output.extend(mask.shape().iter().map(|dimension| *dimension as u32));
            }
            None => output.push(0),
        }
        Ok(())
    }
}

/// Bounded, fair inference scheduler for one rank-local pipeline stage.
///
/// The scheduler owns an independent [`PipelineCache`] for each active request.
/// Submitted work is drained in round-robin request order while preserving the
/// exact transition order within each request. Because ranks are independent
/// processes, every rank must register requests and submit matching work
/// descriptors. [`Self::run_queued`] performs an exact collective descriptor
/// comparison before point-to-point traffic begins, turning divergent schedules
/// into an error instead of an unmatched send/receive deadlock.
///
/// Sampling remains outside this type: callers consume final-rank logits, call
/// [`PipelineModel::sample_and_synchronize`], and either enqueue the selected
/// decode token or call [`Self::finish_request`] after EOS. This keeps sampling
/// policy separate from cache ownership and pipeline scheduling.
#[derive(Debug)]
pub struct PipelineInferenceScheduler {
    topology: ParallelTopology,
    model_kind: ModelKind,
    global_layer_range: Range<usize>,
    architecture_fingerprint: String,
    scheduler: FairScheduler<ScheduledPipelineMicrobatch, PipelineRequestState>,
}

impl PipelineInferenceScheduler {
    /// Binds a scheduler to one loaded pipeline stage.
    pub fn new(model: &PipelineModel, limits: SchedulerLimits) -> Result<Self, Error> {
        Ok(Self {
            topology: model.topology,
            model_kind: model.info.model_kind,
            global_layer_range: model.info.global_layer_range.clone(),
            architecture_fingerprint: model.prompt_cache_architecture_fingerprint()?,
            scheduler: FairScheduler::new(limits)?,
        })
    }

    /// Registers a request with a fresh device-resident cache.
    pub fn register_request(
        &mut self,
        model: &PipelineModel,
        request: RequestId,
    ) -> Result<(), Error> {
        self.validate_model(model)?;
        self.scheduler.validate_registration(request)?;
        let cache = model.new_cache()?;
        self.validate_cache(model, &cache)?;
        self.scheduler.register(
            request,
            PipelineRequestState {
                cache,
                batch_size: None,
                last_phase: None,
            },
        )
    }

    /// Registers a request with a fresh cache under an explicit residency policy.
    pub fn register_request_with_options(
        &mut self,
        model: &PipelineModel,
        request: RequestId,
        policy: CacheResidencyPolicy,
    ) -> Result<(), Error> {
        self.validate_model(model)?;
        self.scheduler.validate_registration(request)?;
        let cache = model.new_cache_with_options(policy)?;
        self.validate_cache(model, &cache)?;
        self.scheduler.register(
            request,
            PipelineRequestState {
                cache,
                batch_size: None,
                last_phase: None,
            },
        )
    }

    /// Registers a request with a restored or caller-created stage-local cache.
    pub fn register_request_with_cache(
        &mut self,
        model: &PipelineModel,
        request: RequestId,
        cache: PipelineCache,
    ) -> Result<(), Error> {
        self.validate_model(model)?;
        self.scheduler.validate_registration(request)?;
        self.validate_cache(model, &cache)?;
        self.scheduler.register(
            request,
            PipelineRequestState {
                cache,
                batch_size: None,
                last_phase: None,
            },
        )
    }

    /// Submits one rank-local microbatch and returns its ordered work identity.
    ///
    /// Requests may contain multiple prompt chunks. Once decode begins, later
    /// prefill work is rejected. Decode work always has sequence length one and
    /// every transition for a request must retain its original batch size.
    pub fn enqueue(&mut self, input: PipelineMicrobatchInput) -> Result<WorkId, Error> {
        let first_stage = self.topology.pipeline_parallel_rank == 0;
        match &input.ingress {
            ScheduledPipelineIngress::Tokens(Some(tokens)) if first_stage => {
                if tokens.ndim() != 2
                    || tokens.shape() != [input.step.batch_size(), input.step.sequence_length()]
                {
                    return Err(Error::Parallel(format!(
                        "pipeline stage zero microbatch expected token ids shaped [{}, {}], got {:?}",
                        input.step.batch_size(),
                        input.step.sequence_length(),
                        tokens.shape()
                    )));
                }
            }
            ScheduledPipelineIngress::Tokens(None) if first_stage => {
                return Err(Error::Parallel(
                    "pipeline stage zero microbatch requires token ids or prepared typed ingress"
                        .into(),
                ));
            }
            ScheduledPipelineIngress::Tokens(Some(_)) => {
                return Err(Error::Parallel(format!(
                    "pipeline stage {} microbatch must receive hidden activations rather than token ids",
                    self.topology.pipeline_parallel_rank
                )));
            }
            ScheduledPipelineIngress::Prepared(PreparedPipelineIngress::Payload(_))
                if !first_stage =>
            {
                return Err(Error::Parallel(format!(
                    "pipeline stage {} must retain only the prepared-input identity, not its payload",
                    self.topology.pipeline_parallel_rank
                )));
            }
            ScheduledPipelineIngress::Prepared(PreparedPipelineIngress::Identity(_))
                if first_stage =>
            {
                return Err(Error::Parallel(
                    "pipeline stage zero requires the owned prepared-input payload, not only its identity"
                        .into(),
                ));
            }
            ScheduledPipelineIngress::Prepared(_)
                if input.phase != PipelineInferencePhase::Prefill =>
            {
                return Err(Error::Parallel(
                    "prepared typed ingress is valid only for pipeline prefill work".into(),
                ));
            }
            ScheduledPipelineIngress::Tokens(None) | ScheduledPipelineIngress::Prepared(_) => {}
        }
        if input.phase == PipelineInferencePhase::Decode && input.step.sequence_length() != 1 {
            return Err(Error::Parallel(format!(
                "pipeline decode microbatch sequence length must be one, got {}",
                input.step.sequence_length()
            )));
        }

        let request = self.scheduler.request_state(input.request).ok_or_else(|| {
            Error::Parallel(format!(
                "pipeline request {} is not active",
                input.request.value()
            ))
        })?;
        if request.last_phase == Some(PipelineInferencePhase::Decode)
            && input.phase == PipelineInferencePhase::Prefill
        {
            return Err(Error::Parallel(format!(
                "pipeline request {} cannot return to prefill after decode began",
                input.request.value()
            )));
        }
        if let Some(batch_size) = request.batch_size {
            if batch_size != input.step.batch_size() {
                return Err(Error::Parallel(format!(
                    "pipeline request {} changed batch size from {batch_size} to {}",
                    input.request.value(),
                    input.step.batch_size()
                )));
            }
        }
        let phase = input.phase;
        let batch_size = input.step.batch_size();
        let request_id = input.request;
        let work = self.scheduler.enqueue(
            request_id,
            ScheduledPipelineMicrobatch {
                phase: input.phase,
                step: input.step,
                ingress: input.ingress,
                mask: input.mask,
            },
        )?;
        let request = self.scheduler.request_state_mut(request_id)?;
        request.batch_size.get_or_insert(batch_size);
        request.last_phase = Some(phase);
        Ok(work)
    }

    /// Drains the current queue using fair round-robin request ordering.
    ///
    /// A collective preflight compares every work descriptor exactly across
    /// ranks before any point-to-point send or receive is issued. Stage zero can
    /// then advance to a later request while downstream stages finish an earlier
    /// request, filling the pipeline without mixing request caches.
    pub fn run_queued(
        &mut self,
        model: &mut PipelineModel,
        group: &Group,
        stream: &Stream,
    ) -> Result<Vec<PipelineMicrobatchOutput>, Error> {
        self.validate_model(model)?;
        model.validate_group(group)?;
        let protocol = 0x5049_5045_0002_0000u64 | self.model_kind as u64;
        self.scheduler
            .drain_distributed(protocol, group, stream, |_, work, request| {
                match &work.ingress {
                    ScheduledPipelineIngress::Tokens(tokens) => model.forward_pipeline(
                        tokens.as_ref(),
                        work.step,
                        work.mask.as_ref(),
                        &mut request.cache,
                        group,
                        stream,
                    ),
                    ScheduledPipelineIngress::Prepared(PreparedPipelineIngress::Payload(input)) => {
                        input.with_model_input(|input| {
                            model.prefill_pipeline(
                                Some(input),
                                work.step,
                                work.mask.as_ref(),
                                &mut request.cache,
                                group,
                                stream,
                            )
                        })
                    }
                    ScheduledPipelineIngress::Prepared(PreparedPipelineIngress::Identity(_)) => {
                        model.prefill_pipeline(
                            None,
                            work.step,
                            work.mask.as_ref(),
                            &mut request.cache,
                            group,
                            stream,
                        )
                    }
                }
            })
            .map(|completed| {
                completed
                    .into_iter()
                    .map(|completed| {
                        let (work, input, logits) = completed.into_parts();
                        PipelineMicrobatchOutput {
                            work,
                            phase: input.phase,
                            step: input.step,
                            logits,
                        }
                    })
                    .collect()
            })
    }

    /// Drains the current queue over topology-derived Cartesian subgroups.
    ///
    /// Global schedule consensus uses [`crate::CartesianExecution::world`],
    /// pipeline payloads use matching-coordinate PP lanes, and stage-local TP
    /// and EP work uses the corresponding Cartesian subgroups.
    pub fn run_queued_cartesian(
        &mut self,
        model: &mut PipelineModel,
        cartesian: &crate::CartesianExecution<'_>,
        stream: &Stream,
    ) -> Result<Vec<PipelineMicrobatchOutput>, Error> {
        self.validate_model(model)?;
        if cartesian.topology() != self.topology {
            return Err(Error::Parallel(format!(
                "pipeline scheduler topology {:?} does not match Cartesian execution topology {:?}",
                self.topology,
                cartesian.topology()
            )));
        }
        let protocol = 0x5049_5045_0003_0000u64 | self.model_kind as u64;
        self.scheduler
            .drain_distributed(
                protocol,
                cartesian.world(),
                stream,
                |_, work, request| match &work.ingress {
                    ScheduledPipelineIngress::Tokens(tokens) => model.forward_cartesian(
                        tokens.as_ref(),
                        work.step,
                        work.mask.as_ref(),
                        &mut request.cache,
                        cartesian,
                        stream,
                    ),
                    ScheduledPipelineIngress::Prepared(PreparedPipelineIngress::Payload(input)) => {
                        input.with_model_input(|input| {
                            model.prefill_cartesian(
                                Some(input),
                                work.step,
                                work.mask.as_ref(),
                                &mut request.cache,
                                cartesian,
                                stream,
                            )
                        })
                    }
                    ScheduledPipelineIngress::Prepared(PreparedPipelineIngress::Identity(_)) => {
                        model.prefill_cartesian(
                            None,
                            work.step,
                            work.mask.as_ref(),
                            &mut request.cache,
                            cartesian,
                            stream,
                        )
                    }
                },
            )
            .map(|completed| {
                completed
                    .into_iter()
                    .map(|completed| {
                        let (work, input, logits) = completed.into_parts();
                        PipelineMicrobatchOutput {
                            work,
                            phase: input.phase,
                            step: input.step,
                            logits,
                        }
                    })
                    .collect()
            })
    }

    /// Marks a request complete after EOS and releases its cache.
    ///
    /// Any speculative work still queued for the request is discarded.
    pub fn finish_request(&mut self, request: RequestId) -> Result<(), Error> {
        self.scheduler.finish(request)
    }

    /// Cancels a request, releases its cache, and discards its queued work.
    pub fn cancel_request(&mut self, request: RequestId) -> Result<(), Error> {
        self.scheduler.cancel(request)
    }

    /// Releases an idle active request and returns its cache to the caller.
    ///
    /// This is the handoff used before explicit prompt-cache persistence. Work
    /// must be drained or cancelled first.
    pub fn release_request_cache(&mut self, request: RequestId) -> Result<PipelineCache, Error> {
        Ok(self.scheduler.release(request)?.cache)
    }

    /// Removes a terminal identity so a caller may explicitly reuse it.
    pub fn forget_terminal_request(&mut self, request: RequestId) -> Result<RequestStatus, Error> {
        self.scheduler.forget_terminal(request)
    }

    /// Returns the known request lifecycle state.
    pub fn request_status(&self, request: RequestId) -> Option<RequestStatus> {
        self.scheduler.request_status(request)
    }

    /// Returns the number of queued transitions for one active request.
    pub fn queued_for_request(&self, request: RequestId) -> usize {
        self.scheduler.queued_for_request(request)
    }

    /// Returns a current observability snapshot.
    pub fn report(&self) -> SchedulerReport {
        self.scheduler.report()
    }

    /// Returns the failure that invalidated this scheduler, if any.
    pub fn poison_reason(&self) -> Option<&str> {
        self.scheduler.poison_reason()
    }

    fn validate_model(&self, model: &PipelineModel) -> Result<(), Error> {
        if model.topology != self.topology
            || model.info.model_kind != self.model_kind
            || model.info.global_layer_range != self.global_layer_range
            || model.prompt_cache_architecture_fingerprint()? != self.architecture_fingerprint
        {
            return Err(Error::Parallel(
                "pipeline scheduler is bound to a different model stage".into(),
            ));
        }
        Ok(())
    }

    fn validate_cache(&self, model: &PipelineModel, cache: &PipelineCache) -> Result<(), Error> {
        if cache.model_kind() != self.model_kind
            || cache.global_layers() != model.info.global_layer_range.clone().collect::<Vec<_>>()
        {
            return Err(Error::Parallel(format!(
                "pipeline request cache {:?} layers {:?} do not match {:?} layers {:?}",
                cache.model_kind(),
                cache.global_layers(),
                self.model_kind,
                model.info.global_layer_range
            )));
        }
        Ok(())
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
        }
    }
}

struct LlamaStage {
    args: llama::ModelArgs,
    layer_adapter: crate::architectures::llama::layerwise::LlamaLayerwiseAdapter,
    range: Range<usize>,
    embedding: Option<MaybeQuantized<nn::Embedding>>,
    output_embedding: Option<MaybeQuantized<nn::Embedding>>,
    layers: Vec<llama::TransformerBlock>,
    dense_layers: Option<PipelineLayerStorage>,
    norm: Option<nn::RmsNorm>,
    lm_head: Option<MaybeQuantized<nn::Linear>>,
    parallel_embedding: Option<crate::nn::parallel::VocabParallelEmbedding>,
    parallel_output_embedding: Option<crate::nn::parallel::VocabParallelEmbedding>,
    parallel_lm_head: Option<crate::nn::parallel::VocabParallelLmHead>,
    parallel_layout: Option<crate::runtime::distributed::parallel::LocalModelLayout>,
    parallel_kv_heads: Option<Vec<i32>>,
}

struct DeepSeekStage {
    args: deepseek_v3::ModelArgs,
    layer_adapter: crate::architectures::deepseek_v3::layerwise::DeepSeekV3LayerwiseAdapter,
    range: Range<usize>,
    embedding: Option<MaybeQuantized<nn::Embedding>>,
    layers: Vec<deepseek_v3::DecoderLayer>,
    dense_layers: Option<PipelineLayerStorage>,
    norm: Option<nn::RmsNorm>,
    lm_head: Option<MaybeQuantized<nn::Linear>>,
    parallel_embedding: Option<crate::nn::parallel::VocabParallelEmbedding>,
    parallel_lm_head: Option<crate::nn::parallel::VocabParallelLmHead>,
    parallel_layout: Option<crate::runtime::distributed::parallel::LocalModelLayout>,
    expert_assignment: Option<ExpertAssignment>,
    expert_storage: PipelineExpertStorage,
    routing_statistics: RoutingStatistics,
}

struct GemmaStage {
    args: gemma4::ModelArgs,
    layer_adapter: Gemma4LayerwiseAdapter,
    range: Range<usize>,
    has_multimodal_ingress: bool,
    media_units: Vec<GemmaPipelineMediaUnit>,
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
    parallel_per_layer_projection: Option<crate::nn::parallel::ParallelLinear>,
    parallel_lm_head: Option<crate::nn::parallel::VocabParallelLmHead>,
    parallel_layout: Option<crate::runtime::distributed::parallel::LocalModelLayout>,
}

#[derive(Debug, Clone, Copy)]
struct GemmaPipelineMediaUnit {
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
    parallel_embedding: Option<crate::nn::parallel::VocabParallelEmbedding>,
    parallel_output_embedding: Option<crate::nn::parallel::VocabParallelEmbedding>,
    parallel_lm_head: Option<crate::nn::parallel::VocabParallelLmHead>,
    parallel_layout: Option<crate::runtime::distributed::parallel::LocalModelLayout>,
    expert_assignment: Option<ExpertAssignment>,
    expert_cache: Option<ExpertCache>,
    routing_statistics: RoutingStatistics,
}

struct Qwen3VlStage {
    args: qwen3_vl::ModelArgs,
    layer_adapter: Qwen3VlLayerwiseAdapter,
    range: Range<usize>,
    vision_layers: Vec<Qwen3VlLayer>,
    layers: Vec<Qwen3VlLayer>,
    dense_layers: Option<PipelineLayerStorage>,
    output_embedding: Option<MaybeQuantized<nn::Embedding>>,
    parallel_output_embedding: Option<crate::nn::parallel::VocabParallelEmbedding>,
    parallel_layout: Option<crate::runtime::distributed::parallel::LocalModelLayout>,
    expert_assignment: Option<ExpertAssignment>,
    expert_storage: PipelineExpertStorage,
    routing_statistics: RoutingStatistics,
}

struct GptOssStage {
    args: gpt_oss::ModelArgs,
    layer_adapter: crate::architectures::gpt_oss::layerwise::GptOssLayerwiseAdapter,
    range: Range<usize>,
    embedding: Option<MaybeQuantized<nn::Embedding>>,
    layers: Vec<gpt_oss::TransformerBlock>,
    dense_layers: Option<PipelineLayerStorage>,
    norm: Option<nn::RmsNorm>,
    lm_head: Option<MaybeQuantized<nn::Linear>>,
    parallel_embedding: Option<crate::nn::parallel::VocabParallelEmbedding>,
    parallel_lm_head: Option<crate::nn::parallel::VocabParallelLmHead>,
    parallel_layout: Option<crate::runtime::distributed::parallel::LocalModelLayout>,
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
    parallel_embedding: Option<crate::nn::parallel::VocabParallelEmbedding>,
    parallel_output_embedding: Option<crate::nn::parallel::VocabParallelEmbedding>,
    parallel_lm_head: Option<crate::nn::parallel::VocabParallelLmHead>,
    parallel_layout: Option<crate::runtime::distributed::parallel::LocalModelLayout>,
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
    parallel_embedding: Option<crate::nn::parallel::VocabParallelEmbedding>,
    parallel_output_embedding: Option<crate::nn::parallel::VocabParallelEmbedding>,
    parallel_lm_head: Option<crate::nn::parallel::VocabParallelLmHead>,
    parallel_layout: Option<crate::runtime::distributed::parallel::LocalModelLayout>,
    parallel_geometry: Option<Vec<nemotron_h::ParallelLayerGeometry>>,
    expert_assignment: Option<ExpertAssignment>,
    expert_storage: PipelineExpertStorage,
    routing_statistics: RoutingStatistics,
}

struct QwenHybridStage {
    args: qwen_hybrid::ModelArgs,
    layer_adapter: QwenHybridLayerwiseAdapter,
    range: Range<usize>,
    embedding: Option<MaybeQuantized<nn::Embedding>>,
    output_embedding: Option<MaybeQuantized<nn::Embedding>>,
    layers: Vec<qwen_hybrid::TransformerBlock>,
    dense_layers: Option<PipelineLayerStorage>,
    norm: Option<qwen_hybrid::Qwen3NextRmsNorm>,
    lm_head: Option<MaybeQuantized<nn::Linear>>,
    parallel_embedding: Option<crate::nn::parallel::VocabParallelEmbedding>,
    parallel_output_embedding: Option<crate::nn::parallel::VocabParallelEmbedding>,
    parallel_lm_head: Option<crate::nn::parallel::VocabParallelLmHead>,
    parallel_layout: Option<crate::runtime::distributed::parallel::LocalModelLayout>,
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
    parallel_embedding: Option<crate::nn::parallel::VocabParallelEmbedding>,
    parallel_output_embedding: Option<crate::nn::parallel::VocabParallelEmbedding>,
    parallel_lm_head: Option<crate::nn::parallel::VocabParallelLmHead>,
    parallel_layout: Option<crate::runtime::distributed::parallel::LocalModelLayout>,
    parallel_cache_geometry: Option<Vec<kimi_linear::KimiLayerCacheGeometry>>,
    expert_assignment: Option<ExpertAssignment>,
    expert_storage: PipelineExpertStorage,
    routing_statistics: RoutingStatistics,
}

struct InklingStage {
    args: inkling::ModelArgs,
    layer_adapter: InklingLayerwiseAdapter,
    range: Range<usize>,
    embedding: Option<MaybeQuantized<nn::Embedding>>,
    embed_norm: Option<nn::RmsNorm>,
    layers: Vec<inkling::DecoderLayer>,
    dense_layers: Option<PipelineLayerStorage>,
    norm: Option<nn::RmsNorm>,
    lm_head: Option<MaybeQuantized<nn::Linear>>,
    parallel_embedding: Option<crate::nn::parallel::VocabParallelEmbedding>,
    parallel_lm_head: Option<crate::nn::parallel::VocabParallelLmHead>,
    parallel_layout: Option<crate::runtime::distributed::parallel::LocalModelLayout>,
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
    fn checkpoint_diagnostics(&self) -> Result<Option<WeightStoreDiagnostics>, Error>;
    fn expert_cache_report(&self) -> Result<Option<ExpertCacheReport>, Error>;

    /// Returns the exact cache identity and local semantic state schedule.
    fn prompt_cache_model_identity(
        &self,
        topology: ParallelTopology,
    ) -> Result<PromptCacheModelIdentity, Error>;

    /// Executes this stage's local decoder range.
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
        input: crate::api::input::ModelInput<'_>,
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
    fn prompt_cache_model_identity(
        &self,
        topology: ParallelTopology,
    ) -> Result<PromptCacheModelIdentity, Error>;
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
        _input: crate::api::input::ModelInput<'_>,
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

    fn checkpoint_diagnostics(&self) -> Result<Option<WeightStoreDiagnostics>, Error> {
        if let Some(diagnostics) = self
            .0
            .dense_layers()
            .map(PipelineLayerStorage::checkpoint_diagnostics)
            .transpose()?
        {
            return Ok(Some(diagnostics));
        }
        self.0
            .expert_cache()
            .map(|cache| {
                cache
                    .report()
                    .map(|report| report.residency.weight_store().clone())
                    .map_err(Error::from)
            })
            .transpose()
    }

    fn expert_cache_report(&self) -> Result<Option<ExpertCacheReport>, Error> {
        self.0
            .expert_cache()
            .map(ExpertCache::report)
            .transpose()
            .map_err(Error::from)
    }

    fn prompt_cache_model_identity(
        &self,
        topology: ParallelTopology,
    ) -> Result<PromptCacheModelIdentity, Error> {
        self.0.prompt_cache_model_identity(topology)
    }

    fn forward(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        self.0.forward(input, step, mask, cache, stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill(
        &mut self,
        input: crate::api::input::ModelInput<'_>,
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
    DenseDiskStream(crate::runtime::residency::dense_stream::DenseDiskStreamLoadOptions),
}

enum PipelineLayerController {
    LayerwiseHost(ResidentLayerGroup),
    DenseDiskStream(Box<DenseStreamController>),
}

struct PipelineLayerStorage {
    store: SharedWeightStore,
    residency: ResidencyManager,
    controller: PipelineLayerController,
    units: Vec<OffloadUnitId>,
    execution_offset: usize,
    independent_expert_prefix: Option<&'static str>,
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
    fn prepare(
        &self,
        local_index: usize,
        prefill: bool,
    ) -> Result<
        (
            Option<crate::runtime::residency::manager::ResidentUnitLease>,
            crate::runtime::residency::manager::ResidentUnitLease,
        ),
        Error,
    > {
        self.prepare_absolute(self.execution_offset + local_index, prefill)
    }

    fn prepare_absolute(
        &self,
        unit_index: usize,
        prefill: bool,
    ) -> Result<
        (
            Option<crate::runtime::residency::manager::ResidentUnitLease>,
            crate::runtime::residency::manager::ResidentUnitLease,
        ),
        Error,
    > {
        if unit_index >= self.units.len() {
            return Err(Error::Parallel(format!(
                "pipeline unit index {unit_index} exceeds {} planned units",
                self.units.len()
            )));
        }
        match &self.controller {
            PipelineLayerController::LayerwiseHost(group) => {
                group.prepare(&self.residency, unit_index)?;
                Ok((
                    None,
                    self.residency
                        .acquire(&self.units[unit_index], MemoryTier::Device)?,
                ))
            }
            PipelineLayerController::DenseDiskStream(controller) => controller.prepare(
                &self.residency,
                "pipeline_stage",
                &self.units,
                unit_index,
                prefill,
            ),
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
            PipelineLayerController::DenseDiskStream(controller) => {
                Ok(Some(controller.report(&self.residency)?))
            }
        }
    }

    fn residency_report(&self) -> Result<ResidencyReport, Error> {
        Ok(self.residency.report()?)
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

    fn checkpoint_diagnostics(&self) -> Result<WeightStoreDiagnostics, Error> {
        Ok(self.store.diagnostics()?)
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
    for (local_index, (global_layer, cache)) in range.zip(caches.iter_mut()).enumerate() {
        if let Some(dense) = dense_layers {
            let (_host, lease) = dense.prepare(local_index, prefill)?;
            let mut layer = new_layer(global_layer, stream)?;
            if let Some(prefix) = dense.independent_expert_prefix {
                crate::runtime::checkpoint::binding::populate_module_from_lease_excluding(
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
            eval(
                std::iter::once(&hidden)
                    .chain(cache.retained_arrays())
                    .chain(forwarded.retained.iter()),
            )?;
            stream.synchronize()?;
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
    topology: ParallelTopology,
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
        topology: ParallelTopology,
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
            crate::architectures::llama::model::prompt_cache_architecture_fingerprint(&self.args),
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

    fn prompt_cache_model_identity(
        &self,
        topology: ParallelTopology,
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
            crate::architectures::deepseek_v3::model::prompt_cache_architecture_fingerprint(
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

impl PipelineStageSemantics for GemmaStage {
    fn model_kind(&self) -> ModelKind {
        ModelKind::Gemma4
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
                vec![1, 1, step.sequence_length, step.sequence_length],
                1 + self.multimodal_mask_windows.len(),
            ));
        }
        shapes
    }

    fn dense_layers(&self) -> Option<&PipelineLayerStorage> {
        self.dense_layers.as_ref()
    }

    fn prompt_cache_model_identity(
        &self,
        topology: ParallelTopology,
    ) -> Result<PromptCacheModelIdentity, Error> {
        let complete = if self.parallel_layout.is_some() {
            self.layer_adapter.parallel_text_cache_layout()?
        } else {
            crate::architectures::gemma4::model::prompt_cache_layer_layout(&self.args)?
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
            crate::architectures::gemma4::model::prompt_cache_architecture_fingerprint(&self.args),
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
        GemmaStage::forward(self, input, step, mask, cache, stream)
    }

    fn prefill(
        &mut self,
        input: crate::api::input::ModelInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        if expert_group.is_some() {
            return Err(Error::Parallel(
                "Gemma 4 multimodal pipeline ingress does not activate expert exchange".into(),
            ));
        }
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
            ),
            _ => self.forward(
                PipelineStageInput::Hidden(&payload),
                step,
                mask,
                cache,
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
        if expert_group.is_some() {
            return Err(Error::Parallel(
                "Gemma 4 text TP+PP does not activate expert exchange".into(),
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
        topology: ParallelTopology,
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

impl PipelineStageSemantics for Qwen3VlStage {
    fn model_kind(&self) -> ModelKind {
        if self.args.text_config.is_moe() {
            ModelKind::Qwen3VlMoe
        } else {
            ModelKind::Qwen3Vl
        }
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
        topology: ParallelTopology,
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
        input: crate::api::input::ModelInput<'_>,
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
        typed: Option<crate::api::input::ModelInput<'_>>,
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
                eval([&forwarded])?;
                stream.synchronize()?;
                Ok(forwarded)
            },
        )?;
        qwen3_vl_set_pipeline_rope_delta(caches, prepared.rope_delta, stream)?;
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
    layout: Option<&crate::runtime::distributed::parallel::LocalModelLayout>,
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
    stream: &Stream,
) -> Result<(), Error> {
    let offset = pipeline_kv_offset(caches);
    for cache in caches {
        let slots = match cache {
            PipelineLayerCache::StateSlots { slots, .. }
            | PipelineLayerCache::KeyValue { slots, .. }
            | PipelineLayerCache::CompressedLatent { slots, .. } => slots,
        };
        if let Some(slot) = slots
            .iter_mut()
            .find(|slot| slot.policy.role == StateTensorRole::PositionDelta)
        {
            slot.value = Some(Array::from_slice(&[rope_delta], &[1]));
            slot.offset = offset;
            eval(slot.value.iter())?;
            stream.synchronize()?;
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
        topology: ParallelTopology,
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
        topology: ParallelTopology,
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

    fn prompt_cache_model_identity(
        &self,
        topology: ParallelTopology,
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
        topology: ParallelTopology,
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
        topology: ParallelTopology,
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
        topology: ParallelTopology,
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
    topology: ParallelTopology,
    info: PipelineStageInfo,
    stage: Box<dyn PipelineStageAdapter>,
    cache_identity: PromptCacheModelIdentity,
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
        topology: ParallelTopology,
        info: PipelineStageInfo,
        stage: impl PipelineStageAdapter + 'static,
    ) -> Result<Self, Error> {
        if stage.model_kind() != info.model_kind {
            return Err(Error::Parallel(format!(
                "pipeline adapter architecture {:?} does not match stage architecture {:?}",
                stage.model_kind(),
                info.model_kind
            )));
        }
        let cache_identity = stage.prompt_cache_model_identity(topology)?;
        if cache_identity.global_layer_start != info.global_layer_range.start
            || cache_identity.global_layer_end != info.global_layer_range.end
            || cache_identity.layer_layout.len() != info.global_layer_range.len()
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

    /// Returns physical checkpoint-read telemetry for a non-resident local stage.
    pub fn checkpoint_diagnostics(&self) -> Result<Option<WeightStoreDiagnostics>, Error> {
        Ok(self
            .stage
            .checkpoint_diagnostics()?
            .or_else(|| self.info.checkpoint_diagnostics.clone()))
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
        Ok(PipelineCache::new(
            self.info.model_kind,
            materialize_pipeline_cache_layers(&self.cache_identity, None)?,
        ))
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
                let layers = materialize_pipeline_cache_layers(
                    &self.cache_identity,
                    Some((manager.clone(), rank)),
                )?;
                Ok(PipelineCache::with_residency_manager(
                    self.info.model_kind,
                    layers,
                    manager,
                ))
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
    ) -> Result<PromptCacheManifest, Error> {
        let identity = self.prompt_cache_model_identity()?;
        validate_prompt_cache_model_identity(&descriptor, &identity)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        if cache.model_kind != self.info.model_kind {
            return Err(Error::Parallel(
                "pipeline prompt cache architecture does not match the stage".into(),
            ));
        }
        let manager = cache.residency_manager.clone().ok_or_else(|| {
            Error::Parallel("pipeline prompt persistence requires a paged cache".into())
        })?;
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
            }
        }
        let expected_offset = i32::try_from(prefix_token_ids.len())
            .map_err(|_| Error::Parallel("pipeline prompt length exceeds i32".into()))?;
        let mut state_arrays = Vec::new();
        for layer in &cache.layers {
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
                    None if slot.policy.required => {
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
        let mut layers =
            materialize_pipeline_cache_layers(&self.cache_identity, Some((manager.clone(), rank)))?;
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
            };
            for slot in slots {
                slot.value = restored_state
                    .remove(&(StateTensorOwner::Layer(global_layer), slot.policy.role));
                if slot.value.is_some() {
                    slot.offset = offset;
                } else if slot.policy.required {
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
        let cache = PipelineCache::with_residency_manager(self.info.model_kind, layers, manager);
        Ok((cache, manifest))
    }

    fn prompt_cache_rank_directory(&self, root: &Path) -> PathBuf {
        root.join(format!("rank-{:05}", self.topology.global_rank))
    }

    /// Returns the canonical cache-relevant architecture identity for this stage.
    pub fn prompt_cache_architecture_fingerprint(&self) -> Result<String, Error> {
        Ok(self.prompt_cache_model_identity()?.architecture_fingerprint)
    }

    /// Returns this stage's exact ordered prompt-cache layout.
    pub fn prompt_cache_layer_layout(
        &self,
    ) -> Result<crate::LayerSchedule<crate::LayerCachePolicy>, Error> {
        Ok(self.prompt_cache_model_identity()?.layer_layout)
    }

    fn prompt_cache_model_identity(&self) -> Result<PromptCacheModelIdentity, Error> {
        Ok(self.cache_identity.clone())
    }

    /// Executes only this stage, without communication.
    ///
    /// This operation is useful for deterministic single-process composition
    /// tests and custom schedulers. Distributed callers normally use
    /// [`Self::forward_pipeline`].
    pub fn forward_stage(
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
            .forward(input, step, mask, &mut cache.layers, stream)
    }

    /// Executes typed multimodal ingress on stage zero without communication.
    ///
    /// This complements [`Self::forward_stage`] for deterministic composition
    /// tests and custom schedulers. Distributed callers normally use
    /// [`Self::prefill_pipeline`] or [`Self::prefill_cartesian`].
    pub fn prefill_stage(
        &mut self,
        input: crate::api::input::ModelInput<'_>,
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
        crate::api::input::validate(input)?;
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

    /// Runs one distributed pipeline microbatch without queue management.
    ///
    /// Stage zero embeds and sends, intermediate stages receive/execute/send,
    /// and the final stage receives and returns logits. Every lazy point-to-
    /// point operation is evaluated and synchronized before the operation
    /// returns. Multi-request inference should normally use
    /// [`PipelineInferenceScheduler::run_queued`], which calls this primitive
    /// in a collectively validated order that can fill different stages.
    pub fn forward_pipeline(
        &mut self,
        tokens: Option<&Array>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut PipelineCache,
        group: &Group,
        stream: &Stream,
    ) -> Result<Option<Array>, Error> {
        self.validate_group(group)?;
        self.forward_pipeline_on_group(
            tokens.map(PipelineIngress::Tokens),
            step,
            mask,
            cache,
            group,
            self.info.predecessor_rank,
            self.info.successor_rank,
            None,
            None,
            stream,
        )
    }

    /// Runs a microbatch over topology-derived pipeline lanes while executing
    /// stage-local TP and EP semantics through their own subgroup contexts.
    pub fn forward_cartesian(
        &mut self,
        tokens: Option<&Array>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut PipelineCache,
        cartesian: &crate::CartesianExecution<'_>,
        stream: &Stream,
    ) -> Result<Option<Array>, Error> {
        if cartesian.topology() != self.topology {
            return Err(Error::Parallel(format!(
                "pipeline model topology {:?} does not match Cartesian execution topology {:?}",
                self.topology,
                cartesian.topology()
            )));
        }
        let pipeline = cartesian.pipeline_group().ok_or_else(|| {
            Error::Parallel("Cartesian pipeline execution requires a PP lane group".into())
        })?;
        let tensor = (self.topology.tensor_parallel_size > 1)
            .then(|| cartesian.tensor_context(stream))
            .transpose()?;
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
            cartesian.expert_group(),
            stream,
        )?;
        // A lane-local barrier keeps later world-wide sampling or consensus
        // collectives from overtaking TP/EP work still running on a later
        // pipeline stage. This is particularly important for backends whose
        // subgroup support is implemented with topology-routed peer exchange.
        let barrier = distributed::all_sum(&Array::from_int(0), pipeline, stream)?;
        eval([&barrier])?;
        stream.synchronize()?;
        Ok(output)
    }

    /// Runs typed multimodal prefill over the world pipeline group.
    ///
    /// Only stage zero supplies `input`; every later stage passes `None` and
    /// receives the architecture-authored payload from its predecessor.
    pub fn prefill_pipeline(
        &mut self,
        input: Option<crate::api::input::ModelInput<'_>>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut PipelineCache,
        group: &Group,
        stream: &Stream,
    ) -> Result<Option<Array>, Error> {
        self.validate_group(group)?;
        self.forward_pipeline_on_group(
            input.map(PipelineIngress::ModelInput),
            step,
            mask,
            cache,
            group,
            self.info.predecessor_rank,
            self.info.successor_rank,
            None,
            None,
            stream,
        )
    }

    /// Runs typed multimodal prefill over topology-derived PP lanes while TP
    /// or EP execution stays scoped to the corresponding stage subgroup.
    pub fn prefill_cartesian(
        &mut self,
        input: Option<crate::api::input::ModelInput<'_>>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut PipelineCache,
        cartesian: &crate::CartesianExecution<'_>,
        stream: &Stream,
    ) -> Result<Option<Array>, Error> {
        if cartesian.topology() != self.topology {
            return Err(Error::Parallel(format!(
                "pipeline model topology {:?} does not match Cartesian execution topology {:?}",
                self.topology,
                cartesian.topology()
            )));
        }
        let pipeline = cartesian.pipeline_group().ok_or_else(|| {
            Error::Parallel("Cartesian pipeline execution requires a PP lane group".into())
        })?;
        let tensor = (self.topology.tensor_parallel_size > 1)
            .then(|| cartesian.tensor_context(stream))
            .transpose()?;
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
            cartesian.expert_group(),
            stream,
        )?;
        let barrier = distributed::all_sum(&Array::from_int(0), pipeline, stream)?;
        eval([&barrier])?;
        stream.synchronize()?;
        Ok(output)
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
        stream: &Stream,
    ) -> Result<Option<Array>, Error> {
        let mut received_payload = None;
        let stage_input = if self.info.is_first {
            Some(
                ingress
                    .ok_or_else(|| Error::Parallel("pipeline stage zero requires input".into()))?,
            )
        } else {
            if ingress.is_some() {
                return Err(Error::Parallel(format!(
                    "pipeline stage {} receives hidden activations and must not receive ingress",
                    self.info.pipeline_stage
                )));
            }
            let peer = predecessor.expect("non-first predecessor");
            let received = distributed::recv(
                &step.activation_shape(self.info.hidden_size),
                self.info.activation_dtype,
                peer,
                group,
                stream,
            )
            .map_err(|error| {
                Error::Parallel(format!(
                    "stage {} failed to receive {:?} {:?} activations from rank {peer}: {error}",
                    self.info.pipeline_stage,
                    step.activation_shape(self.info.hidden_size),
                    self.info.activation_dtype
                ))
            })?;
            eval([&received])?;
            stream.synchronize()?;
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
                        eval([&value])?;
                        Ok(value)
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
            stream.synchronize()?;
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
                    crate::api::input::validate(input)?;
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
        match output {
            PipelineStageOutput::Hidden(payload) => {
                let hidden = &payload.hidden;
                let expected = step.activation_shape(self.info.hidden_size);
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
                eval([&sent])?;
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
                    eval([&sent])?;
                }
                stream.synchronize()?;
                Ok(None)
            }
            PipelineStageOutput::Logits(logits) => Ok(Some(logits)),
        }
    }

    /// Samples on the last stage and broadcasts only the selected token and
    /// EOS/stop state via identically ordered all-sums.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_and_synchronize<S: Sampler>(
        &self,
        logits: Option<&Array>,
        step: PipelineStep,
        sampler: &mut S,
        temperature: f32,
        prng_state: Option<&mut safemlx::random::RandomState>,
        finished: bool,
        group: &Group,
        stream: &Stream,
    ) -> Result<SynchronizedToken, Error> {
        self.validate_group(group)?;
        if !self.info.is_last && logits.is_some() {
            return Err(Error::Parallel(
                "only the last pipeline stage may supply logits".into(),
            ));
        }
        let sampling_rank = self.topology.global_rank_for(ParallelCoordinates {
            tensor: 0,
            pipeline: self.topology.pipeline_parallel_size - 1,
            expert: 0,
        })?;
        sample_and_synchronize(
            logits,
            step.batch_size,
            sampler,
            temperature,
            prng_state,
            finished,
            sampling_rank,
            group,
            stream,
        )
    }

    fn validate_group(&self, group: &Group) -> Result<(), Error> {
        if group.rank() != self.topology.global_rank || group.size() != self.topology.world_size {
            return Err(Error::Parallel(format!(
                "pipeline topology expects group rank {}/{} but received rank {}/{}",
                self.topology.global_rank,
                self.topology.world_size,
                group.rank(),
                group.size()
            )));
        }
        Ok(())
    }
}

fn validate_pipeline_topology(topology: ParallelTopology) -> Result<(), Error> {
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
    topology: ParallelTopology,
    range: Range<usize>,
    model_kind: ModelKind,
    hidden_size: i32,
) -> PipelineStageInfo {
    let stage = topology.pipeline_parallel_rank;
    let last = topology.pipeline_parallel_size - 1;
    PipelineStageInfo {
        topology,
        global_rank: topology.global_rank,
        pipeline_stage: stage,
        pipeline_stages: topology.pipeline_parallel_size,
        is_first: stage == 0,
        is_last: stage == last,
        global_layer_range: range,
        global_expert_count: None,
        local_expert_ids: Vec::new(),
        predecessor_rank: topology
            .pipeline_predecessor()
            .expect("validated topology has valid pipeline predecessor geometry"),
        successor_rank: topology
            .pipeline_successor()
            .expect("validated topology has valid pipeline successor geometry"),
        model_kind,
        hidden_size,
        activation_dtype: Dtype::Float32,
        owned_tensors: Vec::new(),
        local_parameter_bytes: 0,
        planned_owned_parameter_bytes: 0,
        opened_checkpoint_shards: Vec::new(),
        checkpoint_diagnostics: None,
    }
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
    let expected = step.activation_shape(info.hidden_size);
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
    crate::runtime::checkpoint::binding::canonical_checkpoint_name(parameter_name)
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
        eval(
            [&quantized.weight, &quantized.scales]
                .into_iter()
                .chain(quantized.biases.as_ref()),
        )?;
        stream.synchronize()?;
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
    store: &dyn WeightStore,
    bindings: &[crate::runtime::residency::manager::WeightBinding],
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
    store: &dyn WeightStore,
    bindings: &[crate::runtime::residency::manager::WeightBinding],
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
        if module
            .parameters()
            .flatten()
            .keys()
            .any(|name| excluded(name))
        {
            return Err(Error::Quantization(
                "pipeline load-time quantization cannot exclude independent expert parameters"
                    .into(),
            ));
        }
        populate_module_from_dense_arrays_quantized(module, &arrays, quantization, stream)?;
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
        store: &dyn WeightStore,
        bindings: &[crate::runtime::residency::manager::WeightBinding],
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
    fn load_excluding<M: ModuleParameters>(
        &mut self,
        module: &mut M,
        store: &dyn WeightStore,
        bindings: &[crate::runtime::residency::manager::WeightBinding],
        weights_stream: &Stream,
        stream: &Stream,
        excluded: &dyn Fn(&str) -> bool,
    ) -> Result<(), Error> {
        let (bytes, names, dtype) = load_bound_module_excluding(
            module,
            store,
            bindings,
            None,
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
) -> Result<&'a [crate::runtime::residency::manager::WeightBinding], Error> {
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
    store: &dyn WeightStore,
    layout: Option<&crate::runtime::distributed::parallel::LocalModelLayout>,
) -> Result<Vec<crate::runtime::residency::manager::WeightBinding>, Error> {
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
    store: &dyn WeightStore,
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
        &dyn WeightStore,
    ) -> Result<Vec<crate::runtime::residency::manager::WeightBinding>, Error>,
{
    let layer_count = range.len();
    let device_depth = match options {
        PipelineLayerLoadOptions::LayerwiseHost(options) => options.offload.prefetch_depth(),
        PipelineLayerLoadOptions::DenseDiskStream(options) => {
            options.validate()?;
            options.device_lookahead
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
    let mut planned_layer_bytes = 0u64;
    for global_layer in range {
        let layer = make_layer(global_layer, stream)?;
        let bindings = make_bindings(global_layer, &layer, store.as_ref())?;
        let layer_bytes = binding_bytes(&bindings)?;
        planned_layer_bytes = planned_layer_bytes
            .checked_add(layer_bytes)
            .ok_or_else(|| {
                Error::Parallel("pipeline streamed-layer byte total overflowed".into())
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
    }
    let largest = |depth: usize| -> Result<u64, Error> {
        bytes
            .windows(depth)
            .try_fold(0u64, |largest, window| {
                window
                    .iter()
                    .try_fold(0u64, |total, value| total.checked_add(*value))
                    .map(|total| largest.max(total))
            })
            .ok_or_else(|| Error::Parallel("pipeline layer-window byte total overflowed".into()))
    };
    let device_window_bytes = largest(device_depth)?;
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
                if planned_layer_bytes > budget {
                    return Err(Error::Parallel(format!(
                        "pipeline host budget {budget} cannot eagerly hold all {planned_layer_bytes} rank-local layer bytes"
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
                let host_window_bytes = largest(options.host_lookahead)?;
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
                options.host_lookahead.max(options.device_lookahead),
            )?
            .with_eviction_policy(options.eviction_policy)
        }
    };
    let plan = OffloadPlan::new(config, specs)?;
    let residency = ResidencyManager::new_shared(
        Arc::clone(&store),
        plan,
        definitions,
        weights_stream.clone(),
        stream.clone(),
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
            PipelineLayerController::DenseDiskStream(Box::new(DenseStreamController::new(
                &residency,
                options,
                units.len(),
                planned_layer_bytes,
                static_device_bytes,
                [("pipeline_stage".to_string(), units.clone())],
            )?))
        }
    };
    Ok(PipelineLayerStorage {
        store,
        residency,
        controller,
        units,
        execution_offset: 0,
        independent_expert_prefix: None,
        sample_mlx_memory,
        sample_process_memory,
    })
}

/// Loads a pipeline stage using default non-quantizing options.
pub fn load_pipeline_model(
    model_dir: impl AsRef<Path>,
    topology: ParallelTopology,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    load_pipeline_model_with_options(
        model_dir,
        ModelLoadOptions::with_parallel(topology),
        stream,
        weights_stream,
    )
}

/// Loads an executable rank-local Cartesian pipeline stage.
///
/// Llama/Mistral, DeepSeek-V3/R1, Inkling, Kimi Linear, Qwen, Qwen3-VL, GPT-OSS,
/// LFM2, Nemotron-H, Qwen3-Next/Qwen3.5, and Gemma 4 text TP+PP stages, plus
/// DeepSeek-V3/R1, Inkling, Kimi Linear, Qwen, Qwen3-VL-MoE, GPT-OSS, LFM2-MoE,
/// Nemotron-H-MoE, and Qwen3-Next/Qwen3.5-MoE PP+EP stages,
/// support fully resident, host-layerwise, and dense-disk-streamed layers.
/// Non-resident units compose pipeline placement with the authoritative TP
/// semantic layout or EP assignment before residency initialization. Qwen3-MoE,
/// Kimi Linear, Inkling, Qwen3-VL-MoE, GPT-OSS, LFM2-MoE, Nemotron-H-MoE,
/// and Qwen3-Next/Qwen3.5-MoE additionally compose an independent, stage-local
/// expert cache
/// with resident, host-layerwise, or dense-streamed non-expert parameters for
/// PP, TP+PP, PP+EP, and TP+PP+EP. With EP inactive each stage owns all experts
/// for its local layers and executes routes without an expert collective.
pub fn load_pipeline_model_with_options(
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
        let metadata = crate::runtime::checkpoint::load::gguf_metadata(&checkpoint);
        let architecture = pipeline_gguf_architecture(&metadata)?;
        if expert_cache.is_some()
            && !matches!(
                architecture,
                crate::api::GgufArchitecture::DeepSeek2
                    | crate::api::GgufArchitecture::Qwen3Moe
                    | crate::api::GgufArchitecture::Qwen3VlMoe
                    | crate::api::GgufArchitecture::KimiLinear
                    | crate::api::GgufArchitecture::Inkling
                    | crate::api::GgufArchitecture::GptOss
                    | crate::api::GgufArchitecture::Lfm2Moe
                    | crate::api::GgufArchitecture::NemotronHMoe
                    | crate::api::GgufArchitecture::Qwen35Moe
                    | crate::api::GgufArchitecture::Qwen3Next
            )
        {
            return Err(Error::Parallel(format!(
                "pipeline independent expert caching has registered DeepSeek-V3/R1, Qwen3-MoE, Qwen3-VL-MoE, Kimi Linear, Inkling, GPT-OSS, LFM2-MoE, Nemotron-H-MoE, Qwen3-Next-MoE, and Qwen3.5-MoE semantic expert recipes; GGUF architecture {} is not yet registered and no checkpoint payload was materialized",
                architecture.metadata_name()
            )));
        }
        if topology.tensor_parallel_size > 1
            && topology.pipeline_parallel_size > 1
            && topology.expert_parallel_size > 1
            && !matches!(
                architecture,
                crate::api::GgufArchitecture::DeepSeek2
                    | crate::api::GgufArchitecture::Qwen3Moe
                    | crate::api::GgufArchitecture::Qwen3VlMoe
                    | crate::api::GgufArchitecture::KimiLinear
                    | crate::api::GgufArchitecture::Inkling
                    | crate::api::GgufArchitecture::GptOss
                    | crate::api::GgufArchitecture::Lfm2Moe
                    | crate::api::GgufArchitecture::NemotronHMoe
                    | crate::api::GgufArchitecture::Qwen35Moe
                    | crate::api::GgufArchitecture::Qwen3Next
            )
        {
            return Err(Error::Parallel(format!(
                "TP+PP+EP preflight has registered DeepSeek-V3/R1, Qwen3-MoE, Qwen3-VL-MoE, Kimi Linear, Inkling, GPT-OSS, LFM2-MoE, Nemotron-H-MoE, Qwen3-Next-MoE, and Qwen3.5-MoE; GGUF architecture {} has no triple-axis semantic plan and no checkpoint payload was materialized",
                architecture.metadata_name()
            )));
        }
        if topology.expert_parallel_size > 1
            && !matches!(
                architecture,
                crate::api::GgufArchitecture::KimiLinear
                    | crate::api::GgufArchitecture::Inkling
                    | crate::api::GgufArchitecture::DeepSeek2
                    | crate::api::GgufArchitecture::Qwen3Moe
                    | crate::api::GgufArchitecture::Qwen3Vl
                    | crate::api::GgufArchitecture::Qwen3VlMoe
                    | crate::api::GgufArchitecture::GptOss
                    | crate::api::GgufArchitecture::Lfm2Moe
                    | crate::api::GgufArchitecture::NemotronHMoe
                    | crate::api::GgufArchitecture::Qwen35Moe
                    | crate::api::GgufArchitecture::Qwen3Next
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
                crate::api::GgufArchitecture::KimiLinear
                    | crate::api::GgufArchitecture::Inkling
                    | crate::api::GgufArchitecture::DeepSeek2
                    | crate::api::GgufArchitecture::Llama
                    | crate::api::GgufArchitecture::Mistral
                    | crate::api::GgufArchitecture::Qwen2
                    | crate::api::GgufArchitecture::Qwen3
                    | crate::api::GgufArchitecture::Qwen3Moe
                    | crate::api::GgufArchitecture::Qwen3Vl
                    | crate::api::GgufArchitecture::Qwen3VlMoe
                    | crate::api::GgufArchitecture::GptOss
                    | crate::api::GgufArchitecture::Gemma4
                    | crate::api::GgufArchitecture::Lfm2
                    | crate::api::GgufArchitecture::Lfm2Moe
                    | crate::api::GgufArchitecture::NemotronH
                    | crate::api::GgufArchitecture::NemotronHMoe
                    | crate::api::GgufArchitecture::Qwen35
                    | crate::api::GgufArchitecture::Qwen35Moe
                    | crate::api::GgufArchitecture::Qwen3Next
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
        crate::api::structural::validate_gguf(
            architecture,
            &checkpoint,
            &metadata,
            structural_options,
        )
        .into_loader_result()?;
        return match architecture {
            crate::api::GgufArchitecture::Llama | crate::api::GgufArchitecture::Mistral => {
                let prepared = llama::prepare_llama_gguf_checkpoint(
                    &checkpoint,
                    &metadata,
                    None,
                    weights_stream,
                )?;
                let store: SharedWeightStore =
                    Arc::new(GgufWeightStore::new_with_max_mapped_shards(
                        checkpoint,
                        llama::translate_gguf_weight_name,
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
            crate::api::GgufArchitecture::DeepSeek2 => {
                let prepared = deepseek_v3::prepare_gguf_checkpoint(
                    &checkpoint,
                    &metadata,
                    None,
                    weights_stream,
                )?;
                let store: SharedWeightStore =
                    Arc::new(GgufWeightStore::new_with_max_mapped_shards(
                        checkpoint,
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
            crate::api::GgufArchitecture::Gemma4 => {
                let prepared =
                    gemma4::prepare_gemma4_gguf_checkpoint(&checkpoint, &metadata, None)?;
                let store: SharedWeightStore =
                    Arc::new(GgufWeightStore::new_with_max_mapped_shards(
                        checkpoint,
                        gemma4::translate_gguf_weight_name,
                        max_mapped_shards,
                    )?);
                load_gemma_pipeline(
                    GemmaPipelineConfig {
                        args: prepared.args,
                        vision_config: None,
                        image_token_id: None,
                        video_token_id: None,
                        audio_config: None,
                        audio_token_id: None,
                    },
                    store,
                    topology,
                    options.quantization,
                    dense_stream,
                    stream,
                    weights_stream,
                )
            }
            architecture @ (crate::api::GgufArchitecture::Qwen2
            | crate::api::GgufArchitecture::Qwen3
            | crate::api::GgufArchitecture::Qwen3Moe) => {
                let architecture_name = architecture.metadata_name();
                let is_moe = architecture == crate::api::GgufArchitecture::Qwen3Moe;
                let (args, _) = dense_qwen::prepare_gguf_checkpoint(
                    &checkpoint,
                    &metadata,
                    architecture_name,
                    is_moe,
                )?;
                let store: SharedWeightStore =
                    Arc::new(GgufWeightStore::new_with_max_mapped_shards(
                        checkpoint,
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
            crate::api::GgufArchitecture::Qwen3Vl | crate::api::GgufArchitecture::Qwen3VlMoe => {
                let vision_path = qwen3_vl::find_qwen3_vl_mmproj(model_dir)?;
                let vision_checkpoint = GgufCheckpoint::open(vision_path)?;
                let vision_metadata =
                    crate::runtime::checkpoint::load::gguf_metadata(&vision_checkpoint);
                let prepared = qwen3_vl::prepare_qwen3_vl_gguf_checkpoint(
                    &checkpoint,
                    &metadata,
                    &vision_checkpoint,
                    &vision_metadata,
                )?;
                let store = crate::architectures::qwen::vl::layerwise::qwen3_vl_gguf_store(
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
            crate::api::GgufArchitecture::GptOss => {
                let prepared =
                    gpt_oss::prepare_gguf_checkpoint(&checkpoint, &metadata, weights_stream)?;
                let store: SharedWeightStore =
                    Arc::new(GgufWeightStore::new_with_max_mapped_shards(
                        checkpoint,
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
            architecture @ (crate::api::GgufArchitecture::Lfm2
            | crate::api::GgufArchitecture::Lfm2Moe) => {
                let prepared =
                    lfm2::prepare_gguf_checkpoint(&checkpoint, &metadata, weights_stream)?;
                let is_moe = architecture == crate::api::GgufArchitecture::Lfm2Moe;
                let store: SharedWeightStore =
                    Arc::new(GgufWeightStore::new_with_max_mapped_shards(
                        checkpoint,
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
            architecture @ (crate::api::GgufArchitecture::NemotronH
            | crate::api::GgufArchitecture::NemotronHMoe) => {
                let prepared = nemotron_h::prepare_nemotron_h_gguf_checkpoint(
                    &checkpoint,
                    &metadata,
                    weights_stream,
                )?;
                let store: SharedWeightStore =
                    Arc::new(GgufWeightStore::new_with_max_mapped_shards(
                        checkpoint,
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
            crate::api::GgufArchitecture::Qwen35
            | crate::api::GgufArchitecture::Qwen35Moe
            | crate::api::GgufArchitecture::Qwen3Next => {
                let prepared = qwen_hybrid::prepare_qwen35_gguf_checkpoint(
                    &checkpoint,
                    &metadata,
                    weights_stream,
                )?;
                let store: SharedWeightStore =
                    Arc::new(GgufWeightStore::new_with_max_mapped_shards(
                        checkpoint,
                        qwen_hybrid::qwen35_translate_gguf_weight_name,
                        max_mapped_shards,
                    )?);
                load_qwen_hybrid_pipeline(
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
            crate::api::GgufArchitecture::KimiLinear => {
                let prepared = kimi_linear::prepare_gguf_checkpoint(
                    &checkpoint,
                    &metadata,
                    options.quantization,
                    weights_stream,
                )?;
                let store: SharedWeightStore =
                    Arc::new(GgufWeightStore::new_with_max_mapped_shards(
                        checkpoint,
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
            crate::api::GgufArchitecture::Inkling => {
                let prepared =
                    inkling::prepare_gguf_checkpoint_with_mmproj(&checkpoint, &metadata, None)?;
                let store: SharedWeightStore =
                    Arc::new(GgufWeightStore::new_with_max_mapped_shards(
                        checkpoint,
                        inkling::translate_gguf_weight_name,
                        max_mapped_shards,
                    )?);
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
                    | "qwen3_moe"
                    | "qwen3_vl_moe"
                    | "qwen3_vl_moe_text"
                    | "kimi_linear"
                    | "inkling_mm_model"
                    | "gpt_oss"
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
            crate::api::structural::validate_safetensors_load_path(
                ModelKind::Llama,
                model_dir,
                options,
            )?;
            load_llama_pipeline(
                llama::get_llama_model_args(model_dir)?,
                store,
                topology,
                options.quantization,
                dense_stream,
                stream,
                weights_stream,
            )
        }
        Some("deepseek_v3") => {
            crate::api::structural::validate_safetensors_load_path(
                ModelKind::DeepSeekV3,
                model_dir,
                options,
            )?;
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
        Some("gemma4" | "gemma4_text" | "gemma4_unified" | "gemma4_unified_text") => {
            crate::api::structural::validate_safetensors_load_path(
                ModelKind::Gemma4,
                model_dir,
                options,
            )?;
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
                stream,
                weights_stream,
            )
        }
        Some("qwen2" | "qwen3" | "qwen3_moe") => {
            let args = dense_qwen::load_config(model_dir)?;
            crate::api::structural::validate_safetensors_load_path(
                args.model_kind(),
                model_dir,
                options,
            )?;
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
        Some("qwen3_vl" | "qwen3_vl_text" | "qwen3_vl_moe" | "qwen3_vl_moe_text") => {
            let args = qwen3_vl::get_qwen3_vl_model_args(model_dir)?;
            let kind = if args.text_config.is_moe() {
                ModelKind::Qwen3VlMoe
            } else {
                ModelKind::Qwen3Vl
            };
            crate::api::structural::validate_safetensors_load_path(kind, model_dir, options)?;
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
            crate::api::structural::validate_safetensors_load_path(
                ModelKind::GptOss,
                model_dir,
                options,
            )?;
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
            crate::api::structural::validate_safetensors_load_path(
                ModelKind::Lfm2,
                model_dir,
                options,
            )?;
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
            crate::api::structural::validate_safetensors_load_path(
                ModelKind::NemotronH,
                model_dir,
                options,
            )?;
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
            crate::api::structural::validate_safetensors_load_path(
                ModelKind::Qwen3Next,
                model_dir,
                options,
            )?;
            load_qwen_hybrid_pipeline(
                crate::architectures::qwen::hybrid::qwen3_next::get_qwen3_next_model_args(
                    model_dir,
                )?,
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
            crate::api::structural::validate_safetensors_load_path(
                ModelKind::Qwen35,
                model_dir,
                options,
            )?;
            let (args, image_token, video_token, vision) =
                qwen_hybrid::get_qwen3_5_model_args(model_dir)?;
            if image_token.is_some() || video_token.is_some() || vision.is_some() {
                return Err(Error::UnsupportedArchitecture(
                    "Qwen3.5 pipeline execution accepts text-only checkpoints; multimodal ingress must be folded before the decoder pipeline"
                        .into(),
                ));
            }
            load_qwen_hybrid_pipeline(
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
        Some("kimi_linear") => {
            crate::api::structural::validate_safetensors_load_path(
                ModelKind::KimiLinear,
                model_dir,
                options,
            )?;
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
            crate::api::structural::validate_safetensors_load_path(
                ModelKind::Inkling,
                model_dir,
                options,
            )?;
            let args = inkling::get_model_args(model_dir)?;
            if args.audio_config.is_some() || args.vision_config.is_some() {
                return Err(Error::UnsupportedArchitecture(
                    "Inkling pipeline execution accepts text-only checkpoints; image/audio ingress must be folded before the decoder pipeline"
                        .into(),
                ));
            }
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
            "pipeline execution supports Llama-compatible, DeepSeek-V3/R1, Gemma 4 text, Qwen2/Qwen3/Qwen3-MoE, Qwen3-VL/Qwen3-VL-MoE, GPT-OSS, LFM2/LFM2-MoE, Nemotron-H, Kimi Linear, Qwen3-Next/Qwen3.5 text, and Inkling text models, not {model_type}"
        ))),
        None => Err(Error::UnsupportedArchitecture(
            "pipeline model config is missing model_type".into(),
        )),
    }
}

fn pipeline_gguf_architecture(
    metadata: &HashMap<String, GgufMetadataValue>,
) -> Result<crate::api::GgufArchitecture, Error> {
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
    crate::api::GgufArchitecture::resolve(architecture)
}

fn load_llama_pipeline(
    source_args: llama::ModelArgs,
    store: SharedWeightStore,
    topology: ParallelTopology,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    topology.preflight(Some(source_args.attention_schedule.len()), None)?;
    if dense_stream.is_some() && requested_quantization.is_some() {
        return Err(Error::Quantization(
            "load-time quantization is unsupported for non-resident pipeline layers; use checkpoint-native packed weights"
                .into(),
        ));
    }
    let quantize_on_load = requested_quantization
        .map(|requested| {
            crate::runtime::checkpoint::quantization::should_quantize_on_load(
                "Llama pipeline",
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
    }
    let range = topology.layer_range(source_args.attention_schedule.len())?;
    let mut info = base_info(
        topology,
        range.clone(),
        ModelKind::Llama,
        source_args.hidden_size,
    );
    let binding_adapter = crate::architectures::llama::layerwise::LlamaLayerwiseAdapter::new(
        source_args.clone(),
        stream,
    )?;
    let mut stage = LlamaStage::new(target_args.clone(), range, &info, stream)?;
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
                crate::nn::parallel::VocabParallelEmbedding::unloaded(
                    target_args.vocab_size as usize,
                    target_args.hidden_size,
                    target_args.affine_quantization_for("model.embed_tokens.weight"),
                    build,
                    stream,
                )
            })
            .transpose()?;
        stage.parallel_output_embedding =
            (info.is_last && !info.is_first && target_args.tie_word_embeddings)
                .then(|| {
                    crate::nn::parallel::VocabParallelEmbedding::unloaded(
                        target_args.vocab_size as usize,
                        target_args.hidden_size,
                        target_args.affine_quantization_for("model.embed_tokens.weight"),
                        build,
                        stream,
                    )
                })
                .transpose()?;
        stage.parallel_lm_head = (info.is_last && !target_args.tie_word_embeddings)
            .then(|| {
                crate::nn::parallel::VocabParallelLmHead::unloaded(
                    target_args.hidden_size,
                    target_args.vocab_size as usize,
                    target_args.affine_quantization_for("lm_head.weight"),
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
                None,
                stream,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let static_units = pipeline_binding_units(
        &binding_adapter,
        store.as_ref(),
        &selected_pipeline_static_roles([
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
        ]),
    )?;
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
                0,
                global_layer,
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
    let static_device_bytes = loaded.finish(&mut info)?;
    let checkpoint_diagnostics = store.diagnostics()?;
    let materialized_shards = checkpoint_diagnostics.touched_shard_paths.clone();
    if let Some(dense_stream) = dense_stream {
        let streamed_layout = parallel_layout.clone();
        let streamed_adapter = &stage.layer_adapter;
        let dense_layers = build_pipeline_layer_storage(
            Arc::clone(&store),
            stage.range.clone(),
            dense_stream,
            static_device_bytes,
            stream,
            weights_stream,
            |global_layer, stream| {
                streamed_adapter.new_cartesian_layer(
                    0,
                    global_layer,
                    streamed_layout.as_ref(),
                    None,
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
                    None,
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
        args: llama::ModelArgs,
        range: Range<usize>,
        info: &PipelineStageInfo,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let layer_adapter = crate::architectures::llama::layerwise::LlamaLayerwiseAdapter::new(
            args.clone(),
            stream,
        )?;
        let make_embedding = || {
            linear::unloaded_maybe_quantized_embedding(
                args.vocab_size,
                args.hidden_size,
                args.affine_quantization_for("model.embed_tokens.weight"),
                stream,
            )
        };
        let embedding = info.is_first.then(make_embedding).transpose()?;
        let output_embedding = (info.is_last && !info.is_first && args.tie_word_embeddings)
            .then(make_embedding)
            .transpose()?;
        let layers = range
            .clone()
            .map(|layer| llama::TransformerBlock::new_for_layer(&args, layer as i32, stream))
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
                    args.affine_quantization_for("lm_head.weight"),
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
            |global_layer, stream| {
                Ok(llama::TransformerBlock::new_for_layer(
                    args,
                    global_layer as i32,
                    stream,
                )?)
            },
            |global_layer, layer, hidden, cache, stream| match cache {
                PipelineLayerCache::KeyValue {
                    global_layer: cached_layer,
                    cache: PipelineKeyValueCache::Standard(cache),
                    ..
                } if *cached_layer == global_layer => Ok(layer.forward(
                    llama::AttentionInput {
                        x: hidden,
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
                    llama::AttentionInput {
                        x: hidden,
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
                eval([&forwarded])?;
                stream.synchronize()?;
                Ok(forwarded)
            },
        )?;
        if let Some(norm) = &mut self.norm {
            hidden = norm.forward(&hidden, stream)?;
            let sharded = self
                .parallel_lm_head
                .as_mut()
                .ok_or_else(|| {
                    Error::Parallel("last DeepSeek TP+PP stage has no head shard".into())
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
                eval([&forwarded])?;
                stream.synchronize()?;
                Ok(forwarded)
            },
        )?;
        if let Some(norm) = &mut self.norm {
            hidden = norm.forward(&hidden, stream)?;
            Ok(PipelineStageOutput::Logits(
                self.lm_head
                    .as_mut()
                    .expect("last DeepSeek PP+EP stage head")
                    .forward(&hidden, stream)?,
            ))
        } else {
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
                layer_adapter.new_cartesian_layer(
                    0,
                    global_layer,
                    parallel_layout.as_ref(),
                    None,
                    stream,
                )
            },
            |global_layer, layer, hidden, cache, stream| {
                let forwarded = match cache {
                    PipelineLayerCache::KeyValue {
                        global_layer: cached_layer,
                        cache: PipelineKeyValueCache::Standard(cache),
                        ..
                    } if *cached_layer == global_layer => layer.forward_tensor_parallel(
                        hidden,
                        mask,
                        Some(cache),
                        allow_sliding_prefill,
                        group,
                        stream,
                    )?,
                    PipelineLayerCache::KeyValue {
                        global_layer: cached_layer,
                        cache: PipelineKeyValueCache::Paged(cache),
                        ..
                    } if *cached_layer == global_layer => layer.forward_tensor_parallel(
                        hidden,
                        mask,
                        Some(cache),
                        allow_sliding_prefill,
                        group,
                        stream,
                    )?,
                    _ => {
                        return Err(Error::Parallel(format!(
                            "Llama TP+PP cache does not match global layer {global_layer}"
                        )))
                    }
                };
                eval([&forwarded])?;
                stream.synchronize()?;
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
    topology: ParallelTopology,
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
            crate::runtime::checkpoint::quantization::should_quantize_on_load(
                "dense-Qwen pipeline",
                source_args.weight_quantization(),
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    if dense_stream.is_some() && quantize_on_load.is_some() {
        return Err(Error::Quantization(
            "load-time quantization is unsupported for non-resident dense-Qwen pipeline layers; use checkpoint-native packed weights"
                .into(),
        ));
    }
    if expert_cache_options.is_some() && quantize_on_load.is_some() {
        return Err(Error::Quantization(
            "load-time quantization is unsupported for independently cached pipeline experts; use checkpoint-native packed expert weights"
                .into(),
        ));
    }
    let mut target_args = source_args.clone();
    if let Some(quantization) = quantize_on_load {
        target_args.quantization = Some(quantization);
        target_args.quantization_config = None;
        target_args.quantized_weight_configs = None;
    }
    let range = topology.layer_range(source_args.attention_schedule.len())?;
    let mut info = base_info(
        topology,
        range.clone(),
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
                crate::nn::parallel::VocabParallelEmbedding::unloaded(
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
                    crate::nn::parallel::VocabParallelEmbedding::unloaded(
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
                crate::nn::parallel::VocabParallelLmHead::unloaded(
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
    let static_units = pipeline_binding_units(
        &binding_adapter,
        store.as_ref(),
        &selected_pipeline_static_roles([
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
        ]),
    )?;
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
        let cache = ExpertCache::new_shared(
            Arc::clone(&store),
            entries,
            options,
            weights_stream.clone(),
            stream.clone(),
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
    let checkpoint_diagnostics = store.diagnostics()?;
    let materialized_shards = checkpoint_diagnostics.touched_shard_paths.clone();
    info.opened_checkpoint_shards = materialized_shards;
    info.checkpoint_diagnostics = Some(checkpoint_diagnostics);
    PipelineModel::from_adapter(topology, info, PipelineStage(stage))
}

#[allow(clippy::too_many_arguments)]
fn load_qwen3_vl_pipeline(
    source_args: qwen3_vl::ModelArgs,
    store: SharedWeightStore,
    topology: ParallelTopology,
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
            crate::runtime::checkpoint::quantization::should_quantize_on_load(
                "Qwen3-VL pipeline",
                source_quantization,
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    if dense_stream.is_some() && quantize_on_load.is_some() {
        return Err(Error::Quantization(
            "load-time quantization is unsupported for non-resident Qwen3-VL pipeline layers; use checkpoint-native packed weights"
                .into(),
        ));
    }
    if expert_cache_options.is_some() && quantize_on_load.is_some() {
        return Err(Error::Quantization(
            "load-time quantization is unsupported for independently cached Qwen3-VL pipeline experts; use checkpoint-native packed expert weights"
                .into(),
        ));
    }
    let mut target_args = source_args.clone();
    if let Some(quantization) = quantize_on_load {
        target_args.text_config.quantization = Some(quantization);
        target_args.text_config.quantization_config = None;
        target_args.text_config.quantized_weight_configs = None;
    }
    let range = topology.layer_range(source_args.text_config.num_hidden_layers as usize)?;
    let kind = if source_args.text_config.is_moe() {
        ModelKind::Qwen3VlMoe
    } else {
        ModelKind::Qwen3Vl
    };
    let mut info = base_info(
        topology,
        range.clone(),
        kind,
        source_args.text_config.hidden_size,
    );
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
                crate::nn::parallel::VocabParallelEmbedding::unloaded(
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
    if info.is_first {
        stage.vision_layers = (0..target_args.vision_config.layer_count())
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
    let static_units = pipeline_binding_units(
        &binding_adapter,
        store.as_ref(),
        &selected_pipeline_static_roles([
            ("vision", info.is_first),
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
        ]),
    )?;
    let mut loaded = PipelineLoadAccumulator::new("Qwen3-VL");
    if info.is_first {
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
        loaded.load(
            stage.layer_adapter.vision_mut(),
            store.as_ref(),
            &bindings,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
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
    if info.is_first {
        for (index, layer) in stage.vision_layers.iter_mut().enumerate() {
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
    let checkpoint_diagnostics = store.diagnostics()?;
    let materialized_shards = checkpoint_diagnostics.touched_shard_paths.clone();
    if let Some(options) = dense_stream {
        let streamed_layout = parallel_layout.clone();
        let streamed_assignment = stage.expert_assignment.clone();
        let streamed_adapter = &stage.layer_adapter;
        let dense_layers = build_pipeline_layer_storage(
            Arc::clone(&store),
            stage.range.clone(),
            options,
            static_bytes,
            stream,
            weights_stream,
            |global_layer, stream| {
                streamed_adapter.new_cartesian_layer(
                    1,
                    global_layer,
                    streamed_layout.as_ref(),
                    streamed_assignment.as_ref(),
                    stream,
                )
            },
            |global_layer, layer, store| {
                binding_adapter.cartesian_layer_bindings(
                    1,
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
        let cache = ExpertCache::new_shared(
            Arc::clone(&store),
            entries,
            options,
            weights_stream.clone(),
            stream.clone(),
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
                eval([&forwarded])?;
                stream.synchronize()?;
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
                eval([&forwarded])?;
                stream.synchronize()?;
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
    layout: Option<&crate::runtime::distributed::parallel::LocalModelLayout>,
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
    let execute = |routes: &crate::runtime::distributed::expert::DispatchedRoutes,
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
    let execute = |routes: &crate::runtime::distributed::expert::DispatchedRoutes,
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
    let execute = |routes: &crate::runtime::distributed::expert::DispatchedRoutes,
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
    let execute = |routes: &crate::runtime::distributed::expert::DispatchedRoutes,
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
    let execute = |routes: &crate::runtime::distributed::expert::DispatchedRoutes,
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
    let execute = |routes: &crate::runtime::distributed::expert::DispatchedRoutes,
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
    let execute = |routes: &crate::runtime::distributed::expert::DispatchedRoutes,
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
    let execute = |routes: &crate::runtime::distributed::expert::DispatchedRoutes,
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
    topology: ParallelTopology,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    expert_cache_options: Option<ExpertCacheLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    let binding_adapter = if expert_cache_options.is_some() {
        crate::architectures::gpt_oss::layerwise::GptOssLayerwiseAdapter::new_external_experts(
            source_args.clone(),
            stream,
        )?
    } else {
        crate::architectures::gpt_oss::layerwise::GptOssLayerwiseAdapter::new(
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
            crate::runtime::checkpoint::quantization::should_quantize_on_load(
                "GPT-OSS pipeline dense matrices",
                source_args.quantization,
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    if dense_stream.is_some() && quantize_on_load.is_some() {
        return Err(Error::Quantization(
            "load-time quantization is unsupported for non-resident GPT-OSS pipeline layers; use checkpoint-native packed weights"
                .into(),
        ));
    }
    if expert_cache_options.is_some() && quantize_on_load.is_some() {
        return Err(Error::Quantization(
            "load-time quantization is unsupported for independently cached GPT-OSS pipeline experts; use checkpoint-native MXFP4 weights"
                .into(),
        ));
    }
    let mut target_args = source_args.clone();
    if let Some(quantization) = quantize_on_load {
        target_args.quantization = Some(quantization);
        target_args.quantized_weight_configs = None;
    }
    let range = topology.layer_range(source_args.attention_schedule.len())?;
    let mut info = base_info(
        topology,
        range.clone(),
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
                crate::nn::parallel::VocabParallelEmbedding::unloaded(
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
                crate::nn::parallel::VocabParallelLmHead::unloaded(
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
    let static_units = pipeline_binding_units(
        &binding_adapter,
        store.as_ref(),
        &selected_pipeline_static_roles([
            (
                "embedding",
                stage.embedding.is_some() || stage.parallel_embedding.is_some(),
            ),
            ("norm", stage.norm.is_some()),
            (
                "output",
                stage.lm_head.is_some() || stage.parallel_lm_head.is_some(),
            ),
        ]),
    )?;
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
    let checkpoint_diagnostics = store.diagnostics()?;
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
        let entries = crate::architectures::gpt_oss::layerwise::gpt_oss_expert_catalog_cartesian(
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
        let cache = ExpertCache::new_shared(
            Arc::clone(&store),
            entries,
            options,
            weights_stream.clone(),
            stream.clone(),
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
            crate::architectures::gpt_oss::layerwise::GptOssLayerwiseAdapter::new_external_experts(
                args.clone(),
                stream,
            )?
        } else {
            crate::architectures::gpt_oss::layerwise::GptOssLayerwiseAdapter::new(
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
                eval([&forwarded])?;
                stream.synchronize()?;
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
                eval([&forwarded])?;
                stream.synchronize()?;
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
    topology: ParallelTopology,
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
            crate::runtime::checkpoint::quantization::should_quantize_on_load(
                "LFM2 pipeline",
                source_args.weight_quantization,
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    if dense_stream.is_some() && quantize_on_load.is_some() {
        return Err(Error::Quantization(
            "load-time quantization is unsupported for non-resident LFM2 pipeline layers; use checkpoint-native packed weights"
                .into(),
        ));
    }
    if expert_cache_options.is_some() && quantize_on_load.is_some() {
        return Err(Error::Quantization(
            "load-time quantization is unsupported for independently cached LFM2 pipeline experts; use checkpoint-native packed weights"
                .into(),
        ));
    }
    let mut target_args = source_args.clone();
    if let Some(quantization) = quantize_on_load {
        target_args.weight_quantization = Some(quantization);
        target_args.quantized_weight_configs = None;
    }
    let range = topology.layer_range(source_args.layer_schedule.len())?;
    let mut info = base_info(
        topology,
        range.clone(),
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
                crate::nn::parallel::VocabParallelEmbedding::unloaded(
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
                    crate::nn::parallel::VocabParallelEmbedding::unloaded(
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
                crate::nn::parallel::VocabParallelLmHead::unloaded(
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
    let static_units = pipeline_binding_units(
        &binding_adapter,
        store.as_ref(),
        &selected_pipeline_static_roles([
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
        ]),
    )?;
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
    let checkpoint_diagnostics = store.diagnostics()?;
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
        let entries = crate::architectures::lfm2::layerwise::lfm2_expert_catalog(
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
            let cache = ExpertCache::new_shared(
                Arc::clone(&store),
                entries,
                options,
                weights_stream.clone(),
                stream.clone(),
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
                let mut local = crate::nn::convolution::CausalConv1dCache {
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
                let mut local = crate::nn::convolution::CausalConv1dCache {
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
                let mut local = crate::nn::convolution::CausalConv1dCache {
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
                let mut local = crate::nn::convolution::CausalConv1dCache {
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
                let mut local = crate::nn::convolution::CausalConv1dCache {
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
                eval([&forwarded])?;
                stream.synchronize()?;
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
                eval([&forwarded])?;
                stream.synchronize()?;
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
                let parts = [crate::api::input::InputPart::text_token_ids(tokens)];
                Some(self.prepare_multimodal_ingress(
                    crate::api::input::ModelInput::new(&parts),
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
                    } if *cached == global_layer && !policy.key_value.owns_state() => layer
                        .forward_tensor_parallel(
                            hidden,
                            mask,
                            Option::<&mut ConcatKeyValueCache>::None,
                            offset,
                            per_layer_input.as_ref(),
                            &mut shared_kv,
                            group,
                            stream,
                        )?,
                    PipelineLayerCache::KeyValue {
                        global_layer: cached,
                        cache: PipelineKeyValueCache::Standard(cache),
                        ..
                    } if *cached == global_layer && policy.key_value.owns_state() => layer
                        .forward_tensor_parallel(
                            hidden,
                            mask,
                            Some(cache),
                            offset,
                            per_layer_input.as_ref(),
                            &mut shared_kv,
                            group,
                            stream,
                        )?,
                    PipelineLayerCache::KeyValue {
                        global_layer: cached,
                        cache: PipelineKeyValueCache::Paged(cache),
                        ..
                    } if *cached == global_layer && policy.key_value.owns_state() => layer
                        .forward_tensor_parallel(
                            hidden,
                            mask,
                            Some(cache),
                            offset,
                            per_layer_input.as_ref(),
                            &mut shared_kv,
                            group,
                            stream,
                        )?,
                    _ => {
                        return Err(Error::Parallel(format!(
                            "Gemma TP+PP cache does not match global layer {global_layer}"
                        )))
                    }
                };
                eval([&forwarded])?;
                stream.synchronize()?;
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
                        crate::runtime::distributed::topology::balanced_contiguous_range(
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
    topology: ParallelTopology,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    expert_cache_options: Option<ExpertCacheLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    let mut binding_adapter = if expert_cache_options.is_some() {
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
    let existing = source_args.quantization.map(WeightQuantization::Affine);
    let quantize_on_load = requested_quantization
        .map(|requested| {
            crate::runtime::checkpoint::quantization::should_quantize_on_load(
                "Nemotron-H pipeline",
                existing,
                requested,
            )
            .and_then(|required| match (required, requested) {
                (false, _) => Ok(None),
                (true, WeightQuantization::Affine(affine)) => Ok(Some(affine)),
                (true, _) => Err(Error::Quantization(
                    "Nemotron-H pipeline load-time quantization supports MLX affine weights".into(),
                )),
            })
        })
        .transpose()?
        .flatten();
    if dense_stream.is_some() && quantize_on_load.is_some() {
        return Err(Error::Quantization(
            "load-time quantization is unsupported for non-resident Nemotron-H pipeline layers; use checkpoint-native packed weights"
                .into(),
        ));
    }
    if expert_cache_options.is_some() && quantize_on_load.is_some() {
        return Err(Error::Quantization(
            "load-time quantization is unsupported for independently cached Nemotron-H pipeline experts; use checkpoint-native packed weights"
                .into(),
        ));
    }
    let mut target_args = source_args.clone();
    if let Some(affine) = quantize_on_load {
        target_args.quantization = Some(affine);
        target_args.quantized_weights = None;
        target_args.quantized_weight_configs = None;
    }
    let range = topology.layer_range(source_args.num_hidden_layers as usize)?;
    let mut info = base_info(
        topology,
        range.clone(),
        ModelKind::NemotronH,
        source_args.hidden_size,
    );
    let mut stage = NemotronHStage::new(
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
        stage
            .layer_adapter
            .configure_cartesian_layout(build, &layout, stream)?;
        stage.parallel_geometry = stage.layer_adapter.parallel_geometry().map(<[_]>::to_vec);
        stage.parallel_embedding = info
            .is_first
            .then(|| {
                crate::nn::parallel::VocabParallelEmbedding::unloaded(
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
                    crate::nn::parallel::VocabParallelEmbedding::unloaded(
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
                crate::nn::parallel::VocabParallelLmHead::unloaded(
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
    let requested = quantize_on_load.map(WeightQuantization::Affine);
    let static_units = pipeline_binding_units(
        &binding_adapter,
        store.as_ref(),
        &selected_pipeline_static_roles([
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
        ]),
    )?;
    let mut loaded = PipelineLoadAccumulator::new("Nemotron-H");
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
            if expert_cache_options.is_some() {
                loaded.load_excluding(
                    layer,
                    store.as_ref(),
                    &bindings,
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
    if let Some(options) = expert_cache_options {
        let entries =
            crate::architectures::nemotron_h::layerwise::nemotron_h_expert_catalog_for_layers(
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
            let cache = ExpertCache::new_shared(
                Arc::clone(&store),
                entries,
                options,
                weights_stream.clone(),
                stream.clone(),
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
    let checkpoint_diagnostics = store.diagnostics()?;
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
                eval([&forwarded])?;
                stream.synchronize()?;
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
                            "last tied Nemotron-H TP+PP stage has no embedding shard".into(),
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
                eval([&forwarded])?;
                stream.synchronize()?;
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
                        .expect("last tied Nemotron-H PP+EP stage output embedding"),
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
fn load_qwen_hybrid_pipeline(
    source_args: qwen_hybrid::ModelArgs,
    store: SharedWeightStore,
    topology: ParallelTopology,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    expert_cache_options: Option<ExpertCacheLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    let mut binding_adapter = if expert_cache_options.is_some() {
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
            crate::runtime::checkpoint::quantization::should_quantize_on_load(
                "Qwen hybrid pipeline",
                source_args.quantization,
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    if (dense_stream.is_some() || expert_cache_options.is_some()) && quantize_on_load.is_some() {
        return Err(Error::Quantization(
            "load-time quantization is unsupported for non-resident Qwen hybrid pipeline weights; use checkpoint-native packed weights"
                .into(),
        ));
    }
    let mut target_args = source_args.clone();
    if let Some(quantization) = quantize_on_load {
        target_args.quantization = Some(quantization);
        target_args.quantization_config = None;
        target_args.quantized_weight_configs = None;
    }
    let range = topology.layer_range(source_args.num_hidden_layers as usize)?;
    let kind = if source_args.model_type == "qwen3_next" {
        ModelKind::Qwen3Next
    } else {
        ModelKind::Qwen35
    };
    let mut info = base_info(topology, range.clone(), kind, source_args.hidden_size);
    let mut stage = QwenHybridStage::new(
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
        binding_adapter.configure_cartesian_layout(build, &layout, stream)?;
        stage
            .layer_adapter
            .configure_cartesian_layout(build, &layout, stream)?;
        stage.parallel_geometry = stage.layer_adapter.parallel_geometry().map(<[_]>::to_vec);
        stage.parallel_embedding = info
            .is_first
            .then(|| {
                crate::nn::parallel::VocabParallelEmbedding::unloaded(
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
                    crate::nn::parallel::VocabParallelEmbedding::unloaded(
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
                crate::nn::parallel::VocabParallelLmHead::unloaded(
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
    stage.layers = stage
        .range
        .clone()
        .map(|global_layer| {
            match stage.layer_adapter.new_cartesian_layer(
                0,
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
    let static_units = pipeline_binding_units(
        &binding_adapter,
        store.as_ref(),
        &selected_pipeline_static_roles([
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
        ]),
    )?;
    let mut loaded = PipelineLoadAccumulator::new("Qwen hybrid");
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
            let descriptor = QwenHybridLayer::Text(Box::new(layer.clone()));
            let bindings = binding_adapter.cartesian_layer_bindings(
                0,
                global_layer,
                &descriptor,
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
        stage.dense_layers = Some(build_pipeline_layer_storage(
            Arc::clone(&store),
            stage.range.clone(),
            options,
            static_bytes,
            stream,
            weights_stream,
            |global_layer, stream| match streamed_adapter.new_cartesian_layer(
                0,
                global_layer,
                streamed_layout.as_ref(),
                streamed_assignment.as_ref(),
                stream,
            )? {
                QwenHybridLayer::Text(block) => Ok(*block),
                QwenHybridLayer::Vision(_) => Err(Error::Parallel(
                    "Qwen hybrid pipeline received a streamed vision unit".into(),
                )),
            },
            |global_layer, layer, store| {
                let descriptor = QwenHybridLayer::Text(Box::new(layer.clone()));
                binding_adapter.cartesian_layer_bindings(
                    0,
                    global_layer,
                    &descriptor,
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
                Error::Parallel("Qwen hybrid pipeline planned bytes overflowed".into())
            })?;
    } else {
        info.planned_owned_parameter_bytes = static_bytes;
    }
    if let Some(options) = expert_cache_options {
        let entries =
            crate::architectures::qwen::hybrid::layerwise::qwen_hybrid_expert_catalog_for_layers(
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
            let cache = ExpertCache::new_shared(
                Arc::clone(&store),
                entries,
                options,
                weights_stream.clone(),
                stream.clone(),
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
    let checkpoint_diagnostics = store.diagnostics()?;
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
    fn new(
        args: qwen_hybrid::ModelArgs,
        range: Range<usize>,
        info: &PipelineStageInfo,
        external_experts: bool,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let layer_adapter = if external_experts {
            QwenHybridLayerwiseAdapter::new_text_external_experts(args.clone(), stream)?
        } else {
            QwenHybridLayerwiseAdapter::new_text(args.clone(), stream)?
        };
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
                0,
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
                eval([&forwarded])?;
                stream.synchronize()?;
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
                            "last tied Qwen hybrid TP+PP stage has no embedding shard".into(),
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
                0,
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
                eval([&forwarded])?;
                stream.synchronize()?;
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
                        .expect("last tied Qwen hybrid PP+EP stage output embedding"),
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
fn load_kimi_linear_pipeline(
    source_args: kimi_linear::ModelArgs,
    store: SharedWeightStore,
    topology: ParallelTopology,
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
            crate::runtime::checkpoint::quantization::should_quantize_on_load(
                "Kimi Linear pipeline",
                source_args.quantization,
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    if (dense_stream.is_some() || expert_cache_options.is_some()) && quantize_on_load.is_some() {
        return Err(Error::Quantization(
            "load-time quantization is unsupported for non-resident Kimi Linear pipeline layers; use checkpoint-native packed weights"
                .into(),
        ));
    }
    let mut target_args = source_args.clone();
    if let Some(quantization) = quantize_on_load {
        target_args.quantization = Some(quantization);
        target_args.quantized_weight_configs = None;
    }
    let range = topology.layer_range(source_args.num_hidden_layers as usize)?;
    let mut info = base_info(
        topology,
        range.clone(),
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
                crate::nn::parallel::VocabParallelEmbedding::unloaded(
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
                    crate::nn::parallel::VocabParallelEmbedding::unloaded(
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
                crate::nn::parallel::VocabParallelLmHead::unloaded(
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
    let static_units = pipeline_binding_units(
        &binding_adapter,
        store.as_ref(),
        &selected_pipeline_static_roles([
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
        ]),
    )?;
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
    let checkpoint_diagnostics = store.diagnostics()?;
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
        let entries = crate::architectures::kimi_linear::layerwise::kimi_expert_catalog_for_layers(
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
            let cache = ExpertCache::new_shared(
                Arc::clone(&store),
                entries,
                options,
                weights_stream.clone(),
                stream.clone(),
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
                q_conv: crate::nn::convolution::CausalConv1dCache {
                    state: slots[0].value.take(),
                    offset,
                },
                k_conv: crate::nn::convolution::CausalConv1dCache {
                    state: slots[1].value.take(),
                    offset,
                },
                v_conv: crate::nn::convolution::CausalConv1dCache {
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
                    q_conv: crate::nn::convolution::CausalConv1dCache {
                        state: slots[0].value.take(),
                        offset,
                    },
                    k_conv: crate::nn::convolution::CausalConv1dCache {
                        state: slots[1].value.take(),
                        offset,
                    },
                    v_conv: crate::nn::convolution::CausalConv1dCache {
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
                eval([&forwarded])?;
                stream.synchronize()?;
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
                eval([&forwarded])?;
                stream.synchronize()?;
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
    topology: ParallelTopology,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    expert_cache_options: Option<ExpertCacheLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    if requested_quantization.is_some() {
        return Err(Error::Quantization(
            "Inkling pipeline load-time requantization is unsupported; use checkpoint-native encodings"
                .into(),
        ));
    }
    let binding_adapter = if expert_cache_options.is_some() {
        InklingLayerwiseAdapter::new_external_experts(args.clone(), stream)?
    } else {
        InklingLayerwiseAdapter::new(args.clone(), stream)?
    };
    let expert_assignment = binding_adapter.expert_parallel_assignment(topology)?;
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
        ModelKind::Inkling,
        args.text_config.hidden_size,
    );
    let mut stage = InklingStage::new(
        args.clone(),
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
                crate::nn::parallel::VocabParallelEmbedding::unloaded_with_dtype(
                    args.text_config.vocab_size as usize,
                    args.text_config.hidden_size,
                    args.text_config
                        .weight_quantization_for("model.embed_tokens.weight"),
                    args.text_config.weight_dtype(),
                    build,
                    stream,
                )
            })
            .transpose()?;
        stage.parallel_lm_head = info
            .is_last
            .then(|| {
                crate::nn::parallel::VocabParallelLmHead::unloaded_with_dtype(
                    args.text_config.hidden_size,
                    args.text_config.vocab_size as usize,
                    args.text_config.weight_quantization_for("lm_head.weight"),
                    args.text_config.weight_dtype(),
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
        .map(|layer| {
            layer.and_then(|layer| match layer {
                InklingLayer::Text(layer) => Ok(*layer),
                InklingLayer::Vision(_) => Err(Error::Parallel(
                    "Inkling text pipeline constructed a vision layer".into(),
                )),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let static_units = pipeline_binding_units(
        &stage.layer_adapter,
        store.as_ref(),
        &selected_pipeline_static_roles([
            (
                "embedding",
                stage.embedding.is_some() || stage.parallel_embedding.is_some(),
            ),
            ("embed_norm", stage.embed_norm.is_some()),
            ("norm", stage.norm.is_some()),
            (
                "output",
                stage.lm_head.is_some() || stage.parallel_lm_head.is_some(),
            ),
        ]),
    )?;
    let mut loaded = PipelineLoadAccumulator::new("Inkling");
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
            None,
            weights_stream,
            stream,
        )?;
    } else if let Some(module) = &mut stage.embedding {
        loaded.load(
            module,
            store.as_ref(),
            pipeline_static_bindings(&static_units, "embedding")?,
            None,
            weights_stream,
            stream,
        )?;
    }
    if let Some(module) = &mut stage.embed_norm {
        loaded.load(
            module,
            store.as_ref(),
            pipeline_static_bindings(&static_units, "embed_norm")?,
            None,
            weights_stream,
            stream,
        )?;
    }
    if let Some(module) = &mut stage.norm {
        loaded.load(
            module,
            store.as_ref(),
            pipeline_static_bindings(&static_units, "norm")?,
            None,
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
            None,
            weights_stream,
            stream,
        )?;
    } else if let Some(module) = &mut stage.lm_head {
        loaded.load(
            module,
            store.as_ref(),
            pipeline_static_bindings(&static_units, "output")?,
            None,
            weights_stream,
            stream,
        )?;
    }
    if dense_stream.is_none() {
        for (global_layer, layer) in stage.range.clone().zip(&mut stage.layers) {
            let runtime_layer = InklingLayer::Text(Box::new(layer.clone()));
            let bindings = stage.layer_adapter.cartesian_layer_bindings(
                0,
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
                    weights_stream,
                    stream,
                    &|name| name.starts_with("moe.experts."),
                )?;
            } else {
                loaded.load(
                    layer,
                    store.as_ref(),
                    &bindings,
                    None,
                    weights_stream,
                    stream,
                )?;
            }
        }
    }
    let static_bytes = loaded.finish_with_default(&mut info, args.text_config.weight_dtype())?;
    let checkpoint_diagnostics = store.diagnostics()?;
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
            stream,
            weights_stream,
            |global_layer, stream| {
                streamed_adapter
                    .new_cartesian_layer(
                        0,
                        global_layer,
                        streamed_layout.as_ref(),
                        streamed_assignment.as_ref(),
                        stream,
                    )
                    .and_then(|layer| match layer {
                        InklingLayer::Text(layer) => Ok(*layer),
                        InklingLayer::Vision(_) => Err(Error::Parallel(
                            "Inkling text pipeline constructed a vision layer".into(),
                        )),
                    })
            },
            |global_layer, layer, store| {
                let runtime_layer = InklingLayer::Text(Box::new(layer.clone()));
                streamed_adapter.cartesian_layer_bindings(
                    0,
                    global_layer,
                    &runtime_layer,
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
        let entries = crate::architectures::inkling::layerwise::inkling_expert_catalog(
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
            let cache = ExpertCache::new_shared(
                Arc::clone(&store),
                entries,
                options,
                weights_stream.clone(),
                stream.clone(),
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
        crate::nn::convolution::CausalConv1dCache {
            state: slots[0].value.take(),
            offset,
        },
        crate::nn::convolution::CausalConv1dCache {
            state: slots[1].value.take(),
            offset,
        },
        crate::nn::convolution::CausalConv1dCache {
            state: slots[2].value.take(),
            offset,
        },
        crate::nn::convolution::CausalConv1dCache {
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
    fn new(
        args: inkling::ModelArgs,
        range: Range<usize>,
        info: &PipelineStageInfo,
        external_experts: bool,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let layer_adapter = if external_experts {
            InklingLayerwiseAdapter::new_external_experts(args.clone(), stream)?
        } else {
            InklingLayerwiseAdapter::new(args.clone(), stream)?
        };
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
            embedding: info.is_first.then_some(embed_tokens),
            embed_norm: info.is_first.then_some(embed_norm),
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
            crate::nn::convolution::CausalConv1dCache {
                state: slots[0].value.take(),
                offset,
            },
            crate::nn::convolution::CausalConv1dCache {
                state: slots[1].value.take(),
                offset,
            },
            crate::nn::convolution::CausalConv1dCache {
                state: slots[2].value.take(),
                offset,
            },
            crate::nn::convolution::CausalConv1dCache {
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
                let embedded = self
                    .parallel_embedding
                    .as_mut()
                    .ok_or_else(|| {
                        Error::Parallel("first Inkling TP+PP stage has no embedding shard".into())
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
                        0,
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
                eval([&forwarded])?;
                stream.synchronize()?;
                Ok(forwarded)
            },
        )?;
        if let Some(norm) = &mut self.norm {
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
            Ok(PipelineStageOutput::Logits(logits))
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
                    .new_cartesian_layer(0, global_layer, None, Some(&expert_assignment), stream)
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
                eval([&forwarded])?;
                stream.synchronize()?;
                Ok(forwarded)
            },
        )?;
        if let Some(norm) = &mut self.norm {
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
            Ok(PipelineStageOutput::Logits(logits))
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
    vision_config: Option<crate::api::gemma4_vision::Gemma4VisionConfig>,
    image_token_id: Option<i32>,
    video_token_id: Option<i32>,
    audio_config: Option<crate::api::gemma4_audio::Gemma4AudioConfig>,
    audio_token_id: Option<i32>,
}

fn load_gemma_pipeline(
    source: GemmaPipelineConfig,
    store: SharedWeightStore,
    topology: ParallelTopology,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
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
    topology.preflight(Some(source_args.layer_schedule.len()), None)?;
    let quantize_on_load = requested_quantization
        .map(|requested| {
            crate::runtime::checkpoint::quantization::should_quantize_on_load(
                "Gemma pipeline",
                source_args.weight_quantization(),
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    if dense_stream.is_some() && quantize_on_load.is_some() {
        return Err(Error::Quantization(
            "load-time quantization is unsupported for non-resident Gemma pipeline layers; use checkpoint-native packed weights"
                .into(),
        ));
    }
    let mut target_args = source_args.clone();
    if let Some(quantization) = quantize_on_load {
        target_args.quantized = true;
        target_args.weight_quantization = Some(quantization);
        target_args.quantization_group_size = quantization.group_size();
        target_args.quantization_bits = quantization.bits();
        target_args.quantized_weights = None;
        target_args.quantized_weight_configs = None;
    }
    let ranges = gemma_pipeline_ranges(&source_args, topology.pipeline_parallel_size)?;
    let range = ranges
        .get(topology.pipeline_parallel_rank)
        .cloned()
        .ok_or_else(|| Error::Parallel("Gemma pipeline rank has no planned layer range".into()))?;
    let mut info = base_info(
        topology,
        range.clone(),
        ModelKind::Gemma4,
        source_args.hidden_size,
    );
    let mut stage = GemmaStage::new(
        target_args.clone(),
        vision_config.clone(),
        image_token_id,
        video_token_id,
        audio_config.clone(),
        audio_token_id,
        range,
        &info,
        stream,
    )?;
    let binding_adapter = Gemma4LayerwiseAdapter::new_pipeline(
        source_args.clone(),
        vision_config,
        image_token_id,
        video_token_id,
        audio_config,
        audio_token_id,
        stream,
    )?;
    let parallel_layout = if topology.tensor_parallel_size > 1 {
        let build = ParallelBuildContext::new(topology, ShardingPolicy::Require);
        let mut planner = build.planner();
        binding_adapter.register_parallel_parameters(build, &mut planner, stream)?;
        let (_, layout) = planner.finish()?;
        stage
            .layer_adapter
            .configure_parallel_static(build, &layout, stream)?;

        let vocabulary = crate::runtime::distributed::topology::balanced_contiguous_range(
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
            let range = crate::runtime::distributed::topology::balanced_contiguous_range(
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
                Some(crate::nn::parallel::ParallelLinear::unloaded(
                    target_args.hidden_size,
                    target_args.num_hidden_layers * target_args.hidden_size_per_layer_input,
                    false,
                    target_args
                        .quantization_for("model.language_model.per_layer_model_projection.weight"),
                    crate::nn::parallel::LinearParallelism::Column,
                    build,
                    stream,
                )?);
        }
        stage.parallel_lm_head = (info.is_last && !target_args.tie_word_embeddings)
            .then(|| {
                crate::nn::parallel::VocabParallelLmHead::unloaded(
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
                stream,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let static_units = pipeline_binding_units(
        &binding_adapter,
        store.as_ref(),
        &selected_pipeline_static_roles([
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
            (
                "vision",
                info.is_first && stage.layer_adapter.pipeline_static_mut("vision").is_some(),
            ),
            (
                "vision_embed",
                info.is_first
                    && stage
                        .layer_adapter
                        .pipeline_static_mut("vision_embed")
                        .is_some(),
            ),
            (
                "audio",
                info.is_first && stage.layer_adapter.pipeline_static_mut("audio").is_some(),
            ),
            (
                "audio_embed",
                info.is_first
                    && stage
                        .layer_adapter
                        .pipeline_static_mut("audio_embed")
                        .is_some(),
            ),
        ]),
    )?;
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
    let checkpoint_diagnostics = store.diagnostics()?;
    let materialized_shards = checkpoint_diagnostics.touched_shard_paths.clone();
    if let Some(options) = dense_stream {
        let streamed_layout = parallel_layout.clone();
        let streamed_adapter = &stage.layer_adapter;
        let media_units = stage.media_units.clone();
        let media_count = media_units.len();
        let text_start = stage.range.start;
        let unit_count = media_count + stage.range.len();
        stage.dense_layers = Some(
            build_pipeline_layer_storage::<Gemma4Layer, _, _>(
                Arc::clone(&store),
                0..unit_count,
                options,
                static_bytes,
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
            .with_execution_offset(media_count)?,
        );
        let layer_bytes = stage.dense_layers.as_ref().unwrap().planned_layer_bytes()?;
        info.planned_owned_parameter_bytes = static_bytes
            .checked_add(layer_bytes)
            .ok_or_else(|| Error::Parallel("Gemma pipeline planned bytes overflowed".into()))?;
    } else {
        info.planned_owned_parameter_bytes = static_bytes;
    }
    info.opened_checkpoint_shards = materialized_shards;
    info.checkpoint_diagnostics = Some(checkpoint_diagnostics);
    PipelineModel::from_adapter(topology, info, PipelineStage(stage))
}

impl GemmaStage {
    #[allow(clippy::too_many_arguments)]
    fn new(
        args: gemma4::ModelArgs,
        vision_config: Option<crate::api::gemma4_vision::Gemma4VisionConfig>,
        image_token_id: Option<i32>,
        video_token_id: Option<i32>,
        audio_config: Option<crate::api::gemma4_audio::Gemma4AudioConfig>,
        audio_token_id: Option<i32>,
        range: Range<usize>,
        info: &PipelineStageInfo,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let has_multimodal_ingress = vision_config.is_some() || audio_config.is_some();
        let layer_adapter = if info.is_first && has_multimodal_ingress {
            Gemma4LayerwiseAdapter::new_pipeline(
                args.clone(),
                vision_config,
                image_token_id,
                video_token_id,
                audio_config,
                audio_token_id,
                stream,
            )?
        } else {
            Gemma4LayerwiseAdapter::new_text(args.clone(), stream)?
        };
        let media_units = layer_adapter
            .pipeline_media_groups()
            .into_iter()
            .flat_map(|(group, depth)| {
                (0..depth).map(move |index| GemmaPipelineMediaUnit { group, index })
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
        })
    }

    fn prepare_multimodal_ingress(
        &mut self,
        input: crate::api::input::ModelInput<'_>,
        step: PipelineStep,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(Array, PipelineAuxiliaryState), Error> {
        if !self.has_multimodal_ingress || self.media_layer_count != self.media_units.len() {
            return Err(Error::UnsupportedArchitecture(
                "Gemma 4 pipeline typed ingress requires configured stage-zero media semantics"
                    .into(),
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
        for ordinal in active_indices {
            let unit = self.media_units[ordinal];
            let retained = if let Some(storage) = self.dense_layers.as_ref() {
                let (_host, lease) = storage.prepare_absolute(ordinal, prefill)?;
                let mut layer = match self.parallel_layout.as_ref() {
                    Some(layout) => self
                        .layer_adapter
                        .new_parallel_layer(unit.group, unit.index, layout, stream)?,
                    None => self
                        .layer_adapter
                        .new_layer(unit.group, unit.index, stream)?,
                };
                populate_module_from_lease(&mut layer, &lease)?;
                let retained = self.layer_adapter.forward_pipeline_media_layer(
                    unit.group, unit.index, &mut layer, &mut state, execution, stream,
                )?;
                storage.trim_after_absolute(ordinal)?;
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
            eval(retained.iter())?;
            if self.dense_layers.is_some() {
                stream.synchronize()?;
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
            auxiliary.push(prepared.full_mask.ok_or_else(|| {
                Error::Parallel("Gemma multimodal ingress did not produce a full mask".into())
            })?);
            if prepared.sliding_masks.len() != self.multimodal_mask_windows.len() {
                return Err(Error::Parallel(format!(
                    "Gemma multimodal ingress produced {} sliding masks, expected {}",
                    prepared.sliding_masks.len(),
                    self.multimodal_mask_windows.len()
                )));
            }
            auxiliary.extend(prepared.sliding_masks);
        }
        Ok((prepared.hidden, PipelineAuxiliaryState::new(auxiliary)))
    }

    fn multimodal_mask<'a>(
        &self,
        auxiliary: &'a PipelineAuxiliaryState,
        step: PipelineStep,
        policy: crate::runtime::attention::AttentionPolicy,
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
                "Gemma stage cache has {} entries, expected {}",
                caches.len(),
                self.layers.len()
            )));
        }
        let prepared_ingress = match input {
            PipelineStageInput::Tokens(tokens) if self.has_multimodal_ingress => {
                let parts = [crate::api::input::InputPart::text_token_ids(tokens)];
                Some(self.prepare_multimodal_ingress(
                    crate::api::input::ModelInput::new(&parts),
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
        let args = &self.args;
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
                let policy = *args
                    .layer_policy(global_layer)
                    .expect("validated Gemma pipeline range");
                Ok(gemma4::TransformerBlock::new(
                    args,
                    policy,
                    global_layer,
                    stream,
                )?)
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
                    } if *cached == global_layer && !policy.key_value.owns_state() => layer
                        .forward(
                            gemma4::AttentionInput {
                                x: hidden,
                                mask,
                                cache: Option::<&mut ConcatKeyValueCache>::None,
                                position_offset: offset,
                                per_layer_input: per_layer_input.as_ref(),
                                shared_kv: Some(&mut shared_kv),
                                disable_generated_mask: false,
                                generated_sliding_window: None,
                            },
                            stream,
                        )?,
                    PipelineLayerCache::KeyValue {
                        global_layer: cached,
                        cache: PipelineKeyValueCache::Standard(cache),
                        ..
                    } if *cached == global_layer && policy.key_value.owns_state() => layer
                        .forward(
                            gemma4::AttentionInput {
                                x: hidden,
                                mask,
                                cache: Some(cache),
                                position_offset: offset,
                                per_layer_input: per_layer_input.as_ref(),
                                shared_kv: Some(&mut shared_kv),
                                disable_generated_mask: false,
                                generated_sliding_window: None,
                            },
                            stream,
                        )?,
                    PipelineLayerCache::KeyValue {
                        global_layer: cached,
                        cache: PipelineKeyValueCache::Paged(cache),
                        ..
                    } if *cached == global_layer && policy.key_value.owns_state() => layer
                        .forward(
                            gemma4::AttentionInput {
                                x: hidden,
                                mask,
                                cache: Some(cache),
                                position_offset: offset,
                                per_layer_input: per_layer_input.as_ref(),
                                shared_kv: Some(&mut shared_kv),
                                disable_generated_mask: false,
                                generated_sliding_window: None,
                            },
                            stream,
                        )?,
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
    topology: ParallelTopology,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    expert_cache_options: Option<ExpertCacheLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    let binding_adapter = if expert_cache_options.is_some() {
        crate::architectures::deepseek_v3::layerwise::DeepSeekV3LayerwiseAdapter::new_external_experts(
            source_args.clone(),
            stream,
        )?
    } else {
        crate::architectures::deepseek_v3::layerwise::DeepSeekV3LayerwiseAdapter::new(
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
    if (dense_stream.is_some() || expert_cache_options.is_some())
        && requested_quantization.is_some()
    {
        return Err(Error::Quantization(
            "load-time quantization is unsupported for non-resident pipeline layers; use checkpoint-native packed weights"
                .into(),
        ));
    }
    if requested_quantization.is_some() && source_args.native_fp8_config().is_some() {
        return Err(Error::Quantization(
            "native DeepSeek block-FP8 pipeline weights cannot be implicitly requantized".into(),
        ));
    }
    let quantize_on_load = requested_quantization
        .map(|requested| {
            crate::runtime::checkpoint::quantization::should_quantize_on_load(
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
    let range = topology.layer_range(source_args.layer_schedule.len())?;
    let mut info = base_info(
        topology,
        range.clone(),
        ModelKind::DeepSeekV3,
        source_args.hidden_size,
    );
    let mut stage = DeepSeekStage::new(
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
        stage.parallel_embedding = info
            .is_first
            .then(|| {
                crate::nn::parallel::VocabParallelEmbedding::unloaded(
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
                crate::nn::parallel::VocabParallelLmHead::unloaded(
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
    let static_units = pipeline_binding_units(
        &binding_adapter,
        store.as_ref(),
        &selected_pipeline_static_roles([
            (
                "embedding",
                stage.embedding.is_some() || stage.parallel_embedding.is_some(),
            ),
            ("norm", stage.norm.is_some()),
            (
                "output",
                stage.lm_head.is_some() || stage.parallel_lm_head.is_some(),
            ),
        ]),
    )?;
    let mut loaded = PipelineLoadAccumulator::new("DeepSeek");
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
    let checkpoint_diagnostics = store.diagnostics()?;
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
        info.planned_owned_parameter_bytes = static_device_bytes
            .checked_add(layer_bytes)
            .ok_or_else(|| {
                Error::Parallel("pipeline planned-owned byte total overflowed".into())
            })?;
    } else {
        info.planned_owned_parameter_bytes = static_device_bytes;
    }
    if let Some(options) = expert_cache_options {
        let entries = crate::architectures::deepseek_v3::layerwise::deepseek_expert_catalog(
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
            let cache = ExpertCache::new_shared(
                Arc::clone(&store),
                entries,
                options,
                weights_stream.clone(),
                stream.clone(),
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

impl DeepSeekStage {
    fn new(
        args: deepseek_v3::ModelArgs,
        range: Range<usize>,
        info: &PipelineStageInfo,
        external_experts: bool,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let layer_adapter = if external_experts {
            crate::architectures::deepseek_v3::layerwise::DeepSeekV3LayerwiseAdapter::new_external_experts(
                args.clone(),
                stream,
            )?
        } else {
            crate::architectures::deepseek_v3::layerwise::DeepSeekV3LayerwiseAdapter::new(
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
            hidden = norm.forward(&hidden, stream)?;
            let logits = self
                .lm_head
                .as_mut()
                .expect("last stage head")
                .forward(&hidden, stream)?;
            PipelineStageOutput::Logits(logits)
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
            let quantized = crate::nn::moe::quantize_expert_bank(&weight, quantization, stream)?;
            scales = Some(quantized.scales);
            biases = quantized.biases;
            quantized.weight
        } else {
            weight
        };
        eval(
            [&weight]
                .into_iter()
                .chain(fp8_scale.as_ref())
                .chain(scales.as_ref())
                .chain(biases.as_ref()),
        )?;
        stream.synchronize()?;
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

/// Runs stage-local execution while retaining global observer layer names.
pub fn forward_stage_with_observer(
    model: &mut PipelineModel,
    input: PipelineStageInput<'_>,
    step: PipelineStep,
    mask: Option<&Array>,
    cache: &mut PipelineCache,
    stream: &Stream,
    observer: &mut impl ActivationObserver,
) -> Result<PipelineStageOutput, Error> {
    // Common boundary observations are stable; architecture layers already
    // retain global identity in `stage_info` and the normal detailed adapters
    // can be extended without changing orchestration.
    let output = model.forward_stage(input, step, mask, cache, stream)?;
    match &output {
        PipelineStageOutput::Hidden(hidden) => observer.observe(
            &format!(
                "model.layers.{}.pipeline_stage_output",
                model.info.global_layer_range.end - 1
            ),
            &hidden.hidden,
        )?,
        PipelineStageOutput::Logits(logits) => observer.observe("lm_head.logits", logits)?,
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{runtime::distributed::topology::DeviceAssignment, test_utils::SyntheticGguf};
    use safemlx::{
        distributed::{self, Backend},
        module::Param,
        ops::{indexing::TryIndexOp, ones_dtype},
        Device, DeviceType, ExecutionContext,
    };
    use std::{
        fs,
        net::TcpListener,
        process::{Child, Command, Output, Stdio},
        thread,
        time::{Duration, Instant},
    };

    fn topology(world: usize, rank: usize, pp: usize) -> ParallelTopology {
        ParallelTopology::from_rank(
            world,
            rank,
            1,
            pp,
            1,
            DeviceAssignment::new(DeviceType::Cpu, 0),
        )
        .unwrap()
    }

    #[test]
    fn stage_roles_and_neighbor_ranks_are_explicit() {
        let first = base_info(topology(3, 0, 3), 0..2, ModelKind::Llama, 8);
        assert!(first.is_first);
        assert!(!first.is_last);
        assert_eq!(first.predecessor_rank, None);
        assert_eq!(first.successor_rank, Some(1));

        let middle = base_info(topology(3, 1, 3), 2..4, ModelKind::Llama, 8);
        assert!(!middle.is_first);
        assert!(!middle.is_last);
        assert_eq!(middle.predecessor_rank, Some(0));
        assert_eq!(middle.successor_rank, Some(2));

        let last = base_info(topology(3, 2, 3), 4..5, ModelKind::Llama, 8);
        assert!(!last.is_first);
        assert!(last.is_last);
        assert_eq!(last.predecessor_rank, Some(1));
        assert_eq!(last.successor_rank, None);
    }

    #[test]
    fn boundary_and_tied_embedding_ownership_is_not_replicated() {
        let first = base_info(topology(3, 0, 3), 0..1, ModelKind::Llama, 8);
        let middle = base_info(topology(3, 1, 3), 1..2, ModelKind::Llama, 8);
        let last = base_info(topology(3, 2, 3), 2..3, ModelKind::Llama, 8);
        assert!(owns_embedding_weight(&first, false));
        assert!(!owns_embedding_weight(&middle, false));
        assert!(!owns_embedding_weight(&last, false));
        assert!(owns_embedding_weight(&first, true));
        assert!(!owns_embedding_weight(&middle, true));
        assert!(owns_embedding_weight(&last, true));
    }

    #[test]
    fn pipeline_validation_accepts_pairwise_and_triple_axis_geometry() {
        assert!(validate_pipeline_topology(topology(2, 0, 2)).is_ok());
        assert!(validate_pipeline_topology(topology(1, 0, 1)).is_err());
        let hybrid =
            ParallelTopology::from_rank(4, 0, 2, 2, 1, DeviceAssignment::new(DeviceType::Cpu, 0))
                .unwrap();
        assert!(validate_pipeline_topology(hybrid).is_ok());
        let triple =
            ParallelTopology::from_rank(8, 0, 2, 2, 2, DeviceAssignment::new(DeviceType::Cpu, 0))
                .unwrap();
        assert!(validate_pipeline_topology(triple).is_ok());
    }

    #[test]
    fn activation_shape_validation_is_role_aware() {
        let later = base_info(topology(2, 1, 2), 1..2, ModelKind::Llama, 8);
        let step = PipelineStep::new(1, 3).unwrap();
        assert!(validate_hidden_metadata(&later, &[1, 3, 8], Dtype::Float32, step).is_ok());
        assert!(validate_hidden_metadata(&later, &[1, 3, 7], Dtype::Float32, step).is_err());
        assert!(validate_hidden_metadata(&later, &[1, 3, 8], Dtype::Float16, step).is_err());
    }

    #[test]
    fn cache_reports_only_local_global_layers() {
        let cache = PipelineCache::new(
            ModelKind::Llama,
            vec![
                PipelineLayerCache::KeyValue {
                    global_layer: 3,
                    cache: PipelineKeyValueCache::Standard(ConcatKeyValueCache::new()),
                    slots: Vec::new(),
                },
                PipelineLayerCache::KeyValue {
                    global_layer: 4,
                    cache: PipelineKeyValueCache::Standard(ConcatKeyValueCache::new()),
                    slots: Vec::new(),
                },
            ],
        );
        assert_eq!(cache.global_layers(), vec![3, 4]);
    }

    #[test]
    fn semantic_cache_layout_materializes_without_architecture_dispatch() {
        let layout = crate::LayerSchedule::new(
            3,
            vec![
                crate::LayerCachePolicy::NoState,
                crate::LayerCachePolicy::key_value(crate::AttentionPolicy::Full, 2, 4).unwrap(),
                crate::LayerCachePolicy::compressed_latent_rotary(
                    crate::AttentionPolicy::Full,
                    4,
                    2,
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let identity = pipeline_prompt_cache_identity(
            topology(2, 0, 2),
            "synthetic",
            "synthetic",
            "synthetic".into(),
            5,
            2..5,
            layout,
        );
        let layers = materialize_pipeline_cache_layers(&identity, None).unwrap();
        assert!(matches!(
            layers.as_slice(),
            [
                PipelineLayerCache::StateSlots {
                    global_layer: 2,
                    slots,
                },
                PipelineLayerCache::KeyValue {
                    global_layer: 3,
                    ..
                },
                PipelineLayerCache::CompressedLatent {
                    global_layer: 4,
                    ..
                }
            ] if slots.is_empty()
        ));

        let unsupported = pipeline_prompt_cache_identity(
            topology(2, 0, 2),
            "synthetic",
            "synthetic",
            "synthetic-fixed".into(),
            1,
            0..1,
            crate::LayerSchedule::new(
                1,
                vec![crate::LayerCachePolicy::FixedState {
                    tensors: vec![crate::runtime::cache::residency::StateTensorPolicy {
                        role: crate::runtime::cache::residency::StateTensorRole::Recurrent,
                        shape: vec![crate::runtime::cache::residency::StateTensorDimension::Scalar],
                        dtype: crate::runtime::cache::residency::StateTensorDtype::Float32,
                        required: true,
                    }],
                }],
            )
            .unwrap(),
        );
        let fixed = materialize_pipeline_cache_layers(&unsupported, None).unwrap();
        let PipelineLayerCache::StateSlots {
            global_layer,
            slots,
        } = &fixed[0]
        else {
            panic!("fixed state must materialize descriptor-backed slots")
        };
        assert_eq!(*global_layer, 0);
        assert_eq!(slots.len(), 1);
        assert_eq!(
            slots[0].policy().role,
            crate::runtime::cache::residency::StateTensorRole::Recurrent
        );
        assert!(slots[0].value().is_none());
        assert_eq!(slots[0].offset(), 0);
    }

    #[test]
    fn semantic_state_slots_reset_and_reject_inconsistent_offsets() {
        let policy = StateTensorPolicy::new(
            StateTensorRole::Recurrent,
            vec![crate::runtime::cache::residency::StateTensorDimension::Scalar],
            crate::runtime::cache::residency::StateTensorDtype::Float32,
        )
        .unwrap();
        let mut first = PipelineStateSlot::empty(policy.clone());
        first.value = Some(Array::from_f32(1.0));
        first.offset = 3;
        let mut second = PipelineStateSlot::empty(policy);
        second.offset = 2;
        let mut cache = PipelineCache::new(
            ModelKind::Lfm2,
            vec![
                PipelineLayerCache::StateSlots {
                    global_layer: 4,
                    slots: vec![first],
                },
                PipelineLayerCache::StateSlots {
                    global_layer: 5,
                    slots: vec![second],
                },
            ],
        );
        let error = pipeline_state_offset("synthetic", &cache.layers)
            .expect_err("different semantic state offsets must fail closed");
        assert!(error.to_string().contains("global layer 5"));
        cache.reset().unwrap();
        assert_eq!(
            pipeline_state_offset("synthetic", &cache.layers).unwrap(),
            0
        );
        for layer in cache.layers() {
            let PipelineLayerCache::StateSlots { slots, .. } = layer else {
                unreachable!()
            };
            assert_eq!(slots[0].policy().role, StateTensorRole::Recurrent);
            assert!(slots[0].value().is_none());
            assert_eq!(slots[0].offset(), 0);
        }
    }

    #[test]
    fn recurrent_pipeline_layouts_use_only_semantic_cache_variants() {
        use crate::runtime::cache::residency::{StateTensorDimension, StateTensorDtype};

        let tensor = |role| {
            StateTensorPolicy::new(
                role,
                vec![
                    StateTensorDimension::Batch,
                    StateTensorDimension::fixed(1).unwrap(),
                ],
                StateTensorDtype::Float32,
            )
            .unwrap()
        };
        let kimi_kda = crate::LayerCachePolicy::FixedState {
            tensors: vec![
                tensor(StateTensorRole::Convolution { slot: 0 }),
                tensor(StateTensorRole::Convolution { slot: 1 }),
                tensor(StateTensorRole::Convolution { slot: 2 }),
                tensor(StateTensorRole::Recurrent),
            ],
        };
        let nemotron_mamba = crate::LayerCachePolicy::FixedState {
            tensors: vec![
                tensor(StateTensorRole::Convolution { slot: 0 }),
                tensor(StateTensorRole::Recurrent),
            ],
        };
        let qwen_linear = nemotron_mamba.clone();
        let inkling = crate::LayerCachePolicy::key_value_with_fixed_state(
            crate::AttentionPolicy::Sliding {
                window: std::num::NonZeroU32::new(8).unwrap(),
            },
            2,
            4,
            (0..4)
                .map(|slot| tensor(StateTensorRole::Convolution { slot }))
                .collect(),
        )
        .unwrap();
        let layout = crate::LayerSchedule::new(
            5,
            vec![
                kimi_kda,
                crate::LayerCachePolicy::compressed_latent_rotary(
                    crate::AttentionPolicy::Full,
                    4,
                    2,
                )
                .unwrap(),
                nemotron_mamba,
                qwen_linear,
                inkling,
            ],
        )
        .unwrap();
        let identity = pipeline_prompt_cache_identity(
            topology(2, 0, 2),
            "hybrid",
            "hybrid",
            "hybrid-layout".into(),
            5,
            0..5,
            layout,
        );
        let layers = materialize_pipeline_cache_layers(&identity, None).unwrap();
        assert!(matches!(
            &layers[0],
            PipelineLayerCache::StateSlots { slots, .. } if slots.len() == 4
        ));
        assert!(matches!(
            &layers[1],
            PipelineLayerCache::CompressedLatent { slots, .. } if slots.is_empty()
        ));
        assert!(matches!(
            &layers[2],
            PipelineLayerCache::StateSlots { slots, .. } if slots.len() == 2
        ));
        assert!(matches!(
            &layers[3],
            PipelineLayerCache::StateSlots { slots, .. } if slots.len() == 2
        ));
        assert!(matches!(
            &layers[4],
            PipelineLayerCache::KeyValue {
                cache: PipelineKeyValueCache::Standard(cache),
                slots,
                ..
            } if cache.max_size() == Some(8) && slots.len() == 4
        ));
    }

    fn gpu_topology(rank: usize) -> ParallelTopology {
        ParallelTopology::from_rank(2, rank, 1, 2, 1, DeviceAssignment::new(DeviceType::Gpu, 0))
            .unwrap()
    }

    fn tp_pp_gpu_topology(rank: usize) -> ParallelTopology {
        ParallelTopology::from_rank(4, rank, 2, 2, 1, DeviceAssignment::new(DeviceType::Gpu, 0))
            .unwrap()
    }

    fn pp_ep_gpu_topology(rank: usize) -> ParallelTopology {
        ParallelTopology::from_rank(4, rank, 1, 2, 2, DeviceAssignment::new(DeviceType::Gpu, 0))
            .unwrap()
    }

    fn tp_pp_ep_gpu_topology(rank: usize) -> ParallelTopology {
        ParallelTopology::from_rank(8, rank, 2, 2, 2, DeviceAssignment::new(DeviceType::Gpu, 0))
            .unwrap()
    }

    fn initialize_parameters(module: &mut impl ModuleParameters, stream: &Stream) {
        for (name, parameter) in module.parameters_mut().flatten() {
            let shape = parameter.shape().to_vec();
            let dtype = parameter.dtype();
            *parameter = if name.ends_with("_scales") {
                Array::full::<u8>(&shape, Array::from_slice(&[127u8], &[]), stream).unwrap()
            } else if name.ends_with("_blocks") {
                Array::full::<u8>(&shape, Array::from_slice(&[0x11u8], &[]), stream).unwrap()
            } else if name.ends_with("norm.weight") {
                ones_dtype(&shape, dtype, stream).unwrap()
            } else {
                Array::full::<f32>(&shape, Array::from_f32(0.01), stream).unwrap()
            };
        }
    }

    fn assert_close(left: &Array, right: &Array) {
        assert_close_with_tolerance(left, right, 1e-5);
    }

    fn assert_close_with_tolerance(left: &Array, right: &Array, tolerance: f32) {
        let left = left.evaluated().unwrap();
        let right = right.evaluated().unwrap();
        assert_eq!(left.as_array().shape(), right.as_array().shape());
        for (left, right) in left.as_slice::<f32>().iter().zip(right.as_slice::<f32>()) {
            assert!(
                (left - right).abs() <= tolerance,
                "{left} != {right} within {tolerance}"
            );
        }
    }

    fn assert_module_parameter_parity(left: &impl ModuleParameters, right: &impl ModuleParameters) {
        let left = left.parameters().flatten();
        let right = right.parameters().flatten();
        assert_eq!(left.len(), right.len());
        for (name, left) in left {
            let right = right.get(&name).expect("matching parameter");
            assert_eq!(left.shape(), right.shape(), "shape mismatch for {name}");
            assert_eq!(left.dtype(), right.dtype(), "dtype mismatch for {name}");
            assert_close(left, right);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn assert_nonresident_qwen_layer_recipe_parity(
        args: dense_qwen::DecoderConfig,
        store: SharedWeightStore,
        topology: ParallelTopology,
        assignment: Option<ExpertAssignment>,
        residency: PipelineLayerLoadOptions,
        global_layer: usize,
        stream: &Stream,
        weights_stream: &Stream,
    ) {
        let adapter =
            dense_qwen::layerwise::DenseQwenLayerwiseAdapter::new(args.clone(), stream).unwrap();
        let layout = if topology.tensor_parallel_size > 1 {
            let build = ParallelBuildContext::new(topology, ShardingPolicy::Require);
            let mut planner = build.planner();
            adapter
                .register_parallel_parameters(build, &mut planner, stream)
                .unwrap();
            Some(planner.finish().unwrap().1)
        } else {
            None
        };
        let mut resident = adapter
            .new_cartesian_layer(
                0,
                global_layer,
                layout.as_ref(),
                assignment.as_ref(),
                stream,
            )
            .unwrap();
        let bindings = adapter
            .cartesian_layer_bindings(
                0,
                global_layer,
                &resident,
                store.as_ref(),
                layout.as_ref(),
                assignment.as_ref(),
                stream,
            )
            .unwrap();
        load_bound_module(
            &mut resident,
            store.as_ref(),
            &bindings,
            None,
            weights_stream,
            stream,
        )
        .unwrap();

        let dense = build_pipeline_layer_storage(
            Arc::clone(&store),
            global_layer..global_layer + 1,
            residency,
            0,
            stream,
            weights_stream,
            |layer, stream| {
                adapter.new_cartesian_layer(0, layer, layout.as_ref(), assignment.as_ref(), stream)
            },
            |layer, module, store| {
                adapter.cartesian_layer_bindings(
                    0,
                    layer,
                    module,
                    store,
                    layout.as_ref(),
                    assignment.as_ref(),
                    stream,
                )
            },
        )
        .unwrap();
        let (forward, group) = match &dense.controller {
            PipelineLayerController::LayerwiseHost(_) => (None, None),
            PipelineLayerController::DenseDiskStream(controller) => (
                Some(controller.forward_guard(true, &dense.residency).unwrap()),
                Some(controller.group_guard(&dense.residency, "pipeline_stage")),
            ),
        };
        let (_host, lease) = dense.prepare(0, true).unwrap();
        let mut streamed = adapter
            .new_cartesian_layer(
                0,
                global_layer,
                layout.as_ref(),
                assignment.as_ref(),
                stream,
            )
            .unwrap();
        populate_module_from_lease(&mut streamed, &lease).unwrap();
        assert_module_parameter_parity(&resident, &streamed);
        drop(lease);
        dense.trim_after(0).unwrap();
        if let Some(group) = group {
            group.complete().unwrap();
        }
        if let Some(forward) = forward {
            forward.complete().unwrap();
        }
    }

    fn write_parameter_fixture(
        dir: &Path,
        config: &serde_json::Value,
        model: &impl ModuleParameters,
    ) {
        let arrays = model
            .parameters()
            .flatten()
            .iter()
            .map(|(name, value)| {
                (
                    crate::runtime::checkpoint::binding::canonical_checkpoint_name(name),
                    (*value).clone(),
                )
            })
            .collect::<Vec<_>>();
        Array::save_safetensors(
            arrays.iter().map(|(name, value)| (name.as_str(), value)),
            None,
            dir.join("model.safetensors"),
        )
        .unwrap();
        fs::write(dir.join("config.json"), serde_json::to_vec(config).unwrap()).unwrap();
    }

    fn write_split_qwen3_moe_fixture(
        dir: &Path,
        config: &serde_json::Value,
        model: &dense_qwen::Model,
        stream: &Stream,
    ) {
        let mut arrays = Vec::new();
        for (name, value) in model.parameters().flatten() {
            let name = crate::runtime::checkpoint::binding::canonical_checkpoint_name(&name);
            if let Some(prefix) = name.strip_suffix(".mlp.experts.gate_up_proj") {
                for expert in 0..model.args.num_experts {
                    let selected = value.try_index_device(expert, stream).unwrap();
                    let intermediate = model.args.moe_intermediate_size;
                    arrays.push((
                        format!("{prefix}.mlp.experts.{expert}.gate_proj.weight"),
                        selected
                            .try_index_device((..intermediate, ..), stream)
                            .unwrap(),
                    ));
                    arrays.push((
                        format!("{prefix}.mlp.experts.{expert}.up_proj.weight"),
                        selected
                            .try_index_device((intermediate.., ..), stream)
                            .unwrap(),
                    ));
                }
                continue;
            }
            if let Some(prefix) = name.strip_suffix(".mlp.experts.down_proj") {
                for expert in 0..model.args.num_experts {
                    arrays.push((
                        format!("{prefix}.mlp.experts.{expert}.down_proj.weight"),
                        value.try_index_device(expert, stream).unwrap(),
                    ));
                }
                continue;
            }
            arrays.push((name, value.clone()));
        }
        Array::save_safetensors(
            arrays.iter().map(|(name, value)| (name.as_str(), value)),
            None,
            dir.join("model.safetensors"),
        )
        .unwrap();
        fs::write(dir.join("config.json"), serde_json::to_vec(config).unwrap()).unwrap();
    }

    fn run_pipeline_sequence(
        first: &mut PipelineModel,
        last: &mut PipelineModel,
        token_batches: &[Array],
        stream: &Stream,
    ) -> Vec<Array> {
        let mut first_cache = first.new_cache().unwrap();
        let mut last_cache = last.new_cache().unwrap();
        let mut outputs = Vec::with_capacity(token_batches.len());
        for tokens in token_batches {
            let step = PipelineStep::new(1, tokens.shape()[1]).unwrap();
            let hidden = match first
                .forward_stage(
                    PipelineStageInput::Tokens(tokens),
                    step,
                    None,
                    &mut first_cache,
                    stream,
                )
                .unwrap()
            {
                PipelineStageOutput::Hidden(hidden) => hidden,
                PipelineStageOutput::Logits(_) => panic!("first stage produced logits"),
            };
            let logits = match last
                .forward_stage(
                    PipelineStageInput::Hidden(&hidden),
                    step,
                    None,
                    &mut last_cache,
                    stream,
                )
                .unwrap()
            {
                PipelineStageOutput::Logits(logits) => logits,
                PipelineStageOutput::Hidden(_) => panic!("last stage produced hidden state"),
            };
            eval([&logits]).unwrap();
            outputs.push(logits);
        }
        outputs
    }

    fn dense_stream_options() -> crate::runtime::residency::dense_stream::DenseDiskStreamLoadOptions
    {
        crate::runtime::residency::dense_stream::DenseDiskStreamLoadOptions::new(
            u64::MAX,
            u64::MAX,
            1,
            1,
            1,
        )
        .unwrap()
    }

    fn dense_qwen_config(model_type: &str, tied: bool) -> serde_json::Value {
        let is_moe = model_type == "qwen3_moe";
        let mut config = serde_json::json!({
            "architectures": [match model_type {
                "qwen2" => "Qwen2ForCausalLM",
                "qwen3" => "Qwen3ForCausalLM",
                "qwen3_moe" => "Qwen3MoeForCausalLM",
                _ => panic!("unsupported dense-Qwen test model type {model_type}"),
            }],
            "model_type": model_type,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "intermediate_size": if is_moe { 0 } else { 64 },
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "rms_norm_eps": 0.000001,
            "vocab_size": 32,
            "max_position_embeddings": 128,
            "rope_theta": 10000.0,
            "tie_word_embeddings": tied,
            "attention_bias": model_type == "qwen2",
            "mlp_bias": false,
            "moe_intermediate_size": if is_moe { 32 } else { 0 },
            "num_experts": if is_moe { 4 } else { 0 },
            "num_experts_per_tok": if is_moe { 2 } else { 0 },
            "norm_topk_prob": is_moe
        });
        if model_type == "qwen2" {
            config["use_sliding_window"] = serde_json::json!(true);
            config["sliding_window"] = serde_json::json!(3);
            config["max_window_layers"] = serde_json::json!(1);
        }
        config
    }

    fn qwen_hybrid_config() -> serde_json::Value {
        serde_json::json!({
            "architectures": ["Qwen3NextForCausalLM"],
            "model_type": "qwen3_next",
            "vocab_size": 32,
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "max_position_embeddings": 128,
            "intermediate_size": 32,
            "num_experts": 0,
            "linear_conv_kernel_dim": 3,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "tie_word_embeddings": false
        })
    }

    fn qwen_hybrid_moe_config(model_type: &str) -> serde_json::Value {
        let mut config = qwen_hybrid_config();
        config["model_type"] = serde_json::json!(model_type);
        config["architectures"] = serde_json::json!([if model_type == "qwen3_next" {
            "Qwen3NextForCausalLM"
        } else {
            "Qwen3_5MoeForCausalLM"
        }]);
        config["num_key_value_heads"] = serde_json::json!(2);
        config["intermediate_size"] = serde_json::json!(0);
        config["moe_intermediate_size"] = serde_json::json!(8);
        config["shared_expert_intermediate_size"] = serde_json::json!(8);
        config["num_experts"] = serde_json::json!(3);
        config["num_experts_per_tok"] = serde_json::json!(1);
        config["norm_topk_prob"] = serde_json::json!(true);
        config
    }

    fn gpt_oss_config() -> serde_json::Value {
        serde_json::json!({
            "model_type": "gpt_oss",
            "hidden_size": 64,
            "intermediate_size": 96,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "vocab_size": 64,
            "num_local_experts": 2,
            "num_experts_per_tok": 1,
            "rms_norm_eps": 0.00001,
            "sliding_window": 3,
            "max_position_embeddings": 128,
            "rope_theta": 150000.0,
            "layer_types": ["sliding_attention", "full_attention"],
            "quantization_config": {"quant_method": "mxfp4"},
            "swiglu_limit": 7.0
        })
    }

    fn lfm2_config(moe: bool) -> serde_json::Value {
        serde_json::json!({
            "model_type": if moe { "lfm2_moe" } else { "lfm2" },
            "vocab_size": 16,
            "hidden_size": 12,
            "intermediate_size": 17,
            "num_hidden_layers": 2,
            "num_attention_heads": 6,
            "num_key_value_heads": 3,
            "max_position_embeddings": 64,
            "norm_eps": 0.00001,
            "layer_types": ["conv", "full_attention"],
            "conv_L_cache": 3,
            "conv_bias": true,
            "block_auto_adjust_ff_dim": false,
            "tie_word_embeddings": false,
            "moe_intermediate_size": if moe { 9 } else { 0 },
            "num_dense_layers": if moe { 1 } else { 0 },
            "num_experts": if moe { 2 } else { 0 },
            "num_experts_per_tok": if moe { 1 } else { 0 },
            "norm_topk_prob": moe,
            "use_expert_bias": moe
        })
    }

    fn lfm2_gguf_fixture(model: &lfm2::Model, stream: &Stream) -> SyntheticGguf {
        let arrays = model
            .parameters()
            .flatten()
            .into_iter()
            .map(|(name, value)| {
                let canonical =
                    crate::runtime::checkpoint::binding::canonical_checkpoint_name(&name);
                let gguf = match canonical.as_str() {
                    "model.embed_tokens.weight" => "token_embd.weight".into(),
                    "model.embedding_norm.weight" => "token_embd_norm.weight".into(),
                    "lm_head.weight" => "output.weight".into(),
                    name => name
                        .replace("model.layers.", "blk.")
                        .replace(".conv.conv.", ".shortconv.conv.")
                        .replace(".conv.in_proj.", ".shortconv.in_proj.")
                        .replace(".conv.out_proj.", ".shortconv.out_proj.")
                        .replace(".self_attn.q_layernorm.", ".attn_q_norm.")
                        .replace(".self_attn.k_layernorm.", ".attn_k_norm.")
                        .replace(".self_attn.q_proj.", ".attn_q.")
                        .replace(".self_attn.k_proj.", ".attn_k.")
                        .replace(".self_attn.v_proj.", ".attn_v.")
                        .replace(".self_attn.out_proj.", ".attn_output.")
                        .replace(".operator_norm.", ".attn_norm.")
                        .replace(".feed_forward.w1.", ".ffn_gate.")
                        .replace(".feed_forward.w2.", ".ffn_down.")
                        .replace(".feed_forward.w3.", ".ffn_up."),
                };
                let value = if canonical.ends_with(".conv.conv.weight") {
                    value
                        .reshape(&[value.shape()[0], value.shape()[2]], stream)
                        .unwrap()
                } else {
                    value.clone()
                };
                (gguf, value)
            })
            .collect::<HashMap<_, _>>();
        let metadata = HashMap::from([
            (
                "general.architecture".into(),
                GgufMetadataValue::String("lfm2".into()),
            ),
            ("general.file_type".into(), GgufMetadataValue::Uint32(0)),
            ("lfm2.block_count".into(), GgufMetadataValue::Uint32(2)),
            (
                "lfm2.embedding_length".into(),
                GgufMetadataValue::Uint32(12),
            ),
            (
                "lfm2.feed_forward_length".into(),
                GgufMetadataValue::Uint32(17),
            ),
            (
                "lfm2.attention.head_count".into(),
                GgufMetadataValue::Uint32(6),
            ),
            (
                "lfm2.attention.head_count_kv".into(),
                GgufMetadataValue::Array(safemlx::ops::GgufMetadataArray::Uint32(vec![0, 3])),
            ),
            (
                "lfm2.attention.layer_norm_rms_epsilon".into(),
                GgufMetadataValue::Float32(0.00001),
            ),
            ("lfm2.context_length".into(), GgufMetadataValue::Uint32(64)),
            (
                "lfm2.shortconv.l_cache".into(),
                GgufMetadataValue::Uint32(3),
            ),
            (
                "lfm2.rope.freq_base".into(),
                GgufMetadataValue::Float32(10_000.0),
            ),
            ("lfm2.vocab_size".into(), GgufMetadataValue::Uint32(16)),
        ]);
        SyntheticGguf::dense(&arrays, &metadata)
    }

    fn lfm2_moe_gguf_fixture(model: &lfm2::Model, stream: &Stream) -> SyntheticGguf {
        let mut arrays = HashMap::new();
        for (name, value) in model.parameters().flatten() {
            let canonical = crate::runtime::checkpoint::binding::canonical_checkpoint_name(&name);
            let layer_name = |name: &str| {
                name.replace("model.layers.", "blk.")
                    .replace(".conv.conv.", ".shortconv.conv.")
                    .replace(".conv.in_proj.", ".shortconv.in_proj.")
                    .replace(".conv.out_proj.", ".shortconv.out_proj.")
                    .replace(".self_attn.q_layernorm.", ".attn_q_norm.")
                    .replace(".self_attn.k_layernorm.", ".attn_k_norm.")
                    .replace(".self_attn.q_proj.", ".attn_q.")
                    .replace(".self_attn.k_proj.", ".attn_k.")
                    .replace(".self_attn.v_proj.", ".attn_v.")
                    .replace(".self_attn.out_proj.", ".attn_output.")
                    .replace(".operator_norm.", ".attn_norm.")
                    .replace(".feed_forward.gate.", ".ffn_gate_inp.")
                    .replace(".feed_forward.experts.down_proj", ".ffn_down_exps.weight")
                    .replace(".feed_forward.w1.", ".ffn_gate.")
                    .replace(".feed_forward.w2.", ".ffn_down.")
                    .replace(".feed_forward.w3.", ".ffn_up.")
            };
            match canonical.as_str() {
                "model.embed_tokens.weight" => {
                    arrays.insert("token_embd.weight".into(), value.clone());
                }
                "model.embedding_norm.weight" => {
                    arrays.insert("token_embd_norm.weight".into(), value.clone());
                }
                "lm_head.weight" => {
                    arrays.insert("output.weight".into(), value.clone());
                }
                name if name.ends_with("feed_forward.experts.gate_up_proj") => {
                    let prefix = name.trim_end_matches("feed_forward.experts.gate_up_proj");
                    let width = value.dim(1) / 2;
                    arrays.insert(
                        layer_name(&format!("{prefix}ffn_gate_exps.weight")),
                        value.try_index_device((.., ..width, ..), stream).unwrap(),
                    );
                    arrays.insert(
                        layer_name(&format!("{prefix}ffn_up_exps.weight")),
                        value.try_index_device((.., width.., ..), stream).unwrap(),
                    );
                }
                name if name.ends_with("feed_forward.expert_bias") => {
                    let prefix = name.trim_end_matches("feed_forward.expert_bias");
                    arrays.insert(
                        layer_name(&format!("{prefix}ffn_exp_probs_b.bias")),
                        value.clone(),
                    );
                }
                name => {
                    let value = if name.ends_with(".conv.conv.weight") {
                        value
                            .reshape(&[value.shape()[0], value.shape()[2]], stream)
                            .unwrap()
                    } else {
                        value.clone()
                    };
                    arrays.insert(layer_name(name), value);
                }
            }
        }
        let metadata = HashMap::from([
            (
                "general.architecture".into(),
                GgufMetadataValue::String("lfm2moe".into()),
            ),
            ("general.file_type".into(), GgufMetadataValue::Uint32(0)),
            ("lfm2moe.block_count".into(), GgufMetadataValue::Uint32(2)),
            (
                "lfm2moe.embedding_length".into(),
                GgufMetadataValue::Uint32(12),
            ),
            (
                "lfm2moe.feed_forward_length".into(),
                GgufMetadataValue::Uint32(17),
            ),
            (
                "lfm2moe.expert_feed_forward_length".into(),
                GgufMetadataValue::Uint32(9),
            ),
            (
                "lfm2moe.leading_dense_block_count".into(),
                GgufMetadataValue::Uint32(1),
            ),
            ("lfm2moe.expert_count".into(), GgufMetadataValue::Uint32(2)),
            (
                "lfm2moe.expert_used_count".into(),
                GgufMetadataValue::Uint32(1),
            ),
            (
                "lfm2moe.expert_weights_norm".into(),
                GgufMetadataValue::Uint32(1),
            ),
            (
                "lfm2moe.attention.head_count".into(),
                GgufMetadataValue::Uint32(6),
            ),
            (
                "lfm2moe.attention.head_count_kv".into(),
                GgufMetadataValue::Array(safemlx::ops::GgufMetadataArray::Uint32(vec![0, 3])),
            ),
            (
                "lfm2moe.attention.layer_norm_rms_epsilon".into(),
                GgufMetadataValue::Float32(0.00001),
            ),
            (
                "lfm2moe.context_length".into(),
                GgufMetadataValue::Uint32(64),
            ),
            (
                "lfm2moe.shortconv.l_cache".into(),
                GgufMetadataValue::Uint32(3),
            ),
            (
                "lfm2moe.rope.freq_base".into(),
                GgufMetadataValue::Float32(10_000.0),
            ),
            ("lfm2moe.vocab_size".into(), GgufMetadataValue::Uint32(16)),
        ]);
        SyntheticGguf::dense(&arrays, &metadata)
    }

    fn dense_qwen_gguf_fixture(model: &dense_qwen::Model, stream: &Stream) -> SyntheticGguf {
        let args = &model.args;
        let is_moe = args.is_moe();
        let architecture = if is_moe {
            "qwen3moe"
        } else {
            args.model_type.as_str()
        };
        let mut arrays = HashMap::new();
        for (name, value) in model.parameters().flatten() {
            if let Some(prefix) = name.strip_suffix(".mlp.experts.gate_up_proj") {
                let intermediate = args.moe_intermediate_size;
                let prefix = prefix.replace("model.layers.", "blk.");
                arrays.insert(
                    format!("{prefix}.ffn_gate_exps.weight"),
                    value
                        .try_index_device((.., ..intermediate, ..), stream)
                        .unwrap(),
                );
                arrays.insert(
                    format!("{prefix}.ffn_up_exps.weight"),
                    value
                        .try_index_device((.., intermediate.., ..), stream)
                        .unwrap(),
                );
                continue;
            }
            if let Some(prefix) = name.strip_suffix(".mlp.experts.down_proj") {
                arrays.insert(
                    format!(
                        "{}.ffn_down_exps.weight",
                        prefix.replace("model.layers.", "blk.")
                    ),
                    value.clone(),
                );
                continue;
            }
            let name = name
                .replace("model.layers.", "blk.")
                .replace("self_attn.q_norm", "attn_q_norm")
                .replace("self_attn.k_norm", "attn_k_norm")
                .replace("self_attn.q_proj", "attn_q")
                .replace("self_attn.k_proj", "attn_k")
                .replace("self_attn.v_proj", "attn_v")
                .replace("self_attn.o_proj", "attn_output")
                .replace("input_layernorm", "attn_norm")
                .replace("post_attention_layernorm", "ffn_norm")
                .replace("mlp.gate.weight", "ffn_gate_inp.weight")
                .replace("mlp.gate_proj", "ffn_gate")
                .replace("mlp.down_proj", "ffn_down")
                .replace("mlp.up_proj", "ffn_up")
                .replace("model.embed_tokens", "token_embd")
                .replace("model.norm", "output_norm")
                .replace("lm_head", "output");
            arrays.insert(name, value.clone());
        }
        let key = |suffix: &str| format!("{architecture}.{suffix}");
        let mut metadata = HashMap::from([
            (
                "general.architecture".into(),
                GgufMetadataValue::String(architecture.into()),
            ),
            ("general.file_type".into(), GgufMetadataValue::Uint32(0)),
            (
                key("embedding_length"),
                GgufMetadataValue::Uint32(args.hidden_size as u32),
            ),
            (
                key("block_count"),
                GgufMetadataValue::Uint32(args.num_hidden_layers as u32),
            ),
            (
                key("attention.head_count"),
                GgufMetadataValue::Uint32(args.num_attention_heads as u32),
            ),
            (
                key("attention.head_count_kv"),
                GgufMetadataValue::Uint32(args.num_key_value_heads as u32),
            ),
            (
                key("attention.key_length"),
                GgufMetadataValue::Uint32(args.head_dim as u32),
            ),
            (
                key("attention.layer_norm_rms_epsilon"),
                GgufMetadataValue::Float32(args.rms_norm_eps),
            ),
            (
                key("context_length"),
                GgufMetadataValue::Uint32(args.max_position_embeddings as u32),
            ),
            (
                key("rope.freq_base"),
                GgufMetadataValue::Float32(args.rope_theta),
            ),
            (
                key("vocab_size"),
                GgufMetadataValue::Uint32(args.vocab_size as u32),
            ),
        ]);
        if is_moe {
            metadata.extend([
                (
                    key("expert_feed_forward_length"),
                    GgufMetadataValue::Uint32(args.moe_intermediate_size as u32),
                ),
                (
                    key("expert_count"),
                    GgufMetadataValue::Uint32(args.num_experts as u32),
                ),
                (
                    key("expert_used_count"),
                    GgufMetadataValue::Uint32(args.num_experts_per_tok as u32),
                ),
            ]);
        } else {
            metadata.insert(
                key("feed_forward_length"),
                GgufMetadataValue::Uint32(args.intermediate_size as u32),
            );
        }
        if let Some(window) = args
            .attention_schedule
            .iter()
            .find_map(|policy| policy.window())
        {
            metadata.insert(
                key("attention.sliding_window"),
                GgufMetadataValue::Uint32(window.get()),
            );
            metadata.insert(
                key("attention.sliding_window_pattern"),
                GgufMetadataValue::Array(safemlx::ops::GgufMetadataArray::Bool(
                    args.attention_schedule
                        .iter()
                        .map(|policy| policy.window().is_some())
                        .collect(),
                )),
            );
        }
        SyntheticGguf::dense(&arrays, &metadata)
    }

    fn gpt_oss_gguf_fixture(stream: &Stream) -> SyntheticGguf {
        let mut arrays = HashMap::new();
        let mut insert = |name: String, shape: &[i32], value: f32| {
            arrays.insert(
                name,
                Array::full::<f32>(shape, Array::from_f32(value), stream).unwrap(),
            );
        };
        insert("token_embd.weight".into(), &[64, 64], 0.01);
        for layer in 0..2 {
            let prefix = format!("blk.{layer}");
            insert(format!("{prefix}.attn_norm.weight"), &[64], 1.0);
            insert(format!("{prefix}.attn_post_norm.weight"), &[64], 1.0);
            insert(format!("{prefix}.attn_q.weight"), &[128, 64], 0.01);
            insert(format!("{prefix}.attn_q.bias"), &[128], 0.001);
            insert(format!("{prefix}.attn_k.weight"), &[64, 64], 0.01);
            insert(format!("{prefix}.attn_k.bias"), &[64], 0.001);
            insert(format!("{prefix}.attn_v.weight"), &[64, 64], 0.01);
            insert(format!("{prefix}.attn_v.bias"), &[64], 0.001);
            insert(format!("{prefix}.attn_output.weight"), &[64, 128], 0.01);
            insert(format!("{prefix}.attn_output.bias"), &[64], 0.001);
            insert(format!("{prefix}.attn_sinks.weight"), &[4], 0.001);
            insert(format!("{prefix}.ffn_gate_inp.weight"), &[2, 64], 0.01);
            insert(format!("{prefix}.ffn_gate_inp.bias"), &[2], 0.001);
            insert(format!("{prefix}.ffn_gate_exps.weight"), &[2, 96, 64], 0.01);
            insert(format!("{prefix}.ffn_gate_exps.bias"), &[2, 96], 0.001);
            insert(format!("{prefix}.ffn_up_exps.weight"), &[2, 96, 64], 0.01);
            insert(format!("{prefix}.ffn_up_exps.bias"), &[2, 96], 0.001);
            insert(format!("{prefix}.ffn_down_exps.weight"), &[2, 64, 96], 0.01);
            insert(format!("{prefix}.ffn_down_exps.bias"), &[2, 64], 0.001);
        }
        insert("output_norm.weight".into(), &[64], 1.0);
        insert("output.weight".into(), &[64, 64], 0.01);
        let metadata = HashMap::from([
            (
                "general.architecture".into(),
                GgufMetadataValue::String("gpt-oss".into()),
            ),
            ("general.file_type".into(), GgufMetadataValue::Uint32(39)),
            (
                "gpt-oss.embedding_length".into(),
                GgufMetadataValue::Uint32(64),
            ),
            ("gpt-oss.block_count".into(), GgufMetadataValue::Uint32(2)),
            (
                "gpt-oss.expert_feed_forward_length".into(),
                GgufMetadataValue::Uint32(96),
            ),
            (
                "gpt-oss.attention.head_count".into(),
                GgufMetadataValue::Uint32(4),
            ),
            (
                "gpt-oss.attention.head_count_kv".into(),
                GgufMetadataValue::Uint32(2),
            ),
            (
                "gpt-oss.attention.key_length".into(),
                GgufMetadataValue::Uint32(32),
            ),
            (
                "gpt-oss.attention.layer_norm_rms_epsilon".into(),
                GgufMetadataValue::Float32(0.00001),
            ),
            (
                "gpt-oss.attention.sliding_window".into(),
                GgufMetadataValue::Uint32(3),
            ),
            (
                "gpt-oss.context_length".into(),
                GgufMetadataValue::Uint32(128),
            ),
            (
                "gpt-oss.rope.freq_base".into(),
                GgufMetadataValue::Float32(150000.0),
            ),
            ("gpt-oss.expert_count".into(), GgufMetadataValue::Uint32(2)),
            (
                "gpt-oss.expert_used_count".into(),
                GgufMetadataValue::Uint32(1),
            ),
            ("gpt-oss.vocab_size".into(), GgufMetadataValue::Uint32(64)),
        ]);
        SyntheticGguf::with_packed_tensors(&arrays, &metadata, |name, _| {
            (name.contains("ffn_gate_exps.weight")
                || name.contains("ffn_up_exps.weight")
                || name.contains("ffn_down_exps.weight"))
            .then_some(safemlx_gguf::GgmlType::MxFp4)
        })
    }

    #[test]
    fn qwen_tp_pp_preflight_materializes_only_stage_local_tensor_shards() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = gpu.stream();
        for tied in [false, true] {
            let config = dense_qwen_config("qwen3", tied);
            let args = dense_qwen::config_from_hf_value(&config).unwrap();
            let mut source = dense_qwen::Model::new(args, stream).unwrap();
            initialize_parameters(&mut source, stream);
            let directory = tempfile::tempdir().unwrap();
            write_parameter_fixture(directory.path(), &config, &source);

            let models = (0..4)
                .map(|rank| {
                    load_pipeline_model_with_options(
                        directory.path(),
                        ModelLoadOptions::with_parallel(tp_pp_gpu_topology(rank)),
                        stream,
                        cpu.stream(),
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();
            let streamed = (0..4)
                .map(|rank| {
                    load_pipeline_model_with_options(
                        directory.path(),
                        ModelLoadOptions::with_parallel(tp_pp_gpu_topology(rank))
                            .with_weight_residency(WeightResidency::dense_disk_stream(
                                dense_stream_options(),
                            )),
                        stream,
                        cpu.stream(),
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();
            for (rank, model) in models.iter().enumerate() {
                assert_eq!(model.stage_info().topology, tp_pp_gpu_topology(rank));
                assert_eq!(
                    model.stage_info().global_layer_range,
                    rank / 2..rank / 2 + 1
                );
                assert_eq!(
                    model
                        .prompt_cache_model_identity()
                        .unwrap()
                        .topology
                        .tensor_parallel,
                    Some((2, rank % 2))
                );
                assert_eq!(
                    model
                        .prompt_cache_model_identity()
                        .unwrap()
                        .topology
                        .pipeline,
                    Some((2, rank / 2))
                );
                assert_eq!(model.new_cache().unwrap().layers.len(), 1);
                let streamed_info = streamed[rank].stage_info();
                let report = streamed[rank].dense_stream_report().unwrap().unwrap();
                assert_eq!(
                    report.planned_layer_count(),
                    streamed_info.global_layer_range.len()
                );
                assert_eq!(
                    streamed_info.planned_owned_parameter_bytes,
                    model.stage_info().local_parameter_bytes as u64
                );
                assert!(
                    streamed_info.local_parameter_bytes < model.stage_info().local_parameter_bytes
                );
            }
            assert!(models[0]
                .stage_info()
                .owned_tensors
                .iter()
                .any(|name| name.contains("embed_tokens")));
            assert!(
                !models[2]
                    .stage_info()
                    .owned_tensors
                    .iter()
                    .any(|name| name.contains("embed_tokens"))
                    || tied
            );
            assert!(models[2]
                .stage_info()
                .owned_tensors
                .iter()
                .any(|name| name.contains(if tied { "embed_tokens" } else { "lm_head" })));
            assert!(models
                .iter()
                .all(|model| model.stage_info().local_parameter_bytes > 0));
        }
    }

    #[test]
    fn qwen_pp_ep_preflight_places_stage_local_expert_banks() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = gpu.stream();
        let config = dense_qwen_config("qwen3_moe", true);
        let args = dense_qwen::config_from_hf_value(&config).unwrap();
        let mut source = dense_qwen::Model::new(args, stream).unwrap();
        initialize_parameters(&mut source, stream);
        let directory = tempfile::tempdir().unwrap();
        write_split_qwen3_moe_fixture(directory.path(), &config, &source, stream);

        for rank in 0..4 {
            let model = load_pipeline_model_with_options(
                directory.path(),
                ModelLoadOptions::with_parallel(pp_ep_gpu_topology(rank)),
                stream,
                cpu.stream(),
            )
            .unwrap();
            assert_eq!(
                model.stage_info().global_layer_range,
                rank / 2..rank / 2 + 1
            );
            assert_eq!(model.stage_info().global_expert_count, Some(4));
            assert_eq!(
                model.stage_info().local_expert_ids,
                if rank % 2 == 0 {
                    vec![0, 1]
                } else {
                    vec![2, 3]
                }
            );
            let identity = model.prompt_cache_model_identity().unwrap();
            assert_eq!(identity.topology.pipeline, Some((2, rank / 2)));
            assert_eq!(identity.topology.expert_parallel, Some((2, rank % 2)));
            assert_eq!(model.new_cache().unwrap().layers.len(), 1);
            let streamed = load_pipeline_model_with_options(
                directory.path(),
                ModelLoadOptions::with_parallel(pp_ep_gpu_topology(rank)).with_weight_residency(
                    WeightResidency::dense_disk_stream(dense_stream_options()),
                ),
                stream,
                cpu.stream(),
            )
            .unwrap();
            let report = streamed.dense_stream_report().unwrap().unwrap();
            assert_eq!(
                report.planned_layer_count(),
                streamed.stage_info().global_layer_range.len()
            );
            assert_eq!(
                streamed.stage_info().planned_owned_parameter_bytes,
                model.stage_info().local_parameter_bytes as u64
            );
            assert!(
                streamed.stage_info().local_parameter_bytes
                    < model.stage_info().local_parameter_bytes
            );
        }
    }

    #[test]
    fn dense_qwen_pipeline_matches_resident_decode_and_dense_streaming() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = gpu.stream();
        let token_batches = [
            Array::from_slice(&[1u32, 2], &[1, 2]),
            Array::from_slice(&[3u32], &[1, 1]),
        ];

        for model_type in ["qwen2", "qwen3", "qwen3_moe"] {
            for tied in [false, true] {
                let config = dense_qwen_config(model_type, tied);
                let args = dense_qwen::config_from_hf_value(&config).unwrap();
                let mut source = dense_qwen::Model::new(args, stream).unwrap();
                initialize_parameters(&mut source, stream);
                let expected_model_kind = source.args.model_kind();
                let dir = tempfile::tempdir().unwrap();
                if model_type == "qwen3_moe" {
                    write_split_qwen3_moe_fixture(dir.path(), &config, &source, stream);
                } else {
                    write_parameter_fixture(dir.path(), &config, &source);
                }

                let mut reference = source;
                let mut reference_cache = reference.new_cache();
                let expected = token_batches
                    .iter()
                    .map(|tokens| {
                        reference
                            .forward(
                                dense_qwen::ModelInput {
                                    inputs: tokens,
                                    mask: None,
                                    cache: &mut reference_cache,
                                },
                                stream,
                            )
                            .unwrap()
                    })
                    .collect::<Vec<_>>();

                let load = |rank, dense_stream| {
                    let options = ModelLoadOptions::with_parallel(gpu_topology(rank));
                    let options = if dense_stream {
                        options.with_weight_residency(WeightResidency::dense_disk_stream(
                            dense_stream_options(),
                        ))
                    } else {
                        options
                    };
                    load_pipeline_model_with_options(dir.path(), options, stream, cpu.stream())
                        .unwrap()
                };
                let mut first = load(0, false);
                let mut last = load(1, false);
                assert_eq!(first.stage_info().model_kind, expected_model_kind);
                assert_eq!(first.stage_info().global_layer_range, 0..1);
                assert_eq!(last.stage_info().global_layer_range, 1..2);
                assert_eq!(first.prompt_cache_layer_layout().unwrap().len(), 1);
                assert_eq!(last.prompt_cache_layer_layout().unwrap().len(), 1);
                let actual = run_pipeline_sequence(&mut first, &mut last, &token_batches, stream);
                for (actual, expected) in actual.iter().zip(&expected) {
                    assert_close(actual, expected);
                }

                let mut streamed_first = load(0, true);
                let mut streamed_last = load(1, true);
                let streamed = run_pipeline_sequence(
                    &mut streamed_first,
                    &mut streamed_last,
                    &token_batches,
                    stream,
                );
                for (actual, expected) in streamed.iter().zip(&expected) {
                    assert_close(actual, expected);
                }
                for stage in [&streamed_first, &streamed_last] {
                    let report = stage.dense_stream_report().unwrap().unwrap();
                    assert_eq!(report.planned_layer_count(), 1);
                    assert_eq!(report.prefill_forwards(), 1);
                    assert_eq!(report.decode_forwards(), 1);
                }
            }
        }
    }

    #[test]
    fn qwen_hybrid_pipeline_matches_resident_recurrent_decode_and_streaming() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = gpu.stream();
        let config = qwen_hybrid_config();
        let args = qwen_hybrid::model_args_from_config_value(&config).unwrap();
        let mut source = qwen_hybrid::Model::new(args, None, None, None, stream).unwrap();
        initialize_parameters(&mut source, stream);
        let dir = tempfile::tempdir().unwrap();
        write_parameter_fixture(dir.path(), &config, &source);
        let mut reference = crate::architectures::qwen::hybrid::qwen3_next::load_qwen3_next_model(
            dir.path(),
            stream,
            cpu.stream(),
        )
        .unwrap();
        let mut reference_cache = qwen_hybrid::Cache::new(&reference.args).unwrap();
        let token_batches = [
            Array::from_slice(&[1u32, 2], &[1, 2]),
            Array::from_slice(&[3u32], &[1, 1]),
        ];
        let expected = token_batches
            .iter()
            .map(|tokens| {
                reference
                    .forward(
                        qwen_hybrid::ModelInput {
                            inputs: tokens,
                            inputs_embeds: None,
                            mask: None,
                            cache: Some(&mut reference_cache),
                        },
                        stream,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let load = |rank, streamed| {
            let options = ModelLoadOptions::with_parallel(gpu_topology(rank));
            let options = if streamed {
                options.with_weight_residency(WeightResidency::dense_disk_stream(
                    dense_stream_options(),
                ))
            } else {
                options
            };
            load_pipeline_model_with_options(dir.path(), options, stream, cpu.stream()).unwrap()
        };
        for streamed in [false, true] {
            let mut first = load(0, streamed);
            let mut last = load(1, streamed);
            assert_eq!(first.stage_info().model_kind, ModelKind::Qwen3Next);
            assert!(matches!(
                &first.new_cache().unwrap().layers()[0],
                PipelineLayerCache::StateSlots { slots, .. } if slots.len() == 2
            ));
            assert!(matches!(
                &last.new_cache().unwrap().layers()[0],
                PipelineLayerCache::KeyValue { slots, .. } if slots.is_empty()
            ));
            let actual = run_pipeline_sequence(&mut first, &mut last, &token_batches, stream);
            for (actual, expected) in actual.iter().zip(&expected) {
                assert_close(actual, expected);
            }
        }
    }

    #[test]
    fn gpt_oss_pipeline_matches_resident_mixed_attention_and_dense_streaming() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = gpu.stream();
        let config = gpt_oss_config();
        let args = gpt_oss::model_args_from_config_value(&config).unwrap();
        let mut source = gpt_oss::Model::new(args, stream).unwrap();
        initialize_parameters(&mut source, stream);
        let dir = tempfile::tempdir().unwrap();
        write_parameter_fixture(dir.path(), &config, &source);

        let token_batches = [
            Array::from_slice(&[1u32, 2], &[1, 2]),
            Array::from_slice(&[3u32], &[1, 1]),
        ];
        let mut reference = gpt_oss::load_model(dir.path(), stream, cpu.stream()).unwrap();
        let mut reference_cache = reference.new_cache();
        let expected = token_batches
            .iter()
            .map(|tokens| {
                reference
                    .forward(tokens, &mut reference_cache, stream)
                    .unwrap()
            })
            .collect::<Vec<_>>();

        let load = |rank, dense_stream| {
            let options = ModelLoadOptions::with_parallel(gpu_topology(rank));
            let options = if dense_stream {
                options.with_weight_residency(WeightResidency::dense_disk_stream(
                    dense_stream_options(),
                ))
            } else {
                options
            };
            load_pipeline_model_with_options(dir.path(), options, stream, cpu.stream()).unwrap()
        };
        let mut first = load(0, false);
        let mut last = load(1, false);
        assert_eq!(first.stage_info().model_kind, ModelKind::GptOss);
        assert_eq!(first.stage_info().global_layer_range, 0..1);
        assert_eq!(last.stage_info().global_layer_range, 1..2);
        let first_layout = first.prompt_cache_layer_layout().unwrap();
        let last_layout = last.prompt_cache_layer_layout().unwrap();
        assert!(matches!(
            first_layout.get(0),
            Some(crate::LayerCachePolicy::KeyValue {
                attention: crate::AttentionPolicy::Sliding { window },
                ..
            }) if window.get() == 3
        ));
        assert!(matches!(
            last_layout.get(0),
            Some(crate::LayerCachePolicy::KeyValue {
                attention: crate::AttentionPolicy::Full,
                ..
            })
        ));
        let actual = run_pipeline_sequence(&mut first, &mut last, &token_batches, stream);
        for (actual, expected) in actual.iter().zip(&expected) {
            assert_close(actual, expected);
        }

        let mut streamed_first = load(0, true);
        let mut streamed_last = load(1, true);
        let streamed = run_pipeline_sequence(
            &mut streamed_first,
            &mut streamed_last,
            &token_batches,
            stream,
        );
        for (actual, expected) in streamed.iter().zip(&expected) {
            assert_close(actual, expected);
        }
        for stage in [&streamed_first, &streamed_last] {
            let report = stage.dense_stream_report().unwrap().unwrap();
            assert_eq!(report.planned_layer_count(), 1);
            assert_eq!(report.prefill_forwards(), 1);
            assert_eq!(report.decode_forwards(), 1);
        }
    }

    #[test]
    fn dense_qwen_and_gpt_oss_gguf_pipelines_match_resident_decoders() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = gpu.stream();
        let tokens = [Array::from_slice(&[1u32, 2], &[1, 2])];

        let qwen_config = dense_qwen_config("qwen2", false);
        let qwen_args = dense_qwen::config_from_hf_value(&qwen_config).unwrap();
        let mut qwen_source = dense_qwen::Model::new(qwen_args, stream).unwrap();
        initialize_parameters(&mut qwen_source, stream);
        let qwen_fixture = dense_qwen_gguf_fixture(&qwen_source, stream);
        let mut qwen_reference =
            dense_qwen::load_gguf(qwen_fixture.path(), stream, cpu.stream()).unwrap();
        let mut qwen_cache = qwen_reference.new_cache();
        let qwen_expected = qwen_reference
            .forward(
                dense_qwen::ModelInput {
                    inputs: &tokens[0],
                    mask: None,
                    cache: &mut qwen_cache,
                },
                stream,
            )
            .unwrap();
        let mut qwen_first = load_pipeline_model_with_options(
            qwen_fixture.path(),
            ModelLoadOptions::with_parallel(gpu_topology(0)),
            stream,
            cpu.stream(),
        )
        .unwrap();
        let mut qwen_last = load_pipeline_model_with_options(
            qwen_fixture.path(),
            ModelLoadOptions::with_parallel(gpu_topology(1)),
            stream,
            cpu.stream(),
        )
        .unwrap();
        let qwen_actual = run_pipeline_sequence(&mut qwen_first, &mut qwen_last, &tokens, stream);
        assert_close(&qwen_actual[0], &qwen_expected);

        let gpt_fixture = gpt_oss_gguf_fixture(stream);
        let mut gpt_reference =
            gpt_oss::load_gguf(gpt_fixture.path(), stream, cpu.stream()).unwrap();
        let mut gpt_cache = gpt_reference.new_cache();
        let gpt_expected = gpt_reference
            .forward(&tokens[0], &mut gpt_cache, stream)
            .unwrap();
        let mut gpt_first = load_pipeline_model_with_options(
            gpt_fixture.path(),
            ModelLoadOptions::with_parallel(gpu_topology(0)),
            stream,
            cpu.stream(),
        )
        .unwrap();
        let mut gpt_last = load_pipeline_model_with_options(
            gpt_fixture.path(),
            ModelLoadOptions::with_parallel(gpu_topology(1)),
            stream,
            cpu.stream(),
        )
        .unwrap();
        let gpt_actual = run_pipeline_sequence(&mut gpt_first, &mut gpt_last, &tokens, stream);
        assert_close(&gpt_actual[0], &gpt_expected);
    }

    #[test]
    fn dense_qwen_gguf_tp_pp_uses_rank_local_selections() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = gpu.stream();
        let config = dense_qwen_config("qwen3", false);
        let args = dense_qwen::config_from_hf_value(&config).unwrap();
        let mut source = dense_qwen::Model::new(args, stream).unwrap();
        initialize_parameters(&mut source, stream);
        let fixture = dense_qwen_gguf_fixture(&source, stream);
        for rank in 0..4 {
            let model = load_pipeline_model_with_options(
                fixture.path(),
                ModelLoadOptions::with_parallel(tp_pp_gpu_topology(rank)),
                stream,
                cpu.stream(),
            )
            .unwrap();
            let info = model.stage_info();
            assert_eq!(info.global_layer_range, rank / 2..rank / 2 + 1);
            assert_eq!(info.topology.tensor_parallel_rank, rank % 2);
            assert!(!info.opened_checkpoint_shards.is_empty());
            assert!(info.local_parameter_bytes > 0);
            let reads = model.checkpoint_diagnostics().unwrap().unwrap();
            assert!(reads.physical_reads > 0);
            assert!(reads.physical_read_bytes > 0);

            let streamed = load_pipeline_model_with_options(
                fixture.path(),
                ModelLoadOptions::with_parallel(tp_pp_gpu_topology(rank)).with_weight_residency(
                    WeightResidency::dense_disk_stream(dense_stream_options()),
                ),
                stream,
                cpu.stream(),
            )
            .unwrap();
            let report = streamed.dense_stream_report().unwrap().unwrap();
            assert_eq!(report.planned_layer_count(), 1);
            assert_eq!(
                streamed.stage_info().planned_owned_parameter_bytes,
                model.stage_info().local_parameter_bytes as u64
            );
            assert!(
                streamed.stage_info().local_parameter_bytes
                    < model.stage_info().local_parameter_bytes
            );
            let streamed_reads = streamed.checkpoint_diagnostics().unwrap().unwrap();
            assert!(streamed_reads.physical_read_bytes < reads.physical_read_bytes);
        }
    }

    #[test]
    fn gpt_oss_tp_pp_preflight_owns_packed_shards_boundaries_and_local_caches() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = gpu.stream();
        let config = gpt_oss_config();
        let args = gpt_oss::model_args_from_config_value(&config).unwrap();
        let mut source = gpt_oss::Model::new(args, stream).unwrap();
        initialize_parameters(&mut source, stream);
        let safetensors = tempfile::tempdir().unwrap();
        write_parameter_fixture(safetensors.path(), &config, &source);
        let gguf = gpt_oss_gguf_fixture(stream);

        for path in [safetensors.path(), gguf.path()] {
            for rank in 0..4 {
                let topology = tp_pp_gpu_topology(rank);
                let resident = load_pipeline_model_with_options(
                    path,
                    ModelLoadOptions::with_parallel(topology),
                    stream,
                    cpu.stream(),
                )
                .unwrap();
                let streamed = load_pipeline_model_with_options(
                    path,
                    ModelLoadOptions::with_parallel(topology).with_weight_residency(
                        WeightResidency::dense_disk_stream(dense_stream_options()),
                    ),
                    stream,
                    cpu.stream(),
                )
                .unwrap();

                for model in [&resident, &streamed] {
                    let info = model.stage_info();
                    assert_eq!(info.topology, topology);
                    assert_eq!(
                        info.global_layer_range,
                        topology.pipeline_parallel_rank..topology.pipeline_parallel_rank + 1
                    );
                    let layout = model.prompt_cache_layer_layout().unwrap();
                    assert!(matches!(
                        layout.get(0),
                        Some(crate::LayerCachePolicy::KeyValue {
                            num_key_value_heads,
                            head_dim,
                            ..
                        }) if num_key_value_heads.get() == 1 && head_dim.get() == 32
                    ));
                    if info.is_first {
                        assert!(info
                            .owned_tensors
                            .iter()
                            .any(|name| name == "model.embed_tokens.weight"));
                    }
                    if info.is_last {
                        assert!(info
                            .owned_tensors
                            .iter()
                            .any(|name| name == "lm_head.weight"));
                    }
                }
                let report = streamed.dense_stream_report().unwrap().unwrap();
                assert_eq!(
                    report.planned_layer_count(),
                    streamed.stage_info().global_layer_range.len()
                );
                assert_eq!(
                    streamed.stage_info().planned_owned_parameter_bytes,
                    resident.stage_info().local_parameter_bytes as u64
                );
                assert!(
                    streamed.stage_info().local_parameter_bytes
                        < resident.stage_info().local_parameter_bytes
                );
                let resident_reads = resident.checkpoint_diagnostics().unwrap().unwrap();
                let streamed_reads = streamed.checkpoint_diagnostics().unwrap().unwrap();
                assert!(
                    streamed_reads.physical_read_bytes <= resident_reads.physical_read_bytes,
                    "streamed GPT-OSS planning exceeded resident checkpoint reads"
                );
            }
        }
    }

    #[test]
    fn gpt_oss_pp_ep_preflight_owns_stage_local_packed_experts() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = gpu.stream();
        let config = gpt_oss_config();
        let args = gpt_oss::model_args_from_config_value(&config).unwrap();
        let mut source = gpt_oss::Model::new(args, stream).unwrap();
        initialize_parameters(&mut source, stream);
        let safetensors = tempfile::tempdir().unwrap();
        write_parameter_fixture(safetensors.path(), &config, &source);
        let gguf = gpt_oss_gguf_fixture(stream);

        for path in [safetensors.path(), gguf.path()] {
            for rank in 0..4 {
                let topology = pp_ep_gpu_topology(rank);
                let resident = load_pipeline_model_with_options(
                    path,
                    ModelLoadOptions::with_parallel(topology),
                    stream,
                    cpu.stream(),
                )
                .unwrap();
                let streamed = load_pipeline_model_with_options(
                    path,
                    ModelLoadOptions::with_parallel(topology).with_weight_residency(
                        WeightResidency::dense_disk_stream(dense_stream_options()),
                    ),
                    stream,
                    cpu.stream(),
                )
                .unwrap();

                for model in [&resident, &streamed] {
                    let info = model.stage_info();
                    assert_eq!(info.topology, topology);
                    assert_eq!(info.global_expert_count, Some(2));
                    assert_eq!(info.local_expert_ids, vec![topology.expert_parallel_rank]);
                    assert_eq!(
                        info.global_layer_range,
                        topology.pipeline_parallel_rank..topology.pipeline_parallel_rank + 1
                    );
                    assert!(info.local_parameter_bytes > 0);
                    let layout = model.prompt_cache_layer_layout().unwrap();
                    assert_eq!(layout.len(), 1);
                }
                let report = streamed.dense_stream_report().unwrap().unwrap();
                assert_eq!(report.planned_layer_count(), 1);
                assert_eq!(
                    streamed.stage_info().planned_owned_parameter_bytes,
                    resident.stage_info().local_parameter_bytes as u64
                );
                assert!(
                    streamed.stage_info().local_parameter_bytes
                        < resident.stage_info().local_parameter_bytes
                );
                let resident_reads = resident.checkpoint_diagnostics().unwrap().unwrap();
                let streamed_reads = streamed.checkpoint_diagnostics().unwrap().unwrap();
                assert!(
                    streamed_reads.physical_read_bytes <= resident_reads.physical_read_bytes,
                    "streamed GPT-OSS PP+EP planning exceeded resident checkpoint reads"
                );
            }
        }
    }

    #[test]
    fn gpt_oss_gguf_tp_ep_preflight_uses_sharded_nonexperts_and_owned_experts() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = gpu.stream();
        let fixture = gpt_oss_gguf_fixture(stream);

        for rank in 0..4 {
            let topology = ParallelTopology::from_rank(
                4,
                rank,
                2,
                1,
                2,
                DeviceAssignment::new(DeviceType::Gpu, 0),
            )
            .unwrap();
            let residency = WeightResidency::with_expert_cache(
                crate::NonExpertWeightResidency::DenseDiskStream(dense_stream_options()),
                crate::runtime::residency::expert_cache::ExpertCacheLoadOptions::default(),
            );
            let loaded =
                crate::architectures::distributed::expert::load_expert_parallel_model_with_options(
                    fixture.path(),
                    ModelLoadOptions::with_parallel(topology).with_weight_residency(residency),
                    stream,
                    cpu.stream(),
                )
                .unwrap();
            assert_eq!(loaded.info().topology, topology);
            assert_eq!(
                loaded.info().assignment.local_global_expert_ids(),
                &[topology.expert_parallel_rank]
            );
            assert!(loaded.info().owned_expert_bytes > 0);
            let identity = loaded.prompt_cache_layer_layout().unwrap();
            assert!(matches!(
                identity.get(0),
                Some(crate::LayerCachePolicy::KeyValue {
                    num_key_value_heads,
                    head_dim,
                    ..
                }) if num_key_value_heads.get() == 1 && head_dim.get() == 32
            ));
            let report = loaded.dense_stream_report().unwrap().unwrap();
            assert_eq!(report.planned_layer_count(), 2);
        }
    }

    #[test]
    fn lfm2_cartesian_pipeline_preflight_composes_hybrid_state_and_experts() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = gpu.stream();

        for moe in [false, true] {
            let config = lfm2_config(moe);
            let args = lfm2::model_args_from_config_value(&config).unwrap();
            let mut source = lfm2::Model::new(args, stream).unwrap();
            initialize_parameters(&mut source, stream);
            let fixture = tempfile::tempdir().unwrap();
            write_parameter_fixture(fixture.path(), &config, &source);

            for rank in 0..4 {
                let topology = if moe {
                    pp_ep_gpu_topology(rank)
                } else {
                    tp_pp_gpu_topology(rank)
                };
                for residency in [
                    WeightResidency::fully_resident(),
                    WeightResidency::dense_disk_stream(dense_stream_options()),
                ] {
                    let loaded = load_pipeline_model_with_options(
                        fixture.path(),
                        ModelLoadOptions::with_parallel(topology).with_weight_residency(residency),
                        stream,
                        cpu.stream(),
                    )
                    .unwrap();
                    let info = loaded.stage_info();
                    assert_eq!(info.topology, topology);
                    assert_eq!(
                        info.global_layer_range,
                        topology.pipeline_parallel_rank..topology.pipeline_parallel_rank + 1
                    );
                    let layout = loaded.prompt_cache_layer_layout().unwrap();
                    assert_eq!(layout.len(), 1);
                    if moe {
                        assert_eq!(info.global_expert_count, Some(2));
                        let expected = if topology.pipeline_parallel_rank == 1 {
                            vec![topology.expert_parallel_rank]
                        } else {
                            Vec::new()
                        };
                        assert_eq!(info.local_expert_ids, expected);
                    } else {
                        assert_eq!(info.global_expert_count, None);
                        match layout.get(0).unwrap() {
                            crate::LayerCachePolicy::KeyValue {
                                num_key_value_heads,
                                ..
                            } => assert!((1..=2).contains(&num_key_value_heads.get())),
                            crate::LayerCachePolicy::FixedState { tensors } => {
                                assert_eq!(tensors.len(), 1)
                            }
                            other => panic!("unexpected LFM2 TP+PP cache policy {other:?}"),
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn lfm2_moe_triple_axis_preflight_composes_resident_and_cached_ownership() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = gpu.stream();
        let config = lfm2_config(true);
        let args = lfm2::model_args_from_config_value(&config).unwrap();
        let mut source = lfm2::Model::new(args, stream).unwrap();
        initialize_parameters(&mut source, stream);
        let fixture = tempfile::tempdir().unwrap();
        write_parameter_fixture(fixture.path(), &config, &source);

        for rank in 0..8 {
            let topology = tp_pp_ep_gpu_topology(rank);
            for residency in [
                WeightResidency::fully_resident(),
                WeightResidency::with_expert_cache(
                    crate::NonExpertWeightResidency::DenseDiskStream(dense_stream_options()),
                    crate::runtime::residency::expert_cache::ExpertCacheLoadOptions::default(),
                ),
            ] {
                let loaded = load_pipeline_model_with_options(
                    fixture.path(),
                    ModelLoadOptions::with_parallel(topology).with_weight_residency(residency),
                    stream,
                    cpu.stream(),
                )
                .unwrap();
                let info = loaded.stage_info();
                assert_eq!(info.topology, topology);
                assert_eq!(
                    info.global_layer_range,
                    topology.pipeline_parallel_rank..topology.pipeline_parallel_rank + 1
                );
                let expected_experts = if topology.pipeline_parallel_rank == 1 {
                    vec![topology.expert_parallel_rank]
                } else {
                    Vec::new()
                };
                assert_eq!(info.local_expert_ids, expected_experts);
                assert_eq!(loaded.prompt_cache_layer_layout().unwrap().len(), 1);
                if residency.expert_cache().is_some() {
                    let report = loaded.expert_cache_report().unwrap();
                    assert_eq!(report.is_some(), !expected_experts.is_empty());
                    if let Some(report) = report {
                        assert_eq!(report.owned_experts, expected_experts.len());
                    }
                    assert!(loaded.dense_stream_report().unwrap().is_some());
                }
            }
        }
    }

    #[test]
    fn qwen_hybrid_moe_triple_axis_preflight_composes_state_and_expert_residency() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = gpu.stream();

        for model_type in ["qwen3_next", "qwen3_5_moe_text"] {
            let config = qwen_hybrid_moe_config(model_type);
            let args = qwen_hybrid::model_args_from_config_value(&config).unwrap();
            let mut source = qwen_hybrid::Model::new(args, None, None, None, stream).unwrap();
            initialize_parameters(&mut source, stream);
            let fixture = tempfile::tempdir().unwrap();
            write_parameter_fixture(fixture.path(), &config, &source);

            for rank in 0..12 {
                let topology = ParallelTopology::from_rank(
                    12,
                    rank,
                    2,
                    2,
                    3,
                    DeviceAssignment::new(DeviceType::Gpu, 0),
                )
                .unwrap();
                for residency in [
                    WeightResidency::fully_resident(),
                    WeightResidency::with_expert_cache(
                        crate::NonExpertWeightResidency::DenseDiskStream(dense_stream_options()),
                        crate::runtime::residency::expert_cache::ExpertCacheLoadOptions::default(),
                    ),
                ] {
                    let loaded = load_pipeline_model_with_options(
                        fixture.path(),
                        ModelLoadOptions::with_parallel(topology).with_weight_residency(residency),
                        stream,
                        cpu.stream(),
                    )
                    .unwrap();
                    let info = loaded.stage_info();
                    assert_eq!(info.topology, topology);
                    assert_eq!(
                        info.global_layer_range,
                        topology.pipeline_parallel_rank..topology.pipeline_parallel_rank + 1
                    );
                    assert_eq!(info.local_expert_ids, vec![topology.expert_parallel_rank]);
                    let layout = loaded.prompt_cache_layer_layout().unwrap();
                    assert_eq!(layout.len(), 1);
                    if topology.pipeline_parallel_rank == 0 {
                        assert!(matches!(
                            layout.get(0),
                            Some(crate::LayerCachePolicy::FixedState { tensors }) if tensors.len() == 2
                        ));
                    } else {
                        assert!(matches!(
                            layout.get(0),
                            Some(crate::LayerCachePolicy::KeyValue { .. })
                        ));
                    }
                    if residency.expert_cache().is_some() {
                        let report = loaded.expert_cache_report().unwrap().unwrap();
                        assert_eq!(report.owned_experts, 1);
                        assert!(loaded.dense_stream_report().unwrap().is_some());
                    }
                }
            }
        }
    }

    #[test]
    fn qwen3_moe_gguf_pp_ep_streams_only_stage_local_experts() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = gpu.stream();
        let config = dense_qwen_config("qwen3_moe", true);
        let args = dense_qwen::config_from_hf_value(&config).unwrap();
        let mut source = dense_qwen::Model::new(args, stream).unwrap();
        initialize_parameters(&mut source, stream);
        let fixture = dense_qwen_gguf_fixture(&source, stream);

        for rank in 0..4 {
            let resident = load_pipeline_model_with_options(
                fixture.path(),
                ModelLoadOptions::with_parallel(pp_ep_gpu_topology(rank)),
                stream,
                cpu.stream(),
            )
            .unwrap();
            let streamed = load_pipeline_model_with_options(
                fixture.path(),
                ModelLoadOptions::with_parallel(pp_ep_gpu_topology(rank)).with_weight_residency(
                    WeightResidency::dense_disk_stream(dense_stream_options()),
                ),
                stream,
                cpu.stream(),
            )
            .unwrap();
            assert_eq!(
                streamed.stage_info().local_expert_ids,
                resident.stage_info().local_expert_ids
            );
            assert_eq!(
                streamed.stage_info().planned_owned_parameter_bytes,
                resident.stage_info().local_parameter_bytes as u64
            );
            assert!(
                streamed.stage_info().local_parameter_bytes
                    < resident.stage_info().local_parameter_bytes
            );
            let report = streamed.dense_stream_report().unwrap().unwrap();
            assert_eq!(report.planned_layer_count(), 1);
            let resident_reads = resident.checkpoint_diagnostics().unwrap().unwrap();
            let streamed_reads = streamed.checkpoint_diagnostics().unwrap().unwrap();
            assert!(streamed_reads.physical_read_bytes < resident_reads.physical_read_bytes);
        }
    }

    #[test]
    fn combined_qwen_streamed_recipes_match_resident_safetensors_and_gguf_layers() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = gpu.stream();

        let dense_config = dense_qwen_config("qwen3", false);
        let dense_args = dense_qwen::config_from_hf_value(&dense_config).unwrap();
        let mut dense_source = dense_qwen::Model::new(dense_args.clone(), stream).unwrap();
        initialize_parameters(&mut dense_source, stream);
        let dense_directory = tempfile::tempdir().unwrap();
        write_parameter_fixture(dense_directory.path(), &dense_config, &dense_source);
        assert_nonresident_qwen_layer_recipe_parity(
            dense_args.clone(),
            open_safetensors_weight_store(dense_directory.path(), 1).unwrap(),
            tp_pp_gpu_topology(3),
            None,
            PipelineLayerLoadOptions::DenseDiskStream(dense_stream_options()),
            1,
            stream,
            cpu.stream(),
        );
        assert_nonresident_qwen_layer_recipe_parity(
            dense_args.clone(),
            open_safetensors_weight_store(dense_directory.path(), 1).unwrap(),
            tp_pp_gpu_topology(3),
            None,
            PipelineLayerLoadOptions::LayerwiseHost(LayerwiseLoadOptions::new(
                OffloadConfig::new(None, None, 1).unwrap(),
            )),
            1,
            stream,
            cpu.stream(),
        );

        let dense_gguf = dense_qwen_gguf_fixture(&dense_source, stream);
        let checkpoint = GgufCheckpoint::open(dense_gguf.path()).unwrap();
        let metadata = crate::runtime::checkpoint::load::gguf_metadata(&checkpoint);
        let (gguf_dense_args, _) =
            dense_qwen::prepare_gguf_checkpoint(&checkpoint, &metadata, "qwen3", false).unwrap();
        let dense_store: SharedWeightStore = Arc::new(
            GgufWeightStore::new_with_max_mapped_shards(
                checkpoint,
                |name| dense_qwen::translate_gguf_weight_name(name, false),
                1,
            )
            .unwrap(),
        );
        assert_nonresident_qwen_layer_recipe_parity(
            gguf_dense_args,
            dense_store,
            tp_pp_gpu_topology(3),
            None,
            PipelineLayerLoadOptions::DenseDiskStream(dense_stream_options()),
            1,
            stream,
            cpu.stream(),
        );

        let moe_config = dense_qwen_config("qwen3_moe", true);
        let moe_args = dense_qwen::config_from_hf_value(&moe_config).unwrap();
        let mut moe_source = dense_qwen::Model::new(moe_args.clone(), stream).unwrap();
        initialize_parameters(&mut moe_source, stream);
        let moe_directory = tempfile::tempdir().unwrap();
        write_split_qwen3_moe_fixture(moe_directory.path(), &moe_config, &moe_source, stream);
        let assignment = ExpertAssignment::balanced(4, 2, 1).unwrap();
        assert_nonresident_qwen_layer_recipe_parity(
            moe_args.clone(),
            open_safetensors_weight_store(moe_directory.path(), 1).unwrap(),
            pp_ep_gpu_topology(3),
            Some(assignment.clone()),
            PipelineLayerLoadOptions::DenseDiskStream(dense_stream_options()),
            1,
            stream,
            cpu.stream(),
        );

        let moe_gguf = dense_qwen_gguf_fixture(&moe_source, stream);
        let checkpoint = GgufCheckpoint::open(moe_gguf.path()).unwrap();
        let metadata = crate::runtime::checkpoint::load::gguf_metadata(&checkpoint);
        let (gguf_moe_args, _) =
            dense_qwen::prepare_gguf_checkpoint(&checkpoint, &metadata, "qwen3moe", true).unwrap();
        let moe_store: SharedWeightStore = Arc::new(
            GgufWeightStore::new_with_max_mapped_shards(
                checkpoint,
                |name| dense_qwen::translate_gguf_weight_name(name, true),
                1,
            )
            .unwrap(),
        );
        assert_nonresident_qwen_layer_recipe_parity(
            gguf_moe_args,
            moe_store,
            pp_ep_gpu_topology(3),
            Some(assignment),
            PipelineLayerLoadOptions::DenseDiskStream(dense_stream_options()),
            1,
            stream,
            cpu.stream(),
        );
    }

    const COMBINED_STREAM_WORKER: &str = "SAFEMLX_COMBINED_STREAM_WORKER";
    const COMBINED_STREAM_DENSE_ST: &str = "SAFEMLX_COMBINED_STREAM_DENSE_ST";
    const COMBINED_STREAM_DENSE_GGUF: &str = "SAFEMLX_COMBINED_STREAM_DENSE_GGUF";
    const COMBINED_STREAM_MOE_ST: &str = "SAFEMLX_COMBINED_STREAM_MOE_ST";
    const COMBINED_STREAM_MOE_GGUF: &str = "SAFEMLX_COMBINED_STREAM_MOE_GGUF";
    const COMBINED_STREAM_LLAMA_ST: &str = "SAFEMLX_COMBINED_STREAM_LLAMA_ST";
    const COMBINED_STREAM_LLAMA_GGUF: &str = "SAFEMLX_COMBINED_STREAM_LLAMA_GGUF";
    const COMBINED_STREAM_GPT_OSS_ST: &str = "SAFEMLX_COMBINED_STREAM_GPT_OSS_ST";
    const COMBINED_STREAM_GPT_OSS_GGUF: &str = "SAFEMLX_COMBINED_STREAM_GPT_OSS_GGUF";
    const COMBINED_STREAM_GEMMA_ST: &str = "SAFEMLX_COMBINED_STREAM_GEMMA_ST";
    const COMBINED_STREAM_GEMMA_GGUF: &str = "SAFEMLX_COMBINED_STREAM_GEMMA_GGUF";

    #[allow(clippy::too_many_arguments)]
    fn combined_stream_stage_output(
        model: &mut PipelineModel,
        cache: &mut PipelineCache,
        tokens: &Array,
        topology: ParallelTopology,
        group: &Group,
        tensor_parallel: bool,
        expert_parallel: bool,
        stream: &Stream,
    ) -> Array {
        let step = PipelineStep::new(1, tokens.shape()[1]).unwrap();
        let hidden = Array::full::<f32>(
            &[1, tokens.shape()[1], model.info.hidden_size],
            Array::from_f32(0.02),
            stream,
        )
        .unwrap();
        let auxiliary = PipelineAuxiliaryState::new(
            model
                .auxiliary_shapes(step)
                .into_iter()
                .map(|shape| Array::full::<f32>(&shape, Array::from_f32(0.01), stream).unwrap())
                .collect(),
        );
        let payload = PipelinePayload { hidden, auxiliary };
        let input = if topology.pipeline_parallel_rank == 0 {
            PipelineStageInput::Tokens(tokens)
        } else {
            PipelineStageInput::Hidden(&payload)
        };
        let execution = tensor_parallel
            .then(|| ParallelExecutionContext::tensor_parallel(topology, group, stream).unwrap());
        let output = model
            .stage
            .forward_with_execution(
                input,
                step,
                None,
                &mut cache.layers,
                execution.as_ref(),
                expert_parallel.then_some(group),
                stream,
            )
            .unwrap();
        match output {
            PipelineStageOutput::Hidden(payload) => payload.hidden,
            PipelineStageOutput::Logits(logits) => logits,
        }
    }

    #[test]
    fn combined_pipeline_streaming_ring_worker() {
        let Some(rank) = std::env::var_os(COMBINED_STREAM_WORKER) else {
            return;
        };
        let rank = rank.to_string_lossy().parse::<usize>().unwrap();
        let group = distributed::init(true, Backend::Ring).unwrap();
        assert_eq!((group.rank(), group.size()), (rank, 2));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = cpu.stream();
        let cases = [
            (COMBINED_STREAM_DENSE_ST, true, false),
            (COMBINED_STREAM_DENSE_GGUF, true, true),
            (COMBINED_STREAM_MOE_ST, false, false),
            (COMBINED_STREAM_MOE_GGUF, false, true),
            (COMBINED_STREAM_LLAMA_ST, true, false),
            (COMBINED_STREAM_LLAMA_GGUF, true, true),
            (COMBINED_STREAM_GPT_OSS_ST, true, false),
            (COMBINED_STREAM_GPT_OSS_GGUF, true, true),
            (COMBINED_STREAM_GPT_OSS_ST, false, false),
            (COMBINED_STREAM_GPT_OSS_GGUF, false, true),
            (COMBINED_STREAM_GEMMA_ST, true, false),
            (COMBINED_STREAM_GEMMA_GGUF, true, true),
        ];
        for (path_variable, tensor_parallel, _gguf) in cases {
            let path = std::env::var_os(path_variable).expect("combined fixture path");
            for pipeline_rank in 0..2 {
                let global_rank = pipeline_rank * 2 + rank;
                let topology = ParallelTopology::from_rank(
                    4,
                    global_rank,
                    if tensor_parallel { 2 } else { 1 },
                    2,
                    if tensor_parallel { 1 } else { 2 },
                    DeviceAssignment::new(DeviceType::Cpu, 0),
                )
                .unwrap();
                let mut resident = load_pipeline_model_with_options(
                    &path,
                    ModelLoadOptions::with_parallel(topology),
                    stream,
                    stream,
                )
                .unwrap();
                let mut streamed = load_pipeline_model_with_options(
                    &path,
                    ModelLoadOptions::with_parallel(topology).with_weight_residency(
                        WeightResidency::dense_disk_stream(dense_stream_options()),
                    ),
                    stream,
                    stream,
                )
                .unwrap();
                let mut resident_cache = resident.new_cache().unwrap();
                let mut streamed_cache = streamed.new_cache().unwrap();
                let gpt_oss_pp_ep = !tensor_parallel
                    && matches!(
                        path_variable,
                        COMBINED_STREAM_GPT_OSS_ST | COMBINED_STREAM_GPT_OSS_GGUF
                    );
                let reference_topology = ParallelTopology::from_rank(
                    2,
                    pipeline_rank,
                    1,
                    2,
                    1,
                    DeviceAssignment::new(DeviceType::Cpu, 0),
                )
                .unwrap();
                let mut reference = gpt_oss_pp_ep
                    .then(|| {
                        load_pipeline_model_with_options(
                            &path,
                            ModelLoadOptions::with_parallel(reference_topology),
                            stream,
                            stream,
                        )
                    })
                    .transpose()
                    .unwrap();
                let mut reference_cache =
                    reference.as_ref().map(|model| model.new_cache().unwrap());
                for tokens in [
                    Array::from_slice(&[1u32, 2], &[1, 2]),
                    Array::from_slice(&[3u32], &[1, 1]),
                ] {
                    let expected = combined_stream_stage_output(
                        &mut resident,
                        &mut resident_cache,
                        &tokens,
                        topology,
                        &group,
                        tensor_parallel,
                        !tensor_parallel,
                        stream,
                    );
                    let actual = combined_stream_stage_output(
                        &mut streamed,
                        &mut streamed_cache,
                        &tokens,
                        topology,
                        &group,
                        tensor_parallel,
                        !tensor_parallel,
                        stream,
                    );
                    assert_close(&actual, &expected);
                    if let (Some(reference), Some(reference_cache)) =
                        (reference.as_mut(), reference_cache.as_mut())
                    {
                        let baseline = combined_stream_stage_output(
                            reference,
                            reference_cache,
                            &tokens,
                            reference_topology,
                            &group,
                            false,
                            false,
                            stream,
                        );
                        assert_close(&expected, &baseline);
                    }
                }
                let report = streamed.dense_stream_report().unwrap().unwrap();
                assert_eq!(
                    report.planned_layer_count(),
                    streamed.stage_info().global_layer_range.len()
                );
                assert_eq!(report.prefill_forwards(), 1);
                assert_eq!(report.decode_forwards(), 1);
            }
        }
    }

    struct CombinedStreamChildren(Vec<Child>);

    impl CombinedStreamChildren {
        fn finish(mut self) -> Vec<Output> {
            self.0
                .drain(..)
                .map(|child| child.wait_with_output().unwrap())
                .collect()
        }
    }

    impl Drop for CombinedStreamChildren {
        fn drop(&mut self) {
            for child in &mut self.0 {
                let _ = child.kill();
            }
            for child in &mut self.0 {
                let _ = child.wait();
            }
        }
    }

    #[test]
    #[ignore = "spawns two Ring workers and opens loopback sockets; run explicitly"]
    fn combined_pipeline_dense_stream_stage_parity_ring() {
        assert!(distributed::is_available(Backend::Ring));
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = gpu.stream();

        let dense_config = dense_qwen_config("qwen3", false);
        let dense_args = dense_qwen::config_from_hf_value(&dense_config).unwrap();
        let mut dense_source = dense_qwen::Model::new(dense_args, stream).unwrap();
        initialize_parameters(&mut dense_source, stream);
        let dense_directory = tempfile::tempdir().unwrap();
        write_parameter_fixture(dense_directory.path(), &dense_config, &dense_source);
        let dense_gguf = dense_qwen_gguf_fixture(&dense_source, stream);

        let moe_config = dense_qwen_config("qwen3_moe", true);
        let moe_args = dense_qwen::config_from_hf_value(&moe_config).unwrap();
        let mut moe_source = dense_qwen::Model::new(moe_args, stream).unwrap();
        initialize_parameters(&mut moe_source, stream);
        let moe_directory = tempfile::tempdir().unwrap();
        write_split_qwen3_moe_fixture(moe_directory.path(), &moe_config, &moe_source, stream);
        let moe_gguf = dense_qwen_gguf_fixture(&moe_source, stream);

        let llama_args = llama_args(true);
        let mut llama_source = llama::ResidentModel::new(llama_args, stream).unwrap();
        initialize_parameters(&mut llama_source, stream);
        let llama_directory = tempfile::tempdir().unwrap();
        write_llama_fixture(llama_directory.path(), &llama_source, false);
        let llama_gguf = llama_gguf_fixture(&llama_source);

        let gpt_oss_config = gpt_oss_config();
        let gpt_oss_args = gpt_oss::model_args_from_config_value(&gpt_oss_config).unwrap();
        let mut gpt_oss_source = gpt_oss::Model::new(gpt_oss_args, stream).unwrap();
        initialize_parameters(&mut gpt_oss_source, stream);
        let gpt_oss_directory = tempfile::tempdir().unwrap();
        write_parameter_fixture(gpt_oss_directory.path(), &gpt_oss_config, &gpt_oss_source);
        let gpt_oss_gguf = gpt_oss_gguf_fixture(stream);

        let gemma_config = gemma_config();
        let gemma_args =
            gemma4::model_args_from_config_value(&gemma_config["text_config"]).unwrap();
        let mut gemma_source = gemma4::Model::new(gemma_args, stream).unwrap();
        initialize_parameters(&mut gemma_source, stream);
        let gemma_directory = tempfile::tempdir().unwrap();
        write_gemma_fixture(gemma_directory.path(), &gemma_source);
        let gemma_gguf = gemma_gguf_fixture(&gemma_source, stream);

        let sockets = (0..2)
            .map(|_| TcpListener::bind(("127.0.0.1", 0)).unwrap())
            .collect::<Vec<_>>();
        let hosts = sockets
            .iter()
            .map(|socket| vec![format!("127.0.0.1:{}", socket.local_addr().unwrap().port())])
            .collect::<Vec<_>>();
        let ring_directory = tempfile::tempdir().unwrap();
        let hostfile = ring_directory.path().join("ring-hosts.json");
        fs::write(&hostfile, serde_json::to_vec(&hosts).unwrap()).unwrap();
        drop(sockets);

        let executable = std::env::current_exe().unwrap();
        let worker_name =
            "architectures::distributed::pipeline::tests::combined_pipeline_streaming_ring_worker";
        let mut children = CombinedStreamChildren(Vec::with_capacity(2));
        for rank in 0..2 {
            children.0.push(
                Command::new(&executable)
                    .args(["--exact", worker_name, "--nocapture"])
                    .env(COMBINED_STREAM_WORKER, rank.to_string())
                    .env("MLX_RANK", rank.to_string())
                    .env("MLX_HOSTFILE", &hostfile)
                    .env(COMBINED_STREAM_DENSE_ST, dense_directory.path())
                    .env(COMBINED_STREAM_DENSE_GGUF, dense_gguf.path())
                    .env(COMBINED_STREAM_MOE_ST, moe_directory.path())
                    .env(COMBINED_STREAM_MOE_GGUF, moe_gguf.path())
                    .env(COMBINED_STREAM_LLAMA_ST, llama_directory.path())
                    .env(COMBINED_STREAM_LLAMA_GGUF, llama_gguf.path())
                    .env(COMBINED_STREAM_GPT_OSS_ST, gpt_oss_directory.path())
                    .env(COMBINED_STREAM_GPT_OSS_GGUF, gpt_oss_gguf.path())
                    .env(COMBINED_STREAM_GEMMA_ST, gemma_directory.path())
                    .env(COMBINED_STREAM_GEMMA_GGUF, gemma_gguf.path())
                    .env_remove("MLX_RING_VERBOSE")
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .spawn()
                    .unwrap(),
            );
        }
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            let statuses = children
                .0
                .iter_mut()
                .map(|child| child.try_wait().unwrap())
                .collect::<Vec<_>>();
            if statuses.iter().all(Option::is_some) {
                break;
            }
            if Instant::now() >= deadline
                || statuses.iter().flatten().any(|status| !status.success())
            {
                for child in &mut children.0 {
                    if child.try_wait().unwrap().is_none() {
                        let _ = child.kill();
                    }
                }
                break;
            }
            thread::sleep(Duration::from_millis(20));
        }
        for (rank, output) in children.finish().into_iter().enumerate() {
            assert!(
                output.status.success(),
                "combined streaming rank {rank} exited with {}\n--- stdout ---\n{}\n--- stderr ---\n{}",
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            );
        }
    }

    #[test]
    fn qwen3_moe_gguf_pipeline_matches_resident_and_dense_streaming() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = gpu.stream();
        let token_batches = [
            Array::from_slice(&[1u32, 2], &[1, 2]),
            Array::from_slice(&[3u32], &[1, 1]),
        ];
        let config = dense_qwen_config("qwen3_moe", false);
        let args = dense_qwen::config_from_hf_value(&config).unwrap();
        let mut source = dense_qwen::Model::new(args, stream).unwrap();
        initialize_parameters(&mut source, stream);
        let fixture = dense_qwen_gguf_fixture(&source, stream);

        let mut reference = dense_qwen::load_gguf(fixture.path(), stream, cpu.stream()).unwrap();
        let mut reference_cache = reference.new_cache();
        let expected = token_batches
            .iter()
            .map(|tokens| {
                reference
                    .forward(
                        dense_qwen::ModelInput {
                            inputs: tokens,
                            mask: None,
                            cache: &mut reference_cache,
                        },
                        stream,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();

        for streamed in [false, true] {
            let load = |rank| {
                let options = ModelLoadOptions::with_parallel(gpu_topology(rank));
                let options = if streamed {
                    options.with_weight_residency(WeightResidency::dense_disk_stream(
                        dense_stream_options(),
                    ))
                } else {
                    options
                };
                load_pipeline_model_with_options(fixture.path(), options, stream, cpu.stream())
                    .unwrap()
            };
            let mut first = load(0);
            let mut last = load(1);
            assert_eq!(first.stage_info().model_kind, ModelKind::Qwen3);
            let actual = run_pipeline_sequence(&mut first, &mut last, &token_batches, stream);
            for (actual, expected) in actual.iter().zip(&expected) {
                assert_close(actual, expected);
            }
            if streamed {
                for stage in [&first, &last] {
                    let report = stage.dense_stream_report().unwrap().unwrap();
                    assert_eq!(report.planned_layer_count(), 1);
                    assert_eq!(report.prefill_forwards(), 1);
                    assert_eq!(report.decode_forwards(), 1);
                }
            }
        }
    }

    #[test]
    fn lfm2_gguf_pipeline_matches_resident_hybrid_decoder() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = gpu.stream();
        let args = lfm2::model_args_from_config_value(&lfm2_config(false)).unwrap();
        let mut source = lfm2::Model::new(args, stream).unwrap();
        initialize_parameters(&mut source, stream);
        let fixture = lfm2_gguf_fixture(&source, stream);
        let mut reference = lfm2::load_gguf(fixture.path(), stream, cpu.stream()).unwrap();
        let mut reference_cache = reference.new_cache();
        let token_batches = [
            Array::from_slice(&[1u32, 2], &[1, 2]),
            Array::from_slice(&[3u32], &[1, 1]),
        ];
        let mut expected = Vec::new();
        for tokens in &token_batches {
            expected.push(
                reference
                    .forward_logits(tokens, Some(&mut reference_cache), false, stream)
                    .unwrap(),
            );
        }
        let mut first = load_pipeline_model_with_options(
            fixture.path(),
            ModelLoadOptions::with_parallel(gpu_topology(0)),
            stream,
            cpu.stream(),
        )
        .unwrap();
        let mut last = load_pipeline_model_with_options(
            fixture.path(),
            ModelLoadOptions::with_parallel(gpu_topology(1)),
            stream,
            cpu.stream(),
        )
        .unwrap();
        let actual = run_pipeline_sequence(&mut first, &mut last, &token_batches, stream);
        for (actual, expected) in actual.iter().zip(expected) {
            assert_close(actual, &expected);
        }
    }

    #[test]
    fn lfm2_gguf_combined_pipeline_preflight_uses_rank_local_recipes() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = gpu.stream();

        let dense_args = lfm2::model_args_from_config_value(&lfm2_config(false)).unwrap();
        let mut dense_source = lfm2::Model::new(dense_args, stream).unwrap();
        initialize_parameters(&mut dense_source, stream);
        let dense = lfm2_gguf_fixture(&dense_source, stream);

        let moe_args = lfm2::model_args_from_config_value(&lfm2_config(true)).unwrap();
        let mut moe_source = lfm2::Model::new(moe_args, stream).unwrap();
        initialize_parameters(&mut moe_source, stream);
        let moe = lfm2_moe_gguf_fixture(&moe_source, stream);

        for rank in 0..4 {
            for (fixture, topology) in [
                (dense.path(), tp_pp_gpu_topology(rank)),
                (moe.path(), pp_ep_gpu_topology(rank)),
            ] {
                let loaded = load_pipeline_model_with_options(
                    fixture,
                    ModelLoadOptions::with_parallel(topology).with_weight_residency(
                        WeightResidency::dense_disk_stream(dense_stream_options()),
                    ),
                    stream,
                    cpu.stream(),
                )
                .unwrap();
                assert_eq!(loaded.stage_info().topology, topology);
                assert_eq!(
                    loaded.stage_info().global_layer_range,
                    topology.pipeline_parallel_rank..topology.pipeline_parallel_rank + 1
                );
                let report = loaded.dense_stream_report().unwrap().unwrap();
                assert_eq!(report.planned_layer_count(), 1);
                let diagnostics = loaded.checkpoint_diagnostics().unwrap().unwrap();
                assert!(diagnostics.physical_reads > 0);
            }
        }

        for rank in 0..8 {
            let topology = tp_pp_ep_gpu_topology(rank);
            let loaded = load_pipeline_model_with_options(
                moe.path(),
                ModelLoadOptions::with_parallel(topology).with_weight_residency(
                    WeightResidency::with_expert_cache(
                        crate::NonExpertWeightResidency::DenseDiskStream(dense_stream_options()),
                        crate::runtime::residency::expert_cache::ExpertCacheLoadOptions::default(),
                    ),
                ),
                stream,
                cpu.stream(),
            )
            .unwrap();
            assert_eq!(loaded.stage_info().topology, topology);
            assert!(loaded.dense_stream_report().unwrap().is_some());
            let report = loaded.expert_cache_report().unwrap();
            assert_eq!(
                report.as_ref().map(|report| report.owned_experts),
                (topology.pipeline_parallel_rank == 1).then_some(1)
            );
        }
    }

    #[test]
    fn dense_qwen_and_gpt_oss_pipeline_requantization_uses_shared_bindings() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = gpu.stream();
        let tokens = [Array::from_slice(&[1u32, 2], &[1, 2])];

        let qwen_config = dense_qwen_config("qwen2", false);
        let qwen_args = dense_qwen::config_from_hf_value(&qwen_config).unwrap();
        let mut qwen_source = dense_qwen::Model::new(qwen_args, stream).unwrap();
        initialize_parameters(&mut qwen_source, stream);
        let qwen_dir = tempfile::tempdir().unwrap();
        write_parameter_fixture(qwen_dir.path(), &qwen_config, &qwen_source);
        let affine: WeightQuantization =
            crate::runtime::checkpoint::quantization::AffineQuantization::new(32, 4)
                .unwrap()
                .into();
        let qwen_load = |rank, quantization| {
            let options = match quantization {
                Some(quantization) => ModelLoadOptions::with_quantization(quantization)
                    .with_parallel_topology(gpu_topology(rank)),
                None => ModelLoadOptions::with_parallel(gpu_topology(rank)),
            };
            load_pipeline_model_with_options(qwen_dir.path(), options, stream, cpu.stream())
                .unwrap()
        };
        let mut qwen_dense_first = qwen_load(0, None);
        let mut qwen_dense_last = qwen_load(1, None);
        let mut qwen_quant_first = qwen_load(0, Some(affine));
        let mut qwen_quant_last = qwen_load(1, Some(affine));
        assert!(
            qwen_quant_first.stage_info().local_parameter_bytes
                < qwen_dense_first.stage_info().local_parameter_bytes
        );
        assert!(
            qwen_quant_last.stage_info().local_parameter_bytes
                < qwen_dense_last.stage_info().local_parameter_bytes
        );
        let qwen_expected =
            run_pipeline_sequence(&mut qwen_dense_first, &mut qwen_dense_last, &tokens, stream);
        let qwen_actual =
            run_pipeline_sequence(&mut qwen_quant_first, &mut qwen_quant_last, &tokens, stream);
        assert!(qwen_actual[0]
            .all_close(&qwen_expected[0], Some(2e-3), Some(2e-3), None, stream)
            .unwrap()
            .item::<bool>(stream));

        let gpt_config = gpt_oss_config();
        let gpt_args = gpt_oss::model_args_from_config_value(&gpt_config).unwrap();
        let mut gpt_source = gpt_oss::Model::new(gpt_args, stream).unwrap();
        initialize_parameters(&mut gpt_source, stream);
        let gpt_dir = tempfile::tempdir().unwrap();
        write_parameter_fixture(gpt_dir.path(), &gpt_config, &gpt_source);
        let gpt_load = |rank, quantized| {
            let options = if quantized {
                ModelLoadOptions::with_quantization(WeightQuantization::MxFp4)
                    .with_parallel_topology(gpu_topology(rank))
            } else {
                ModelLoadOptions::with_parallel(gpu_topology(rank))
            };
            load_pipeline_model_with_options(gpt_dir.path(), options, stream, cpu.stream()).unwrap()
        };
        let mut gpt_dense_first = gpt_load(0, false);
        let mut gpt_dense_last = gpt_load(1, false);
        let mut gpt_quant_first = gpt_load(0, true);
        let mut gpt_quant_last = gpt_load(1, true);
        assert!(
            gpt_quant_first.stage_info().local_parameter_bytes
                < gpt_dense_first.stage_info().local_parameter_bytes
        );
        assert!(
            gpt_quant_last.stage_info().local_parameter_bytes
                < gpt_dense_last.stage_info().local_parameter_bytes
        );
        let gpt_expected =
            run_pipeline_sequence(&mut gpt_dense_first, &mut gpt_dense_last, &tokens, stream);
        let gpt_actual =
            run_pipeline_sequence(&mut gpt_quant_first, &mut gpt_quant_last, &tokens, stream);
        let gpt_actual_values = gpt_actual[0].evaluated().unwrap();
        let gpt_expected_values = gpt_expected[0].evaluated().unwrap();
        assert_eq!(
            gpt_actual_values.as_array().shape(),
            gpt_expected_values.as_array().shape()
        );
        assert!(gpt_actual_values
            .as_slice::<f32>()
            .iter()
            .all(|value| value.is_finite()));

        let error = load_pipeline_model_with_options(
            gpt_dir.path(),
            ModelLoadOptions::with_quantization(affine).with_parallel_topology(gpu_topology(0)),
            stream,
            cpu.stream(),
        )
        .expect_err("GPT-OSS expert banks cannot transcode to affine");
        assert!(error
            .to_string()
            .contains("cannot be implicitly dequantized"));
    }

    #[test]
    fn qwen3_moe_pipeline_requantization_uses_shared_expert_bindings() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = gpu.stream();
        let tokens = [Array::from_slice(&[1u32, 2], &[1, 2])];
        let config = dense_qwen_config("qwen3_moe", false);
        let args = dense_qwen::config_from_hf_value(&config).unwrap();
        let mut source = dense_qwen::Model::new(args, stream).unwrap();
        initialize_parameters(&mut source, stream);
        let dir = tempfile::tempdir().unwrap();
        write_parameter_fixture(dir.path(), &config, &source);
        let affine: WeightQuantization =
            crate::runtime::checkpoint::quantization::AffineQuantization::new(32, 4)
                .unwrap()
                .into();
        let load = |rank, quantization| {
            let options = match quantization {
                Some(quantization) => ModelLoadOptions::with_quantization(quantization)
                    .with_parallel_topology(gpu_topology(rank)),
                None => ModelLoadOptions::with_parallel(gpu_topology(rank)),
            };
            load_pipeline_model_with_options(dir.path(), options, stream, cpu.stream()).unwrap()
        };
        let mut dense_first = load(0, None);
        let mut dense_last = load(1, None);
        let mut quantized_first = load(0, Some(affine));
        let mut quantized_last = load(1, Some(affine));
        assert!(
            quantized_first.stage_info().local_parameter_bytes
                < dense_first.stage_info().local_parameter_bytes
        );
        assert!(
            quantized_last.stage_info().local_parameter_bytes
                < dense_last.stage_info().local_parameter_bytes
        );
        let expected = run_pipeline_sequence(&mut dense_first, &mut dense_last, &tokens, stream);
        let actual =
            run_pipeline_sequence(&mut quantized_first, &mut quantized_last, &tokens, stream);
        assert!(actual[0]
            .all_close(&expected[0], Some(2e-3), Some(2e-3), None, stream)
            .unwrap()
            .item::<bool>(stream));
    }

    fn llama_args(tied: bool) -> llama::ModelArgs {
        llama::ModelArgs {
            model_type: "llama".into(),
            hidden_size: 8,
            num_hidden_layers: 2,
            intermediate_size: 16,
            num_attention_heads: 2,
            rms_norm_eps: 1e-5,
            vocab_size: 16,
            num_key_value_heads: 2,
            max_position_embeddings: 64,
            rope_theta: 10_000.0,
            rope_traditional: false,
            head_dim: 4,
            tie_word_embeddings: tied,
            attention_bias: false,
            mlp_bias: false,
            rope_scaling: None,
            attention_schedule: crate::runtime::attention::LayerSchedule::all_full(4).unwrap(),
            quantization: None,
            quantization_config: None,
            quantized_weights: None,
            quantized_weight_configs: None,
        }
    }

    fn gemma_config() -> serde_json::Value {
        serde_json::json!({
            "model_type": "gemma4",
            "tie_word_embeddings": true,
            "text_config": {
                "model_type": "gemma4",
                "hidden_size": 8,
                "num_hidden_layers": 4,
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "rms_norm_eps": 1e-6,
                "vocab_size": 32,
                "pad_token_id": 0,
                "num_key_value_heads": 2,
                "max_position_embeddings": 128,
                "rope_theta": 10000.0,
                "head_dim": 4,
                "attention_bias": false,
                "hidden_size_per_layer_input": 4,
                "vocab_size_per_layer_input": 32,
                "num_kv_shared_layers": 1,
                "layer_types": [
                    "sliding_attention",
                    "full_attention",
                    "sliding_attention",
                    "full_attention"
                ],
                "sliding_window": 8,
                "final_logit_softcapping": 4.0
            }
        })
    }

    fn quantizable_gemma_config() -> serde_json::Value {
        let mut config = gemma_config();
        let text = config["text_config"]
            .as_object_mut()
            .expect("Gemma fixture text config");
        text.insert("hidden_size".into(), serde_json::json!(32));
        text.insert("intermediate_size".into(), serde_json::json!(64));
        text.insert("head_dim".into(), serde_json::json!(16));
        text.insert("hidden_size_per_layer_input".into(), serde_json::json!(32));
        config
    }

    fn quantizable_gemma_moe_config() -> serde_json::Value {
        let mut config = quantizable_gemma_config();
        let text = config["text_config"]
            .as_object_mut()
            .expect("Gemma fixture text config");
        text.insert("enable_moe_block".into(), serde_json::json!(true));
        text.insert("num_experts".into(), serde_json::json!(4));
        text.insert("top_k_experts".into(), serde_json::json!(2));
        text.insert("moe_intermediate_size".into(), serde_json::json!(32));
        config
    }

    fn write_gemma_fixture_with_config(
        dir: &Path,
        model: &gemma4::Model,
        config: &serde_json::Value,
    ) {
        let arrays = model
            .parameters()
            .flatten()
            .iter()
            .map(|(name, value)| {
                (
                    crate::runtime::checkpoint::binding::canonical_checkpoint_name(name),
                    (*value).clone(),
                )
            })
            .collect::<Vec<_>>();
        Array::save_safetensors(
            arrays.iter().map(|(name, value)| (name.as_str(), value)),
            None,
            dir.join("model.safetensors"),
        )
        .unwrap();
        fs::write(dir.join("config.json"), serde_json::to_vec(config).unwrap()).unwrap();
    }

    fn write_gemma_fixture(dir: &Path, model: &gemma4::Model) {
        write_gemma_fixture_with_config(dir, model, &gemma_config());
    }

    fn gemma_gguf_fixture(model: &gemma4::Model, stream: &Stream) -> SyntheticGguf {
        let args = &model.args;
        let as_u32 = |value: i32| u32::try_from(value).expect("positive Gemma fixture geometry");
        let feed_forward = args
            .layer_schedule
            .iter()
            .map(|policy| policy.intermediate_size.get())
            .collect::<Vec<_>>();
        let kv_heads = args
            .layer_schedule
            .iter()
            .map(|policy| policy.num_key_value_heads.get())
            .collect::<Vec<_>>();
        let sliding_pattern = args
            .layer_schedule
            .iter()
            .map(|policy| policy.attention.window().is_some())
            .collect::<Vec<_>>();
        let head_dim = args
            .layer_schedule
            .get(0)
            .expect("Gemma fixture has layers")
            .head_dim
            .get();
        let sliding_window = args
            .layer_schedule
            .iter()
            .find_map(|policy| policy.attention.window())
            .map_or(8, |window| window.get());
        let shared_layers = args
            .layer_schedule
            .iter()
            .filter(|policy| policy.key_value == gemma4::KeyValuePolicy::Shared)
            .count() as u32;
        let arrays = model
            .parameters()
            .flatten()
            .iter()
            .map(|(name, value)| {
                let canonical =
                    crate::runtime::checkpoint::binding::canonical_checkpoint_name(name);
                let gguf = if canonical.starts_with("model.language_model.embed_tokens_per_layer.")
                {
                    canonical.replacen(
                        "model.language_model.embed_tokens_per_layer",
                        "per_layer_token_embd",
                        1,
                    )
                } else if canonical.starts_with("model.language_model.per_layer_model_projection.")
                {
                    canonical.replacen(
                        "model.language_model.per_layer_model_projection",
                        "per_layer_model_proj",
                        1,
                    )
                } else if canonical.starts_with("model.language_model.per_layer_projection_norm.") {
                    canonical.replacen(
                        "model.language_model.per_layer_projection_norm",
                        "per_layer_proj_norm",
                        1,
                    )
                } else if canonical.starts_with("model.language_model.embed_tokens.") {
                    canonical.replacen("model.language_model.embed_tokens", "token_embd", 1)
                } else if canonical.starts_with("model.language_model.norm.") {
                    canonical.replacen("model.language_model.norm", "output_norm", 1)
                } else if canonical.starts_with("lm_head.") {
                    canonical.replacen("lm_head", "output", 1)
                } else {
                    let rest = canonical
                        .strip_prefix("model.language_model.layers.")
                        .expect("Gemma fixture parameter has a known root");
                    let (layer, parameter) = rest.split_once('.').unwrap();
                    let parameter = if parameter == "layer_scalar" {
                        "layer_output_scale.weight".to_string()
                    } else {
                        [
                            ("self_attn.q_norm", "attn_q_norm"),
                            ("self_attn.k_norm", "attn_k_norm"),
                            ("self_attn.q_proj", "attn_q"),
                            ("self_attn.k_proj", "attn_k"),
                            ("self_attn.v_proj", "attn_v"),
                            ("self_attn.o_proj", "attn_output"),
                            ("input_layernorm", "attn_norm"),
                            ("post_attention_layernorm", "post_attention_norm"),
                            ("pre_feedforward_layernorm", "ffn_norm"),
                            ("post_feedforward_layernorm", "post_ffw_norm"),
                            ("mlp.gate_proj", "ffn_gate"),
                            ("mlp.down_proj", "ffn_down"),
                            ("mlp.up_proj", "ffn_up"),
                            ("router.proj", "ffn_gate_inp"),
                            ("experts.switch_glu.gate_proj", "ffn_gate_exps"),
                            ("experts.switch_glu.up_proj", "ffn_up_exps"),
                            ("experts.switch_glu.down_proj", "ffn_down_exps"),
                            ("post_feedforward_layernorm_1", "post_ffw_norm_1"),
                            ("pre_feedforward_layernorm_2", "pre_ffw_norm_2"),
                            ("post_feedforward_layernorm_2", "post_ffw_norm_2"),
                            ("per_layer_input_gate", "inp_gate"),
                            ("per_layer_projection", "proj"),
                            ("post_per_layer_input_norm", "post_norm"),
                        ]
                        .into_iter()
                        .find_map(|(target, source)| {
                            (parameter == target || parameter.starts_with(&format!("{target}.")))
                                .then(|| parameter.replacen(target, source, 1))
                        })
                        .or_else(|| {
                            (parameter == "router.scale").then(|| "ffn_gate_inp.scale".to_string())
                        })
                        .or_else(|| {
                            (parameter == "router.per_expert_scale")
                                .then(|| "ffn_down_exps.scale".to_string())
                        })
                        .expect("Gemma fixture block parameter has a GGUF spelling")
                    };
                    format!("blk.{layer}.{parameter}")
                };
                (gguf, (*value).clone())
            })
            .collect::<HashMap<_, _>>();
        let mut arrays = arrays;
        if args.num_experts.is_some() {
            for layer in 0..args.num_hidden_layers {
                let gate = arrays.remove(&format!("blk.{layer}.ffn_gate_exps.weight"));
                let up = arrays.remove(&format!("blk.{layer}.ffn_up_exps.weight"));
                match (gate, up) {
                    (Some(gate), Some(up)) => {
                        let fused = safemlx::ops::concatenate_axis(&[gate, up], 1, stream).unwrap();
                        arrays.insert(format!("blk.{layer}.ffn_gate_up_exps.weight"), fused);
                    }
                    (None, None) => {}
                    _ => panic!("Gemma fixture has incomplete expert gate/up tensors"),
                }
            }
        }
        let mut metadata = HashMap::from([
            (
                "general.architecture".into(),
                GgufMetadataValue::String("gemma4".into()),
            ),
            ("general.file_type".into(), GgufMetadataValue::Uint32(0)),
            (
                "gemma4.block_count".into(),
                GgufMetadataValue::Uint32(as_u32(args.num_hidden_layers)),
            ),
            (
                "gemma4.embedding_length".into(),
                GgufMetadataValue::Uint32(as_u32(args.hidden_size)),
            ),
            (
                "gemma4.embedding_length_per_layer_input".into(),
                GgufMetadataValue::Uint32(as_u32(args.hidden_size_per_layer_input)),
            ),
            (
                "gemma4.feed_forward_length".into(),
                GgufMetadataValue::Array(safemlx::ops::GgufMetadataArray::Uint32(feed_forward)),
            ),
            (
                "gemma4.attention.head_count".into(),
                GgufMetadataValue::Uint32(as_u32(args.num_attention_heads)),
            ),
            (
                "gemma4.attention.head_count_kv".into(),
                GgufMetadataValue::Array(safemlx::ops::GgufMetadataArray::Uint32(kv_heads)),
            ),
            (
                "gemma4.attention.key_length".into(),
                GgufMetadataValue::Uint32(head_dim),
            ),
            (
                "gemma4.attention.key_length_swa".into(),
                GgufMetadataValue::Uint32(head_dim),
            ),
            (
                "gemma4.attention.sliding_window_pattern".into(),
                GgufMetadataValue::Array(safemlx::ops::GgufMetadataArray::Bool(sliding_pattern)),
            ),
            (
                "gemma4.attention.sliding_window".into(),
                GgufMetadataValue::Uint32(sliding_window),
            ),
            (
                "gemma4.attention.shared_kv_layers".into(),
                GgufMetadataValue::Uint32(shared_layers),
            ),
            (
                "gemma4.attention.layer_norm_rms_epsilon".into(),
                GgufMetadataValue::Float32(1e-6),
            ),
            (
                "gemma4.context_length".into(),
                GgufMetadataValue::Uint32(as_u32(args.max_position_embeddings)),
            ),
            (
                "gemma4.vocab_size".into(),
                GgufMetadataValue::Uint32(as_u32(args.vocab_size)),
            ),
            (
                "gemma4.final_logit_softcapping".into(),
                GgufMetadataValue::Float32(4.0),
            ),
            (
                "gemma4.rope.freq_base".into(),
                GgufMetadataValue::Float32(10_000.0),
            ),
            (
                "gemma4.rope.freq_base_swa".into(),
                GgufMetadataValue::Float32(10_000.0),
            ),
        ]);
        if let (Some(num_experts), Some(top_k), Some(intermediate)) = (
            args.num_experts,
            args.top_k_experts,
            args.moe_intermediate_size,
        ) {
            metadata.insert(
                "gemma4.expert_count".into(),
                GgufMetadataValue::Uint32(as_u32(num_experts)),
            );
            metadata.insert(
                "gemma4.expert_used_count".into(),
                GgufMetadataValue::Uint32(as_u32(top_k)),
            );
            metadata.insert(
                "gemma4.expert_feed_forward_length".into(),
                GgufMetadataValue::Uint32(as_u32(intermediate)),
            );
        }
        SyntheticGguf::dense(&arrays, &metadata)
    }

    #[test]
    fn dependency_planner_preserves_atomic_units_and_rejects_too_many_stages() {
        let ranges =
            dependency_safe_layer_ranges(7, 3, &[true, false, false, true, true, true]).unwrap();
        assert_eq!(ranges, [0..1, 1..4, 4..7]);
        let error = dependency_safe_layer_ranges(4, 3, &[false, false, true])
            .expect_err("two atomic units cannot occupy three stages");
        assert!(error.to_string().contains("dependency-safe decoder units"));

        let (args, _, _, _, _, _) = gemma4::model_config_from_value(&gemma_config()).unwrap();
        let ranges = gemma_pipeline_ranges(&args, 2).unwrap();
        assert_eq!(ranges, [0..1, 1..4]);
        assert!(gemma_pipeline_ranges(&args, 3).is_err());
    }

    #[test]
    fn gemma_pipeline_preserves_per_layer_inputs_shared_kv_and_decode_state() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = gpu.stream();
        let (args, _, _, _, _, _) = gemma4::model_config_from_value(&gemma_config()).unwrap();
        let mut source = gemma4::Model::new(args.clone(), stream).unwrap();
        initialize_parameters(&mut source, stream);
        let dir = tempfile::tempdir().unwrap();
        write_gemma_fixture(dir.path(), &source);
        let mut reference = gemma4::load_gemma4_model(dir.path(), stream, cpu.stream()).unwrap();
        let mut first = load_pipeline_model_with_options(
            dir.path(),
            ModelLoadOptions::with_parallel(gpu_topology(0)),
            stream,
            cpu.stream(),
        )
        .unwrap();
        let mut last = load_pipeline_model_with_options(
            dir.path(),
            ModelLoadOptions::with_parallel(gpu_topology(1)),
            stream,
            cpu.stream(),
        )
        .unwrap();
        assert_eq!(first.stage_info().model_kind, ModelKind::Gemma4);
        assert_eq!(first.stage_info().global_layer_range, 0..1);
        assert_eq!(last.stage_info().global_layer_range, 1..4);

        let mut reference_cache = Vec::<Option<ConcatKeyValueCache>>::new();
        let mut first_cache = first.new_cache().unwrap();
        let mut last_cache = last.new_cache().unwrap();
        for tokens in [
            Array::from_slice(&[1u32, 2], &[1, 2]),
            Array::from_slice(&[3u32], &[1, 1]),
        ] {
            let expected = reference
                .forward_logits(
                    gemma4::ModelInput {
                        inputs: &tokens,
                        inputs_embeds: None,
                        per_layer_input_ids: None,
                        mask: None,
                        sliding_masks: None,
                        cache: &mut reference_cache,
                    },
                    false,
                    stream,
                )
                .unwrap();
            let step = PipelineStep::new(1, tokens.shape()[1]).unwrap();
            let payload = match first
                .forward_stage(
                    PipelineStageInput::Tokens(&tokens),
                    step,
                    None,
                    &mut first_cache,
                    stream,
                )
                .unwrap()
            {
                PipelineStageOutput::Hidden(payload) => payload,
                PipelineStageOutput::Logits(_) => panic!("first stage produced logits"),
            };
            assert_eq!(payload.auxiliary.tensors().len(), 1);
            assert_eq!(
                payload.auxiliary.tensors()[0].shape(),
                [1, tokens.shape()[1], 4, 4]
            );
            let actual = match last
                .forward_stage(
                    PipelineStageInput::Hidden(&payload),
                    step,
                    None,
                    &mut last_cache,
                    stream,
                )
                .unwrap()
            {
                PipelineStageOutput::Logits(logits) => logits,
                PipelineStageOutput::Hidden(_) => panic!("last stage produced hidden state"),
            };
            assert_close(&actual, &expected);
        }
        assert_eq!(first_cache.global_layers(), [0]);
        assert_eq!(last_cache.global_layers(), [1, 2, 3]);
        assert!(matches!(
            last_cache.layers().last(),
            Some(PipelineLayerCache::StateSlots {
                global_layer: 3,
                slots
            }) if slots.is_empty()
        ));
    }

    #[test]
    fn gemma_tp_pp_preflight_owns_dependency_safe_shards_and_local_caches() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = gpu.stream();
        let config = gemma_config();
        let args = gemma4::model_args_from_config_value(&config["text_config"]).unwrap();
        let mut source = gemma4::Model::new(args, stream).unwrap();
        initialize_parameters(&mut source, stream);
        let safetensors = tempfile::tempdir().unwrap();
        write_gemma_fixture(safetensors.path(), &source);
        let gguf = gemma_gguf_fixture(&source, stream);

        for path in [safetensors.path(), gguf.path()] {
            for rank in 0..4 {
                let topology = tp_pp_gpu_topology(rank);
                let resident = load_pipeline_model_with_options(
                    path,
                    ModelLoadOptions::with_parallel(topology),
                    stream,
                    cpu.stream(),
                )
                .unwrap();
                let streamed = load_pipeline_model_with_options(
                    path,
                    ModelLoadOptions::with_parallel(topology).with_weight_residency(
                        WeightResidency::dense_disk_stream(dense_stream_options()),
                    ),
                    stream,
                    cpu.stream(),
                )
                .unwrap();
                for model in [&resident, &streamed] {
                    assert_eq!(model.stage_info().topology, topology);
                    let local_layers = model.stage_info().global_layer_range.len();
                    assert!(local_layers > 0);
                    let layout = model.prompt_cache_layer_layout().unwrap();
                    assert_eq!(layout.len(), local_layers);
                    for policy in layout.iter() {
                        if let crate::LayerCachePolicy::KeyValue {
                            num_key_value_heads,
                            ..
                        } = policy
                        {
                            assert_eq!(num_key_value_heads.get(), 1);
                        }
                    }
                }
                let report = streamed.dense_stream_report().unwrap().unwrap();
                assert_eq!(
                    report.planned_layer_count(),
                    resident.stage_info().global_layer_range.len()
                );
                assert_eq!(
                    streamed.stage_info().planned_owned_parameter_bytes,
                    resident.stage_info().local_parameter_bytes as u64
                );
            }
        }
    }

    #[test]
    fn gemma_gguf_pipeline_uses_the_same_dependency_plan_and_matches_resident() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = gpu.stream();
        let (args, _, _, _, _, _) = gemma4::model_config_from_value(&gemma_config()).unwrap();
        let mut reference = gemma4::Model::new(args, stream).unwrap();
        initialize_parameters(&mut reference, stream);
        let fixture = gemma_gguf_fixture(&reference, stream);
        let mut first = load_pipeline_model_with_options(
            fixture.path(),
            ModelLoadOptions::with_parallel(gpu_topology(0)),
            stream,
            cpu.stream(),
        )
        .unwrap();
        let mut last = load_pipeline_model_with_options(
            fixture.path(),
            ModelLoadOptions::with_parallel(gpu_topology(1)),
            stream,
            cpu.stream(),
        )
        .unwrap();
        assert_eq!(first.stage_info().global_layer_range, 0..1);
        assert_eq!(last.stage_info().global_layer_range, 1..4);
        let tokens = Array::from_slice(&[1u32, 2], &[1, 2]);
        let mut reference_cache = Vec::<Option<ConcatKeyValueCache>>::new();
        let expected = reference
            .forward_logits(
                gemma4::ModelInput {
                    inputs: &tokens,
                    inputs_embeds: None,
                    per_layer_input_ids: None,
                    mask: None,
                    sliding_masks: None,
                    cache: &mut reference_cache,
                },
                false,
                stream,
            )
            .unwrap();
        let step = PipelineStep::new(1, 2).unwrap();
        let mut first_cache = first.new_cache().unwrap();
        let payload = match first
            .forward_stage(
                PipelineStageInput::Tokens(&tokens),
                step,
                None,
                &mut first_cache,
                stream,
            )
            .unwrap()
        {
            PipelineStageOutput::Hidden(payload) => payload,
            PipelineStageOutput::Logits(_) => panic!("first stage produced logits"),
        };
        let mut last_cache = last.new_cache().unwrap();
        let actual = match last
            .forward_stage(
                PipelineStageInput::Hidden(&payload),
                step,
                None,
                &mut last_cache,
                stream,
            )
            .unwrap()
        {
            PipelineStageOutput::Logits(logits) => logits,
            PipelineStageOutput::Hidden(_) => panic!("last stage produced hidden state"),
        };
        assert_close(&actual, &expected);
        assert!(!first.stage_info().opened_checkpoint_shards.is_empty());
        assert!(!last.stage_info().opened_checkpoint_shards.is_empty());
    }

    #[test]
    fn gemma_dense_stream_pipeline_matches_fully_resident_stages() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = gpu.stream();
        let (args, _, _, _, _, _) = gemma4::model_config_from_value(&gemma_config()).unwrap();
        let mut source = gemma4::Model::new(args, stream).unwrap();
        initialize_parameters(&mut source, stream);
        let dir = tempfile::tempdir().unwrap();
        write_gemma_fixture(dir.path(), &source);
        let dense_options = || {
            crate::runtime::residency::dense_stream::DenseDiskStreamLoadOptions::new(
                u64::MAX,
                u64::MAX,
                1,
                1,
                1,
            )
            .unwrap()
        };
        let mut resident_first = load_pipeline_model_with_options(
            dir.path(),
            ModelLoadOptions::with_parallel(gpu_topology(0)),
            stream,
            cpu.stream(),
        )
        .unwrap();
        let mut resident_last = load_pipeline_model_with_options(
            dir.path(),
            ModelLoadOptions::with_parallel(gpu_topology(1)),
            stream,
            cpu.stream(),
        )
        .unwrap();
        let mut dense_first = load_pipeline_model_with_options(
            dir.path(),
            ModelLoadOptions::with_parallel(gpu_topology(0))
                .with_weight_residency(WeightResidency::dense_disk_stream(dense_options())),
            stream,
            cpu.stream(),
        )
        .unwrap();
        let mut dense_last = load_pipeline_model_with_options(
            dir.path(),
            ModelLoadOptions::with_parallel(gpu_topology(1))
                .with_weight_residency(WeightResidency::dense_disk_stream(dense_options())),
            stream,
            cpu.stream(),
        )
        .unwrap();
        let tokens = Array::from_slice(&[1u32, 2], &[1, 2]);
        let step = PipelineStep::new(1, 2).unwrap();
        let run = |first: &mut PipelineModel, last: &mut PipelineModel| {
            let mut first_cache = first.new_cache().unwrap();
            let mut last_cache = last.new_cache().unwrap();
            let payload = match first
                .forward_stage(
                    PipelineStageInput::Tokens(&tokens),
                    step,
                    None,
                    &mut first_cache,
                    stream,
                )
                .unwrap()
            {
                PipelineStageOutput::Hidden(payload) => payload,
                PipelineStageOutput::Logits(_) => panic!("first stage produced logits"),
            };
            match last
                .forward_stage(
                    PipelineStageInput::Hidden(&payload),
                    step,
                    None,
                    &mut last_cache,
                    stream,
                )
                .unwrap()
            {
                PipelineStageOutput::Logits(logits) => logits,
                PipelineStageOutput::Hidden(_) => panic!("last stage produced hidden state"),
            }
        };
        let expected = run(&mut resident_first, &mut resident_last);
        let actual = run(&mut dense_first, &mut dense_last);
        assert_close(&actual, &expected);
        assert!(dense_first.dense_stream_report().unwrap().is_some());
        assert!(dense_last.dense_stream_report().unwrap().is_some());
    }

    #[test]
    fn gemma_pipeline_requantizes_shared_bindings_and_matches_dense_execution() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = gpu.stream();
        let quantization =
            crate::runtime::checkpoint::quantization::AffineQuantization::new(32, 4).unwrap();

        for tied in [true, false] {
            let mut config = quantizable_gemma_config();
            config["tie_word_embeddings"] = serde_json::json!(tied);
            let (args, _, _, _, _, _) = gemma4::model_config_from_value(&config).unwrap();
            let mut source = gemma4::Model::new(args, stream).unwrap();
            initialize_parameters(&mut source, stream);
            let dir = tempfile::tempdir().unwrap();
            write_gemma_fixture_with_config(dir.path(), &source, &config);

            let load = |rank, quantized| {
                let options = if quantized {
                    ModelLoadOptions::with_quantization(quantization)
                        .with_parallel_topology(gpu_topology(rank))
                } else {
                    ModelLoadOptions::with_parallel(gpu_topology(rank))
                };
                load_pipeline_model_with_options(dir.path(), options, stream, cpu.stream()).unwrap()
            };
            let mut dense_first = load(0, false);
            let mut dense_last = load(1, false);
            let mut quantized_first = load(0, true);
            let mut quantized_last = load(1, true);
            assert!(
                quantized_first.stage_info().local_parameter_bytes
                    < dense_first.stage_info().local_parameter_bytes
            );
            assert!(
                quantized_last.stage_info().local_parameter_bytes
                    < dense_last.stage_info().local_parameter_bytes
            );

            let run = |first: &mut PipelineModel, last: &mut PipelineModel| {
                let mut first_cache = first.new_cache().unwrap();
                let mut last_cache = last.new_cache().unwrap();
                let mut outputs = Vec::new();
                for tokens in [
                    Array::from_slice(&[1u32, 2], &[1, 2]),
                    Array::from_slice(&[3u32], &[1, 1]),
                ] {
                    let step = PipelineStep::new(1, tokens.shape()[1]).unwrap();
                    let payload = match first
                        .forward_stage(
                            PipelineStageInput::Tokens(&tokens),
                            step,
                            None,
                            &mut first_cache,
                            stream,
                        )
                        .unwrap()
                    {
                        PipelineStageOutput::Hidden(payload) => payload,
                        PipelineStageOutput::Logits(_) => panic!("first stage produced logits"),
                    };
                    let logits = match last
                        .forward_stage(
                            PipelineStageInput::Hidden(&payload),
                            step,
                            None,
                            &mut last_cache,
                            stream,
                        )
                        .unwrap()
                    {
                        PipelineStageOutput::Logits(logits) => logits,
                        PipelineStageOutput::Hidden(_) => panic!("last stage produced hidden"),
                    };
                    eval([&logits]).unwrap();
                    outputs.push(logits);
                }
                outputs
            };
            let expected = run(&mut dense_first, &mut dense_last);
            let actual = run(&mut quantized_first, &mut quantized_last);
            for (actual, expected) in actual.iter().zip(&expected) {
                let actual = actual.evaluated().unwrap();
                let expected = expected.evaluated().unwrap();
                assert_eq!(actual.as_array().shape(), expected.as_array().shape());
                for (actual, expected) in actual
                    .as_slice::<f32>()
                    .iter()
                    .zip(expected.as_slice::<f32>())
                {
                    assert!((actual - expected).abs() <= 2e-3, "{actual} != {expected}");
                }
            }

            let gguf = gemma_gguf_fixture(&source, stream);
            let load_gguf = |rank, quantized| {
                let options = if quantized {
                    ModelLoadOptions::with_quantization(quantization)
                        .with_parallel_topology(gpu_topology(rank))
                } else {
                    ModelLoadOptions::with_parallel(gpu_topology(rank))
                };
                load_pipeline_model_with_options(gguf.path(), options, stream, cpu.stream())
                    .unwrap()
            };
            let mut dense_gguf_first = load_gguf(0, false);
            let mut dense_gguf_last = load_gguf(1, false);
            let mut quantized_gguf_first = load_gguf(0, true);
            let mut quantized_gguf_last = load_gguf(1, true);
            assert!(
                quantized_gguf_first.stage_info().local_parameter_bytes
                    < dense_gguf_first.stage_info().local_parameter_bytes
            );
            assert!(
                quantized_gguf_last.stage_info().local_parameter_bytes
                    < dense_gguf_last.stage_info().local_parameter_bytes
            );
            let expected = run(&mut dense_gguf_first, &mut dense_gguf_last);
            let actual = run(&mut quantized_gguf_first, &mut quantized_gguf_last);
            for (actual, expected) in actual.iter().zip(&expected) {
                let actual = actual.evaluated().unwrap();
                let expected = expected.evaluated().unwrap();
                for (actual, expected) in actual
                    .as_slice::<f32>()
                    .iter()
                    .zip(expected.as_slice::<f32>())
                {
                    assert!((actual - expected).abs() <= 2e-3, "{actual} != {expected}");
                }
            }

            let dense_stream =
                crate::runtime::residency::dense_stream::DenseDiskStreamLoadOptions::new(
                    u64::MAX,
                    u64::MAX,
                    1,
                    1,
                    1,
                )
                .unwrap();
            let error = load_pipeline_model_with_options(
                dir.path(),
                ModelLoadOptions::with_quantization(quantization)
                    .with_parallel_topology(gpu_topology(0))
                    .with_weight_residency(WeightResidency::dense_disk_stream(dense_stream)),
                stream,
                cpu.stream(),
            )
            .expect_err("dense streaming must reject on-load Gemma quantization");
            assert!(error
                .to_string()
                .contains("non-resident Gemma pipeline layers"));
        }
    }

    #[test]
    fn gemma_pipeline_requantizes_fused_gguf_expert_bindings() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = gpu.stream();
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let quantization =
            crate::runtime::checkpoint::quantization::AffineQuantization::new(32, 4).unwrap();
        let config = quantizable_gemma_moe_config();
        let (args, _, _, _, _, _) = gemma4::model_config_from_value(&config).unwrap();
        let mut source = gemma4::Model::new(args, stream).unwrap();
        initialize_parameters(&mut source, stream);
        let gguf = gemma_gguf_fixture(&source, stream);

        let load = |rank, quantized| {
            let options = if quantized {
                ModelLoadOptions::with_quantization(quantization)
                    .with_parallel_topology(gpu_topology(rank))
            } else {
                ModelLoadOptions::with_parallel(gpu_topology(rank))
            };
            load_pipeline_model_with_options(gguf.path(), options, stream, cpu.stream()).unwrap()
        };
        let mut dense_first = load(0, false);
        let mut dense_last = load(1, false);
        let mut quantized_first = load(0, true);
        let mut quantized_last = load(1, true);
        let mxfp4_first = load_pipeline_model_with_options(
            gguf.path(),
            ModelLoadOptions::with_quantization(WeightQuantization::MxFp4)
                .with_parallel_topology(gpu_topology(0)),
            stream,
            cpu.stream(),
        )
        .unwrap();
        assert!(
            quantized_first.stage_info().local_parameter_bytes
                < dense_first.stage_info().local_parameter_bytes
        );
        assert!(
            quantized_last.stage_info().local_parameter_bytes
                < dense_last.stage_info().local_parameter_bytes
        );
        assert!(
            mxfp4_first.stage_info().local_parameter_bytes
                < dense_first.stage_info().local_parameter_bytes
        );
        for stage in [&quantized_first, &quantized_last] {
            assert!(
                stage
                    .stage_info()
                    .owned_tensors
                    .iter()
                    .any(|name| name.contains("gate_up")),
                "missing fused expert binding in {:?}",
                stage.stage_info().owned_tensors
            );
        }

        let run = |first: &mut PipelineModel, last: &mut PipelineModel| {
            let tokens = Array::from_slice(&[1u32, 2], &[1, 2]);
            let step = PipelineStep::new(1, 2).unwrap();
            let mut first_cache = first.new_cache().unwrap();
            let mut last_cache = last.new_cache().unwrap();
            let payload = match first
                .forward_stage(
                    PipelineStageInput::Tokens(&tokens),
                    step,
                    None,
                    &mut first_cache,
                    stream,
                )
                .unwrap()
            {
                PipelineStageOutput::Hidden(payload) => payload,
                PipelineStageOutput::Logits(_) => panic!("first stage produced logits"),
            };
            match last
                .forward_stage(
                    PipelineStageInput::Hidden(&payload),
                    step,
                    None,
                    &mut last_cache,
                    stream,
                )
                .unwrap()
            {
                PipelineStageOutput::Logits(logits) => logits,
                PipelineStageOutput::Hidden(_) => panic!("last stage produced hidden"),
            }
        };
        let expected_logits = run(&mut dense_first, &mut dense_last);
        let expected = expected_logits.evaluated().unwrap();
        let actual_logits = run(&mut quantized_first, &mut quantized_last);
        let actual = actual_logits.evaluated().unwrap();
        for (actual, expected) in actual
            .as_slice::<f32>()
            .iter()
            .zip(expected.as_slice::<f32>())
        {
            assert!((actual - expected).abs() <= 2e-3, "{actual} != {expected}");
        }
    }

    fn write_llama_fixture(
        dir: &Path,
        model: &llama::ResidentModel,
        include_unrelated_tensor: bool,
    ) {
        write_llama_compatible_fixture(
            dir,
            model,
            include_unrelated_tensor,
            "llama",
            model.args.tie_word_embeddings,
        );
    }

    fn write_llama_compatible_fixture(
        dir: &Path,
        model: &llama::ResidentModel,
        include_unrelated_tensor: bool,
        model_type: &str,
        tie_word_embeddings: bool,
    ) {
        let params = model.parameters().flatten();
        let mut arrays = params
            .iter()
            .map(|(name, value)| {
                (
                    crate::runtime::checkpoint::binding::canonical_checkpoint_name(name),
                    (*value).clone(),
                )
            })
            .collect::<Vec<_>>();
        if include_unrelated_tensor {
            arrays.push((
                "unrelated.weight".to_string(),
                Array::from_slice(&[1.0f32], &[1]),
            ));
        }
        Array::save_safetensors(
            arrays.iter().map(|(name, value)| (name.as_str(), value)),
            None,
            dir.join("model.safetensors"),
        )
        .unwrap();
        fs::write(
            dir.join("config.json"),
            serde_json::to_vec(&serde_json::json!({
                "model_type": model_type,
                "hidden_size": 8,
                "num_hidden_layers": 2,
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-5,
                "vocab_size": 16,
                "max_position_embeddings": 64,
                "rope_theta": 10000.0,
                "rope_traditional": false,
                "head_dim": 4,
                "tie_word_embeddings": tie_word_embeddings,
                "attention_bias": false,
                "mlp_bias": false
            }))
            .unwrap(),
        )
        .unwrap();
    }

    fn llama_gguf_fixture(model: &llama::ResidentModel) -> SyntheticGguf {
        llama_compatible_gguf_fixture(model, "llama")
    }

    fn llama_compatible_gguf_fixture(
        model: &llama::ResidentModel,
        architecture: &str,
    ) -> SyntheticGguf {
        let arrays = model
            .parameters()
            .flatten()
            .into_iter()
            .map(|(name, value)| {
                let canonical =
                    crate::runtime::checkpoint::binding::canonical_checkpoint_name(&name);
                let gguf = match canonical.as_str() {
                    "model.embed_tokens.weight" => "token_embd.weight".into(),
                    "model.norm.weight" => "output_norm.weight".into(),
                    "lm_head.weight" => "output.weight".into(),
                    name => name
                        .replace("model.layers.", "blk.")
                        .replace(".input_layernorm.", ".attn_norm.")
                        .replace(".post_attention_layernorm.", ".ffn_norm.")
                        .replace(".self_attn.q_proj.", ".attn_q.")
                        .replace(".self_attn.k_proj.", ".attn_k.")
                        .replace(".self_attn.v_proj.", ".attn_v.")
                        .replace(".self_attn.o_proj.", ".attn_output.")
                        .replace(".mlp.gate_proj.", ".ffn_gate.")
                        .replace(".mlp.up_proj.", ".ffn_up.")
                        .replace(".mlp.down_proj.", ".ffn_down."),
                };
                (gguf, value.clone())
            })
            .collect::<HashMap<_, _>>();
        let metadata = HashMap::from([
            (
                "general.architecture".into(),
                GgufMetadataValue::String(architecture.into()),
            ),
            ("general.file_type".into(), GgufMetadataValue::Uint32(0)),
            (
                format!("{architecture}.block_count"),
                GgufMetadataValue::Uint32(2),
            ),
            (
                format!("{architecture}.embedding_length"),
                GgufMetadataValue::Uint32(8),
            ),
            (
                format!("{architecture}.attention.head_count"),
                GgufMetadataValue::Uint32(2),
            ),
            (
                format!("{architecture}.attention.head_count_kv"),
                GgufMetadataValue::Uint32(2),
            ),
            (
                format!("{architecture}.attention.key_length"),
                GgufMetadataValue::Uint32(4),
            ),
            (
                format!("{architecture}.feed_forward_length"),
                GgufMetadataValue::Uint32(16),
            ),
            (
                format!("{architecture}.attention.layer_norm_rms_epsilon"),
                GgufMetadataValue::Float32(1e-5),
            ),
            (
                format!("{architecture}.context_length"),
                GgufMetadataValue::Uint32(64),
            ),
            (
                format!("{architecture}.vocab_size"),
                GgufMetadataValue::Uint32(16),
            ),
        ]);
        SyntheticGguf::dense(&arrays, &metadata)
    }

    #[test]
    fn llama_gguf_pipeline_two_rank_outputs_and_caches_match_resident_model() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = gpu.stream();
        let mut source = llama::ResidentModel::new(llama_args(true), stream).unwrap();
        initialize_parameters(&mut source, stream);
        let fixture = llama_gguf_fixture(&source);
        let mut reference = llama::load_llama_gguf(fixture.path(), stream, cpu.stream()).unwrap();
        let mut first = load_pipeline_model_with_options(
            fixture.path(),
            ModelLoadOptions::with_parallel(gpu_topology(0)),
            stream,
            cpu.stream(),
        )
        .unwrap();
        let mut last = load_pipeline_model_with_options(
            fixture.path(),
            ModelLoadOptions::with_parallel(gpu_topology(1)),
            stream,
            cpu.stream(),
        )
        .unwrap();
        assert_eq!(first.stage_info().global_layer_range, 0..1);
        assert_eq!(last.stage_info().global_layer_range, 1..2);

        let mut reference_cache = reference.new_cache();
        let mut first_cache = first.new_cache().unwrap();
        let mut last_cache = last.new_cache().unwrap();
        for tokens in [
            Array::from_slice(&[1u32, 2], &[1, 2]),
            Array::from_slice(&[3u32], &[1, 1]),
        ] {
            let expected = reference
                .forward(
                    llama::ModelInput {
                        inputs: &tokens,
                        mask: None,
                        cache: &mut reference_cache,
                    },
                    stream,
                )
                .unwrap();
            let step = PipelineStep::new(1, tokens.shape()[1]).unwrap();
            let hidden = match first
                .forward_stage(
                    PipelineStageInput::Tokens(&tokens),
                    step,
                    None,
                    &mut first_cache,
                    stream,
                )
                .unwrap()
            {
                PipelineStageOutput::Hidden(hidden) => hidden,
                PipelineStageOutput::Logits(_) => panic!("first stage produced logits"),
            };
            let actual = match last
                .forward_stage(
                    PipelineStageInput::Hidden(&hidden),
                    step,
                    None,
                    &mut last_cache,
                    stream,
                )
                .unwrap()
            {
                PipelineStageOutput::Logits(logits) => logits,
                PipelineStageOutput::Hidden(_) => panic!("last stage produced hidden state"),
            };
            assert_close(&actual, &expected);
        }
        assert_eq!(reference_cache[0].as_ref().unwrap().offset(), 3);
        assert_eq!(first_cache.global_layers(), [0]);
        assert_eq!(last_cache.global_layers(), [1]);
    }

    #[test]
    fn llama_and_mistral_tp_pp_preflight_owns_sharded_boundaries_and_caches() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = gpu.stream();
        let mut args = llama_args(false);
        args.model_type = "mistral".into();
        let mut source = llama::ResidentModel::new(args, stream).unwrap();
        initialize_parameters(&mut source, stream);
        let safetensors = tempfile::tempdir().unwrap();
        write_llama_compatible_fixture(safetensors.path(), &source, false, "mistral", false);
        let gguf = llama_compatible_gguf_fixture(&source, "mistral");

        for path in [safetensors.path(), gguf.path()] {
            for rank in 0..4 {
                let topology = tp_pp_gpu_topology(rank);
                for residency in [
                    WeightResidency::fully_resident(),
                    WeightResidency::layerwise_host(LayerwiseLoadOptions::new(
                        OffloadConfig::new(None, None, 1).unwrap(),
                    )),
                    WeightResidency::dense_disk_stream(dense_stream_options()),
                ] {
                    let model = load_pipeline_model_with_options(
                        path,
                        ModelLoadOptions::with_parallel(topology).with_weight_residency(residency),
                        stream,
                        cpu.stream(),
                    )
                    .unwrap();
                    let info = model.stage_info();
                    assert_eq!(info.topology, topology);
                    assert_eq!(
                        info.global_layer_range,
                        topology.pipeline_parallel_rank..topology.pipeline_parallel_rank + 1
                    );
                    assert_eq!(
                        info.predecessor_rank,
                        topology.pipeline_predecessor().unwrap()
                    );
                    assert_eq!(info.successor_rank, topology.pipeline_successor().unwrap());
                    let layout = model.prompt_cache_layer_layout().unwrap();
                    assert_eq!(layout.len(), 1);
                    assert!(matches!(
                        layout.get(0),
                        Some(crate::LayerCachePolicy::KeyValue {
                            num_key_value_heads,
                            head_dim,
                            ..
                        }) if num_key_value_heads.get() == 1 && head_dim.get() == 4
                    ));
                    assert_eq!(
                        model.dense_stream_report().unwrap().is_some(),
                        matches!(residency.layers(), LayerWeightResidency::DenseDiskStream(_))
                    );
                    assert_eq!(
                        model.parameter_residency_report().unwrap().is_some(),
                        !matches!(residency.layers(), LayerWeightResidency::FullyResident)
                    );
                    if matches!(residency.layers(), LayerWeightResidency::LayerwiseHost(_)) {
                        let report = model.parameter_residency_report().unwrap().unwrap();
                        assert!(report.initialized());
                        assert!(report.units().iter().all(|unit| {
                            unit.planned_tier() == MemoryTier::Host
                                && unit.host_resident()
                                && !unit.device_resident()
                        }));
                    }
                    if info.is_first {
                        assert!(info
                            .owned_tensors
                            .iter()
                            .any(|name| name == "model.embed_tokens.weight"));
                    }
                    if info.is_last {
                        assert!(info
                            .owned_tensors
                            .iter()
                            .any(|name| name == "lm_head.weight"));
                    }
                }
            }
        }
    }

    fn llama_pipeline_dense_requirements(
        model_dir: &Path,
        topology: ParallelTopology,
        stream: &Stream,
        weights_stream: &Stream,
    ) -> (u64, u64, u64) {
        let sizing = load_pipeline_model_with_options(
            model_dir,
            ModelLoadOptions::with_parallel(topology).with_weight_residency(
                WeightResidency::dense_disk_stream(
                    crate::runtime::residency::dense_stream::DenseDiskStreamLoadOptions::new(
                        u64::MAX,
                        u64::MAX,
                        1,
                        1,
                        1,
                    )
                    .unwrap(),
                ),
            ),
            stream,
            weights_stream,
        )
        .unwrap();
        let report = sizing.dense_stream_report().unwrap().unwrap();
        let static_bytes = report.pinned_static_device_bytes();
        let layer_bytes = report.planned_layer_bytes();
        let device_bytes = static_bytes.checked_add(layer_bytes).unwrap();
        (device_bytes, layer_bytes, static_bytes)
    }

    fn sampled_dense_options(
        device_budget_bytes: u64,
        host_budget_bytes: u64,
    ) -> crate::runtime::residency::dense_stream::DenseDiskStreamLoadOptions {
        let mut options = crate::runtime::residency::dense_stream::DenseDiskStreamLoadOptions::new(
            device_budget_bytes,
            host_budget_bytes,
            1,
            1,
            1,
        )
        .unwrap();
        options.sample_mlx_memory = true;
        options.sample_process_memory = true;
        options
    }

    #[test]
    fn pipeline_dense_stream_respects_strict_loading() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = gpu.stream();
        let mut reference = llama::ResidentModel::new(llama_args(true), stream).unwrap();
        initialize_parameters(&mut reference, stream);
        let dir = tempfile::tempdir().unwrap();
        write_llama_fixture(dir.path(), &reference, true);

        let mut dense = crate::runtime::residency::dense_stream::DenseDiskStreamLoadOptions::new(
            u64::MAX,
            u64::MAX,
            1,
            1,
            1,
        )
        .unwrap();
        let strict_error = load_pipeline_model_with_options(
            dir.path(),
            ModelLoadOptions::with_parallel(gpu_topology(0))
                .with_weight_residency(WeightResidency::dense_disk_stream(dense)),
            stream,
            cpu.stream(),
        )
        .expect_err("strict pipeline loading must reject an unrelated checkpoint tensor");
        assert!(
            matches!(
                &strict_error,
                Error::StrictLoadValidation { missing, unused }
                    if missing.is_empty()
                        && unused.len() == 1
                        && unused[0] == "unrelated.weight"
            ),
            "unexpected strict pipeline diagnostic: {strict_error:?}"
        );

        dense.strict_loading = false;
        let model = load_pipeline_model_with_options(
            dir.path(),
            ModelLoadOptions::with_parallel(gpu_topology(0))
                .with_weight_residency(WeightResidency::dense_disk_stream(dense)),
            stream,
            cpu.stream(),
        )
        .expect("non-strict pipeline loading must ignore unrelated checkpoint tensors");
        assert!(model.dense_stream_report().unwrap().is_some());
    }

    #[test]
    fn llama_pipeline_dense_stream_is_stage_local_cold_and_matches_decode() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = gpu.stream();
        let mut reference = llama::ResidentModel::new(llama_args(true), stream).unwrap();
        initialize_parameters(&mut reference, stream);
        let dir = tempfile::tempdir().unwrap();
        write_llama_fixture(dir.path(), &reference, false);
        let first_requirements =
            llama_pipeline_dense_requirements(dir.path(), gpu_topology(0), stream, cpu.stream());
        let last_requirements =
            llama_pipeline_dense_requirements(dir.path(), gpu_topology(1), stream, cpu.stream());
        let mut first = load_pipeline_model_with_options(
            dir.path(),
            ModelLoadOptions::with_parallel(gpu_topology(0)).with_weight_residency(
                WeightResidency::dense_disk_stream(sampled_dense_options(
                    first_requirements.0,
                    first_requirements.1,
                )),
            ),
            stream,
            cpu.stream(),
        )
        .unwrap();
        let mut last = load_pipeline_model_with_options(
            dir.path(),
            ModelLoadOptions::with_parallel(gpu_topology(1)).with_weight_residency(
                WeightResidency::dense_disk_stream(sampled_dense_options(
                    last_requirements.0,
                    last_requirements.1,
                )),
            ),
            stream,
            cpu.stream(),
        )
        .unwrap();
        for (model, expected_id) in [
            (&first, "pipeline.layer.00000"),
            (&last, "pipeline.layer.00001"),
        ] {
            let report = model.dense_stream_report().unwrap().unwrap();
            let layers = report.residency().units();
            assert_eq!(layers.len(), 1);
            assert_eq!(layers[0].id().as_str(), expected_id);
            assert_eq!(layers[0].planned_tier(), MemoryTier::Disk);
            assert!(!layers[0].host_resident());
            assert!(!layers[0].device_resident());
            assert!(report.residency().offload().mlx_memory().is_none());
            assert!(!report.residency().offload().process_sampled());
        }

        let mut reference_cache = reference.new_cache();
        let mut first_cache = first.new_cache().unwrap();
        let mut last_cache = last.new_cache().unwrap();
        for tokens in [
            Array::from_slice(&[1u32, 2], &[1, 2]),
            Array::from_slice(&[3u32], &[1, 1]),
            Array::from_slice(&[4u32], &[1, 1]),
        ] {
            let sequence = tokens.shape()[1];
            let expected = reference
                .forward(
                    llama::ModelInput {
                        inputs: &tokens,
                        mask: None,
                        cache: &mut reference_cache,
                    },
                    stream,
                )
                .unwrap();
            let step = PipelineStep::new(1, sequence).unwrap();
            let hidden = match first
                .forward_stage(
                    PipelineStageInput::Tokens(&tokens),
                    step,
                    None,
                    &mut first_cache,
                    stream,
                )
                .unwrap()
            {
                PipelineStageOutput::Hidden(hidden) => hidden,
                PipelineStageOutput::Logits(_) => panic!("first stage produced logits"),
            };
            let actual = match last
                .forward_stage(
                    PipelineStageInput::Hidden(&hidden),
                    step,
                    None,
                    &mut last_cache,
                    stream,
                )
                .unwrap()
            {
                PipelineStageOutput::Logits(logits) => logits,
                PipelineStageOutput::Hidden(_) => panic!("last stage produced hidden state"),
            };
            assert_close(&actual, &expected);
        }
        for (model, (device_budget, host_budget, static_bytes)) in
            [(&first, first_requirements), (&last, last_requirements)]
        {
            let report = model.dense_stream_report().unwrap().unwrap();
            assert!(report.residency().offload().mlx_memory().is_some());
            assert!(report.residency().offload().process_sampled());
            assert_eq!(report.pinned_static_device_bytes(), static_bytes);
            assert_eq!(report.planned_layer_bytes(), host_budget);
            assert!(
                report
                    .device_layers()
                    .peak_layer_bytes()
                    .checked_add(static_bytes)
                    .unwrap()
                    <= device_budget
            );
            assert!(report.host_layers().peak_layer_bytes() <= host_budget);
        }
    }

    #[test]
    fn llama_pipeline_nonresident_layers_reject_undersized_local_budgets() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = gpu.stream();
        let mut reference = llama::ResidentModel::new(llama_args(true), stream).unwrap();
        initialize_parameters(&mut reference, stream);
        let dir = tempfile::tempdir().unwrap();
        write_llama_fixture(dir.path(), &reference, false);
        let (device_bytes, layer_bytes, static_bytes) =
            llama_pipeline_dense_requirements(dir.path(), gpu_topology(0), stream, cpu.stream());

        let host_budget = layer_bytes.checked_sub(1).unwrap();
        let host_error = load_pipeline_model_with_options(
            dir.path(),
            ModelLoadOptions::with_parallel(gpu_topology(0)).with_weight_residency(
                WeightResidency::dense_disk_stream(sampled_dense_options(
                    device_bytes,
                    host_budget,
                )),
            ),
            stream,
            cpu.stream(),
        )
        .expect_err("an undersized local host budget must fail");
        assert!(matches!(
            host_error,
            Error::Parallel(message)
                if message == format!(
                    "pipeline host budget {host_budget} cannot hold the largest protected local layer window ({layer_bytes} bytes)"
                )
        ));

        let device_budget = device_bytes.checked_sub(1).unwrap();
        let device_error = load_pipeline_model_with_options(
            dir.path(),
            ModelLoadOptions::with_parallel(gpu_topology(0)).with_weight_residency(
                WeightResidency::dense_disk_stream(sampled_dense_options(
                    device_budget,
                    layer_bytes,
                )),
            ),
            stream,
            cpu.stream(),
        )
        .expect_err("an undersized local device budget must fail");
        assert!(matches!(
            device_error,
            Error::Parallel(message)
                if message == format!(
                    "pipeline device budget {device_budget} cannot hold {static_bytes} pinned static bytes plus the largest local layer window ({layer_bytes} bytes, {device_bytes} total)"
                )
        ));

        let host_error = load_pipeline_model_with_options(
            dir.path(),
            ModelLoadOptions::with_parallel(gpu_topology(0)).with_weight_residency(
                WeightResidency::layerwise_host(LayerwiseLoadOptions::new(
                    OffloadConfig::new(Some(device_bytes), Some(host_budget), 1).unwrap(),
                )),
            ),
            stream,
            cpu.stream(),
        )
        .expect_err("host-layerwise residency must hold every rank-local layer on host");
        assert!(matches!(
            host_error,
            Error::Parallel(message)
                if message == format!(
                    "pipeline host budget {host_budget} cannot eagerly hold all {layer_bytes} rank-local layer bytes"
                )
        ));

        let device_error = load_pipeline_model_with_options(
            dir.path(),
            ModelLoadOptions::with_parallel(gpu_topology(0)).with_weight_residency(
                WeightResidency::layerwise_host(LayerwiseLoadOptions::new(
                    OffloadConfig::new(Some(device_budget), Some(layer_bytes), 1).unwrap(),
                )),
            ),
            stream,
            cpu.stream(),
        )
        .expect_err("host-layerwise residency must fit static weights and its device window");
        assert!(matches!(
            device_error,
            Error::Parallel(message)
                if message == format!(
                    "pipeline device budget {device_budget} cannot hold {static_bytes} pinned static bytes plus the largest local layer window ({layer_bytes} bytes, {device_bytes} total)"
                )
        ));
    }

    fn llama_pipeline_stages(
        source: &llama::ResidentModel,
        stream: &Stream,
    ) -> (PipelineModel, PipelineModel) {
        let first_topology = gpu_topology(0);
        let last_topology = gpu_topology(1);
        let first_info = base_info(
            first_topology,
            0..1,
            ModelKind::Llama,
            source.args.hidden_size,
        );
        let last_info = base_info(
            last_topology,
            1..2,
            ModelKind::Llama,
            source.args.hidden_size,
        );
        let first = LlamaStage {
            args: source.args.clone(),
            layer_adapter: crate::architectures::llama::layerwise::LlamaLayerwiseAdapter::new(
                source.args.clone(),
                stream,
            )
            .unwrap(),
            range: 0..1,
            embedding: Some(source.model.embed_tokens.clone()),
            output_embedding: None,
            layers: vec![source.model.layers[0].clone()],
            dense_layers: None,
            norm: None,
            lm_head: None,
            parallel_embedding: None,
            parallel_output_embedding: None,
            parallel_lm_head: None,
            parallel_layout: None,
            parallel_kv_heads: None,
        };
        let last = LlamaStage {
            args: source.args.clone(),
            layer_adapter: crate::architectures::llama::layerwise::LlamaLayerwiseAdapter::new(
                source.args.clone(),
                stream,
            )
            .unwrap(),
            range: 1..2,
            embedding: None,
            output_embedding: source
                .args
                .tie_word_embeddings
                .then(|| source.model.embed_tokens.clone()),
            layers: vec![source.model.layers[1].clone()],
            dense_layers: None,
            norm: Some(source.model.norm.clone()),
            lm_head: source.lm_head.clone(),
            parallel_embedding: None,
            parallel_output_embedding: None,
            parallel_lm_head: None,
            parallel_layout: None,
            parallel_kv_heads: None,
        };
        (
            PipelineModel::from_adapter(first_topology, first_info, PipelineStage(first)).unwrap(),
            PipelineModel::from_adapter(last_topology, last_info, PipelineStage(last)).unwrap(),
        )
    }

    #[test]
    fn pipeline_program_enforces_phase_lifecycle_and_cache_handoff() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = context.stream();
        let mut source = llama::ResidentModel::new(llama_args(false), stream).unwrap();
        initialize_parameters(&mut source, stream);
        let (first, _) = llama_pipeline_stages(&source, stream);
        let mut scheduler =
            PipelineInferenceScheduler::new(&first, SchedulerLimits::new(2, 4).unwrap()).unwrap();
        let first_request = RequestId::new(11);
        let second_request = RequestId::new(22);
        scheduler.register_request(&first, first_request).unwrap();
        scheduler.register_request(&first, second_request).unwrap();
        assert!(scheduler
            .register_request(&first, RequestId::new(33))
            .unwrap_err()
            .to_string()
            .contains("active-request capacity"));

        scheduler
            .enqueue(
                PipelineMicrobatchInput::new(
                    first_request,
                    PipelineInferencePhase::Prefill,
                    PipelineStep::new(1, 2).unwrap(),
                )
                .with_tokens(Array::from_slice(&[1u32, 2], &[1, 2])),
            )
            .unwrap();
        scheduler
            .enqueue(
                PipelineMicrobatchInput::new(
                    first_request,
                    PipelineInferencePhase::Decode,
                    PipelineStep::new(1, 1).unwrap(),
                )
                .with_tokens(Array::from_slice(&[3u32], &[1, 1])),
            )
            .unwrap();
        assert!(scheduler
            .enqueue(
                PipelineMicrobatchInput::new(
                    first_request,
                    PipelineInferencePhase::Prefill,
                    PipelineStep::new(1, 1).unwrap(),
                )
                .with_tokens(Array::from_slice(&[4u32], &[1, 1])),
            )
            .unwrap_err()
            .to_string()
            .contains("cannot return to prefill"));
        scheduler
            .enqueue(
                PipelineMicrobatchInput::new(
                    second_request,
                    PipelineInferencePhase::Prefill,
                    PipelineStep::new(1, 3).unwrap(),
                )
                .with_tokens(Array::from_slice(&[4u32, 5, 6], &[1, 3])),
            )
            .unwrap();
        scheduler
            .enqueue(
                PipelineMicrobatchInput::new(
                    second_request,
                    PipelineInferencePhase::Decode,
                    PipelineStep::new(1, 1).unwrap(),
                )
                .with_tokens(Array::from_slice(&[7u32], &[1, 1])),
            )
            .unwrap();
        assert!(scheduler
            .enqueue(
                PipelineMicrobatchInput::new(
                    second_request,
                    PipelineInferencePhase::Decode,
                    PipelineStep::new(1, 1).unwrap(),
                )
                .with_tokens(Array::from_slice(&[0u32], &[1, 1])),
            )
            .unwrap_err()
            .to_string()
            .contains("queue capacity"));

        scheduler.cancel_request(first_request).unwrap();
        scheduler.finish_request(second_request).unwrap();
        assert_eq!(
            scheduler.request_status(first_request),
            Some(RequestStatus::Cancelled)
        );
        assert_eq!(
            scheduler.request_status(second_request),
            Some(RequestStatus::Finished)
        );
        assert_eq!(
            scheduler.report(),
            SchedulerReport {
                active_requests: 0,
                queued_work: 0,
                peak_queued_work: 4,
                submitted_work: 4,
                completed_work: 0,
                failed_work: 0,
                discarded_work: 4,
                finished_requests: 1,
                cancelled_requests: 1,
                drain_cycles: 0,
                poisoned: false,
            }
        );
        assert_eq!(
            scheduler.forget_terminal_request(second_request).unwrap(),
            RequestStatus::Finished
        );
        scheduler.register_request(&first, second_request).unwrap();
        let released = scheduler.release_request_cache(second_request).unwrap();
        assert_eq!(released.global_layers(), vec![0]);
        assert_eq!(scheduler.request_status(second_request), None);
    }

    #[test]
    fn sequential_llama_pipeline_matches_tied_and_untied_prefill_decode() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = context.stream();
        for tied in [false, true] {
            let mut reference = llama::ResidentModel::new(llama_args(tied), stream).unwrap();
            initialize_parameters(&mut reference, stream);
            let (mut first, mut last) = llama_pipeline_stages(&reference, stream);
            let mut reference_cache = reference.new_cache();
            let mut first_cache = first.new_cache().unwrap();
            let mut last_cache = last.new_cache().unwrap();
            let prompt = Array::from_slice(&[1u32, 2], &[1, 2]);

            for (tokens, sequence) in [
                (&prompt, 2),
                (&Array::from_slice(&[3u32], &[1, 1]), 1),
                (&Array::from_slice(&[4u32], &[1, 1]), 1),
            ] {
                let reference_logits = reference
                    .forward(
                        llama::ModelInput {
                            inputs: tokens,
                            mask: None,
                            cache: &mut reference_cache,
                        },
                        stream,
                    )
                    .unwrap();
                let hidden = match first
                    .forward_stage(
                        PipelineStageInput::Tokens(tokens),
                        PipelineStep::new(1, sequence).unwrap(),
                        None,
                        &mut first_cache,
                        stream,
                    )
                    .unwrap()
                {
                    PipelineStageOutput::Hidden(hidden) => hidden,
                    PipelineStageOutput::Logits(_) => panic!("first stage produced logits"),
                };
                let pipeline_logits = match last
                    .forward_stage(
                        PipelineStageInput::Hidden(&hidden),
                        PipelineStep::new(1, sequence).unwrap(),
                        None,
                        &mut last_cache,
                        stream,
                    )
                    .unwrap()
                {
                    PipelineStageOutput::Logits(logits) => logits,
                    PipelineStageOutput::Hidden(_) => panic!("last stage produced hidden state"),
                };
                assert_close(&pipeline_logits, &reference_logits);
            }
        }
    }

    fn deepseek_args() -> deepseek_v3::ModelArgs {
        deepseek_v3::ModelArgs {
            model_type: "deepseek_v3".into(),
            hidden_size: 8,
            intermediate_size: 16,
            moe_intermediate_size: 4,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            vocab_size: 16,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 64,
            rope_theta: 10_000.0,
            rope_scaling: None,
            q_lora_rank: Some(4),
            kv_lora_rank: 4,
            qk_nope_head_dim: 2,
            qk_rope_head_dim: 2,
            v_head_dim: 2,
            layer_schedule: crate::runtime::attention::LayerSchedule::new(
                2,
                vec![
                    deepseek_v3::LayerPolicy::DenseMlp,
                    deepseek_v3::LayerPolicy::SparseMoe,
                ],
            )
            .unwrap(),
            n_routed_experts: 4,
            n_shared_experts: 1,
            num_experts_per_tok: 2,
            n_group: 2,
            topk_group: 1,
            topk_method: "noaux_tc".into(),
            scoring_func: "sigmoid".into(),
            norm_topk_prob: true,
            routed_scaling_factor: 1.5,
            num_nextn_predict_layers: 0,
            quantization_config: None,
            quantization: None,
            quantized_weight_configs: None,
            split_kv_b: false,
            tie_word_embeddings: false,
        }
    }

    fn initialized_deepseek(stream: &Stream) -> deepseek_v3::Model {
        let mut model = deepseek_v3::Model::new(deepseek_args(), stream).unwrap();
        if let deepseek_v3::FeedForward::Moe(moe) = &mut model.model.layers[1].mlp {
            let experts = model.args.n_routed_experts;
            let hidden = model.args.hidden_size;
            let intermediate = model.args.moe_intermediate_size;
            moe.experts.gate_proj = Param::new(Some(
                Array::full::<f32>(
                    &[experts, intermediate, hidden],
                    Array::from_f32(0.01),
                    stream,
                )
                .unwrap(),
            ));
            moe.experts.up_proj = moe.experts.gate_proj.clone();
            moe.experts.down_proj = Param::new(Some(
                Array::full::<f32>(
                    &[experts, hidden, intermediate],
                    Array::from_f32(0.01),
                    stream,
                )
                .unwrap(),
            ));
        } else {
            panic!("second tiny DeepSeek layer must be MoE");
        }
        initialize_parameters(&mut model, stream);
        model
    }

    fn write_deepseek_fixture(dir: &Path, model: &deepseek_v3::Model, stream: &Stream) {
        let mut arrays = Vec::<(String, Array)>::new();
        for (name, value) in model.parameters().flatten() {
            let name = crate::runtime::checkpoint::binding::canonical_checkpoint_name(&name);
            let packed_projection = ["gate_proj", "up_proj", "down_proj"]
                .into_iter()
                .find(|projection| name.ends_with(&format!(".mlp.experts.{projection}")));
            if let Some(projection) = packed_projection {
                let suffix = format!(".experts.{projection}");
                let prefix = name.strip_suffix(&suffix).unwrap();
                for expert in 0..model.args.n_routed_experts {
                    arrays.push((
                        format!("{prefix}.experts.{expert}.{projection}.weight"),
                        value.try_index_device(expert, stream).unwrap(),
                    ));
                }
            } else {
                arrays.push((name, value.clone()));
            }
        }
        Array::save_safetensors(
            arrays.iter().map(|(name, value)| (name.as_str(), value)),
            None,
            dir.join("model.safetensors"),
        )
        .unwrap();
        fs::write(
            dir.join("config.json"),
            serde_json::to_vec(&serde_json::json!({
                "model_type": "deepseek_v3",
                "hidden_size": 8,
                "intermediate_size": 16,
                "moe_intermediate_size": 4,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "vocab_size": 16,
                "rms_norm_eps": 1e-6,
                "max_position_embeddings": 64,
                "rope_theta": 10000.0,
                "q_lora_rank": 4,
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
                "routed_scaling_factor": 1.5,
                "num_nextn_predict_layers": 0,
                "split_kv_b": false,
                "tie_word_embeddings": false
            }))
            .unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn deepseek_pipeline_dense_stream_crosses_dense_to_moe_boundary() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = gpu.stream();
        let mut reference = initialized_deepseek(stream);
        let dir = tempfile::tempdir().unwrap();
        write_deepseek_fixture(dir.path(), &reference, stream);
        let dense = crate::runtime::residency::dense_stream::DenseDiskStreamLoadOptions::new(
            u64::MAX,
            u64::MAX,
            1,
            1,
            1,
        )
        .unwrap();
        let mut first = load_pipeline_model_with_options(
            dir.path(),
            ModelLoadOptions::with_parallel(gpu_topology(0))
                .with_weight_residency(WeightResidency::dense_disk_stream(dense)),
            stream,
            cpu.stream(),
        )
        .unwrap();
        let mut last = load_pipeline_model_with_options(
            dir.path(),
            ModelLoadOptions::with_parallel(gpu_topology(1))
                .with_weight_residency(WeightResidency::dense_disk_stream(dense)),
            stream,
            cpu.stream(),
        )
        .unwrap();
        assert_eq!(
            first
                .dense_stream_report()
                .unwrap()
                .unwrap()
                .residency()
                .units()[0]
                .id()
                .as_str(),
            "pipeline.layer.00000"
        );
        assert_eq!(
            last.dense_stream_report()
                .unwrap()
                .unwrap()
                .residency()
                .units()[0]
                .id()
                .as_str(),
            "pipeline.layer.00001"
        );

        let mut reference_cache = reference.new_cache();
        let mut first_cache = first.new_cache().unwrap();
        let mut last_cache = last.new_cache().unwrap();
        for tokens in [
            Array::from_slice(&[1u32, 2], &[1, 2]),
            Array::from_slice(&[3u32], &[1, 1]),
            Array::from_slice(&[4u32], &[1, 1]),
        ] {
            let step = PipelineStep::new(1, tokens.shape()[1]).unwrap();
            let expected = reference
                .forward(
                    deepseek_v3::ModelInput {
                        inputs: &tokens,
                        mask: None,
                        cache: Some(&mut reference_cache),
                    },
                    stream,
                )
                .unwrap();
            let hidden = match first
                .forward_stage(
                    PipelineStageInput::Tokens(&tokens),
                    step,
                    None,
                    &mut first_cache,
                    stream,
                )
                .unwrap()
            {
                PipelineStageOutput::Hidden(hidden) => hidden,
                PipelineStageOutput::Logits(_) => panic!("first stage produced logits"),
            };
            let actual = match last
                .forward_stage(
                    PipelineStageInput::Hidden(&hidden),
                    step,
                    None,
                    &mut last_cache,
                    stream,
                )
                .unwrap()
            {
                PipelineStageOutput::Logits(logits) => logits,
                PipelineStageOutput::Hidden(_) => panic!("last stage produced hidden state"),
            };
            assert_close(&actual, &expected);
        }
    }

    #[test]
    fn sequential_deepseek_pipeline_matches_local_moe_prefill_decode() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = context.stream();
        let mut reference = initialized_deepseek(stream);

        let first_topology = gpu_topology(0);
        let last_topology = gpu_topology(1);
        let mut first = PipelineModel::from_adapter(
            first_topology,
            base_info(first_topology, 0..1, ModelKind::DeepSeekV3, 8),
            PipelineStage(DeepSeekStage {
                args: reference.args.clone(),
                layer_adapter:
                    crate::architectures::deepseek_v3::layerwise::DeepSeekV3LayerwiseAdapter::new(
                        reference.args.clone(),
                        stream,
                    )
                    .unwrap(),
                range: 0..1,
                embedding: Some(reference.model.embed_tokens.clone()),
                layers: vec![reference.model.layers[0].clone()],
                dense_layers: None,
                norm: None,
                lm_head: None,
                parallel_embedding: None,
                parallel_lm_head: None,
                parallel_layout: None,
                expert_assignment: None,
                expert_storage: PipelineExpertStorage::LayerLocal,
                routing_statistics: RoutingStatistics::default(),
            }),
        )
        .unwrap();
        let mut last = PipelineModel::from_adapter(
            last_topology,
            base_info(last_topology, 1..2, ModelKind::DeepSeekV3, 8),
            PipelineStage(DeepSeekStage {
                args: reference.args.clone(),
                layer_adapter:
                    crate::architectures::deepseek_v3::layerwise::DeepSeekV3LayerwiseAdapter::new(
                        reference.args.clone(),
                        stream,
                    )
                    .unwrap(),
                range: 1..2,
                embedding: None,
                layers: vec![reference.model.layers[1].clone()],
                dense_layers: None,
                norm: Some(reference.model.norm.clone()),
                lm_head: Some(reference.lm_head.clone()),
                parallel_embedding: None,
                parallel_lm_head: None,
                parallel_layout: None,
                expert_assignment: None,
                expert_storage: PipelineExpertStorage::LayerLocal,
                routing_statistics: RoutingStatistics::default(),
            }),
        )
        .unwrap();
        let mut reference_cache = reference.new_cache();
        let mut first_cache = first.new_cache().unwrap();
        let mut last_cache = last.new_cache().unwrap();
        let prompt = Array::from_slice(&[1u32, 2], &[1, 2]);

        for (tokens, sequence) in [
            (&prompt, 2),
            (&Array::from_slice(&[3u32], &[1, 1]), 1),
            (&Array::from_slice(&[4u32], &[1, 1]), 1),
        ] {
            let reference_logits = reference
                .forward(
                    deepseek_v3::ModelInput {
                        inputs: tokens,
                        mask: None,
                        cache: Some(&mut reference_cache),
                    },
                    stream,
                )
                .unwrap();
            let hidden = match first
                .forward_stage(
                    PipelineStageInput::Tokens(tokens),
                    PipelineStep::new(1, sequence).unwrap(),
                    None,
                    &mut first_cache,
                    stream,
                )
                .unwrap()
            {
                PipelineStageOutput::Hidden(hidden) => hidden,
                PipelineStageOutput::Logits(_) => panic!("first stage produced logits"),
            };
            let pipeline_logits = match last
                .forward_stage(
                    PipelineStageInput::Hidden(&hidden),
                    PipelineStep::new(1, sequence).unwrap(),
                    None,
                    &mut last_cache,
                    stream,
                )
                .unwrap()
            {
                PipelineStageOutput::Logits(logits) => logits,
                PipelineStageOutput::Hidden(_) => panic!("last stage produced hidden state"),
            };
            assert_close(&pipeline_logits, &reference_logits);
        }
        assert_eq!(last_cache.global_layers(), vec![1]);
    }
}
