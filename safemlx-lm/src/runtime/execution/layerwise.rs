//! Architecture-independent execution of decoder models from resident layers.
//!
//! [`crate::runtime::execution::layerwise::LayerwiseModel`] owns checkpoint
//! storage, residency, bounded device
//! windows, and synchronization. Model-family behavior is supplied by an
//! [`crate::runtime::execution::layerwise::ArchitectureAdapter`].

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    ops::Range,
    path::Path,
    sync::{Arc, Mutex},
};

use safemlx::{module::ModuleParameters, transforms::async_eval_with_event, Array, Stream};

use crate::{
    error::Error,
    runtime::cache::residency::{
        validate_prompt_cache_model_identity, PagedCacheOptions, PromptCacheDescriptor,
        PromptCacheManifest, PromptCacheModelIdentity, PromptCacheOptions, PromptCacheTopology,
    },
    runtime::checkpoint::binding::{
        binding_bytes, build_module_bindings, populate_module_from_lease, ModuleBindingError,
    },
    runtime::checkpoint::bounded_quantization::{
        BoundedQuantizationPlan, BoundedQuantizationReport, BoundedQuantizationTarget,
        BoundedQuantizedWeightStore,
    },
    runtime::checkpoint::quantization::WeightQuantization,
    runtime::checkpoint::recipe::RecipeDtype,
    runtime::checkpoint::store::{MemoryWeightStore, SafetensorsWeightStore, WeightStore},
    runtime::execution::inspection::{ActivationObserver, ActivationObserverProxy},
    runtime::residency::dense_stream::{
        BackgroundLayerPrefetch, BackgroundPrefetchReport, DenseDiskStreamLoadOptions,
        DENSE_TRANSFER_WINDOW,
    },
    runtime::residency::manager::{
        OffloadUnit, ResidencyError, ResidencyManager, ResidencyReport, ResidentLayerGroup,
        ResidentTransfer, ResidentUnitLease,
    },
    runtime::residency::policy::{
        MemoryTier, OffloadConfig, OffloadPlan, OffloadReport, OffloadUnitId, OffloadUnitSpec,
        ResidencyPolicy, TransferDirection,
    },
};

/// Type-erased checkpoint store accepted by the generalized execution engine.
pub type SharedWeightStore = Arc<dyn WeightStore + Send + Sync>;

pub(crate) fn open_safetensors_weight_store(
    model_dir: &Path,
    max_mapped_shards: usize,
) -> Result<SharedWeightStore, Error> {
    Ok(Arc::new(
        SafetensorsWeightStore::open_with_max_mapped_shards(model_dir, max_mapped_shards)?,
    ))
}

/// Captures a completed load-time transformation as an immutable checkpoint.
///
/// Quantization changes one dense source tensor into a packed parameter group,
/// so it cannot be represented as a one-to-one lazy binding. The transformation
/// is performed once, then this store hands the resulting arrays to the same
/// generalized residency and execution engine used by native packed artifacts.
pub(crate) fn transformed_module_weight_store(
    module: &impl ModuleParameters,
) -> Result<SharedWeightStore, Error> {
    let mut arrays = BTreeMap::new();
    for (parameter_name, value) in module.parameters().flatten() {
        let checkpoint_name =
            crate::runtime::checkpoint::binding::canonical_checkpoint_name(&parameter_name);
        if arrays
            .insert(checkpoint_name.clone(), value.clone())
            .is_some()
        {
            return Err(Error::Quantization(format!(
                "load-time transformation produced duplicate checkpoint tensor {checkpoint_name:?}"
            )));
        }
    }
    Ok(Arc::new(MemoryWeightStore::new(arrays)?))
}

pub(crate) fn validate_gguf_layerwise_source(
    checkpoint: &safemlx::ops::GgufCheckpoint,
    metadata: &std::collections::HashMap<String, safemlx::ops::GgufMetadataValue>,
    options: LayerWeightResidency,
) -> Result<crate::api::GgufArchitecture, Error> {
    let architecture_name = match metadata.get("general.architecture") {
        Some(safemlx::ops::GgufMetadataValue::String(name)) => name,
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
    let architecture = crate::api::GgufArchitecture::resolve(architecture_name)?;
    let residency = options.weight_residency();
    crate::api::structural::validate_gguf(
        architecture,
        checkpoint,
        metadata,
        crate::api::ModelLoadOptions::default().with_weight_residency(residency),
    )
    .into_loader_result()?;
    Ok(architecture)
}

/// Loader controls for a host-backed layerwise execution engine.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct LayerwiseLoadOptions {
    /// Residency budgets and maximum device-layer window.
    pub offload: OffloadConfig,
    /// Maximum number of checkpoint payload shards retained as mappings.
    pub max_mapped_shards: usize,
    /// Reject checkpoint tensors unrelated to the adapter's parameter tree.
    pub strict_loading: bool,
    /// Sample MLX allocator memory when a forward pass completes.
    pub sample_mlx_memory: bool,
    /// Sample process memory metrics when a forward pass completes.
    pub sample_process_memory: bool,
}

impl LayerwiseLoadOptions {
    /// Creates strict options with the default mapped-shard bound.
    pub fn new(offload: OffloadConfig) -> Self {
        Self {
            offload,
            ..Self::default()
        }
    }
}

impl Default for LayerwiseLoadOptions {
    fn default() -> Self {
        Self {
            offload: OffloadConfig::default(),
            max_mapped_shards: crate::runtime::checkpoint::store::DEFAULT_MAX_MAPPED_SHARDS,
            strict_loading: true,
            sample_mlx_memory: false,
            sample_process_memory: false,
        }
    }
}

/// Placement policy for ordinary architecture execution units.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Default)]
pub enum LayerWeightResidency {
    /// Construct every rank-local module once and retain it on the execution device.
    #[default]
    FullyResident,
    /// Eagerly materialize units on a host stream and use a bounded device window.
    LayerwiseHost(LayerwiseLoadOptions),
    /// Leave units cold on disk and use finite host and device caches.
    DenseDiskStream(DenseDiskStreamLoadOptions),
}

impl LayerWeightResidency {
    /// Returns the backend shard/reader cache bound carried by this policy.
    pub(crate) const fn max_mapped_shards(self) -> usize {
        match self {
            Self::FullyResident => crate::runtime::checkpoint::store::DEFAULT_MAX_MAPPED_SHARDS,
            Self::LayerwiseHost(options) => options.max_mapped_shards,
            Self::DenseDiskStream(options) => options.max_mapped_shards,
        }
    }

    /// Returns whether whole-artifact admission rejects unrelated tensors.
    pub(crate) const fn strict_loading(self) -> bool {
        match self {
            Self::FullyResident => true,
            Self::LayerwiseHost(options) => options.strict_loading,
            Self::DenseDiskStream(options) => options.strict_loading,
        }
    }
}

/// Routed-expert placement relative to ordinary layer units.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub enum ExpertWeightResidency {
    /// Keep routed experts in the ordinary layer residency unit.
    #[default]
    WithLayer,
    /// Catalog routed experts as independent atomic residency units.
    IndependentCache(crate::runtime::residency::expert_cache::ExpertCacheLoadOptions),
}

/// Placement of non-expert parameters when routed experts are independent units.
///
/// This is distinct from [`LayerWeightResidency`]: `FullyResident` here pins
/// attention, routers, shared experts, norms, and other non-routed parameters,
/// while routed expert banks remain governed by their independent cache.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum NonExpertWeightResidency {
    /// Keep every non-expert parameter resident on the execution device.
    FullyResident,
    /// Eagerly materialize non-expert units on host behind a device window.
    LayerwiseHost(LayerwiseLoadOptions),
    /// Leave non-expert units cold on disk behind finite tier caches.
    DenseDiskStream(DenseDiskStreamLoadOptions),
}

impl NonExpertWeightResidency {
    /// Returns the corresponding generalized layer policy.
    pub const fn layers(self) -> LayerWeightResidency {
        match self {
            Self::FullyResident => LayerWeightResidency::FullyResident,
            Self::LayerwiseHost(options) => LayerWeightResidency::LayerwiseHost(options),
            Self::DenseDiskStream(options) => LayerWeightResidency::DenseDiskStream(options),
        }
    }
}

/// Composable weight placement selected before checkpoint materialization.
///
/// Its sum-of-products shape distinguishes complete layer residency from
/// non-expert residency plus independently managed routed experts. Whether
/// independent experts apply to a checkpoint is necessarily validated after
/// architecture metadata is read.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum WeightResidency {
    /// Routed experts share the ordinary layer residency unit.
    Layers(LayerWeightResidency),
    /// Routed experts are independent units beside bounded ordinary layers.
    IndependentExperts {
        /// Bounded ordinary-layer policy.
        non_experts: NonExpertWeightResidency,
        /// Expert-granular cache controls.
        cache: crate::runtime::residency::expert_cache::ExpertCacheLoadOptions,
    },
}

impl WeightResidency {
    /// Keeps every rank-owned parameter on the execution device.
    pub const fn fully_resident() -> Self {
        Self::with_layers(LayerWeightResidency::FullyResident)
    }

    /// Eagerly materializes ordinary units on host behind a bounded device window.
    pub const fn layerwise_host(options: LayerwiseLoadOptions) -> Self {
        Self::with_layers(LayerWeightResidency::LayerwiseHost(options))
    }

    /// Leaves ordinary units cold on disk behind finite host/device caches.
    pub const fn dense_disk_stream(options: DenseDiskStreamLoadOptions) -> Self {
        Self::with_layers(LayerWeightResidency::DenseDiskStream(options))
    }

    /// Keeps every owned parameter in its ordinary layer residency unit.
    pub const fn with_layers(layers: LayerWeightResidency) -> Self {
        Self::Layers(layers)
    }

    /// Gives routed experts an independent cache beside the selected ordinary-layer policy.
    pub const fn with_expert_cache(
        non_experts: NonExpertWeightResidency,
        experts: crate::runtime::residency::expert_cache::ExpertCacheLoadOptions,
    ) -> Self {
        Self::IndependentExperts {
            non_experts,
            cache: experts,
        }
    }

    /// Returns the ordinary-layer placement policy.
    pub const fn layers(self) -> LayerWeightResidency {
        match self {
            Self::Layers(layers) => layers,
            Self::IndependentExperts { non_experts, .. } => non_experts.layers(),
        }
    }

    /// Returns routed-expert placement relative to ordinary layers.
    pub const fn experts(self) -> ExpertWeightResidency {
        match self {
            Self::Layers(_) => ExpertWeightResidency::WithLayer,
            Self::IndependentExperts { cache, .. } => {
                ExpertWeightResidency::IndependentCache(cache)
            }
        }
    }

    /// Returns independently cached expert controls, when selected.
    pub const fn expert_cache(
        self,
    ) -> Option<crate::runtime::residency::expert_cache::ExpertCacheLoadOptions> {
        match self {
            Self::Layers(_) => None,
            Self::IndependentExperts { cache, .. } => Some(cache),
        }
    }

    /// Returns non-expert placement paired with an independent expert cache.
    pub const fn non_experts(self) -> Option<NonExpertWeightResidency> {
        match self {
            Self::Layers(_) => None,
            Self::IndependentExperts { non_experts, .. } => Some(non_experts),
        }
    }

    /// Returns whether every non-expert parameter remains resident.
    pub const fn non_experts_are_fully_resident(self) -> bool {
        matches!(
            self,
            Self::Layers(LayerWeightResidency::FullyResident)
                | Self::IndependentExperts {
                    non_experts: NonExpertWeightResidency::FullyResident,
                    ..
                }
        )
    }

    /// Returns whether every ordinary layer remains resident.
    pub const fn is_fully_resident(self) -> bool {
        matches!(self, Self::Layers(LayerWeightResidency::FullyResident))
    }

    /// Returns the common backend shard/reader cache bound.
    pub(crate) const fn max_mapped_shards(self) -> usize {
        self.layers().max_mapped_shards()
    }

    /// Returns whether whole-artifact admission rejects unrelated tensors.
    pub(crate) const fn strict_loading(self) -> bool {
        self.layers().strict_loading()
    }
}

impl From<NonExpertWeightResidency> for LayerWeightResidency {
    fn from(value: NonExpertWeightResidency) -> Self {
        value.layers()
    }
}

impl Default for WeightResidency {
    fn default() -> Self {
        Self::fully_resident()
    }
}

/// Static-parameter placement used by the generalized execution engine.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ExecutionResidency {
    /// Every module is constructed once and all rank-local parameters remain on device.
    FullyResident,
    /// Decoder parameters remain on host behind bounded device windows.
    LayerwiseHost,
    /// Decoder parameters are materialized through bounded disk, host, and device caches.
    DenseDiskStream,
}

impl From<LayerwiseLoadOptions> for LayerWeightResidency {
    fn from(value: LayerwiseLoadOptions) -> Self {
        Self::LayerwiseHost(value)
    }
}

impl From<DenseDiskStreamLoadOptions> for LayerWeightResidency {
    fn from(value: DenseDiskStreamLoadOptions) -> Self {
        Self::DenseDiskStream(value)
    }
}

impl LayerWeightResidency {
    fn sample_mlx_memory(self) -> bool {
        match self {
            Self::FullyResident => false,
            Self::LayerwiseHost(options) => options.sample_mlx_memory,
            Self::DenseDiskStream(options) => options.sample_mlx_memory,
        }
    }

    fn sample_process_memory(self) -> bool {
        match self {
            Self::FullyResident => false,
            Self::LayerwiseHost(options) => options.sample_process_memory,
            Self::DenseDiskStream(options) => options.sample_process_memory,
        }
    }

    fn device_depth(self, layer_count: usize) -> usize {
        match self {
            Self::FullyResident => layer_count,
            Self::LayerwiseHost(options) => options.offload.prefetch_depth(),
            Self::DenseDiskStream(_) => layer_count.min(DENSE_TRANSFER_WINDOW),
        }
    }

    fn offload(self) -> Result<OffloadConfig, Error> {
        match self {
            Self::FullyResident => Ok(OffloadConfig::default()),
            Self::LayerwiseHost(options) => Ok(options.offload),
            Self::DenseDiskStream(options) => {
                options.validate()?;
                Ok(OffloadConfig::new(
                    Some(options.device_budget_bytes),
                    Some(options.host_budget_bytes),
                    options.host_lookahead.max(DENSE_TRANSFER_WINDOW),
                )?
                .with_eviction_policy(options.eviction_policy))
            }
        }
    }

    fn dense(self) -> Option<DenseDiskStreamLoadOptions> {
        match self {
            Self::DenseDiskStream(options) => Some(options),
            Self::FullyResident | Self::LayerwiseHost(_) => None,
        }
    }

    pub(crate) const fn weight_residency(self) -> WeightResidency {
        WeightResidency::with_layers(self)
    }

    const fn is_fully_resident(self) -> bool {
        matches!(self, Self::FullyResident)
    }

    const fn residency(self) -> ExecutionResidency {
        match self {
            Self::FullyResident => ExecutionResidency::FullyResident,
            Self::LayerwiseHost(_) => ExecutionResidency::LayerwiseHost,
            Self::DenseDiskStream(_) => ExecutionResidency::DenseDiskStream,
        }
    }
}

/// Stable dense-stream observations combining residency and worker state.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct DenseDiskStreamReport {
    planned_layer_count: usize,
    planned_layer_bytes: u64,
    pinned_static_device_bytes: u64,
    transfer_stream_index: i32,
    residency: ResidencyReport,
    background: BackgroundPrefetchReport,
    host_layers: DenseTierResidencyReport,
    device_layers: DenseTierResidencyReport,
    groups: Vec<DenseExecutionGroupReport>,
    prefill: DensePassReport,
    decode: DensePassReport,
}

/// Cache activity attributed to one logical residency tier.
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq)]
pub struct DenseCacheMetrics {
    requests: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
    evicted_bytes: u64,
}

impl DenseCacheMetrics {
    /// Returns cache requests targeting the tier.
    pub const fn requests(self) -> u64 {
        self.requests
    }
    /// Returns requests served by an existing tier copy.
    pub const fn hits(self) -> u64 {
        self.hits
    }
    /// Returns requests requiring tier materialization.
    pub const fn misses(self) -> u64 {
        self.misses
    }
    /// Returns copies evicted from the tier.
    pub const fn evictions(self) -> u64 {
        self.evictions
    }
    /// Returns logical bytes evicted from the tier.
    pub const fn evicted_bytes(self) -> u64 {
        self.evicted_bytes
    }

    fn from_report(report: &OffloadReport, tier: MemoryTier) -> Self {
        let prefetch = report.tier_prefetch(tier);
        let evictions = report.tier_evictions(tier);
        Self {
            requests: prefetch.requests(),
            hits: prefetch.hits(),
            misses: prefetch.misses(),
            evictions: evictions.count(),
            evicted_bytes: evictions.bytes(),
        }
    }

    fn saturating_delta(self, earlier: Self) -> Self {
        Self {
            requests: self.requests.saturating_sub(earlier.requests),
            hits: self.hits.saturating_sub(earlier.hits),
            misses: self.misses.saturating_sub(earlier.misses),
            evictions: self.evictions.saturating_sub(earlier.evictions),
            evicted_bytes: self.evicted_bytes.saturating_sub(earlier.evicted_bytes),
        }
    }

    fn saturating_add(&mut self, other: Self) {
        self.requests = self.requests.saturating_add(other.requests);
        self.hits = self.hits.saturating_add(other.hits);
        self.misses = self.misses.saturating_add(other.misses);
        self.evictions = self.evictions.saturating_add(other.evictions);
        self.evicted_bytes = self.evicted_bytes.saturating_add(other.evicted_bytes);
    }
}

/// Streamed-layer occupancy and cache history for one tier.
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq)]
pub struct DenseTierResidencyReport {
    current_layer_count: usize,
    peak_layer_count: usize,
    current_layer_bytes: u64,
    peak_layer_bytes: u64,
    cache: DenseCacheMetrics,
}

impl DenseTierResidencyReport {
    /// Returns currently resident streamed layers.
    pub const fn current_layer_count(self) -> usize {
        self.current_layer_count
    }
    /// Returns the peak number of simultaneously resident streamed layers.
    pub const fn peak_layer_count(self) -> usize {
        self.peak_layer_count
    }
    /// Returns current streamed-layer bytes in the tier.
    pub const fn current_layer_bytes(self) -> u64 {
        self.current_layer_bytes
    }
    /// Returns peak streamed-layer bytes in the tier.
    pub const fn peak_layer_bytes(self) -> u64 {
        self.peak_layer_bytes
    }
    /// Returns cumulative cache activity for the tier.
    pub const fn cache(self) -> DenseCacheMetrics {
        self.cache
    }
}

/// Point-in-time occupancy for one named execution stack.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct DenseExecutionGroupReport {
    id: String,
    planned_layers: usize,
    planned_bytes: u64,
    completed_executions: u64,
    host_layers: usize,
    host_bytes: u64,
    peak_host_layers: usize,
    peak_host_bytes: u64,
    device_layers: usize,
    device_bytes: u64,
    peak_device_layers: usize,
    peak_device_bytes: u64,
}

impl DenseExecutionGroupReport {
    /// Returns the stable execution-group identifier.
    pub fn id(&self) -> &str {
        &self.id
    }
    /// Returns disk-planned layers in the group.
    pub const fn planned_layers(&self) -> usize {
        self.planned_layers
    }
    /// Returns logical checkpoint bytes in the group.
    pub const fn planned_bytes(&self) -> u64 {
        self.planned_bytes
    }
    /// Returns successfully completed executions of this group.
    pub const fn completed_executions(&self) -> u64 {
        self.completed_executions
    }
    /// Returns current host-resident group layers.
    pub const fn host_layers(&self) -> usize {
        self.host_layers
    }
    /// Returns current host-resident group bytes.
    pub const fn host_bytes(&self) -> u64 {
        self.host_bytes
    }
    /// Returns the peak number of host-resident layers observed for the group.
    pub const fn peak_host_layers(&self) -> usize {
        self.peak_host_layers
    }
    /// Returns peak host-resident layer bytes observed for the group.
    pub const fn peak_host_bytes(&self) -> u64 {
        self.peak_host_bytes
    }
    /// Returns current device-resident group layers.
    pub const fn device_layers(&self) -> usize {
        self.device_layers
    }
    /// Returns current device-resident group bytes.
    pub const fn device_bytes(&self) -> u64 {
        self.device_bytes
    }
    /// Returns the peak number of device-resident layers observed for the group.
    pub const fn peak_device_layers(&self) -> usize {
        self.peak_device_layers
    }
    /// Returns peak device-resident layer bytes observed for the group.
    pub const fn peak_device_bytes(&self) -> u64 {
        self.peak_device_bytes
    }
}

/// Cache and logical transfer activity from completed prefill or decode forwards.
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq)]
pub struct DensePassReport {
    forwards: u64,
    host_cache: DenseCacheMetrics,
    device_cache: DenseCacheMetrics,
    peak_host_layers: usize,
    peak_host_bytes: u64,
    peak_device_layers: usize,
    peak_device_bytes: u64,
    disk_to_host_bytes: u64,
    disk_to_device_bytes: u64,
    host_to_device_bytes: u64,
}

impl DensePassReport {
    /// Returns completed forwards in this pass category.
    pub const fn forwards(self) -> u64 {
        self.forwards
    }
    /// Returns host-cache activity during completed forwards.
    pub const fn host_cache(self) -> DenseCacheMetrics {
        self.host_cache
    }
    /// Returns device-cache activity during completed forwards.
    pub const fn device_cache(self) -> DenseCacheMetrics {
        self.device_cache
    }
    /// Returns peak host-resident streamed layers observed during these forwards.
    pub const fn peak_host_layers(self) -> usize {
        self.peak_host_layers
    }
    /// Returns peak host-resident streamed-layer bytes during these forwards.
    pub const fn peak_host_bytes(self) -> u64 {
        self.peak_host_bytes
    }
    /// Returns peak device-resident streamed layers observed during these forwards.
    pub const fn peak_device_layers(self) -> usize {
        self.peak_device_layers
    }
    /// Returns peak device-resident streamed-layer bytes during these forwards.
    pub const fn peak_device_bytes(self) -> u64 {
        self.peak_device_bytes
    }
    /// Returns logical disk-to-host bytes during completed forwards.
    pub const fn disk_to_host_bytes(self) -> u64 {
        self.disk_to_host_bytes
    }
    /// Returns logical disk-to-device bytes during completed forwards.
    pub const fn disk_to_device_bytes(self) -> u64 {
        self.disk_to_device_bytes
    }
    /// Returns logical host-to-device bytes during completed forwards.
    pub const fn host_to_device_bytes(self) -> u64 {
        self.host_to_device_bytes
    }
}

impl DenseDiskStreamReport {
    pub(crate) fn with_materialization(
        mut self,
        materialization: Option<BoundedQuantizationReport>,
    ) -> Self {
        self.residency = self.residency.with_materialization(materialization);
        self
    }

    /// Returns the number of disk-planned execution units.
    pub const fn planned_layer_count(&self) -> usize {
        self.planned_layer_count
    }
    /// Returns the logical checkpoint bytes in disk-planned execution units.
    pub const fn planned_layer_bytes(&self) -> u64 {
        self.planned_layer_bytes
    }
    /// Returns pinned static parameter bytes outside the streamed-layer totals.
    pub const fn pinned_static_device_bytes(&self) -> u64 {
        self.pinned_static_device_bytes
    }
    /// Returns the distinct MLX stream used for device weight transfers.
    pub const fn transfer_stream_index(&self) -> i32 {
        self.transfer_stream_index
    }
    /// Returns the complete logical tier and checkpoint-store report.
    pub const fn residency(&self) -> &ResidencyReport {
        &self.residency
    }
    /// Returns bounded background worker observations.
    pub const fn background(&self) -> BackgroundPrefetchReport {
        self.background
    }
    /// Returns streamed host-layer occupancy and cache history.
    pub const fn host_layers(&self) -> DenseTierResidencyReport {
        self.host_layers
    }
    /// Returns streamed device-layer occupancy and cache history.
    pub const fn device_layers(&self) -> DenseTierResidencyReport {
        self.device_layers
    }
    /// Returns point-in-time observations for each named execution group.
    pub fn execution_groups(&self) -> &[DenseExecutionGroupReport] {
        &self.groups
    }
    /// Returns completed prefill activity.
    pub const fn prefill(&self) -> DensePassReport {
        self.prefill
    }
    /// Returns completed decode activity.
    pub const fn decode(&self) -> DensePassReport {
        self.decode
    }
    /// Returns completed multi-token forward passes.
    pub const fn prefill_forwards(&self) -> u64 {
        self.prefill.forwards
    }
    /// Returns completed single-token forward passes.
    pub const fn decode_forwards(&self) -> u64 {
        self.decode.forwards
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct DenseCounterSnapshot {
    host_cache: DenseCacheMetrics,
    device_cache: DenseCacheMetrics,
    disk_to_host_bytes: u64,
    disk_to_device_bytes: u64,
    host_to_device_bytes: u64,
}

impl DenseCounterSnapshot {
    fn from_report(report: &OffloadReport) -> Self {
        Self {
            host_cache: DenseCacheMetrics::from_report(report, MemoryTier::Host),
            device_cache: DenseCacheMetrics::from_report(report, MemoryTier::Device),
            disk_to_host_bytes: report.transfer(TransferDirection::DiskToHost).bytes(),
            disk_to_device_bytes: report.transfer(TransferDirection::DiskToDevice).bytes(),
            host_to_device_bytes: report.transfer(TransferDirection::HostToDevice).bytes(),
        }
    }

    fn delta(self, earlier: Self) -> DensePassReport {
        DensePassReport {
            forwards: 1,
            host_cache: self.host_cache.saturating_delta(earlier.host_cache),
            device_cache: self.device_cache.saturating_delta(earlier.device_cache),
            peak_host_layers: 0,
            peak_host_bytes: 0,
            peak_device_layers: 0,
            peak_device_bytes: 0,
            disk_to_host_bytes: self
                .disk_to_host_bytes
                .saturating_sub(earlier.disk_to_host_bytes),
            disk_to_device_bytes: self
                .disk_to_device_bytes
                .saturating_sub(earlier.disk_to_device_bytes),
            host_to_device_bytes: self
                .host_to_device_bytes
                .saturating_sub(earlier.host_to_device_bytes),
        }
    }
}

impl DensePassReport {
    fn accumulate(&mut self, other: Self) {
        self.forwards = self.forwards.saturating_add(other.forwards);
        self.host_cache.saturating_add(other.host_cache);
        self.device_cache.saturating_add(other.device_cache);
        self.peak_host_layers = self.peak_host_layers.max(other.peak_host_layers);
        self.peak_host_bytes = self.peak_host_bytes.max(other.peak_host_bytes);
        self.peak_device_layers = self.peak_device_layers.max(other.peak_device_layers);
        self.peak_device_bytes = self.peak_device_bytes.max(other.peak_device_bytes);
        self.disk_to_host_bytes = self
            .disk_to_host_bytes
            .saturating_add(other.disk_to_host_bytes);
        self.disk_to_device_bytes = self
            .disk_to_device_bytes
            .saturating_add(other.disk_to_device_bytes);
        self.host_to_device_bytes = self
            .host_to_device_bytes
            .saturating_add(other.host_to_device_bytes);
    }
}

#[derive(Debug)]
struct DensePassState {
    active: Option<DensePassActivity>,
    prefill: DensePassReport,
    decode: DensePassReport,
}

#[derive(Debug, Clone, Copy)]
struct DensePassActivity {
    prefill: bool,
    start: DenseCounterSnapshot,
    peaks: DensePassReport,
}

#[derive(Debug, Clone)]
struct DenseExecutionGroupPlan {
    id: String,
    units: Vec<OffloadUnitId>,
}

#[derive(Debug, Default, Clone, Copy)]
struct DenseExecutionGroupState {
    completed_executions: u64,
    peak_host_layers: usize,
    peak_host_bytes: u64,
    peak_device_layers: usize,
    peak_device_bytes: u64,
}

pub(crate) struct DenseStreamController {
    options: DenseDiskStreamLoadOptions,
    background: Option<BackgroundLayerPrefetch>,
    planned_layer_count: usize,
    planned_layer_bytes: u64,
    pinned_static_device_bytes: u64,
    transfer_stream_index: i32,
    groups: Vec<DenseExecutionGroupPlan>,
    group_activity: Mutex<BTreeMap<String, DenseExecutionGroupState>>,
    pass: Mutex<DensePassState>,
}

impl DenseStreamController {
    pub(crate) fn new(
        manager: &ResidencyManager,
        options: DenseDiskStreamLoadOptions,
        planned_layer_count: usize,
        planned_layer_bytes: u64,
        pinned_static_device_bytes: u64,
        groups: impl IntoIterator<Item = (String, Vec<OffloadUnitId>)>,
    ) -> Result<Self, Error> {
        let background = (options.host_budget_bytes > 0)
            .then(|| {
                BackgroundLayerPrefetch::new(manager.clone(), options.background_queue_capacity)
            })
            .transpose()?;
        let groups = groups
            .into_iter()
            .map(|(id, units)| DenseExecutionGroupPlan { id, units })
            .collect::<Vec<_>>();
        let group_activity = groups
            .iter()
            .map(|group| (group.id.clone(), DenseExecutionGroupState::default()))
            .collect();
        Ok(Self {
            options,
            background,
            planned_layer_count,
            planned_layer_bytes,
            pinned_static_device_bytes,
            transfer_stream_index: manager.device_stream_index()?,
            groups,
            group_activity: Mutex::new(group_activity),
            pass: Mutex::new(DensePassState {
                active: None,
                prefill: DensePassReport::default(),
                decode: DensePassReport::default(),
            }),
        })
    }

    pub(crate) fn transfer_window<'a>(
        &'a self,
        manager: &'a ResidencyManager,
        group: impl Into<String>,
        units: &'a [OffloadUnitId],
        indices: impl IntoIterator<Item = usize>,
        prefill: bool,
    ) -> Result<DenseTransferWindow<'a>, Error> {
        let indices = indices.into_iter().collect::<VecDeque<_>>();
        let mut prior = None;
        for &index in &indices {
            if index >= units.len() || prior.is_some_and(|prior| prior >= index) {
                return Err(LayerwiseModelError::InvalidDenseTransferWindow {
                    index,
                    unit_count: units.len(),
                }
                .into());
            }
            prior = Some(index);
        }
        let mut window = DenseTransferWindow {
            controller: self,
            manager,
            group: group.into(),
            units,
            pending: indices,
            ready: VecDeque::new(),
            prefill,
        };
        window.refill()?;
        Ok(window)
    }

    fn observe_group(
        &self,
        manager: &ResidencyManager,
        group: &str,
        prefill: bool,
    ) -> Result<(), Error> {
        let plan = self
            .groups
            .iter()
            .find(|candidate| candidate.id == group)
            .ok_or_else(|| LayerwiseModelError::UnknownExecutionGroup(group.to_string()))?;
        let ids = plan.units.iter().collect::<BTreeSet<_>>();
        let (_, _, units, _) = manager.telemetry_snapshot()?;
        let group_units = units
            .iter()
            .filter(|unit| ids.contains(unit.id()))
            .collect::<Vec<_>>();
        let host_layers = group_units
            .iter()
            .filter(|unit| unit.host_resident())
            .count();
        let host_bytes = group_units
            .iter()
            .filter(|unit| unit.host_resident())
            .map(|unit| unit.expected_bytes())
            .sum();
        let device_layers = group_units
            .iter()
            .filter(|unit| unit.device_resident())
            .count();
        let device_bytes = group_units
            .iter()
            .filter(|unit| unit.device_resident())
            .map(|unit| unit.expected_bytes())
            .sum();
        let mut activity = self.group_activity.lock().map_err(|_| {
            crate::runtime::residency::dense_stream::DenseStreamError::StatePoisoned
        })?;
        let state = activity
            .get_mut(group)
            .ok_or_else(|| LayerwiseModelError::UnknownExecutionGroup(group.to_string()))?;
        state.peak_host_layers = state.peak_host_layers.max(host_layers);
        state.peak_host_bytes = state.peak_host_bytes.max(host_bytes);
        state.peak_device_layers = state.peak_device_layers.max(device_layers);
        state.peak_device_bytes = state.peak_device_bytes.max(device_bytes);
        drop(activity);

        let streamed = self
            .groups
            .iter()
            .flat_map(|group| group.units.iter())
            .collect::<BTreeSet<_>>();
        let streamed_units = units
            .iter()
            .filter(|unit| streamed.contains(unit.id()))
            .collect::<Vec<_>>();
        let host_layers = streamed_units
            .iter()
            .filter(|unit| unit.host_resident())
            .count();
        let host_bytes = streamed_units
            .iter()
            .filter(|unit| unit.host_resident())
            .map(|unit| unit.expected_bytes())
            .sum();
        let device_layers = streamed_units
            .iter()
            .filter(|unit| unit.device_resident())
            .count();
        let device_bytes = streamed_units
            .iter()
            .filter(|unit| unit.device_resident())
            .map(|unit| unit.expected_bytes())
            .sum();
        let mut pass = self.pass.lock().map_err(|_| {
            crate::runtime::residency::dense_stream::DenseStreamError::StatePoisoned
        })?;
        let active = pass.active.as_mut().ok_or(
            crate::runtime::residency::dense_stream::DenseStreamError::InvalidForwardTelemetry(
                "residency was observed without an active forward",
            ),
        )?;
        if active.prefill != prefill {
            return Err(
                crate::runtime::residency::dense_stream::DenseStreamError::InvalidForwardTelemetry(
                    "residency observation changed pass category",
                )
                .into(),
            );
        }
        active.peaks.peak_host_layers = active.peaks.peak_host_layers.max(host_layers);
        active.peaks.peak_host_bytes = active.peaks.peak_host_bytes.max(host_bytes);
        active.peaks.peak_device_layers = active.peaks.peak_device_layers.max(device_layers);
        active.peaks.peak_device_bytes = active.peaks.peak_device_bytes.max(device_bytes);
        Ok(())
    }

    fn record_group_execution(&self, group: &str) -> Result<(), Error> {
        let mut activity = self.group_activity.lock().map_err(|_| {
            crate::runtime::residency::dense_stream::DenseStreamError::StatePoisoned
        })?;
        let state = activity
            .get_mut(group)
            .ok_or_else(|| LayerwiseModelError::UnknownExecutionGroup(group.to_string()))?;
        state.completed_executions = state.completed_executions.saturating_add(1);
        Ok(())
    }

    pub(crate) fn clear_group(&self, manager: &ResidencyManager, group: &str) -> Result<(), Error> {
        manager.protect_group_window(&format!("dense:{group}:host"), &[], MemoryTier::Host)?;
        manager.protect_group_window(&format!("dense:{group}:device"), &[], MemoryTier::Device)?;
        if let Some(background) = &self.background {
            background.cancel()?;
        }
        Ok(())
    }

    pub(crate) fn forward_guard<'a>(
        &'a self,
        prefill: bool,
        manager: &'a ResidencyManager,
    ) -> Result<DenseStreamForwardGuard<'a>, Error> {
        let (_, offload, _, _) = manager.telemetry_snapshot()?;
        let mut state = self.pass.lock().map_err(|_| {
            crate::runtime::residency::dense_stream::DenseStreamError::StatePoisoned
        })?;
        if state.active.is_some() {
            return Err(
                crate::runtime::residency::dense_stream::DenseStreamError::InvalidForwardTelemetry(
                    "a forward is already active",
                )
                .into(),
            );
        }
        state.active = Some(DensePassActivity {
            prefill,
            start: DenseCounterSnapshot::from_report(&offload),
            peaks: DensePassReport::default(),
        });
        Ok(DenseStreamForwardGuard {
            controller: self,
            manager,
            armed: true,
        })
    }

    fn commit_forward(&self, manager: &ResidencyManager) -> Result<(), Error> {
        if self.options.sample_mlx_memory || self.options.sample_process_memory {
            manager.sample_memory(
                self.options.sample_mlx_memory,
                self.options.sample_process_memory,
            )?;
        }
        let (_, offload, _, _) = manager.telemetry_snapshot()?;
        let current = DenseCounterSnapshot::from_report(&offload);
        let mut state = self.pass.lock().map_err(|_| {
            crate::runtime::residency::dense_stream::DenseStreamError::StatePoisoned
        })?;
        let active = state.active.take().ok_or(
            crate::runtime::residency::dense_stream::DenseStreamError::InvalidForwardTelemetry(
                "a forward was committed without being started",
            ),
        )?;
        let mut delta = current.delta(active.start);
        delta.peak_host_layers = active.peaks.peak_host_layers;
        delta.peak_host_bytes = active.peaks.peak_host_bytes;
        delta.peak_device_layers = active.peaks.peak_device_layers;
        delta.peak_device_bytes = active.peaks.peak_device_bytes;
        if active.prefill {
            state.prefill.accumulate(delta);
        } else {
            state.decode.accumulate(delta);
        }
        Ok(())
    }

    fn abort_forward(&self) {
        if let Ok(mut state) = self.pass.lock() {
            state.active = None;
        }
    }

    pub(crate) fn group_guard<'a>(
        &'a self,
        manager: &'a ResidencyManager,
        group: &str,
    ) -> DenseStreamGroupGuard<'a> {
        DenseStreamGroupGuard {
            controller: self,
            manager,
            group: group.to_string(),
            armed: true,
        }
    }

    pub(crate) fn report(
        &self,
        manager: &ResidencyManager,
    ) -> Result<DenseDiskStreamReport, Error> {
        let residency = manager.report()?;
        let streamed = self
            .groups
            .iter()
            .flat_map(|group| group.units.iter())
            .collect::<BTreeSet<_>>();
        let units = residency
            .units()
            .iter()
            .map(|unit| (unit.id(), unit))
            .collect::<BTreeMap<_, _>>();
        let pinned_device_bytes = residency
            .units()
            .iter()
            .filter(|unit| unit.policy() == ResidencyPolicy::Pinned && unit.device_resident())
            .map(|unit| unit.expected_bytes())
            .sum::<u64>();
        let pinned_device_count = residency
            .units()
            .iter()
            .filter(|unit| unit.policy() == ResidencyPolicy::Pinned && unit.device_resident())
            .count();
        let tier_report = |tier: MemoryTier| {
            let current = residency
                .units()
                .iter()
                .filter(|unit| streamed.contains(unit.id()))
                .filter(|unit| match tier {
                    MemoryTier::Host => unit.host_resident(),
                    MemoryTier::Device => unit.device_resident(),
                    MemoryTier::Disk => false,
                })
                .collect::<Vec<_>>();
            let (pinned_bytes, pinned_count) = if tier == MemoryTier::Device {
                (pinned_device_bytes, pinned_device_count)
            } else {
                (0, 0)
            };
            DenseTierResidencyReport {
                current_layer_count: current.len(),
                peak_layer_count: residency
                    .offload()
                    .peak_resident_units()
                    .get(tier)
                    .saturating_sub(pinned_count),
                current_layer_bytes: current.iter().map(|unit| unit.expected_bytes()).sum(),
                peak_layer_bytes: residency
                    .offload()
                    .peak_resident_bytes()
                    .get(tier)
                    .saturating_sub(pinned_bytes),
                cache: DenseCacheMetrics::from_report(residency.offload(), tier),
            }
        };
        let activity = self.group_activity.lock().map_err(|_| {
            crate::runtime::residency::dense_stream::DenseStreamError::StatePoisoned
        })?;
        let groups = self
            .groups
            .iter()
            .map(|group| {
                let group_units = group
                    .units
                    .iter()
                    .filter_map(|id| units.get(id).copied())
                    .collect::<Vec<_>>();
                let observed = activity.get(&group.id).copied().unwrap_or_default();
                DenseExecutionGroupReport {
                    id: group.id.clone(),
                    planned_layers: group_units.len(),
                    planned_bytes: group_units.iter().map(|unit| unit.expected_bytes()).sum(),
                    completed_executions: observed.completed_executions,
                    host_layers: group_units
                        .iter()
                        .filter(|unit| unit.host_resident())
                        .count(),
                    host_bytes: group_units
                        .iter()
                        .filter(|unit| unit.host_resident())
                        .map(|unit| unit.expected_bytes())
                        .sum(),
                    peak_host_layers: observed.peak_host_layers,
                    peak_host_bytes: observed.peak_host_bytes,
                    device_layers: group_units
                        .iter()
                        .filter(|unit| unit.device_resident())
                        .count(),
                    device_bytes: group_units
                        .iter()
                        .filter(|unit| unit.device_resident())
                        .map(|unit| unit.expected_bytes())
                        .sum(),
                    peak_device_layers: observed.peak_device_layers,
                    peak_device_bytes: observed.peak_device_bytes,
                }
            })
            .collect();
        let pass = self.pass.lock().map_err(|_| {
            crate::runtime::residency::dense_stream::DenseStreamError::StatePoisoned
        })?;
        Ok(DenseDiskStreamReport {
            planned_layer_count: self.planned_layer_count,
            planned_layer_bytes: self.planned_layer_bytes,
            pinned_static_device_bytes: self.pinned_static_device_bytes,
            transfer_stream_index: self.transfer_stream_index,
            host_layers: tier_report(MemoryTier::Host),
            device_layers: tier_report(MemoryTier::Device),
            groups,
            prefill: pass.prefill,
            decode: pass.decode,
            residency,
            background: self
                .background
                .as_ref()
                .map(BackgroundLayerPrefetch::report)
                .transpose()?
                .unwrap_or_default(),
        })
    }
}

/// Caller-thread transfer window retaining the current and next dense unit.
///
/// Entries own [`ResidentTransfer`] guards and therefore remain on the host
/// thread supported by MLX events. A window submits at most two device copies.
/// Callers consume one entry, evaluate and synchronize its compute work, drop
/// that entry, and then call [`Self::refill`] to submit the following layer.
pub(crate) struct DenseTransferWindow<'a> {
    controller: &'a DenseStreamController,
    manager: &'a ResidencyManager,
    group: String,
    units: &'a [OffloadUnitId],
    pending: VecDeque<usize>,
    ready: VecDeque<DensePreparedTransfer>,
    prefill: bool,
}

impl DenseTransferWindow<'_> {
    /// Takes the next transfer after ordering `consumer` behind its event.
    pub(crate) fn next(&mut self, consumer: &Stream) -> Result<DensePreparedTransfer, Error> {
        let transfer = self.ready.pop_front().ok_or({
            LayerwiseModelError::InvalidDenseTransferWindow {
                index: self.units.len(),
                unit_count: self.units.len(),
            }
        })?;
        transfer.transfer.wait_on(consumer)?;
        Ok(transfer)
    }

    /// Reprotects the current/next units and submits one replacement transfer.
    ///
    /// The completed [`DensePreparedTransfer`] must be dropped before this is
    /// called so the fixed two-layer device budget can admit the replacement.
    pub(crate) fn refill(&mut self) -> Result<(), Error> {
        let device_indices = self
            .ready
            .iter()
            .map(DensePreparedTransfer::index)
            .chain(self.pending.iter().copied())
            .take(DENSE_TRANSFER_WINDOW)
            .collect::<Vec<_>>();
        let host_indices = if self.controller.background.is_some() {
            self.ready
                .iter()
                .map(DensePreparedTransfer::index)
                .chain(self.pending.iter().copied())
                .take(self.controller.options.host_lookahead)
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let device_units = device_indices
            .iter()
            .map(|&index| self.units[index].clone())
            .collect::<Vec<_>>();
        let host_units = host_indices
            .iter()
            .map(|&index| self.units[index].clone())
            .collect::<Vec<_>>();
        self.manager.protect_group_window(
            &format!("dense:{}:host", self.group),
            &host_units,
            MemoryTier::Host,
        )?;
        self.manager.protect_group_window(
            &format!("dense:{}:device", self.group),
            &device_units,
            MemoryTier::Device,
        )?;
        if let Some(background) = &self.controller.background {
            for id in &host_units {
                background.submit(id)?;
            }
        }
        while self.ready.len() < DENSE_TRANSFER_WINDOW {
            let Some(index) = self.pending.pop_front() else {
                break;
            };
            let id = &self.units[index];
            let _host = if host_indices.contains(&index) {
                self.controller
                    .background
                    .as_ref()
                    .map(|background| background.acquire(id))
                    .transpose()?
            } else {
                None
            };
            let transfer = self
                .manager
                .acquire_many_with_transfer(&[(id.clone(), 1)], MemoryTier::Device)?;
            self.ready
                .push_back(DensePreparedTransfer { index, transfer });
        }
        self.controller
            .observe_group(self.manager, &self.group, self.prefill)?;
        Ok(())
    }
}

/// One populated-unit dependency taken from a [`DenseTransferWindow`].
pub(crate) struct DensePreparedTransfer {
    index: usize,
    transfer: ResidentTransfer,
}

impl DensePreparedTransfer {
    /// Returns the index in the group's authoritative unit list.
    pub(crate) const fn index(&self) -> usize {
        self.index
    }

    /// Returns the single resident lease protected by this transfer.
    pub(crate) fn lease(&self) -> &ResidentUnitLease {
        self.transfer
            .leases()
            .first()
            .expect("dense transfer always acquires one unit")
    }
}

pub(crate) struct DenseStreamForwardGuard<'a> {
    controller: &'a DenseStreamController,
    manager: &'a ResidencyManager,
    armed: bool,
}

impl DenseStreamForwardGuard<'_> {
    pub(crate) fn complete(mut self) -> Result<(), Error> {
        let result = self.controller.commit_forward(self.manager);
        if result.is_ok() {
            self.armed = false;
        }
        result
    }
}

impl Drop for DenseStreamForwardGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.controller.abort_forward();
        }
    }
}

pub(crate) struct DenseStreamGroupGuard<'a> {
    controller: &'a DenseStreamController,
    manager: &'a ResidencyManager,
    group: String,
    armed: bool,
}

impl DenseStreamGroupGuard<'_> {
    pub(crate) fn complete(mut self) -> Result<(), Error> {
        let result = self
            .controller
            .clear_group(self.manager, &self.group)
            .and_then(|()| self.controller.record_group_execution(&self.group));
        self.armed = false;
        result
    }
}

impl Drop for DenseStreamGroupGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            let _ = self.controller.clear_group(self.manager, &self.group);
        }
    }
}

/// Inspectable parameter-residency metadata for a layerwise model.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct LayerwiseModelMetadata {
    model_type: String,
    quantization: Option<crate::runtime::checkpoint::quantization::WeightQuantization>,
    layer_count: usize,
    static_device_bytes: u64,
    residency: ExecutionResidency,
    layer_parameter_bytes: u64,
    maximum_device_layer_bytes: u64,
    device_layer_capacity: usize,
    materialization: Option<BoundedQuantizationReport>,
}

/// Architecture-neutral information for a rank-local parallel model.
#[derive(Debug, Clone)]
pub struct ParallelModelInfo {
    topology: crate::runtime::distributed::topology::ParallelTopology,
    model_type: String,
    owned_tensors: Vec<String>,
    local_parameter_bytes: u64,
    global_parameter_bytes: u64,
    pinned_device_parameter_bytes: u64,
    maximum_device_parameter_bytes: u64,
}

impl ParallelModelInfo {
    /// Returns the complete distributed topology and this process's coordinates.
    pub const fn topology(&self) -> crate::runtime::distributed::topology::ParallelTopology {
        self.topology
    }

    /// Returns the adapter's normalized model type.
    pub fn model_type(&self) -> &str {
        &self.model_type
    }

    /// Returns exact checkpoint targets owned or replicated by this rank.
    pub fn owned_tensors(&self) -> &[String] {
        &self.owned_tensors
    }

    /// Returns planned rank-local parameter bytes across static and execution units.
    pub const fn local_parameter_bytes(&self) -> u64 {
        self.local_parameter_bytes
    }

    /// Returns the unsharded model parameter bytes represented by this checkpoint.
    pub const fn global_parameter_bytes(&self) -> u64 {
        self.global_parameter_bytes
    }

    /// Returns rank-local parameter bytes permanently pinned on the execution device.
    pub const fn pinned_device_parameter_bytes(&self) -> u64 {
        self.pinned_device_parameter_bytes
    }

    /// Returns the maximum planned rank-local parameter footprint on device.
    pub const fn maximum_device_parameter_bytes(&self) -> u64 {
        self.maximum_device_parameter_bytes
    }
}

impl LayerwiseModelMetadata {
    /// Returns the checkpoint model type supplied by the adapter.
    pub fn model_type(&self) -> &str {
        &self.model_type
    }
    /// Returns checkpoint-native packed quantization metadata, if present.
    pub const fn quantization(
        &self,
    ) -> Option<crate::runtime::checkpoint::quantization::WeightQuantization> {
        self.quantization
    }
    /// Returns the decoder layer count.
    pub const fn layer_count(&self) -> usize {
        self.layer_count
    }
    /// Returns pinned static parameter bytes on the execution device.
    pub const fn static_device_bytes(&self) -> u64 {
        self.static_device_bytes
    }
    /// Returns the selected generalized parameter-residency policy.
    pub const fn residency(&self) -> ExecutionResidency {
        self.residency
    }
    /// Returns the complete rank-local decoder-layer parameter byte total.
    pub const fn layer_parameter_bytes(&self) -> u64 {
        self.layer_parameter_bytes
    }
    /// Returns the largest possible device-resident decoder-layer byte total.
    pub const fn maximum_device_layer_bytes(&self) -> u64 {
        self.maximum_device_layer_bytes
    }
    /// Returns the maximum number of decoder layers retained on device.
    pub const fn device_layer_capacity(&self) -> usize {
        self.device_layer_capacity
    }

    /// Returns bounded load-time materialization telemetry when dense semantic
    /// weights were converted into a packed disk overlay.
    pub const fn materialization(&self) -> Option<&BoundedQuantizationReport> {
        self.materialization.as_ref()
    }
}

/// One pinned static module and its checkpoint bindings.
pub struct StaticUnitBindings {
    id: OffloadUnitId,
    bindings: Vec<crate::runtime::residency::manager::WeightBinding>,
}

impl StaticUnitBindings {
    /// Creates a pinned static unit definition.
    pub(crate) fn new(
        id: impl Into<String>,
        bindings: Vec<crate::runtime::residency::manager::WeightBinding>,
    ) -> Result<Self, Error> {
        Ok(Self {
            id: OffloadUnitId::new(id.into())?,
            bindings,
        })
    }

    /// Returns the stable residency identifier for this static module.
    pub(crate) const fn id(&self) -> &OffloadUnitId {
        &self.id
    }

    /// Returns the authoritative checkpoint bindings for this static module.
    pub(crate) fn bindings(&self) -> &[crate::runtime::residency::manager::WeightBinding] {
        &self.bindings
    }
}

/// Forward state returned by a generalized architecture adapter.
pub struct LayerwiseForwardState<C> {
    /// Initial activation made available to root execution groups.
    pub hidden: Array,
    /// Architecture-owned masks, positions, and auxiliary per-forward state.
    pub context: C,
}

/// One named execution group and the groups whose completed outputs it consumes.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ExecutionGroupSpec {
    id: String,
    dependencies: Vec<String>,
}

impl ExecutionGroupSpec {
    /// Declares a root execution group.
    pub fn root(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            dependencies: Vec::new(),
        }
    }

    /// Declares a group with named input dependencies.
    pub fn with_dependencies(
        id: impl Into<String>,
        dependencies: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            id: id.into(),
            dependencies: dependencies.into_iter().map(Into::into).collect(),
        }
    }

    /// Returns the stable group identifier.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns dependency identifiers in declaration order.
    pub fn dependencies(&self) -> &[String] {
        &self.dependencies
    }
}

/// Validated execution-group dependency graph with one authoritative output.
///
/// Group slots retain declaration order for architecture callbacks. Execution
/// follows the stable topological order derived here, never numeric adjacency.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ExecutionGroupDag {
    groups: Vec<ExecutionGroupSpec>,
    dependencies: Vec<Vec<usize>>,
    execution_order: Vec<usize>,
    output: usize,
}

impl ExecutionGroupDag {
    /// Validates unique names, dependency references, acyclicity, and that
    /// every declared group contributes transitively to `output`.
    pub fn new(groups: Vec<ExecutionGroupSpec>, output: impl AsRef<str>) -> Result<Self, Error> {
        if groups.is_empty() {
            return Err(LayerwiseModelError::EmptyExecutionGraph.into());
        }
        let mut by_id = BTreeMap::new();
        for (index, group) in groups.iter().enumerate() {
            if group.id.is_empty() {
                return Err(LayerwiseModelError::EmptyExecutionGroupId.into());
            }
            if by_id.insert(group.id.clone(), index).is_some() {
                return Err(LayerwiseModelError::DuplicateExecutionGroup(group.id.clone()).into());
            }
        }
        let output_name = output.as_ref();
        let output = by_id.get(output_name).copied().ok_or_else(|| {
            LayerwiseModelError::UnknownExecutionGraphOutput(output_name.to_string())
        })?;
        let mut dependencies = Vec::with_capacity(groups.len());
        let mut dependents = vec![Vec::new(); groups.len()];
        let mut indegree = vec![0usize; groups.len()];
        for (index, group) in groups.iter().enumerate() {
            let mut seen = BTreeSet::new();
            let mut resolved = Vec::with_capacity(group.dependencies.len());
            for dependency in &group.dependencies {
                let dependency_index = by_id.get(dependency).copied().ok_or_else(|| {
                    LayerwiseModelError::UnknownExecutionGroupDependency {
                        group: group.id.clone(),
                        dependency: dependency.clone(),
                    }
                })?;
                if dependency_index == index {
                    return Err(
                        LayerwiseModelError::SelfDependentExecutionGroup(group.id.clone()).into(),
                    );
                }
                if !seen.insert(dependency_index) {
                    return Err(LayerwiseModelError::DuplicateExecutionGroupDependency {
                        group: group.id.clone(),
                        dependency: dependency.clone(),
                    }
                    .into());
                }
                resolved.push(dependency_index);
                dependents[dependency_index].push(index);
            }
            indegree[index] = resolved.len();
            dependencies.push(resolved);
        }
        let mut ready = indegree
            .iter()
            .enumerate()
            .filter_map(|(index, &degree)| (degree == 0).then_some(index))
            .collect::<BTreeSet<_>>();
        let mut execution_order = Vec::with_capacity(groups.len());
        while let Some(index) = ready.pop_first() {
            execution_order.push(index);
            for &dependent in &dependents[index] {
                indegree[dependent] -= 1;
                if indegree[dependent] == 0 {
                    ready.insert(dependent);
                }
            }
        }
        if execution_order.len() != groups.len() {
            return Err(LayerwiseModelError::CyclicExecutionGraph.into());
        }
        let mut contributes = BTreeSet::new();
        let mut pending = vec![output];
        while let Some(index) = pending.pop() {
            if contributes.insert(index) {
                pending.extend(dependencies[index].iter().copied());
            }
        }
        if contributes.len() != groups.len() {
            let disconnected = groups
                .iter()
                .enumerate()
                .filter_map(|(index, group)| {
                    (!contributes.contains(&index)).then_some(group.id.clone())
                })
                .collect();
            return Err(LayerwiseModelError::DisconnectedExecutionGroups { disconnected }.into());
        }
        Ok(Self {
            groups,
            dependencies,
            execution_order,
            output,
        })
    }

    /// Creates a dependency chain in iterator order whose final group is the output.
    pub fn chain(ids: impl IntoIterator<Item = impl Into<String>>) -> Result<Self, Error> {
        let ids = ids.into_iter().map(Into::into).collect::<Vec<String>>();
        let output = ids
            .last()
            .cloned()
            .ok_or(LayerwiseModelError::EmptyExecutionGraph)?;
        let groups = ids
            .iter()
            .enumerate()
            .map(|(index, id)| match index.checked_sub(1) {
                Some(previous) => {
                    ExecutionGroupSpec::with_dependencies(id.clone(), [ids[previous].clone()])
                }
                None => ExecutionGroupSpec::root(id.clone()),
            })
            .collect();
        Self::new(groups, output)
    }

    /// Returns group specifications in stable architecture slot order.
    pub fn groups(&self) -> &[ExecutionGroupSpec] {
        &self.groups
    }

    /// Returns stable topological execution slots.
    pub fn execution_order(&self) -> &[usize] {
        &self.execution_order
    }

    /// Returns dependency slots for an architecture group slot.
    pub fn dependencies(&self, group: usize) -> Option<&[usize]> {
        self.dependencies.get(group).map(Vec::as_slice)
    }

    /// Returns the authoritative output group slot.
    pub const fn output(&self) -> usize {
        self.output
    }

    fn consumer_counts(&self) -> Vec<usize> {
        let mut counts = vec![0; self.groups.len()];
        for dependencies in &self.dependencies {
            for &dependency in dependencies {
                counts[dependency] += 1;
            }
        }
        counts
    }
}

/// Canonical architecture contract for resident, bounded-residency, and future
/// distributed execution.
///
/// Heterogeneous caches, architecture-specific inputs, multiple execution
/// groups, and retained recurrent state are represented directly rather than
/// being forced into a decoder-only KV-cache interface.
pub trait ArchitectureAdapter: Sized {
    /// Borrowed family-specific forward input.
    type Input<'a>;
    /// Complete architecture-owned cache and recurrent state.
    type Cache;
    /// Runtime execution unit. Families with heterogeneous blocks may use an enum.
    type Layer: ModuleParameters;
    /// Masks, positions, prepared media, or other per-forward state.
    type ForwardContext;

    /// Stable architecture identity used by residency metadata.
    fn model_type(&self) -> &str {
        std::any::type_name::<Self>()
    }

    /// Model-wide checkpoint quantization, when one uniform encoding exists.
    fn quantization(&self) -> Option<crate::runtime::checkpoint::quantization::WeightQuantization> {
        None
    }

    /// Returns whether a floating static matrix is represented by a packed
    /// parameter group in this adapter. Multimodal adapters override this for
    /// checkpoint components whose target modules intentionally stay dense.
    fn quantizes_static_binding(
        &self,
        _binding: &crate::runtime::residency::manager::WeightBinding,
    ) -> bool {
        true
    }

    /// Returns the exact cache compatibility identity for replicated or
    /// rank-local parallel execution.
    fn prompt_cache_model_identity(
        &self,
        _topology: Option<crate::runtime::distributed::topology::ParallelTopology>,
    ) -> Result<PromptCacheModelIdentity, Error> {
        Err(Error::Parallel(format!(
            "architecture adapter {} has not declared a prompt-cache identity",
            std::any::type_name::<Self>()
        )))
    }

    /// Persists a validated architecture-owned cache.
    #[allow(clippy::too_many_arguments)]
    fn save_prompt_cache(
        &self,
        _cache: &mut Self::Cache,
        _destination: &Path,
        _descriptor: PromptCacheDescriptor,
        _prefix_token_ids: &[u32],
        _options: &PromptCacheOptions,
        _stream: &Stream,
    ) -> Result<PromptCacheManifest, Error> {
        Err(Error::Parallel(format!(
            "architecture adapter {} has not implemented prompt-cache persistence",
            std::any::type_name::<Self>()
        )))
    }

    /// Restores a validated architecture-owned cache.
    #[allow(clippy::too_many_arguments)]
    fn load_prompt_cache(
        &self,
        _directory: &Path,
        _expected: &PromptCacheDescriptor,
        _identity: &PromptCacheModelIdentity,
        _prefix_token_ids: &[u32],
        _options: PagedCacheOptions,
        _stream: &Stream,
    ) -> Result<(Self::Cache, PromptCacheManifest), Error> {
        Err(Error::Parallel(format!(
            "architecture adapter {} has not implemented prompt-cache restoration",
            std::any::type_name::<Self>()
        )))
    }

    /// Builds bindings for modules that remain pinned on the execution device.
    fn static_units(&self, store: &dyn WeightStore) -> Result<Vec<StaticUnitBindings>, Error>;

    /// Builds only static units selected by their stable architecture id.
    ///
    /// Adapters should override this when binding construction consults a
    /// sharded store, so distributed stages do not open unowned static shards.
    fn selected_static_units(
        &self,
        store: &dyn WeightStore,
        select: &dyn Fn(&str) -> bool,
    ) -> Result<Vec<StaticUnitBindings>, Error> {
        Ok(self
            .static_units(store)?
            .into_iter()
            .filter(|unit| select(unit.id().as_str()))
            .collect())
    }

    /// Assigns pinned leases to the adapter's static modules.
    fn populate_static(&mut self, leases: &[ResidentUnitLease]) -> Result<(), Error>;

    /// Validates or initializes the complete cache before any weight lease is acquired.
    fn validate_cache(&self, cache: &mut Self::Cache) -> Result<(), Error>;

    /// Embeds or prepares the input and creates family-owned forward context.
    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<LayerwiseForwardState<Self::ForwardContext>, Error>;

    /// Embeds or prepares input under an explicit execution context.
    fn begin_forward_with_execution<'a>(
        &mut self,
        input: Self::Input<'a>,
        cache: &mut Self::Cache,
        execution: &crate::runtime::distributed::parallel::ParallelExecutionContext<'_>,
    ) -> Result<LayerwiseForwardState<Self::ForwardContext>, Error> {
        self.begin_forward(input, cache, execution.stream())
    }

    /// Declares the complete named execution-group dependency graph.
    fn execution_graph(&self) -> Result<ExecutionGroupDag, Error>;

    /// Returns whether a group is needed for this particular forward pass.
    ///
    /// This lets multimodal adapters skip vision groups during text-only decode.
    fn should_execute_group(&self, _group: usize, _context: &Self::ForwardContext) -> bool {
        true
    }

    /// Returns the number of ordered units in one group.
    fn layer_count(&self, group: usize) -> Result<usize, Error>;

    /// Creates a metadata-only runtime unit for one group position.
    fn new_layer(&self, group: usize, index: usize, stream: &Stream) -> Result<Self::Layer, Error>;

    /// Describes this architecture's physical checkpoint tensors using typed
    /// logical roles for tensor-parallel planning.
    ///
    /// Adapters without an exact tensor-parallel parameter plan fail closed.
    fn parallel_parameter_groups(
        &self,
        _context: crate::runtime::distributed::parallel::ParallelBuildContext,
    ) -> Result<Vec<crate::runtime::distributed::parallel::ParameterGroupSpec>, Error> {
        Err(Error::Parallel(format!(
            "architecture adapter {} has not declared tensor-parallel parameter roles",
            std::any::type_name::<Self>()
        )))
    }

    /// Registers placement for streamed and pinned parameters. Composite
    /// adapters can override this to reuse the planners of their nested model
    /// families instead of duplicating parameter-name logic.
    fn register_parallel_parameters(
        &self,
        context: crate::runtime::distributed::parallel::ParallelBuildContext,
        planner: &mut crate::runtime::distributed::parallel::ParallelPlanBuilder,
        _stream: &Stream,
    ) -> Result<(), Error> {
        for group in self.parallel_parameter_groups(context)? {
            planner.register(group)?;
        }
        Ok(())
    }

    /// Rebuilds pinned modules whose parameter geometry is rank-local.
    ///
    /// The loader captures the global static bindings before invoking this
    /// hook, then applies the typed layout to those bindings before residency
    /// initialization.
    fn configure_parallel_static(
        &mut self,
        _context: crate::runtime::distributed::parallel::ParallelBuildContext,
        _layout: &crate::runtime::distributed::parallel::LocalModelLayout,
        _stream: &Stream,
    ) -> Result<(), Error> {
        Ok(())
    }

    /// Creates a rank-local runtime unit from planned model geometry.
    fn new_parallel_layer(
        &self,
        _group: usize,
        _index: usize,
        _layout: &crate::runtime::distributed::parallel::LocalModelLayout,
        _stream: &Stream,
    ) -> Result<Self::Layer, Error> {
        Err(Error::Parallel(format!(
            "architecture adapter {} has not implemented rank-local layer construction",
            std::any::type_name::<Self>()
        )))
    }

    /// Creates a rank-local runtime unit whose routed experts follow an
    /// authoritative expert assignment.
    ///
    /// Architectures supporting PP+EP implement this hook instead of exposing
    /// expert-bank representation details to the pipeline runtime.
    fn new_expert_parallel_layer(
        &self,
        _group: usize,
        _index: usize,
        _assignment: &crate::runtime::distributed::expert::ExpertAssignment,
        _stream: &Stream,
    ) -> Result<Self::Layer, Error> {
        Err(Error::Parallel(format!(
            "architecture adapter {} has not implemented expert-local layer construction",
            std::any::type_name::<Self>()
        )))
    }

    /// Creates a layer whose ordinary projections follow the TP layout while
    /// routed experts are restricted to the authoritative EP assignment.
    ///
    /// Architectures opt into triple-axis execution by implementing this one
    /// semantic composition point; pipeline placement remains external.
    fn new_tensor_expert_parallel_layer(
        &self,
        _group: usize,
        _index: usize,
        _layout: &crate::runtime::distributed::parallel::LocalModelLayout,
        _assignment: &crate::runtime::distributed::expert::ExpertAssignment,
        _stream: &Stream,
    ) -> Result<Self::Layer, Error> {
        Err(Error::Parallel(format!(
            "architecture adapter {} has not implemented combined tensor/expert-local layer construction",
            std::any::type_name::<Self>()
        )))
    }

    /// Derives this rank's expert ownership from architecture metadata and the
    /// authoritative Cartesian topology.
    ///
    /// The default accepts an inactive EP axis and fails closed otherwise.
    fn expert_parallel_assignment(
        &self,
        topology: crate::runtime::distributed::topology::ParallelTopology,
    ) -> Result<Option<crate::runtime::distributed::expert::ExpertAssignment>, Error> {
        if topology.expert_parallel_size > 1 {
            Err(Error::Parallel(format!(
                "architecture adapter {} has not declared expert ownership for EP size {}",
                std::any::type_name::<Self>(),
                topology.expert_parallel_size
            )))
        } else {
            Ok(None)
        }
    }

    /// Creates one runtime unit for replicated, TP-local, or EP-local
    /// execution from shared semantic inputs.
    ///
    /// Architecture adapters own the semantic composition of simultaneous TP
    /// and EP; PP remains the outer selection of execution units.
    fn new_cartesian_layer(
        &self,
        group: usize,
        index: usize,
        layout: Option<&crate::runtime::distributed::parallel::LocalModelLayout>,
        assignment: Option<&crate::runtime::distributed::expert::ExpertAssignment>,
        stream: &Stream,
    ) -> Result<Self::Layer, Error> {
        match (layout, assignment) {
            (None, None) => self.new_layer(group, index, stream),
            (Some(layout), None) => self.new_parallel_layer(group, index, layout, stream),
            (None, Some(assignment)) => {
                self.new_expert_parallel_layer(group, index, assignment, stream)
            }
            (Some(layout), Some(assignment)) => {
                self.new_tensor_expert_parallel_layer(group, index, layout, assignment, stream)
            }
        }
    }

    /// Returns the checkpoint prefix for one runtime unit.
    fn layer_checkpoint_prefix(&self, group: usize, index: usize) -> String;

    /// Returns the stable residency unit name for one runtime unit.
    fn layer_unit_name(&self, group: usize, index: usize) -> String;
    /// Populates one temporary execution unit from its protected lease.
    fn populate_layer(
        &self,
        _group: usize,
        _index: usize,
        layer: &mut Self::Layer,
        lease: &ResidentUnitLease,
    ) -> Result<(), Error> {
        Ok(populate_module_from_lease(layer, lease)?)
    }

    /// Builds direct or derived bindings for one runtime unit.
    fn layer_bindings(
        &self,
        group: usize,
        index: usize,
        layer: &Self::Layer,
        store: &dyn WeightStore,
    ) -> Result<Vec<crate::runtime::residency::manager::WeightBinding>, Error> {
        Ok(build_module_bindings(
            layer,
            &self.layer_checkpoint_prefix(group, index),
            store,
        )?)
    }

    /// Builds rank-local bindings for a tensor-parallel execution unit.
    fn parallel_layer_bindings(
        &self,
        group: usize,
        index: usize,
        layer: &Self::Layer,
        store: &dyn WeightStore,
        layout: &crate::runtime::distributed::parallel::LocalModelLayout,
        _stream: &Stream,
    ) -> Result<Vec<crate::runtime::residency::manager::WeightBinding>, Error> {
        build_parallel_module_bindings(
            layer,
            &self.layer_checkpoint_prefix(group, index),
            store,
            layout,
        )
    }

    /// Builds rank-local bindings for an expert-parallel execution unit.
    ///
    /// The adapter owns checkpoint expert layout, packed companions, and the
    /// mapping from global expert ids to its layer representation.
    fn expert_parallel_layer_bindings(
        &self,
        _group: usize,
        _index: usize,
        _layer: &Self::Layer,
        _store: &dyn WeightStore,
        _assignment: &crate::runtime::distributed::expert::ExpertAssignment,
        _stream: &Stream,
    ) -> Result<Vec<crate::runtime::residency::manager::WeightBinding>, Error> {
        Err(Error::Parallel(format!(
            "architecture adapter {} has not implemented expert-local checkpoint bindings",
            std::any::type_name::<Self>()
        )))
    }

    /// Builds bindings for a layer that is simultaneously TP-sharded and
    /// restricted to EP-owned routed experts.
    ///
    /// The default composes the architecture's EP selection recipe with the
    /// shared semantic TP shard plan. Architectures only need to override this
    /// when their checkpoint representation requires a different ordering.
    #[allow(clippy::too_many_arguments)]
    fn tensor_expert_parallel_layer_bindings(
        &self,
        group: usize,
        index: usize,
        layer: &Self::Layer,
        store: &dyn WeightStore,
        layout: &crate::runtime::distributed::parallel::LocalModelLayout,
        assignment: &crate::runtime::distributed::expert::ExpertAssignment,
        stream: &Stream,
    ) -> Result<Vec<crate::runtime::residency::manager::WeightBinding>, Error> {
        let bindings =
            self.expert_parallel_layer_bindings(group, index, layer, store, assignment, stream)?;
        shard_layer_bindings(
            bindings,
            &self.layer_checkpoint_prefix(group, index),
            store,
            layout,
        )
    }

    /// Builds bindings for the same replicated, TP-local, or EP-local unit
    /// geometry selected by [`Self::new_cartesian_layer`].
    #[allow(clippy::too_many_arguments)]
    fn cartesian_layer_bindings(
        &self,
        group: usize,
        index: usize,
        layer: &Self::Layer,
        store: &dyn WeightStore,
        layout: Option<&crate::runtime::distributed::parallel::LocalModelLayout>,
        assignment: Option<&crate::runtime::distributed::expert::ExpertAssignment>,
        stream: &Stream,
    ) -> Result<Vec<crate::runtime::residency::manager::WeightBinding>, Error> {
        match (layout, assignment) {
            (None, None) => {
                // The execution layer can have transformed target geometry
                // (for example load-time affine quantization). Bindings must
                // continue to describe the adapter's source checkpoint
                // geometry and are transformed only during population.
                let source = self.new_layer(group, index, stream)?;
                self.layer_bindings(group, index, &source, store)
            }
            (Some(layout), None) => {
                self.parallel_layer_bindings(group, index, layer, store, layout, stream)
            }
            (None, Some(assignment)) => {
                self.expert_parallel_layer_bindings(group, index, layer, store, assignment, stream)
            }
            (Some(layout), Some(assignment)) => self.tensor_expert_parallel_layer_bindings(
                group, index, layer, store, layout, assignment, stream,
            ),
        }
    }

    /// Returns checkpoint keys consumed by dependent units outside execution groups.
    fn additional_consumed_checkpoint_keys(&self, _store: &dyn WeightStore) -> Vec<String> {
        Vec::new()
    }

    /// Executes one populated unit while inspecting and mutating the complete cache.
    #[allow(clippy::too_many_arguments)]
    fn forward_layer(
        &mut self,
        group: usize,
        index: usize,
        layer: &mut Self::Layer,
        hidden: &Array,
        cache: &mut Self::Cache,
        context: &mut Self::ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error>;

    /// Executes one populated unit with architecture-specific observation.
    ///
    /// The default provides stable unit boundary names. Adapters whose block
    /// math exposes richer observations override this hook without replacing
    /// residency, cache, or graph execution.
    #[allow(clippy::too_many_arguments)]
    fn forward_layer_with_observer<O: ActivationObserver>(
        &mut self,
        group: usize,
        index: usize,
        layer: &mut Self::Layer,
        hidden: &Array,
        cache: &mut Self::Cache,
        context: &mut Self::ForwardContext,
        stream: &Stream,
        observer: &mut O,
    ) -> Result<Array, Error> {
        let prefix = self.layer_checkpoint_prefix(group, index);
        observer.observe(&format!("{prefix}.input"), hidden)?;
        let output = self.forward_layer(group, index, layer, hidden, cache, context, stream)?;
        observer.observe(&format!("{prefix}.output"), &output)?;
        Ok(observer
            .intervene(&format!("{prefix}.output"), &output)?
            .unwrap_or(output))
    }

    /// Executes one unit under an explicit replicated or TP context.
    ///
    /// The default preserves ordinary execution and rejects TP, ensuring a
    /// family cannot become distributed merely because it implements the
    /// resident adapter contract.
    #[allow(clippy::too_many_arguments)]
    fn forward_layer_with_execution(
        &mut self,
        group: usize,
        index: usize,
        layer: &mut Self::Layer,
        hidden: &Array,
        cache: &mut Self::Cache,
        context: &mut Self::ForwardContext,
        execution: &crate::runtime::distributed::parallel::ParallelExecutionContext<'_>,
    ) -> Result<Array, Error> {
        if execution.is_tensor_parallel() {
            return Err(Error::Parallel(format!(
                "architecture adapter {} has not implemented tensor-parallel execution",
                std::any::type_name::<Self>()
            )));
        }
        self.forward_layer(
            group,
            index,
            layer,
            hidden,
            cache,
            context,
            execution.stream(),
        )
    }

    /// Returns every cache/state array that must be evaluated before lease release.
    fn retained_arrays<'a>(
        &self,
        cache: &'a Self::Cache,
        group: usize,
        index: usize,
    ) -> Vec<&'a Array>;

    /// Returns transient forward-context arrays that must be evaluated before lease release.
    fn retained_context_arrays<'a>(
        &self,
        _context: &'a Self::ForwardContext,
        _group: usize,
        _index: usize,
    ) -> Vec<&'a Array> {
        Vec::new()
    }

    /// Selects or assembles the activation consumed by one ready group.
    ///
    /// Root groups receive `initial_hidden`. A group with one dependency uses
    /// that output by default. Multi-input groups must define an exact merge.
    #[allow(clippy::too_many_arguments)]
    fn begin_execution_group(
        &mut self,
        group: usize,
        initial_hidden: &Array,
        dependency_outputs: &[Array],
        _cache: &mut Self::Cache,
        _context: &mut Self::ForwardContext,
        _stream: &Stream,
    ) -> Result<Array, Error> {
        match dependency_outputs {
            [] => Ok(initial_hidden.clone()),
            [dependency] => Ok(dependency.clone()),
            _ => Err(LayerwiseModelError::UnmergedExecutionGroupInputs {
                group,
                inputs: dependency_outputs.len(),
            }
            .into()),
        }
    }

    /// Selects or assembles a ready group under an explicit execution context.
    #[allow(clippy::too_many_arguments)]
    fn begin_execution_group_with_execution(
        &mut self,
        group: usize,
        initial_hidden: &Array,
        dependency_outputs: &[Array],
        cache: &mut Self::Cache,
        context: &mut Self::ForwardContext,
        execution: &crate::runtime::distributed::parallel::ParallelExecutionContext<'_>,
    ) -> Result<Array, Error> {
        self.begin_execution_group(
            group,
            initial_hidden,
            dependency_outputs,
            cache,
            context,
            execution.stream(),
        )
    }

    /// Converts one group's output into the activation consumed by the next group.
    ///
    /// Multimodal adapters use this hook to merge encoded media before entering
    /// a text decoder. Homogeneous adapters keep the activation unchanged.
    fn complete_execution_group(
        &mut self,
        _group: usize,
        hidden: &Array,
        _cache: &mut Self::Cache,
        _context: &mut Self::ForwardContext,
        _stream: &Stream,
    ) -> Result<Array, Error> {
        Ok(hidden.clone())
    }

    /// Converts group output under an explicit execution context.
    fn complete_execution_group_with_execution(
        &mut self,
        group: usize,
        hidden: &Array,
        cache: &mut Self::Cache,
        context: &mut Self::ForwardContext,
        execution: &crate::runtime::distributed::parallel::ParallelExecutionContext<'_>,
    ) -> Result<Array, Error> {
        self.complete_execution_group(group, hidden, cache, context, execution.stream())
    }

    /// Applies final normalization, projections, or family-specific output assembly.
    fn finish(
        &mut self,
        hidden: &Array,
        cache: &mut Self::Cache,
        context: &Self::ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error>;

    /// Produces final output under an explicit execution context.
    fn finish_with_execution(
        &mut self,
        hidden: &Array,
        cache: &mut Self::Cache,
        context: &Self::ForwardContext,
        execution: &crate::runtime::distributed::parallel::ParallelExecutionContext<'_>,
    ) -> Result<Array, Error> {
        self.finish(hidden, cache, context, execution.stream())
    }

    /// Returns whether a checkpoint key is intentionally ignored by strict loading.
    fn ignores_checkpoint_key(&self, _key: &str) -> bool {
        false
    }
}

/// Semantic adapter capability required by bounded load-time quantization.
///
/// Implementations rebuild the same architecture with packed matrix modules
/// while preserving architecture-owned execution choices such as multimodal
/// towers and externally resident experts. Checkpoint format is deliberately
/// absent from this contract: SafeTensors and dense GGUF use the same packed
/// overlay once their stores expose the adapter's semantic recipes.
pub(crate) trait LoadTimeQuantizableAdapter: ArchitectureAdapter {
    /// Rebuilds this adapter with `quantization` as its uniform packed matrix
    /// representation.
    fn load_time_quantized(
        &self,
        quantization: WeightQuantization,
        stream: &Stream,
    ) -> Result<Self, Error>;
}

/// Residency-owned execution engine for generalized adapters.
///
/// Group windows, lease lifetime, retained-state evaluation, stream
/// synchronization, and telemetry stay centralized here. Adapter code owns only
/// architecture math, cache validation, and runtime-unit construction.
pub struct LayerwiseModel<A: ArchitectureAdapter> {
    adapter: A,
    graph: ExecutionGroupDag,
    store: SharedWeightStore,
    residency: ResidencyManager,
    groups: Vec<ResidentLayerGroup>,
    static_leases: Vec<ResidentUnitLease>,
    resident_layers: Option<Vec<Vec<A::Layer>>>,
    // Keep every populated layer's source arrays protected for the model lifetime.
    _resident_layer_leases: Vec<Vec<ResidentUnitLease>>,
    dense_stream: Option<DenseStreamController>,
    sample_mlx_memory: bool,
    sample_process_memory: bool,
    metadata: LayerwiseModelMetadata,
    parallel_layout: Option<crate::runtime::distributed::parallel::LocalModelLayout>,
    parallel_topology: Option<crate::runtime::distributed::topology::ParallelTopology>,
    parallel_info: Option<ParallelModelInfo>,
}

enum PreparedExecutionLayer<'a, L> {
    Resident(&'a mut L),
    Leased {
        layer: L,
        _device: ResidentUnitLease,
    },
    Transferred {
        layer: L,
        _transfer: DensePreparedTransfer,
    },
}

impl<L> PreparedExecutionLayer<'_, L> {
    fn layer_mut(&mut self) -> &mut L {
        match self {
            Self::Resident(layer) => layer,
            Self::Leased { layer, .. } | Self::Transferred { layer, .. } => layer,
        }
    }
}

impl<A: ArchitectureAdapter> LayerwiseModel<A> {
    /// Creates an engine from a validated residency manager and execution groups.
    pub fn new(
        adapter: A,
        graph: ExecutionGroupDag,
        store: SharedWeightStore,
        residency: ResidencyManager,
        groups: Vec<ResidentLayerGroup>,
        static_leases: Vec<ResidentUnitLease>,
    ) -> Result<Self, Error> {
        if groups.len() != graph.groups().len() {
            return Err(LayerwiseModelError::ExecutionGroupCount {
                adapter: graph.groups().len(),
                configured: groups.len(),
            }
            .into());
        }
        for (group_index, group) in groups.iter().enumerate() {
            let expected_id = graph.groups()[group_index].id();
            if group.id() != expected_id {
                return Err(LayerwiseModelError::ExecutionGroupIdentity {
                    slot: group_index,
                    adapter: expected_id.to_string(),
                    configured: group.id().to_string(),
                }
                .into());
            }
            let expected = adapter.layer_count(group_index)?;
            if expected != group.units().len() {
                return Err(LayerwiseModelError::ExecutionGroupLength {
                    group: group.id().to_string(),
                    adapter: expected,
                    configured: group.units().len(),
                }
                .into());
            }
        }
        let layer_count = groups.iter().map(|group| group.units().len()).sum();
        let metadata = LayerwiseModelMetadata {
            model_type: adapter.model_type().into(),
            quantization: adapter.quantization(),
            layer_count,
            static_device_bytes: 0,
            residency: ExecutionResidency::LayerwiseHost,
            layer_parameter_bytes: 0,
            maximum_device_layer_bytes: 0,
            device_layer_capacity: 0,
            materialization: None,
        };
        Ok(Self {
            adapter,
            graph,
            store,
            residency,
            groups,
            static_leases,
            resident_layers: None,
            _resident_layer_leases: Vec::new(),
            dense_stream: None,
            sample_mlx_memory: false,
            sample_process_memory: false,
            metadata,
            parallel_layout: None,
            parallel_topology: None,
            parallel_info: None,
        })
    }

    fn materialize_resident_layers(&mut self, stream: &Stream) -> Result<(), Error> {
        let mut resident_layers = Vec::with_capacity(self.groups.len());
        let mut resident_layer_leases = Vec::with_capacity(self.groups.len());
        for (group_index, group) in self.groups.iter().enumerate() {
            let mut layers = Vec::with_capacity(group.units().len());
            let mut leases = Vec::with_capacity(group.units().len());
            for (index, id) in group.units().iter().enumerate() {
                let mut layer = if let Some(layout) = &self.parallel_layout {
                    self.adapter
                        .new_parallel_layer(group_index, index, layout, stream)?
                } else {
                    self.adapter.new_layer(group_index, index, stream)?
                };
                let lease = self.residency.acquire(id, MemoryTier::Device)?;
                self.adapter
                    .populate_layer(group_index, index, &mut layer, &lease)?;
                layers.push(layer);
                leases.push(lease);
            }
            resident_layers.push(layers);
            resident_layer_leases.push(leases);
        }
        self.resident_layers = Some(resident_layers);
        self._resident_layer_leases = resident_layer_leases;
        Ok(())
    }

    /// Enables optional allocator and process-memory samples after forward.
    pub fn with_memory_sampling(mut self, mlx: bool, process: bool) -> Self {
        self.sample_mlx_memory = mlx;
        self.sample_process_memory = process;
        self
    }

    /// Returns the architecture adapter.
    pub const fn adapter(&self) -> &A {
        &self.adapter
    }

    /// Returns aggregate residency metadata for all execution groups.
    pub const fn metadata(&self) -> &LayerwiseModelMetadata {
        &self.metadata
    }

    /// Returns rank-local parallel placement information when loaded with a
    /// generalized distributed execution group.
    pub const fn parallel_info(&self) -> Option<&ParallelModelInfo> {
        self.parallel_info.as_ref()
    }

    /// Returns the exact typed rank-local parameter layout, when parallel.
    pub const fn parallel_layout(
        &self,
    ) -> Option<&crate::runtime::distributed::parallel::LocalModelLayout> {
        self.parallel_layout.as_ref()
    }

    /// Returns the cache-relevant architecture fingerprint for this execution rank.
    pub fn prompt_cache_architecture_fingerprint(&self) -> Result<String, Error> {
        Ok(self
            .adapter
            .prompt_cache_model_identity(self.parallel_topology)?
            .architecture_fingerprint
            .clone())
    }

    /// Returns the complete cache identity, including every active parallel
    /// coordinate and the rank-local cache layout.
    pub fn prompt_cache_model_identity(&self) -> Result<PromptCacheModelIdentity, Error> {
        self.adapter
            .prompt_cache_model_identity(self.parallel_topology)
    }

    /// Returns this execution rank's exact ordered cache-state layout.
    pub fn prompt_cache_layer_layout(
        &self,
    ) -> Result<crate::runtime::attention::LayerSchedule<crate::LayerCachePolicy>, Error> {
        Ok(self
            .adapter
            .prompt_cache_model_identity(self.parallel_topology)?
            .layer_layout
            .clone())
    }

    /// Persists a compatible architecture-owned prefix cache. Parallel ranks
    /// publish into deterministic subdirectories below the supplied root.
    pub fn save_prompt_cache(
        &self,
        cache: &mut A::Cache,
        root: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        stream: &Stream,
    ) -> Result<PromptCacheManifest, Error> {
        let identity = self
            .adapter
            .prompt_cache_model_identity(self.parallel_topology)?;
        validate_prompt_cache_model_identity(&descriptor, &identity)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let directory = self.prompt_cache_directory(root.as_ref());
        self.adapter.save_prompt_cache(
            cache,
            &directory,
            descriptor,
            prefix_token_ids,
            options,
            stream,
        )
    }

    pub(crate) fn save_prompt_cache_with_validated_identity(
        &self,
        cache: &mut A::Cache,
        directory: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        stream: &Stream,
    ) -> Result<PromptCacheManifest, Error> {
        self.adapter.save_prompt_cache(
            cache,
            directory,
            descriptor,
            prefix_token_ids,
            options,
            stream,
        )
    }

    /// Restores a compatible architecture-owned prefix cache from this rank's
    /// deterministic cache directory.
    pub fn load_prompt_cache(
        &self,
        root: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
        stream: &Stream,
    ) -> Result<(A::Cache, PromptCacheManifest), Error> {
        let identity = self
            .adapter
            .prompt_cache_model_identity(self.parallel_topology)?;
        validate_prompt_cache_model_identity(expected, &identity)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        self.adapter.load_prompt_cache(
            &self.prompt_cache_directory(root.as_ref()),
            expected,
            &identity,
            prefix_token_ids,
            options,
            stream,
        )
    }

    pub(crate) fn load_prompt_cache_with_validated_identity(
        &self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        identity: &PromptCacheModelIdentity,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
        stream: &Stream,
    ) -> Result<(A::Cache, PromptCacheManifest), Error> {
        self.adapter.load_prompt_cache(
            directory,
            expected,
            identity,
            prefix_token_ids,
            options,
            stream,
        )
    }

    fn prompt_cache_directory(&self, root: &Path) -> std::path::PathBuf {
        match self.parallel_topology {
            Some(topology) => root.join(format!("rank-{:05}", topology.global_rank)),
            None => root.to_path_buf(),
        }
    }

    pub(crate) fn prompt_cache_rank_identity(
        &self,
    ) -> Option<crate::runtime::cache::residency::CacheRankIdentity> {
        self.parallel_topology
            .map(PromptCacheTopology::for_parallel_topology)
            .and_then(|topology| topology.cache_rank_identity())
    }

    /// Binds state and persistence identity to an enclosing Cartesian runtime.
    pub(crate) fn bind_parallel_topology(
        &mut self,
        topology: crate::runtime::distributed::topology::ParallelTopology,
    ) {
        self.parallel_topology = Some(topology);
    }

    /// Returns the mutable adapter for loader-time dependent-unit setup.
    pub(crate) fn adapter_mut(&mut self) -> &mut A {
        &mut self.adapter
    }

    /// Returns a shared handle to the persistent checkpoint store.
    pub(crate) fn checkpoint_store_arc(&self) -> SharedWeightStore {
        Arc::clone(&self.store)
    }

    /// Returns the persistent backend-neutral checkpoint store.
    pub fn checkpoint_store(&self) -> &(dyn WeightStore + Send + Sync) {
        self.store.as_ref()
    }

    /// Returns named execution groups in deterministic order.
    pub fn execution_groups(&self) -> &[ResidentLayerGroup] {
        &self.groups
    }

    /// Returns the validated dependency graph governing group execution.
    pub const fn execution_graph(&self) -> &ExecutionGroupDag {
        &self.graph
    }

    /// Returns the reusable residency manager.
    pub const fn residency_manager(&self) -> &ResidencyManager {
        &self.residency
    }

    /// Returns a current residency and transfer report.
    pub fn residency_report(&self) -> Result<ResidencyReport, Error> {
        Ok(self
            .residency
            .report()?
            .with_materialization(self.metadata.materialization.clone()))
    }

    /// Returns dense-stream observations when that experimental policy is active.
    pub fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error> {
        self.dense_stream
            .as_ref()
            .map(|streamer| streamer.report(&self.residency))
            .transpose()
    }

    /// Runs every graph-ready group while centrally enforcing lease safety.
    pub fn forward<'a>(
        &mut self,
        input: A::Input<'a>,
        cache: &mut A::Cache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.forward_with_hooks(
            input,
            cache,
            stream,
            |adapter, group, index, layer, hidden, cache, context, stream| {
                let execution =
                    crate::runtime::distributed::parallel::ParallelExecutionContext::replicated(
                        stream,
                    );
                adapter.forward_layer_with_execution(
                    group, index, layer, hidden, cache, context, &execution,
                )
            },
            |_, _, _| Ok(()),
        )
        .map(|(output, _)| output)
    }

    /// Runs a rank-local layerwise model using the tensor-parallel subgroup.
    pub fn forward_tensor_parallel<'a>(
        &mut self,
        input: A::Input<'a>,
        cache: &mut A::Cache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.forward_tensor_parallel_with_context_hook(
            input,
            cache,
            group,
            stream,
            |_, _, _| Ok(()),
        )
        .map(|(output, _)| output)
    }

    /// Runs TP execution groups and invokes a context hook after each unit.
    pub(crate) fn forward_tensor_parallel_with_context_hook<'a, H>(
        &mut self,
        input: A::Input<'a>,
        cache: &mut A::Cache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
        hook: H,
    ) -> Result<(Array, A::ForwardContext), Error>
    where
        H: FnMut(usize, usize, &mut A::ForwardContext) -> Result<(), Error>,
    {
        self.forward_tensor_parallel_with_hooks(
            input,
            cache,
            group,
            stream,
            |adapter, group, index, layer, hidden, cache, context, execution| {
                adapter.forward_layer_with_execution(
                    group, index, layer, hidden, cache, context, execution,
                )
            },
            hook,
        )
    }

    /// Runs TP execution while allowing routed-expert evaluation to be replaced.
    ///
    /// The embedding, attention, dense/shared projections, cache geometry, and
    /// output head retain their tensor-parallel execution context. The caller
    /// replaces only the selected populated-layer operation and receives the
    /// same TP context, enabling an EP exchange inside a TP-sharded layer.
    #[allow(clippy::type_complexity)]
    pub(crate) fn forward_tensor_parallel_with_layer_executor<'a, F>(
        &mut self,
        input: A::Input<'a>,
        cache: &mut A::Cache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
        executor: F,
    ) -> Result<Array, Error>
    where
        F: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Layer,
            &Array,
            &mut A::Cache,
            &mut A::ForwardContext,
            &crate::runtime::distributed::parallel::ParallelExecutionContext<'_>,
        ) -> Result<Array, Error>,
    {
        self.forward_tensor_parallel_with_hooks(input, cache, group, stream, executor, |_, _, _| {
            Ok(())
        })
        .map(|(output, _)| output)
    }

    /// Runs TP execution with the architecture's ordinary layer semantics and
    /// returns the forward context retained by the pass.
    ///
    /// Embedded prediction heads consume the final decoder hidden state, so
    /// distributed callers need the same context that replicated execution
    /// exposes without replacing any layer operation.
    pub(crate) fn forward_tensor_parallel_with_context<'a>(
        &mut self,
        input: A::Input<'a>,
        cache: &mut A::Cache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<(Array, A::ForwardContext), Error> {
        self.forward_tensor_parallel_with_hooks(
            input,
            cache,
            group,
            stream,
            |adapter, group, index, layer, hidden, cache, context, execution| {
                adapter.forward_layer_with_execution(
                    group, index, layer, hidden, cache, context, execution,
                )
            },
            |_, _, _| Ok(()),
        )
    }

    /// Runs TP execution with a caller-provided layer operation and returns
    /// the architecture context retained by the pass. This is the TP+EP
    /// counterpart of [`Self::forward_with_layer_executor_and_context`].
    #[allow(clippy::type_complexity)]
    pub(crate) fn forward_tensor_parallel_with_layer_executor_and_context<'a, F>(
        &mut self,
        input: A::Input<'a>,
        cache: &mut A::Cache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
        executor: F,
    ) -> Result<(Array, A::ForwardContext), Error>
    where
        F: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Layer,
            &Array,
            &mut A::Cache,
            &mut A::ForwardContext,
            &crate::runtime::distributed::parallel::ParallelExecutionContext<'_>,
        ) -> Result<Array, Error>,
    {
        self.forward_tensor_parallel_with_hooks(input, cache, group, stream, executor, |_, _, _| {
            Ok(())
        })
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn forward_tensor_parallel_with_hooks<'a, F, H>(
        &mut self,
        input: A::Input<'a>,
        cache: &mut A::Cache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
        mut executor: F,
        mut hook: H,
    ) -> Result<(Array, A::ForwardContext), Error>
    where
        F: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Layer,
            &Array,
            &mut A::Cache,
            &mut A::ForwardContext,
            &crate::runtime::distributed::parallel::ParallelExecutionContext<'_>,
        ) -> Result<Array, Error>,
        H: FnMut(usize, usize, &mut A::ForwardContext) -> Result<(), Error>,
    {
        let topology = self.parallel_topology.ok_or_else(|| {
            Error::Parallel("layerwise model was not loaded for tensor-parallel execution".into())
        })?;
        let execution =
            crate::runtime::distributed::parallel::ParallelExecutionContext::tensor_parallel(
                topology, group, stream,
            )?;
        self.adapter.validate_cache(cache)?;
        let LayerwiseForwardState {
            hidden: initial_hidden,
            mut context,
        } = self
            .adapter
            .begin_forward_with_execution(input, cache, &execution)?;
        let prefill = initial_hidden.dim(1) > 1;
        let dense_forward = self
            .dense_stream
            .as_ref()
            .map(|streamer| streamer.forward_guard(prefill, &self.residency))
            .transpose()?;
        let layout = self
            .parallel_layout
            .as_ref()
            .expect("parallel topology has a layout");
        let mut group_outputs: Vec<Option<Array>> = vec![None; self.graph.groups().len()];
        let mut remaining_consumers = self.graph.consumer_counts();
        for group_index in self.graph.execution_order().iter().copied() {
            let resident_group = &self.groups[group_index];
            let dependency_slots = self
                .graph
                .dependencies(group_index)
                .unwrap_or_default()
                .to_vec();
            let dependency_outputs = dependency_slots
                .iter()
                .map(|&dependency| {
                    group_outputs[dependency]
                        .as_ref()
                        .expect("validated topological dependency")
                        .clone()
                })
                .collect::<Vec<_>>();
            let mut hidden = self.adapter.begin_execution_group_with_execution(
                group_index,
                &initial_hidden,
                &dependency_outputs,
                cache,
                &mut context,
                &execution,
            )?;
            for dependency in dependency_slots {
                remaining_consumers[dependency] -= 1;
                if remaining_consumers[dependency] == 0 {
                    group_outputs[dependency] = None;
                }
            }
            let execute_group = self.adapter.should_execute_group(group_index, &context);
            let dense_guard = execute_group.then(|| {
                self.dense_stream
                    .as_ref()
                    .map(|streamer| streamer.group_guard(&self.residency, resident_group.id()))
            });
            if execute_group {
                let mut dense_window = self
                    .dense_stream
                    .as_ref()
                    .map(|streamer| {
                        streamer.transfer_window(
                            &self.residency,
                            resident_group.id(),
                            resident_group.units(),
                            0..resident_group.units().len(),
                            prefill,
                        )
                    })
                    .transpose()?;
                for index in 0..resident_group.units().len() {
                    let id = &resident_group.units()[index];
                    {
                        let mut prepared = if let Some(layers) = &mut self.resident_layers {
                            PreparedExecutionLayer::Resident(&mut layers[group_index][index])
                        } else if let Some(window) = &mut dense_window {
                            let transfer = window.next(stream)?;
                            debug_assert_eq!(transfer.index(), index);
                            let mut layer = self.adapter.new_parallel_layer(
                                group_index,
                                index,
                                layout,
                                stream,
                            )?;
                            self.adapter.populate_layer(
                                group_index,
                                index,
                                &mut layer,
                                transfer.lease(),
                            )?;
                            PreparedExecutionLayer::Transferred {
                                layer,
                                _transfer: transfer,
                            }
                        } else {
                            resident_group.prepare(&self.residency, index)?;
                            let lease = self.residency.acquire(id, MemoryTier::Device)?;
                            let mut layer = self.adapter.new_parallel_layer(
                                group_index,
                                index,
                                layout,
                                stream,
                            )?;
                            self.adapter
                                .populate_layer(group_index, index, &mut layer, &lease)?;
                            PreparedExecutionLayer::Leased {
                                layer,
                                _device: lease,
                            }
                        };
                        hidden = executor(
                            &mut self.adapter,
                            group_index,
                            index,
                            prepared.layer_mut(),
                            &hidden,
                            cache,
                            &mut context,
                            &execution,
                        )?;
                        let hook_result = hook(group_index, index, &mut context);
                        let retained = self.adapter.retained_arrays(cache, group_index, index);
                        let retained_context =
                            self.adapter
                                .retained_context_arrays(&context, group_index, index);
                        async_eval_with_event(
                            std::iter::once(&hidden)
                                .chain(retained)
                                .chain(retained_context),
                        )?
                        .synchronize()?;
                        hook_result?;
                    }
                    if let Some(window) = &mut dense_window {
                        window.refill()?;
                    } else if self.resident_layers.is_none() {
                        let end = index
                            .saturating_add(resident_group.depth())
                            .min(resident_group.units().len());
                        resident_group
                            .trim_to(&self.residency, &resident_group.units()[index..end])?;
                    }
                }
            }
            hidden = self.adapter.complete_execution_group_with_execution(
                group_index,
                &hidden,
                cache,
                &mut context,
                &execution,
            )?;
            async_eval_with_event([&hidden])?.synchronize()?;
            if let Some(Some(guard)) = dense_guard {
                guard.complete()?;
            }
            group_outputs[group_index] = Some(hidden);
        }
        let hidden = group_outputs[self.graph.output()]
            .take()
            .expect("validated execution graph output was executed");
        let output = self
            .adapter
            .finish_with_execution(&hidden, cache, &context, &execution)?;
        async_eval_with_event([&output])?.synchronize()?;
        if let Some(guard) = dense_forward {
            guard.complete()?;
        }
        Ok((output, context))
    }

    /// Runs a generalized forward pass while allowing the caller to replace
    /// execution of each populated layer.
    ///
    /// Residency, prefetch, lease lifetime, retained-array evaluation, and
    /// telemetry remain owned by this engine. Distributed execution uses this
    /// hook to replace only routed-expert evaluation while reusing the same
    /// architecture adapter and checkpoint bindings.
    #[allow(clippy::type_complexity)]
    pub(crate) fn forward_with_layer_executor<'a, F>(
        &mut self,
        input: A::Input<'a>,
        cache: &mut A::Cache,
        stream: &Stream,
        executor: F,
    ) -> Result<Array, Error>
    where
        F: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Layer,
            &Array,
            &mut A::Cache,
            &mut A::ForwardContext,
            &Stream,
        ) -> Result<Array, Error>,
    {
        self.forward_with_hooks(input, cache, stream, executor, |_, _, _| Ok(()))
            .map(|(output, _)| output)
    }

    /// Runs the canonical execution path while exposing stable per-unit inputs
    /// and outputs to an activation observer.
    ///
    /// Observation is deliberately owned by the shared engine so fully
    /// resident, host-layerwise, and disk-streamed policies report identical
    /// names and intervention points.
    pub fn forward_with_observer<'a>(
        &mut self,
        input: A::Input<'a>,
        cache: &mut A::Cache,
        stream: &Stream,
        observer: &mut dyn ActivationObserver,
    ) -> Result<Array, Error> {
        let mut observer = ActivationObserverProxy(observer);
        let output = self.forward_with_layer_executor(
            input,
            cache,
            stream,
            |adapter, group, index, layer, hidden, cache, context, stream| {
                adapter.forward_layer_with_observer(
                    group,
                    index,
                    layer,
                    hidden,
                    cache,
                    context,
                    stream,
                    &mut observer,
                )
            },
        )?;
        observer.observe("model.logits", &output)?;
        Ok(output)
    }

    /// Runs a generalized pass with caller-provided populated-layer execution
    /// and returns the architecture context retained by that pass.
    #[allow(clippy::type_complexity)]
    pub(crate) fn forward_with_layer_executor_and_context<'a, F>(
        &mut self,
        input: A::Input<'a>,
        cache: &mut A::Cache,
        stream: &Stream,
        executor: F,
    ) -> Result<(Array, A::ForwardContext), Error>
    where
        F: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Layer,
            &Array,
            &mut A::Cache,
            &mut A::ForwardContext,
            &Stream,
        ) -> Result<Array, Error>,
    {
        self.forward_with_hooks(input, cache, stream, executor, |_, _, _| Ok(()))
    }

    /// Runs a generalized forward pass and invokes `hook` after each execution unit.
    ///
    /// Realtime autoregressive subgroups use this to turn one unit's logits into
    /// the token consumed by the next unit without moving lease ownership out of
    /// the shared residency engine.
    pub(crate) fn forward_with_context_hook<'a, F>(
        &mut self,
        input: A::Input<'a>,
        cache: &mut A::Cache,
        stream: &Stream,
        hook: F,
    ) -> Result<(Array, A::ForwardContext), Error>
    where
        F: FnMut(usize, usize, &mut A::ForwardContext) -> Result<(), Error>,
    {
        self.forward_with_hooks(
            input,
            cache,
            stream,
            |adapter, group, index, layer, hidden, cache, context, stream| {
                let execution =
                    crate::runtime::distributed::parallel::ParallelExecutionContext::replicated(
                        stream,
                    );
                adapter.forward_layer_with_execution(
                    group, index, layer, hidden, cache, context, &execution,
                )
            },
            hook,
        )
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn forward_with_hooks<'a, F, H>(
        &mut self,
        input: A::Input<'a>,
        cache: &mut A::Cache,
        stream: &Stream,
        mut executor: F,
        mut hook: H,
    ) -> Result<(Array, A::ForwardContext), Error>
    where
        F: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Layer,
            &Array,
            &mut A::Cache,
            &mut A::ForwardContext,
            &Stream,
        ) -> Result<Array, Error>,
        H: FnMut(usize, usize, &mut A::ForwardContext) -> Result<(), Error>,
    {
        self.adapter.validate_cache(cache)?;
        let execution =
            crate::runtime::distributed::parallel::ParallelExecutionContext::replicated(stream);
        let LayerwiseForwardState {
            hidden: initial_hidden,
            mut context,
        } = self
            .adapter
            .begin_forward_with_execution(input, cache, &execution)?;
        let prefill = initial_hidden.dim(1) > 1;
        let dense_forward = self
            .dense_stream
            .as_ref()
            .map(|streamer| streamer.forward_guard(prefill, &self.residency))
            .transpose()?;

        let mut group_outputs: Vec<Option<Array>> = vec![None; self.graph.groups().len()];
        let mut remaining_consumers = self.graph.consumer_counts();
        for group_index in self.graph.execution_order().iter().copied() {
            let group = &self.groups[group_index];
            let dependency_slots = self
                .graph
                .dependencies(group_index)
                .unwrap_or_default()
                .to_vec();
            let dependency_outputs = dependency_slots
                .iter()
                .map(|&dependency| {
                    group_outputs[dependency]
                        .as_ref()
                        .expect("validated topological dependency")
                        .clone()
                })
                .collect::<Vec<_>>();
            let mut hidden = self.adapter.begin_execution_group_with_execution(
                group_index,
                &initial_hidden,
                &dependency_outputs,
                cache,
                &mut context,
                &execution,
            )?;
            for dependency in dependency_slots {
                remaining_consumers[dependency] -= 1;
                if remaining_consumers[dependency] == 0 {
                    group_outputs[dependency] = None;
                }
            }
            let execute_group = self.adapter.should_execute_group(group_index, &context);
            let dense_guard = execute_group.then(|| {
                self.dense_stream
                    .as_ref()
                    .map(|streamer| streamer.group_guard(&self.residency, group.id()))
            });
            if execute_group {
                let mut dense_window = self
                    .dense_stream
                    .as_ref()
                    .map(|streamer| {
                        streamer.transfer_window(
                            &self.residency,
                            group.id(),
                            group.units(),
                            0..group.units().len(),
                            prefill,
                        )
                    })
                    .transpose()?;
                for index in 0..group.units().len() {
                    let id = &group.units()[index];
                    {
                        let mut prepared = if let Some(layers) = &mut self.resident_layers {
                            PreparedExecutionLayer::Resident(&mut layers[group_index][index])
                        } else if let Some(window) = &mut dense_window {
                            let transfer = window.next(stream)?;
                            debug_assert_eq!(transfer.index(), index);
                            let mut layer = self.adapter.new_layer(group_index, index, stream)?;
                            self.adapter.populate_layer(
                                group_index,
                                index,
                                &mut layer,
                                transfer.lease(),
                            )?;
                            PreparedExecutionLayer::Transferred {
                                layer,
                                _transfer: transfer,
                            }
                        } else {
                            group.prepare(&self.residency, index)?;
                            let lease = self.residency.acquire(id, MemoryTier::Device)?;
                            let mut layer = self.adapter.new_layer(group_index, index, stream)?;
                            self.adapter
                                .populate_layer(group_index, index, &mut layer, &lease)?;
                            PreparedExecutionLayer::Leased {
                                layer,
                                _device: lease,
                            }
                        };
                        hidden = executor(
                            &mut self.adapter,
                            group_index,
                            index,
                            prepared.layer_mut(),
                            &hidden,
                            cache,
                            &mut context,
                            stream,
                        )?;
                        let hook_result = hook(group_index, index, &mut context);
                        let retained = self.adapter.retained_arrays(cache, group_index, index);
                        let retained_context =
                            self.adapter
                                .retained_context_arrays(&context, group_index, index);
                        async_eval_with_event(
                            std::iter::once(&hidden)
                                .chain(retained)
                                .chain(retained_context),
                        )?
                        .synchronize()?;
                        hook_result?;
                    }
                    if let Some(window) = &mut dense_window {
                        window.refill()?;
                    } else if self.resident_layers.is_none() {
                        let end = index.saturating_add(group.depth()).min(group.units().len());
                        group.trim_to(&self.residency, &group.units()[index..end])?;
                    }
                }
            }
            hidden = self.adapter.complete_execution_group_with_execution(
                group_index,
                &hidden,
                cache,
                &mut context,
                &execution,
            )?;
            let retained_context =
                self.adapter
                    .retained_context_arrays(&context, group_index, group.units().len());
            async_eval_with_event(std::iter::once(&hidden).chain(retained_context))?
                .synchronize()?;
            if let Some(Some(guard)) = dense_guard {
                guard.complete()?;
            }
            group_outputs[group_index] = Some(hidden);
        }

        let hidden = group_outputs[self.graph.output()]
            .take()
            .expect("validated execution graph output was executed");

        let output = self
            .adapter
            .finish_with_execution(&hidden, cache, &context, &execution)?;
        async_eval_with_event([&output])?.synchronize()?;
        if self.dense_stream.is_none() && (self.sample_mlx_memory || self.sample_process_memory) {
            self.residency
                .sample_memory(self.sample_mlx_memory, self.sample_process_memory)?;
        }
        if let Some(guard) = dense_forward {
            guard.complete()?;
        }
        Ok((output, context))
    }

    /// Clears one named execution group without affecting other groups.
    pub fn clear_device_group(&self, id: &str) -> Result<(), Error> {
        let group = self
            .groups
            .iter()
            .find(|group| group.id() == id)
            .ok_or_else(|| LayerwiseModelError::UnknownExecutionGroup(id.to_string()))?;
        if self.resident_layers.is_some() {
            return Ok(());
        }
        Ok(group.clear(&self.residency)?)
    }

    /// Clears every temporary device execution group.
    pub fn clear_all_device_groups(&self) -> Result<(), Error> {
        if self.resident_layers.is_some() {
            return Ok(());
        }
        for group in &self.groups {
            group.clear(&self.residency)?;
        }
        Ok(())
    }

    /// Returns the number of pinned static leases held by the engine.
    pub fn static_lease_count(&self) -> usize {
        self.static_leases.len()
    }
}

/// Builds a generalized layerwise model with independently bounded groups.
pub(crate) fn load_safetensors_layerwise_model<A, O>(
    model_dir: impl AsRef<Path>,
    adapter: A,
    options: O,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LayerwiseModel<A>, Error>
where
    A: ArchitectureAdapter,
    O: Into<LayerWeightResidency>,
{
    let options = options.into();
    let store = open_safetensors_weight_store(model_dir.as_ref(), options.max_mapped_shards())?;
    load_layerwise_model(store, adapter, options, stream, weights_stream)
}

/// Builds a packed, disk-backed overlay for every quantizable static and
/// execution-unit binding declared by `source_adapter`, then loads the
/// quantized target adapter through the ordinary residency engine.
///
/// The source adapter is used only for semantic checkpoint recipes. No dense
/// module is populated. The target adapter is therefore free to expose packed
/// parameter trees, while every residency budget is computed from the packed
/// store metadata seen by [`load_layerwise_model`].
fn packed_weight_companion_dtypes(module: &impl ModuleParameters) -> BTreeMap<String, RecipeDtype> {
    let parameters = module.parameters().flatten();
    parameters
        .iter()
        .filter(|(_, parameter)| parameter.dtype() == safemlx::Dtype::Uint32)
        .map(|(name, _)| {
            let canonical = crate::runtime::checkpoint::binding::canonical_checkpoint_name(name);
            let scales = canonical
                .strip_suffix(".weight")
                .map(|prefix| format!("{prefix}.scales"))
                .unwrap_or_else(|| format!("{canonical}_scales"));
            let dtype = parameters
                .get(scales.as_str())
                .map(|parameter| RecipeDtype::from(parameter.dtype()))
                .unwrap_or(RecipeDtype::F32);
            (canonical, dtype)
        })
        .collect()
}

pub(crate) fn load_layerwise_model_quantized<A, O>(
    store: SharedWeightStore,
    source_adapter: A,
    target_adapter: A,
    options: O,
    quantization: WeightQuantization,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LayerwiseModel<A>, Error>
where
    A: ArchitectureAdapter,
    O: Into<LayerWeightResidency>,
{
    let mut recipes = BTreeMap::new();
    let mut collect =
        |bindings: &[crate::runtime::residency::manager::WeightBinding],
         selected_local_weights: Option<&BTreeMap<String, RecipeDtype>>| {
            for binding in bindings {
                let recipe = binding.source_recipe();
                let metadata = recipe.infer(store.as_ref())?;
                if !matches!(
                    metadata.dtype(),
                    RecipeDtype::F16 | RecipeDtype::BF16 | RecipeDtype::F32
                ) || metadata.shape().len() < 2
                {
                    continue;
                }
                let canonical_local =
                    crate::runtime::checkpoint::binding::canonical_checkpoint_name(binding.name());
                if selected_local_weights
                    .is_some_and(|selected| !selected.contains_key(&canonical_local))
                {
                    continue;
                }
                let target = binding.logical_target().unwrap_or(binding.checkpoint_key());
                let companion_dtype = selected_local_weights
                    .and_then(|selected| selected.get(&canonical_local))
                    .cloned()
                    .unwrap_or(RecipeDtype::F32);
                match recipes.entry(target.to_string()) {
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        entry.insert((recipe, companion_dtype));
                    }
                    std::collections::btree_map::Entry::Occupied(entry)
                        if entry.get() != &(recipe, companion_dtype) =>
                    {
                        return Err(Error::Quantization(format!(
                        "load-time quantization target {target:?} has conflicting semantic recipes"
                    )));
                    }
                    std::collections::btree_map::Entry::Occupied(_) => {}
                }
            }
            Ok::<(), Error>(())
        };
    for unit in source_adapter.static_units(store.as_ref())? {
        let selected = unit
            .bindings()
            .iter()
            .filter(|binding| target_adapter.quantizes_static_binding(binding))
            .map(|binding| {
                (
                    crate::runtime::checkpoint::binding::canonical_checkpoint_name(binding.name()),
                    RecipeDtype::F32,
                )
            })
            .collect::<BTreeMap<_, _>>();
        collect(unit.bindings(), Some(&selected))?;
    }
    let graph = source_adapter.execution_graph()?;
    for group in 0..graph.groups().len() {
        for index in 0..source_adapter.layer_count(group)? {
            let layer = source_adapter.new_layer(group, index, stream)?;
            let target_layer = target_adapter.new_layer(group, index, stream)?;
            let selected = packed_weight_companion_dtypes(&target_layer);
            collect(
                &source_adapter.layer_bindings(group, index, &layer, store.as_ref())?,
                Some(&selected),
            )?;
        }
    }
    if recipes.is_empty() {
        return Err(Error::Quantization(format!(
            "architecture adapter {} declared no floating matrix bindings for load-time quantization",
            source_adapter.model_type()
        )));
    }
    let targets = recipes
        .into_iter()
        .map(|(target, (recipe, companion_dtype))| {
            let target = BoundedQuantizationTarget::from_recipe(target, recipe)?;
            match quantization {
                WeightQuantization::Affine(_) => {
                    target.with_affine_companion_dtype(companion_dtype)
                }
                WeightQuantization::MxFp4 => Ok(target),
                WeightQuantization::GgufIQuant { .. } => unreachable!(
                    "load-time materialization rejects checkpoint-native GGUF encodings"
                ),
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let working_set_bytes =
        bounded_quantization_working_set(store.as_ref(), &targets, quantization)?;
    let transformed = Arc::new(BoundedQuantizedWeightStore::create(
        Arc::clone(&store),
        BoundedQuantizationPlan::new(quantization, working_set_bytes, targets)?,
        weights_stream,
    )?);
    let report = transformed.report().clone();
    let transformed: SharedWeightStore = transformed;
    let mut model =
        load_layerwise_model(transformed, target_adapter, options, stream, weights_stream)?;
    model.metadata.materialization = Some(report);
    Ok(model)
}

/// Loads an adapter directly or through the shared bounded packed overlay.
///
/// This is the authoritative standalone materialization route for both
/// SafeTensors and dense GGUF stores. Architecture code supplies semantic
/// bindings through the adapter; residency sees only the resulting packed
/// store and therefore budgets packed bytes rather than dense source bytes.
pub(crate) fn load_layerwise_model_with_quantization<A, O>(
    store: SharedWeightStore,
    source_adapter: A,
    options: O,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LayerwiseModel<A>, Error>
where
    A: LoadTimeQuantizableAdapter,
    O: Into<LayerWeightResidency>,
{
    let options = options.into();
    match quantization {
        Some(quantization) => {
            let target_adapter = source_adapter.load_time_quantized(quantization, stream)?;
            load_layerwise_model_quantized(
                store,
                source_adapter,
                target_adapter,
                options,
                quantization,
                stream,
                weights_stream,
            )
        }
        None => load_layerwise_model(store, source_adapter, options, stream, weights_stream),
    }
}

/// Builds a PP-stage-local packed overlay before pipeline residency planning.
///
/// The source adapter contributes semantic recipes for only the stage-owned
/// static roles and decoder range. The target adapter identifies projections
/// whose runtime parameter tree is packed. Complete or rank-selected expert
/// banks use the same matrix-row tiler as ordinary projections; an independent
/// expert store is only involved when that residency policy was requested.
pub(crate) struct PipelineStageQuantizationSelection<'a> {
    static_roles: &'a [&'a str],
    layer_group: usize,
    layer_range: Range<usize>,
}

impl<'a> PipelineStageQuantizationSelection<'a> {
    pub(crate) fn new(
        static_roles: &'a [&'a str],
        layer_group: usize,
        layer_range: Range<usize>,
    ) -> Self {
        Self {
            static_roles,
            layer_group,
            layer_range,
        }
    }
}

pub(crate) fn quantize_pipeline_stage_store<A>(
    store: SharedWeightStore,
    source_adapter: &A,
    target_adapter: &A,
    selection: PipelineStageQuantizationSelection<'_>,
    quantization: WeightQuantization,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(SharedWeightStore, BoundedQuantizationReport), Error>
where
    A: ArchitectureAdapter,
{
    let mut recipes = BTreeMap::new();
    let mut collect =
        |bindings: &[crate::runtime::residency::manager::WeightBinding],
         selected_local_weights: Option<&BTreeMap<String, RecipeDtype>>| {
            for binding in bindings {
                let recipe = binding.source_recipe();
                let metadata = recipe.infer(store.as_ref())?;
                if !matches!(
                    metadata.dtype(),
                    RecipeDtype::F16 | RecipeDtype::BF16 | RecipeDtype::F32
                ) || metadata.shape().len() < 2
                {
                    continue;
                }
                let canonical_local =
                    crate::runtime::checkpoint::binding::canonical_checkpoint_name(binding.name());
                if selected_local_weights
                    .is_some_and(|selected| !selected.contains_key(&canonical_local))
                {
                    continue;
                }
                let target = binding.logical_target().unwrap_or(binding.checkpoint_key());
                let companion_dtype = selected_local_weights
                    .and_then(|selected| selected.get(&canonical_local))
                    .cloned()
                    .unwrap_or(RecipeDtype::F32);
                match recipes.entry(target.to_string()) {
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        entry.insert((recipe, companion_dtype));
                    }
                    std::collections::btree_map::Entry::Occupied(entry)
                        if entry.get() != &(recipe, companion_dtype) =>
                    {
                        return Err(Error::Quantization(format!(
                        "pipeline load-time quantization target {target:?} has conflicting semantic recipes"
                    )));
                    }
                    std::collections::btree_map::Entry::Occupied(_) => {}
                }
            }
            Ok::<(), Error>(())
        };

    for unit in source_adapter.static_units(store.as_ref())? {
        if !selection
            .static_roles
            .iter()
            .any(|role| unit.id().as_str().ends_with(&format!(".static.{role}")))
        {
            continue;
        }
        let selected = unit
            .bindings()
            .iter()
            .filter(|binding| target_adapter.quantizes_static_binding(binding))
            .map(|binding| {
                (
                    crate::runtime::checkpoint::binding::canonical_checkpoint_name(binding.name()),
                    RecipeDtype::F32,
                )
            })
            .collect::<BTreeMap<_, _>>();
        collect(unit.bindings(), Some(&selected))?;
    }

    for index in selection.layer_range {
        let source_layer = source_adapter.new_layer(selection.layer_group, index, stream)?;
        let target_layer = target_adapter.new_layer(selection.layer_group, index, stream)?;
        let selected = packed_weight_companion_dtypes(&target_layer);
        collect(
            &source_adapter.layer_bindings(
                selection.layer_group,
                index,
                &source_layer,
                store.as_ref(),
            )?,
            Some(&selected),
        )?;
    }

    if recipes.is_empty() {
        return Err(Error::Quantization(format!(
            "pipeline architecture adapter {} declared no floating matrix bindings for stage-local load-time quantization",
            source_adapter.model_type()
        )));
    }
    let targets = recipes
        .into_iter()
        .map(|(target, (recipe, companion_dtype))| {
            let target = BoundedQuantizationTarget::from_recipe(target, recipe)?;
            match quantization {
                WeightQuantization::Affine(_) => {
                    target.with_affine_companion_dtype(companion_dtype)
                }
                WeightQuantization::MxFp4 => Ok(target),
                WeightQuantization::GgufIQuant { .. } => unreachable!(
                    "load-time materialization rejects checkpoint-native GGUF encodings"
                ),
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let working_set_bytes =
        bounded_quantization_working_set(store.as_ref(), &targets, quantization)?;
    let transformed = Arc::new(BoundedQuantizedWeightStore::create(
        store,
        BoundedQuantizationPlan::new(quantization, working_set_bytes, targets)?,
        weights_stream,
    )?);
    let report = transformed.report().clone();
    let transformed: SharedWeightStore = transformed;
    Ok((transformed, report))
}

fn bounded_quantization_working_set(
    store: &dyn WeightStore,
    targets: &[BoundedQuantizationTarget],
    quantization: WeightQuantization,
) -> Result<u64, Error> {
    let mut output_bytes = 0u64;
    let mut minimum_tile_bytes = 0u64;
    for target in targets {
        let metadata = target.source().infer(store)?;
        let shape = metadata.shape();
        if shape.len() < 2 {
            return Err(Error::Quantization(format!(
                "load-time quantization target {:?} must be a matrix or matrix bank, got shape {shape:?}",
                target.weight_name()
            )));
        }
        let row_axis = shape.len() - 2;
        let leading = shape[..row_axis]
            .iter()
            .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
            .ok_or_else(|| Error::Quantization("leading matrix count overflowed".into()))?;
        if leading == 0 || shape[row_axis] == 0 {
            return Err(Error::Quantization(format!(
                "load-time quantization target {:?} must contain at least one matrix row",
                target.weight_name()
            )));
        }
        let rows = leading
            .checked_mul(shape[row_axis])
            .ok_or_else(|| Error::Quantization("matrix-bank row count overflowed".into()))?
            as u64;
        let columns = shape[row_axis + 1];
        let group_size = usize::try_from(quantization.group_size())
            .map_err(|_| Error::Quantization("quantization group size is invalid".into()))?;
        if columns % group_size != 0 || columns % 32 != 0 {
            return Err(Error::Quantization(format!(
                "load-time quantization target {:?} input dimension {columns} must be divisible by group_size {group_size} and 32",
                target.weight_name()
            )));
        }
        let groups = (columns / group_size) as u64;
        let packed_row = (columns as u64)
            .checked_mul(quantization.bits() as u64)
            .and_then(|bits| bits.checked_div(8))
            .ok_or_else(|| Error::Quantization("packed row size overflowed".into()))?;
        let companion_row = if matches!(quantization, WeightQuantization::MxFp4) {
            groups
        } else {
            groups
                .checked_mul(target.affine_companion_bytes())
                .ok_or_else(|| Error::Quantization("packed scale row size overflowed".into()))?
        };
        let bias_row = if quantization.has_biases() {
            groups
                .checked_mul(target.affine_companion_bytes())
                .ok_or_else(|| Error::Quantization("packed bias row size overflowed".into()))?
        } else {
            0
        };
        let output_row = packed_row
            .checked_add(companion_row)
            .and_then(|bytes| bytes.checked_add(bias_row))
            .ok_or_else(|| Error::Quantization("packed output row size overflowed".into()))?;
        output_bytes = output_bytes
            .checked_add(
                rows.checked_mul(output_row)
                    .ok_or_else(|| Error::Quantization("packed target size overflowed".into()))?,
            )
            .ok_or_else(|| Error::Quantization("packed model size overflowed".into()))?;
        for matrix in 0..leading {
            let one_row = target
                .source()
                .select_bounded_matrix_rows(store, matrix, 0, 1)?;
            one_row.preflight_bounded(store)?;
            minimum_tile_bytes = minimum_tile_bytes.max(
                one_row
                    .peak_materialization_bytes(store)?
                    .checked_add(output_row)
                    .ok_or_else(|| Error::Quantization("conversion tile size overflowed".into()))?,
            );
        }
    }
    Ok(output_bytes.max(minimum_tile_bytes))
}

/// Builds a generalized layerwise model from an already cataloged checkpoint.
pub fn load_layerwise_model<A, O>(
    store: SharedWeightStore,
    mut adapter: A,
    options: O,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LayerwiseModel<A>, Error>
where
    A: ArchitectureAdapter,
    O: Into<LayerWeightResidency>,
{
    let options = options.into();
    let fully_resident = options.is_fully_resident();
    let dense = options.dense();
    let offload = options.offload()?;
    let mut definitions = Vec::new();
    let mut specs = Vec::new();
    let mut consumed = BTreeSet::new();
    let mut static_device_bytes = 0u64;
    let mut static_ids = Vec::new();
    for unit in adapter.static_units(store.as_ref())? {
        static_ids.push(unit.id.clone());
        add_unit(
            &mut definitions,
            &mut specs,
            &mut consumed,
            unit.id,
            unit.bindings,
            ResidencyPolicy::Pinned,
            MemoryTier::Device,
            &mut static_device_bytes,
        )?;
    }

    let execution_graph = adapter.execution_graph()?;
    let mut groups = Vec::with_capacity(execution_graph.groups().len());
    let mut layer_parameter_bytes = 0u64;
    let mut device_window_bytes = 0u64;
    let mut host_window_bytes = 0u64;
    let mut planned_layer_count = 0usize;
    let mut maximum_group_depth = 0usize;
    for (group_index, group_spec) in execution_graph.groups().iter().enumerate() {
        let layer_count = adapter.layer_count(group_index)?;
        let depth = options.device_depth(layer_count);
        maximum_group_depth = maximum_group_depth.max(depth);
        if depth > layer_count {
            return Err(LayerwiseModelError::InvalidLayerWindow { depth, layer_count }.into());
        }
        if let Some(dense) = dense {
            if dense.host_budget_bytes > 0 && dense.host_lookahead > layer_count {
                return Err(LayerwiseModelError::InvalidHostLayerWindow {
                    depth: dense.host_lookahead,
                    layer_count,
                }
                .into());
            }
        }
        let mut layer_ids = Vec::with_capacity(layer_count);
        let mut layer_bytes = Vec::with_capacity(layer_count);
        for index in 0..layer_count {
            let layer = adapter.new_layer(group_index, index, stream)?;
            let bindings = adapter.layer_bindings(group_index, index, &layer, store.as_ref())?;
            let bytes = binding_bytes(&bindings)?;
            layer_parameter_bytes = layer_parameter_bytes.checked_add(bytes).ok_or(
                LayerwiseModelError::ArithmeticOverflow {
                    context: "host execution-unit byte total",
                },
            )?;
            let id = OffloadUnitId::new(adapter.layer_unit_name(group_index, index))?;
            consumed.extend(
                bindings
                    .iter()
                    .flat_map(|binding| binding.checkpoint_keys().into_iter().map(str::to_string)),
            );
            definitions.push(OffloadUnit::new(id.clone(), bindings)?);
            specs.push(OffloadUnitSpec::new(
                id.clone(),
                bytes,
                if fully_resident {
                    ResidencyPolicy::Pinned
                } else if dense.is_some() {
                    ResidencyPolicy::Cacheable
                } else {
                    ResidencyPolicy::Windowed
                },
                if fully_resident {
                    MemoryTier::Device
                } else if dense.is_some() {
                    MemoryTier::Disk
                } else {
                    MemoryTier::Host
                },
            )?);
            planned_layer_count = planned_layer_count.checked_add(1).ok_or(
                LayerwiseModelError::ArithmeticOverflow {
                    context: "streamed execution-unit count",
                },
            )?;
            layer_ids.push(id);
            layer_bytes.push(bytes);
        }
        let group_device_window = largest_window_bytes(&layer_bytes, depth)?;
        if dense.is_some() {
            device_window_bytes = device_window_bytes.max(group_device_window);
            if let Some(dense) = dense {
                if dense.host_budget_bytes > 0 {
                    host_window_bytes = host_window_bytes
                        .max(largest_window_bytes(&layer_bytes, dense.host_lookahead)?);
                }
            }
        } else {
            device_window_bytes = device_window_bytes.checked_add(group_device_window).ok_or(
                LayerwiseModelError::ArithmeticOverflow {
                    context: "combined device execution-window byte total",
                },
            )?;
        }
        groups.push(ResidentLayerGroup::new(
            group_spec.id().to_string(),
            layer_ids,
            depth,
        )?);
    }

    consumed.extend(store.materialized_source_keys());
    consumed.extend(adapter.additional_consumed_checkpoint_keys(store.as_ref()));

    validate_unused(store.as_ref(), &consumed, options.strict_loading(), |key| {
        adapter.ignores_checkpoint_key(key)
    })?;
    if fully_resident {
        validate_host_budget(offload, 0)?;
    } else if dense.is_some() {
        validate_host_budget(offload, host_window_bytes)?;
    } else {
        validate_host_budget(offload, layer_parameter_bytes)?;
    }
    validate_device_budget(
        offload,
        static_device_bytes,
        device_window_bytes,
        maximum_group_depth,
    )?;

    let plan = OffloadPlan::new(offload, specs)?;
    let residency_stream = if dense.is_some() {
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
    let static_leases = static_ids
        .iter()
        .map(|id| residency.acquire(id, MemoryTier::Device))
        .collect::<Result<Vec<_>, _>>()?;
    adapter.populate_static(&static_leases)?;

    let mut model = LayerwiseModel::new(
        adapter,
        execution_graph,
        store,
        residency,
        groups,
        static_leases,
    )?
    .with_memory_sampling(options.sample_mlx_memory(), options.sample_process_memory());
    if fully_resident {
        model.materialize_resident_layers(stream)?;
    }
    model.metadata = LayerwiseModelMetadata {
        model_type: model.adapter.model_type().into(),
        quantization: model.adapter.quantization(),
        layer_count: planned_layer_count,
        static_device_bytes,
        residency: options.residency(),
        layer_parameter_bytes,
        maximum_device_layer_bytes: device_window_bytes,
        device_layer_capacity: if fully_resident {
            planned_layer_count
        } else {
            maximum_group_depth
        },
        materialization: None,
    };
    if let Some(dense) = dense {
        let execution_groups = model
            .groups
            .iter()
            .map(|group| (group.id().to_string(), group.units().to_vec()))
            .collect::<Vec<_>>();
        model.dense_stream = Some(DenseStreamController::new(
            &model.residency,
            dense,
            planned_layer_count,
            layer_parameter_bytes,
            static_device_bytes,
            execution_groups,
        )?);
    }
    Ok(model)
}

pub(crate) fn load_tensor_parallel_layerwise_model<A, O>(
    store: SharedWeightStore,
    mut adapter: A,
    options: O,
    build: crate::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LayerwiseModel<A>, Error>
where
    A: ArchitectureAdapter,
    O: Into<LayerWeightResidency>,
{
    let options = options.into();
    let fully_resident = options.is_fully_resident();
    let dense = options.dense();
    let offload = options.offload()?;
    let mut planner = build.planner();
    adapter.register_parallel_parameters(build, &mut planner, stream)?;
    let (_, layout) = planner.finish()?;
    if layout.is_empty() {
        return Err(Error::Parallel(format!(
            "architecture adapter {} declared no tensor-parallel execution-group parameters",
            adapter.model_type()
        )));
    }

    let static_units = adapter.static_units(store.as_ref())?;
    adapter.configure_parallel_static(build, &layout, stream)?;

    let mut definitions = Vec::new();
    let mut specs = Vec::new();
    let mut consumed = BTreeSet::new();
    let mut static_device_bytes = 0u64;
    let mut global_parameter_bytes = 0u64;
    let mut static_ids = Vec::new();
    for unit in static_units {
        global_parameter_bytes = global_parameter_bytes
            .checked_add(binding_bytes(&unit.bindings)?)
            .ok_or(LayerwiseModelError::ArithmeticOverflow {
                context: "global static parameter byte total",
            })?;
        let bindings = shard_layer_bindings(unit.bindings, "", store.as_ref(), &layout)?;
        static_ids.push(unit.id.clone());
        add_unit(
            &mut definitions,
            &mut specs,
            &mut consumed,
            unit.id,
            bindings,
            ResidencyPolicy::Pinned,
            MemoryTier::Device,
            &mut static_device_bytes,
        )?;
    }

    let execution_graph = adapter.execution_graph()?;
    let mut groups = Vec::with_capacity(execution_graph.groups().len());
    let mut layer_parameter_bytes = 0u64;
    let mut device_window_bytes = 0u64;
    let mut host_window_bytes = 0u64;
    let mut planned_layer_count = 0usize;
    let mut maximum_group_depth = 0usize;
    for (group_index, group_spec) in execution_graph.groups().iter().enumerate() {
        let layer_count = adapter.layer_count(group_index)?;
        let depth = options.device_depth(layer_count);
        maximum_group_depth = maximum_group_depth.max(depth);
        if depth > layer_count {
            return Err(LayerwiseModelError::InvalidLayerWindow { depth, layer_count }.into());
        }
        if let Some(dense) = dense {
            if dense.host_budget_bytes > 0 && dense.host_lookahead > layer_count {
                return Err(LayerwiseModelError::InvalidHostLayerWindow {
                    depth: dense.host_lookahead,
                    layer_count,
                }
                .into());
            }
        }
        let mut layer_ids = Vec::with_capacity(layer_count);
        let mut layer_bytes = Vec::with_capacity(layer_count);
        for index in 0..layer_count {
            let global_layer = adapter.new_layer(group_index, index, stream)?;
            let global_bindings =
                adapter.layer_bindings(group_index, index, &global_layer, store.as_ref())?;
            global_parameter_bytes = global_parameter_bytes
                .checked_add(binding_bytes(&global_bindings)?)
                .ok_or(LayerwiseModelError::ArithmeticOverflow {
                    context: "global TP execution-unit byte total",
                })?;
            let layer = adapter.new_parallel_layer(group_index, index, &layout, stream)?;
            let bindings = adapter.parallel_layer_bindings(
                group_index,
                index,
                &layer,
                store.as_ref(),
                &layout,
                stream,
            )?;
            let bytes = binding_bytes(&bindings)?;
            layer_parameter_bytes = layer_parameter_bytes.checked_add(bytes).ok_or(
                LayerwiseModelError::ArithmeticOverflow {
                    context: "host TP execution-unit byte total",
                },
            )?;
            let id = OffloadUnitId::new(adapter.layer_unit_name(group_index, index))?;
            consumed.extend(
                bindings
                    .iter()
                    .flat_map(|binding| binding.checkpoint_keys().into_iter().map(str::to_string)),
            );
            definitions.push(OffloadUnit::new(id.clone(), bindings)?);
            specs.push(OffloadUnitSpec::new(
                id.clone(),
                bytes,
                if fully_resident {
                    ResidencyPolicy::Pinned
                } else if dense.is_some() {
                    ResidencyPolicy::Cacheable
                } else {
                    ResidencyPolicy::Windowed
                },
                if fully_resident {
                    MemoryTier::Device
                } else if dense.is_some() {
                    MemoryTier::Disk
                } else {
                    MemoryTier::Host
                },
            )?);
            planned_layer_count = planned_layer_count.checked_add(1).ok_or(
                LayerwiseModelError::ArithmeticOverflow {
                    context: "TP execution-unit count",
                },
            )?;
            layer_ids.push(id);
            layer_bytes.push(bytes);
        }
        let group_device_window = largest_window_bytes(&layer_bytes, depth)?;
        if dense.is_some() {
            device_window_bytes = device_window_bytes.max(group_device_window);
            if let Some(dense) = dense {
                if dense.host_budget_bytes > 0 {
                    host_window_bytes = host_window_bytes
                        .max(largest_window_bytes(&layer_bytes, dense.host_lookahead)?);
                }
            }
        } else {
            device_window_bytes = device_window_bytes.checked_add(group_device_window).ok_or(
                LayerwiseModelError::ArithmeticOverflow {
                    context: "combined TP device execution-window byte total",
                },
            )?;
        }
        groups.push(ResidentLayerGroup::new(
            group_spec.id().to_string(),
            layer_ids,
            depth,
        )?);
    }
    consumed.extend(store.materialized_source_keys());
    consumed.extend(adapter.additional_consumed_checkpoint_keys(store.as_ref()));
    validate_unused(store.as_ref(), &consumed, options.strict_loading(), |key| {
        adapter.ignores_checkpoint_key(key)
    })?;
    if fully_resident {
        validate_host_budget(offload, 0)?;
    } else if dense.is_some() {
        validate_host_budget(offload, host_window_bytes)?;
    } else {
        validate_host_budget(offload, layer_parameter_bytes)?;
    }
    validate_device_budget(
        offload,
        static_device_bytes,
        device_window_bytes,
        maximum_group_depth,
    )?;
    let plan = OffloadPlan::new(offload, specs)?;
    let residency_stream = if dense.is_some() {
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
    let static_leases = static_ids
        .iter()
        .map(|id| residency.acquire(id, MemoryTier::Device))
        .collect::<Result<Vec<_>, _>>()?;
    adapter.populate_static(&static_leases)?;
    let mut model = LayerwiseModel::new(
        adapter,
        execution_graph,
        store,
        residency,
        groups,
        static_leases,
    )?
    .with_memory_sampling(options.sample_mlx_memory(), options.sample_process_memory());
    model.parallel_layout = Some(layout);
    model.parallel_topology = Some(build.topology());
    if fully_resident {
        model.materialize_resident_layers(stream)?;
    }
    model.metadata = LayerwiseModelMetadata {
        model_type: model.adapter.model_type().into(),
        quantization: model.adapter.quantization(),
        layer_count: planned_layer_count,
        static_device_bytes,
        residency: options.residency(),
        layer_parameter_bytes,
        maximum_device_layer_bytes: device_window_bytes,
        device_layer_capacity: if fully_resident {
            planned_layer_count
        } else {
            maximum_group_depth
        },
        materialization: None,
    };
    let local_parameter_bytes = static_device_bytes
        .checked_add(layer_parameter_bytes)
        .ok_or(LayerwiseModelError::ArithmeticOverflow {
            context: "rank-local parallel parameter byte total",
        })?;
    model.parallel_info = Some(ParallelModelInfo {
        topology: build.topology(),
        model_type: model.adapter.model_type().into(),
        owned_tensors: model
            .parallel_layout
            .as_ref()
            .expect("parallel layout was assigned")
            .tensors()
            .map(|(target, _)| target.to_string())
            .collect(),
        local_parameter_bytes,
        global_parameter_bytes,
        pinned_device_parameter_bytes: if fully_resident {
            local_parameter_bytes
        } else {
            static_device_bytes
        },
        maximum_device_parameter_bytes: static_device_bytes
            .checked_add(device_window_bytes)
            .ok_or(LayerwiseModelError::ArithmeticOverflow {
                context: "maximum rank-local device parameter byte total",
            })?,
    });
    if let Some(dense) = dense {
        let execution_groups = model
            .groups
            .iter()
            .map(|group| (group.id().to_string(), group.units().to_vec()))
            .collect::<Vec<_>>();
        model.dense_stream = Some(DenseStreamController::new(
            &model.residency,
            dense,
            planned_layer_count,
            layer_parameter_bytes,
            static_device_bytes,
            execution_groups,
        )?);
    }
    Ok(model)
}

fn packed_semantic_weight_name(name: &str) -> Option<String> {
    name.strip_suffix(".scales")
        .or_else(|| name.strip_suffix(".biases"))
        .map(|prefix| format!("{prefix}.weight"))
}

fn stored_tensor_selection(
    tensor: &crate::runtime::distributed::parallel::LocalTensorLayout,
    stored_shape: &[usize],
) -> Result<crate::runtime::checkpoint::store::TensorSelection, Error> {
    use crate::runtime::checkpoint::store::TensorSelection;
    use crate::runtime::distributed::topology::TensorPlacement;

    let scale_boundary = |axis: usize, boundary: usize| -> Result<usize, Error> {
        let semantic = tensor.global_shape()[axis];
        let stored = stored_shape[axis];
        boundary
            .checked_mul(stored)
            .and_then(|value| value.checked_div(semantic))
            .filter(|scaled| scaled * semantic == boundary * stored)
            .ok_or_else(|| {
                Error::Parallel(format!(
                    "semantic shard boundary {boundary} on axis {axis} is not aligned to packed storage shape {stored_shape:?} derived from {:?}",
                    tensor.global_shape()
                ))
            })
    };

    Ok(match tensor.placement() {
        TensorPlacement::Replicated | TensorPlacement::Local => TensorSelection::Full,
        TensorPlacement::Shard { axis, index, parts } => {
            let stored = stored_shape[*axis];
            if !stored.is_multiple_of(*parts) {
                return Err(Error::Parallel(format!(
                    "packed storage axis {axis} width {stored} cannot be divided among {parts} TP ranks"
                )));
            }
            let width = stored / *parts;
            TensorSelection::Range {
                axis: *axis,
                start: index * width,
                end: (index + 1) * width,
            }
        }
        TensorPlacement::Range { axis, start, end } => TensorSelection::Range {
            axis: *axis,
            start: scale_boundary(*axis, *start)?,
            end: scale_boundary(*axis, *end)?,
        },
        TensorPlacement::Indices { axis, indices } => {
            if stored_shape[*axis] != tensor.global_shape()[*axis] {
                return Err(Error::Parallel(format!(
                    "indexed TP placement on semantic axis {axis} cannot address packed storage shape {stored_shape:?} derived from {:?}",
                    tensor.global_shape()
                )));
            }
            TensorSelection::Indices {
                axis: *axis,
                indices: indices.clone(),
            }
        }
        TensorPlacement::Omit
        | TensorPlacement::Rank { .. }
        | TensorPlacement::PipelineStage { .. } => {
            return Err(Error::Parallel(format!(
                "execution-group binding has non-TP placement {:?}",
                tensor.placement()
            )))
        }
    })
}

pub(crate) fn shard_layer_bindings(
    bindings: Vec<crate::runtime::residency::manager::WeightBinding>,
    prefix: &str,
    store: &dyn WeightStore,
    layout: &crate::runtime::distributed::parallel::LocalModelLayout,
) -> Result<Vec<crate::runtime::residency::manager::WeightBinding>, Error> {
    use crate::runtime::checkpoint::store::TensorSelection;

    let store_keys = store.keys().into_iter().collect::<BTreeSet<_>>();
    let mut output = Vec::with_capacity(bindings.len());
    for binding in bindings {
        let canonical_name = crate::runtime::checkpoint::binding::canonical_checkpoint_name(
            &format!("{prefix}.{}", binding.name()),
        );
        let logical_target = binding.logical_target();
        let tensor = logical_target
            .and_then(|target| layout.tensor(target))
            .or_else(|| {
                logical_target.and_then(|logical| {
                    let canonical_logical =
                        crate::runtime::checkpoint::binding::canonical_checkpoint_name(logical);
                    layout.tensors().find_map(|(target, tensor)| {
                        (crate::runtime::checkpoint::binding::canonical_checkpoint_name(target)
                            == canonical_logical)
                            .then_some(tensor)
                    })
                })
            })
            .or_else(|| layout.tensor(binding.checkpoint_key()))
            .or_else(|| layout.tensor(&canonical_name))
            .or_else(|| {
                layout.tensors().find_map(|(target, tensor)| {
                    let canonical_target =
                        crate::runtime::checkpoint::binding::canonical_checkpoint_name(target);
                    (canonical_target == binding.checkpoint_key()
                        || canonical_target == canonical_name)
                        .then_some(tensor)
                })
            })
            .or_else(|| {
                [
                    logical_target.map(str::to_string),
                    Some(binding.checkpoint_key().to_string()),
                    Some(canonical_name.clone()),
                ]
                .into_iter()
                .flatten()
                .filter_map(|name| packed_semantic_weight_name(&name))
                .find_map(|weight| {
                    layout.tensor(&weight).or_else(|| {
                        let canonical =
                            crate::runtime::checkpoint::binding::canonical_checkpoint_name(&weight);
                        layout.tensor(&canonical)
                    })
                })
            });
        let Some(tensor) = tensor else {
            output.push(binding);
            continue;
        };
        // Direct bindings created by callers can describe a logical checkpoint
        // target that is deliberately absent from this physical store. Preserve
        // that contract by deriving its selection and byte count from semantic
        // layout alone. Physical and derived bindings use store metadata below,
        // which is required when packed storage geometry differs from the
        // semantic weight geometry.
        if binding.recipe().is_none() && !store_keys.contains(binding.checkpoint_key()) {
            let selection = stored_tensor_selection(tensor, tensor.global_shape())?;
            if selection == TensorSelection::Full {
                output.push(binding);
                continue;
            }
            let global_elements = tensor.global_shape().iter().product::<usize>();
            let local_elements = tensor.local_shape().iter().product::<usize>();
            let expected_bytes = binding
                .expected_bytes()
                .checked_mul(local_elements as u64)
                .and_then(|bytes| bytes.checked_div(global_elements as u64))
                .ok_or_else(|| {
                    Error::Parallel(format!(
                        "cannot size rank-local binding {:?}",
                        binding.name()
                    ))
                })?;
            output.push(crate::runtime::residency::manager::WeightBinding::new(
                binding.name(),
                binding.checkpoint_key(),
                selection,
                expected_bytes,
            )?);
            continue;
        }
        let recipe = binding.source_recipe();
        let metadata = recipe.infer(store)?;
        let selection = stored_tensor_selection(tensor, metadata.shape())?;
        if selection == crate::runtime::checkpoint::store::TensorSelection::Full {
            output.push(binding);
            continue;
        }
        let recipe = recipe.select_bounded(store, selection)?;
        let expected_bytes = recipe.infer(store)?.byte_len();
        let sharded = crate::runtime::residency::manager::WeightBinding::from_recipe(
            binding.name(),
            recipe,
            expected_bytes,
        )?;
        output.push(sharded);
    }
    Ok(output)
}

fn build_parallel_module_bindings(
    module: &impl ModuleParameters,
    prefix: &str,
    store: &dyn WeightStore,
    layout: &crate::runtime::distributed::parallel::LocalModelLayout,
) -> Result<Vec<crate::runtime::residency::manager::WeightBinding>, Error> {
    use crate::runtime::checkpoint::store::TensorSelection;
    let keys = store.keys().into_iter().collect::<BTreeSet<_>>();
    let params = module.parameters().flatten();
    let mut names = params.keys().map(ToString::to_string).collect::<Vec<_>>();
    names.sort();
    let mut bindings = Vec::with_capacity(names.len());
    for local_name in names {
        let parameter = params.get(local_name.as_str()).expect("known parameter");
        let destination = if prefix.is_empty() {
            local_name.clone()
        } else {
            format!("{prefix}.{local_name}")
        };
        let canonical =
            crate::runtime::checkpoint::binding::canonical_checkpoint_name(&destination);
        let checkpoint_key = if keys.contains(&destination) {
            destination.clone()
        } else if keys.contains(&canonical) {
            canonical.clone()
        } else {
            return Err(
                crate::runtime::checkpoint::binding::ModuleBindingError::MissingParameter {
                    destination,
                }
                .into(),
            );
        };
        let metadata = store.metadata(&checkpoint_key)?;
        let tensor = layout
            .tensor(&checkpoint_key)
            .or_else(|| layout.tensor(&canonical))
            .or_else(|| {
                packed_semantic_weight_name(&checkpoint_key)
                    .or_else(|| packed_semantic_weight_name(&canonical))
                    .and_then(|weight| {
                        layout.tensor(&weight).or_else(|| {
                            let canonical =
                                crate::runtime::checkpoint::binding::canonical_checkpoint_name(
                                    &weight,
                                );
                            layout.tensor(&canonical)
                        })
                    })
            });
        let (selection, expected_bytes) = if let Some(tensor) = tensor {
            let local_shape = parameter
                .shape()
                .iter()
                .map(|&dim| usize::try_from(dim))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| Error::Parallel(format!("invalid local shape for {destination}")))?;
            let selection = stored_tensor_selection(tensor, &metadata.shape)?;
            let mut selected_shape = metadata.shape.clone();
            match &selection {
                TensorSelection::Full => {}
                TensorSelection::Range { axis, start, end } => {
                    selected_shape[*axis] = end - start;
                }
                TensorSelection::Indices { axis, indices } => {
                    selected_shape[*axis] = indices.len();
                }
                TensorSelection::Contiguous { .. } => {
                    unreachable!("TP packed placement never emits a reshaped contiguous span")
                }
            }
            if selected_shape != local_shape {
                return Err(Error::Parallel(format!(
                    "planned packed local shape {:?} for {destination} does not match runtime {:?}",
                    selected_shape, local_shape
                )));
            }
            let recipe = crate::runtime::checkpoint::recipe::DerivedWeightRecipe::source(
                checkpoint_key.clone(),
                selection.clone(),
            );
            let bytes = recipe.infer(store)?.byte_len();
            (selection, bytes)
        } else {
            let expected = parameter
                .shape()
                .iter()
                .map(|&dim| usize::try_from(dim))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| Error::Parallel(format!("invalid shape for {destination}")))?;
            if metadata.shape != expected {
                return Err(Error::Parallel(format!(
                    "unplanned parameter {destination} has checkpoint shape {:?}, runtime expects {:?}",
                    metadata.shape, expected
                )));
            }
            (TensorSelection::Full, metadata.logical_byte_len as u64)
        };
        bindings.push(crate::runtime::residency::manager::WeightBinding::new(
            local_name,
            checkpoint_key,
            selection,
            expected_bytes,
        )?);
    }
    Ok(bindings)
}

#[allow(clippy::too_many_arguments)]
fn add_unit(
    definitions: &mut Vec<OffloadUnit>,
    specs: &mut Vec<OffloadUnitSpec>,
    consumed: &mut BTreeSet<String>,
    id: OffloadUnitId,
    bindings: Vec<crate::runtime::residency::manager::WeightBinding>,
    policy: ResidencyPolicy,
    tier: MemoryTier,
    byte_total: &mut u64,
) -> Result<(), Error> {
    let bytes = binding_bytes(&bindings)?;
    *byte_total = byte_total
        .checked_add(bytes)
        .ok_or(LayerwiseModelError::ArithmeticOverflow {
            context: "static device byte total",
        })?;
    consumed.extend(
        bindings
            .iter()
            .flat_map(|binding| binding.checkpoint_keys().into_iter().map(str::to_string)),
    );
    definitions.push(OffloadUnit::new(id.clone(), bindings)?);
    specs.push(OffloadUnitSpec::new(id, bytes, policy, tier)?);
    Ok(())
}

fn validate_unused<F>(
    store: &dyn WeightStore,
    consumed: &BTreeSet<String>,
    strict: bool,
    ignored: F,
) -> Result<(), Error>
where
    F: Fn(&str) -> bool,
{
    if !strict {
        return Ok(());
    }
    let unused = store
        .keys()
        .into_iter()
        .filter(|key| !consumed.contains(key))
        .filter(|key| !ignored(key))
        .collect::<Vec<_>>();
    if unused.is_empty() {
        Ok(())
    } else {
        Err(LayerwiseModelError::UnexpectedCheckpointParameters { unused }.into())
    }
}

fn largest_window_bytes(layer_bytes: &[u64], depth: usize) -> Result<u64, Error> {
    let mut largest = 0u64;
    for start in 0..layer_bytes.len() {
        let mut current = 0u64;
        for bytes in layer_bytes.iter().skip(start).take(depth) {
            current =
                current
                    .checked_add(*bytes)
                    .ok_or(LayerwiseModelError::ArithmeticOverflow {
                        context: "device layer window byte total",
                    })?;
        }
        largest = largest.max(current);
    }
    Ok(largest)
}

fn validate_host_budget(config: OffloadConfig, required: u64) -> Result<(), Error> {
    if let Some(budget) = config.host_budget_bytes() {
        if required > budget {
            return Err(LayerwiseModelError::HostBudgetTooSmall { required, budget }.into());
        }
    }
    Ok(())
}

fn validate_device_budget(
    config: OffloadConfig,
    static_bytes: u64,
    window_bytes: u64,
    depth: usize,
) -> Result<(), Error> {
    let required =
        static_bytes
            .checked_add(window_bytes)
            .ok_or(LayerwiseModelError::ArithmeticOverflow {
                context: "static plus device-window byte total",
            })?;
    if let Some(budget) = config.device_budget_bytes() {
        if required > budget {
            return Err(LayerwiseModelError::DeviceBudgetTooSmall {
                static_bytes,
                window_bytes,
                depth,
                required,
                budget,
            }
            .into());
        }
    }
    Ok(())
}

/// Structured failures produced by the generic layerwise execution engine.
#[derive(Debug, thiserror::Error)]
pub enum LayerwiseModelError {
    /// An adapter declared no execution groups.
    #[error("execution-group graph must contain at least one group")]
    EmptyExecutionGraph,
    /// A group has no stable identity.
    #[error("execution-group identifiers must not be empty")]
    EmptyExecutionGroupId,
    /// Two groups use the same stable identity.
    #[error("duplicate execution-group identifier {0:?}")]
    DuplicateExecutionGroup(String),
    /// The declared graph output does not exist.
    #[error("execution-group graph output {0:?} does not exist")]
    UnknownExecutionGraphOutput(String),
    /// A dependency does not exist.
    #[error("execution group {group:?} depends on unknown group {dependency:?}")]
    UnknownExecutionGroupDependency {
        /// Dependent group.
        group: String,
        /// Missing dependency.
        dependency: String,
    },
    /// A group lists itself as an input.
    #[error("execution group {0:?} cannot depend on itself")]
    SelfDependentExecutionGroup(String),
    /// A group repeats one dependency.
    #[error("execution group {group:?} repeats dependency {dependency:?}")]
    DuplicateExecutionGroupDependency {
        /// Dependent group.
        group: String,
        /// Repeated dependency.
        dependency: String,
    },
    /// The graph contains a cycle.
    #[error("execution-group graph contains a dependency cycle")]
    CyclicExecutionGraph,
    /// Some groups do not contribute to the declared graph output.
    #[error("execution groups do not contribute to the graph output: {disconnected:?}")]
    DisconnectedExecutionGroups {
        /// Stable disconnected group identifiers.
        disconnected: Vec<String>,
    },
    /// A multi-input group did not define how to combine its dependencies.
    #[error(
        "execution group slot {group} has {inputs} dependency outputs but no merge implementation"
    )]
    UnmergedExecutionGroupInputs {
        /// Architecture group slot.
        group: usize,
        /// Number of ready dependency outputs.
        inputs: usize,
    },
    /// Adapter and configured execution-group counts differ.
    #[error("adapter declares {adapter} execution groups but {configured} were configured")]
    ExecutionGroupCount {
        /// Adapter-declared count.
        adapter: usize,
        /// Configured count.
        configured: usize,
    },
    /// Adapter and configured group identities differ at one stable slot.
    #[error("execution group slot {slot} is {configured:?}, expected adapter group {adapter:?}")]
    ExecutionGroupIdentity {
        /// Architecture group slot.
        slot: usize,
        /// Adapter-declared identity.
        adapter: String,
        /// Configured residency identity.
        configured: String,
    },
    /// Adapter and configured unit counts differ for one execution group.
    #[error("execution group {group:?} has {configured} configured units but adapter declares {adapter}")]
    ExecutionGroupLength {
        /// Group id.
        group: String,
        /// Adapter-declared count.
        adapter: usize,
        /// Configured count.
        configured: usize,
    },
    /// A requested execution group does not exist.
    #[error("unknown resident execution group {0:?}")]
    UnknownExecutionGroup(String),
    /// The configured ordered layer window was invalid.
    #[error("device layer window depth {depth} must be between 1 and layer count {layer_count}")]
    InvalidLayerWindow {
        /// Requested depth.
        depth: usize,
        /// Decoder layer count.
        layer_count: usize,
    },
    /// The protected host lookahead exceeds an execution group.
    #[error("host layer window depth {depth} must be between 1 and layer count {layer_count}")]
    InvalidHostLayerWindow {
        /// Requested depth.
        depth: usize,
        /// Available ordered units.
        layer_count: usize,
    },
    /// A dense transfer window contained an invalid or unordered unit index.
    #[error(
        "dense transfer window index {index} is out of order or outside {unit_count} planned units"
    )]
    InvalidDenseTransferWindow {
        /// Invalid unit index.
        index: usize,
        /// Available units.
        unit_count: usize,
    },
    /// Strict loading found unrelated checkpoint tensors.
    #[error("strict layerwise loading found unexpected checkpoint parameters: {unused:?}")]
    UnexpectedCheckpointParameters {
        /// Unexpected keys in stable order.
        unused: Vec<String>,
    },
    /// The host cannot retain every decoder layer.
    #[error("host budget {budget} bytes cannot contain all {required} decoder-weight bytes")]
    HostBudgetTooSmall {
        /// Required decoder bytes.
        required: u64,
        /// Configured host budget.
        budget: u64,
    },
    /// The device cannot contain static weights plus the configured window.
    #[error("device budget {budget} bytes cannot contain {static_bytes} static bytes plus the depth-{depth} layer window ({window_bytes} bytes, {required} total)")]
    DeviceBudgetTooSmall {
        /// Pinned static device bytes.
        static_bytes: u64,
        /// Largest consecutive window bytes.
        window_bytes: u64,
        /// Configured layer count.
        depth: usize,
        /// Total required parameter bytes.
        required: u64,
        /// Configured device budget.
        budget: u64,
    },
    /// A cache vector had the wrong number of layers.
    #[error("layerwise cache has {actual} layers, expected {expected}")]
    CacheLengthMismatch {
        /// Model decoder count.
        expected: usize,
        /// Supplied cache count.
        actual: usize,
    },
    /// A cache entry was absent.
    #[error("layerwise cache entry {index} is missing")]
    MissingLayerCache {
        /// Missing decoder index.
        index: usize,
    },
    /// Checked byte or index arithmetic overflowed.
    #[error("layerwise model arithmetic overflow: {context}")]
    ArithmeticOverflow {
        /// Failed calculation.
        context: &'static str,
    },
    /// Module checkpoint binding failed.
    #[error(transparent)]
    ModuleBinding(#[from] ModuleBindingError),
    /// Residency execution failed.
    #[error(transparent)]
    Residency(#[from] ResidencyError),
}

#[cfg(test)]
mod tests {
    #[test]
    fn weight_residency_decomposes_layers_and_experts() {
        let host = LayerwiseLoadOptions::default();
        let experts = crate::runtime::residency::expert_cache::ExpertCacheLoadOptions::default();
        let residency = WeightResidency::with_expert_cache(
            NonExpertWeightResidency::LayerwiseHost(host),
            experts,
        );

        assert_eq!(
            residency.layers(),
            LayerWeightResidency::LayerwiseHost(host)
        );
        assert_eq!(
            residency.experts(),
            ExpertWeightResidency::IndependentCache(experts)
        );
        assert_eq!(residency.expert_cache(), Some(experts));
        assert_eq!(
            residency.non_experts(),
            Some(NonExpertWeightResidency::LayerwiseHost(host))
        );
        assert!(!residency.is_fully_resident());
        assert!(!residency.non_experts_are_fully_resident());

        let resident_non_experts =
            WeightResidency::with_expert_cache(NonExpertWeightResidency::FullyResident, experts);
        assert_eq!(
            resident_non_experts.layers(),
            LayerWeightResidency::FullyResident
        );
        assert_eq!(
            resident_non_experts.experts(),
            ExpertWeightResidency::IndependentCache(experts)
        );
        assert!(resident_non_experts.non_experts_are_fully_resident());
        assert!(!resident_non_experts.is_fully_resident());

        let resident = WeightResidency::fully_resident();
        assert_eq!(resident.layers(), LayerWeightResidency::FullyResident);
        assert_eq!(resident.experts(), ExpertWeightResidency::WithLayer);
        assert!(resident.is_fully_resident());
        assert!(resident.non_experts_are_fully_resident());
        assert_eq!(resident.expert_cache(), None);
        assert_eq!(resident.non_experts(), None);
    }

    use std::fs;

    use safemlx::{
        module::{Module, ModuleParameters},
        ops::ones_dtype,
        Device, DeviceType, ExecutionContext,
    };
    use safetensors::tensor::{serialize_to_file, Dtype as SafeDtype, TensorView};

    use super::*;
    use crate::{
        architectures::llama::layerwise::{
            load_llama_model, LlamaCache, LlamaLoadOptions, LlamaModel,
        },
        architectures::llama::model::{self as llama, ModelArgs},
        runtime::residency::manager::UnitResidencyReport,
        runtime::residency::policy::TransferDirection,
    };

    #[test]
    fn execution_group_dag_orders_branching_multimodal_ingress() {
        let graph = ExecutionGroupDag::new(
            vec![
                ExecutionGroupSpec::with_dependencies(
                    "text_decoder",
                    ["vision_encoder", "audio_encoder"],
                ),
                ExecutionGroupSpec::root("vision_encoder"),
                ExecutionGroupSpec::root("audio_encoder"),
            ],
            "text_decoder",
        )
        .unwrap();

        assert_eq!(graph.execution_order(), &[1, 2, 0]);
        assert_eq!(graph.dependencies(0), Some([1, 2].as_slice()));
        assert_eq!(graph.dependencies(1), Some([].as_slice()));
        assert_eq!(graph.consumer_counts(), [0, 1, 1]);
        assert_eq!(graph.output(), 0);
        assert_eq!(graph.groups()[graph.output()].id(), "text_decoder");
    }

    #[test]
    fn execution_group_dag_rejects_ambiguous_or_invalid_topology() {
        let duplicate = ExecutionGroupDag::new(
            vec![
                ExecutionGroupSpec::root("text"),
                ExecutionGroupSpec::root("text"),
            ],
            "text",
        )
        .unwrap_err();
        assert!(matches!(
            duplicate,
            Error::LayerwiseModel(LayerwiseModelError::DuplicateExecutionGroup(_))
        ));

        let unknown = ExecutionGroupDag::new(
            vec![ExecutionGroupSpec::with_dependencies("text", ["vision"])],
            "text",
        )
        .unwrap_err();
        assert!(matches!(
            unknown,
            Error::LayerwiseModel(LayerwiseModelError::UnknownExecutionGroupDependency { .. })
        ));

        let cycle = ExecutionGroupDag::new(
            vec![
                ExecutionGroupSpec::with_dependencies("vision", ["text"]),
                ExecutionGroupSpec::with_dependencies("text", ["vision"]),
            ],
            "text",
        )
        .unwrap_err();
        assert!(matches!(
            cycle,
            Error::LayerwiseModel(LayerwiseModelError::CyclicExecutionGraph)
        ));

        let disconnected = ExecutionGroupDag::new(
            vec![
                ExecutionGroupSpec::root("text"),
                ExecutionGroupSpec::root("unused_vision"),
            ],
            "text",
        )
        .unwrap_err();
        assert!(matches!(
            disconnected,
            Error::LayerwiseModel(LayerwiseModelError::DisconnectedExecutionGroups { .. })
        ));
    }

    fn load_layerwise_llama(
        model_dir: impl AsRef<Path>,
        offload: OffloadConfig,
        stream: &Stream,
        weights_stream: &Stream,
    ) -> Result<LlamaModel, Error> {
        load_llama_model(
            model_dir,
            LlamaLoadOptions::layerwise_host(LayerwiseLoadOptions::new(offload)),
            stream,
            weights_stream,
        )
    }

    fn args(model_type: &str, tied: bool, sliding_window: Option<i32>) -> ModelArgs {
        ModelArgs {
            model_type: model_type.into(),
            hidden_size: 8,
            num_hidden_layers: 3,
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
            attention_bias: true,
            mlp_bias: true,
            rope_scaling: None,
            attention_schedule: match sliding_window {
                Some(window) => crate::runtime::attention::LayerSchedule::all_sliding(
                    3,
                    u32::try_from(window).unwrap(),
                )
                .unwrap(),
                None => crate::runtime::attention::LayerSchedule::all_full(3).unwrap(),
            },
            quantization: None,
            quantization_config: None,
            quantized_weights: None,
            quantized_weight_configs: None,
        }
    }

    #[test]
    fn execution_group_binding_sharding_preserves_direct_and_derived_selections() {
        use crate::runtime::distributed::{
            parallel::{
                MemberSharding, ParallelPlanBuilder, ParameterGroupSpec, ParameterMemberSpec,
                ParameterRole,
            },
            topology::{DeviceAssignment, ParallelTopology},
        };
        use crate::runtime::{
            checkpoint::{recipe::DerivedWeightRecipe, store::TensorSelection},
            residency::manager::WeightBinding,
        };
        let topology =
            ParallelTopology::from_rank(2, 1, 2, 1, 1, DeviceAssignment::new(DeviceType::Cpu, 0))
                .unwrap();
        let mut planner = ParallelPlanBuilder::new(topology);
        planner
            .register(
                ParameterGroupSpec::new(
                    "group.projection",
                    ParameterRole::ColumnProjection,
                    [ParameterMemberSpec::new(
                        "stack.0.projection.weight",
                        [8, 4],
                        MemberSharding::Equal { axis: 0 },
                    )],
                )
                .unwrap(),
            )
            .unwrap();
        planner
            .register(
                ParameterGroupSpec::new(
                    "group.physical_source",
                    ParameterRole::Replicated,
                    [ParameterMemberSpec::new(
                        "raw.weight",
                        [8, 4],
                        MemberSharding::Replicated,
                    )],
                )
                .unwrap(),
            )
            .unwrap();
        let (_, layout) = planner.finish().unwrap();
        let checkpoint = tempfile::tempdir().unwrap();
        let raw = vec![0u8; 8 * 4 * std::mem::size_of::<f32>()];
        serialize_to_file(
            [(
                "raw.weight",
                TensorView::new(SafeDtype::F32, vec![8, 4], &raw).unwrap(),
            )],
            None,
            &checkpoint.path().join("model.safetensors"),
        )
        .unwrap();
        let store = SafetensorsWeightStore::open(checkpoint.path()).unwrap();
        let direct = WeightBinding::new(
            "projection.weight",
            "stack.0.projection.weight",
            TensorSelection::Full,
            128,
        )
        .unwrap();
        let derived = WeightBinding::from_recipe(
            "checkpoint_alias.weight",
            DerivedWeightRecipe::source("raw.weight", TensorSelection::Full),
            128,
        )
        .unwrap()
        .with_logical_target("stack.0.projection.weight")
        .unwrap();
        let direct = shard_layer_bindings(vec![direct], "stack.0", &store, &layout).unwrap();
        assert_eq!(
            direct[0].selection(),
            &TensorSelection::Range {
                axis: 0,
                start: 4,
                end: 8,
            }
        );
        assert_eq!(direct[0].expected_bytes(), 64);
        let derived = shard_layer_bindings(vec![derived], "stack.0", &store, &layout).unwrap();
        assert_eq!(derived[0].expected_bytes(), 64);
        assert!(matches!(
            derived[0].recipe(),
            Some(DerivedWeightRecipe::Source {
                key,
                selection: TensorSelection::Range {
                    axis: 0,
                    start: 4,
                    end: 8,
                },
            }) if key == "raw.weight"
        ));
    }

    fn initialize(module: &mut impl ModuleParameters, stream: &Stream) {
        let mut names = module
            .parameters()
            .flatten()
            .keys()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        names.sort();
        let mut params = module.parameters_mut().flatten();
        for (index, name) in names.iter().enumerate() {
            let parameter = params.get_mut(name.as_str()).unwrap();
            let shape = parameter.shape().to_vec();
            let dtype = parameter.dtype();
            **parameter = if name.ends_with("layernorm.weight") || name == "model.norm.weight" {
                ones_dtype(&shape, dtype, stream).unwrap()
            } else {
                Array::full::<f32>(&shape, Array::from_f32(0.0025 * (index + 1) as f32), stream)
                    .unwrap()
                    .as_dtype(dtype, stream)
                    .unwrap()
            };
        }
    }

    fn write_fixture(dir: &Path, model: &llama::ResidentModel) {
        let params = model.parameters().flatten();
        let arrays = params
            .iter()
            .map(|(name, value)| {
                (
                    crate::runtime::checkpoint::binding::canonical_checkpoint_name(name),
                    *value,
                )
            })
            .collect::<Vec<_>>();
        Array::save_safetensors(
            arrays.iter().map(|(name, value)| (name.as_str(), *value)),
            None,
            dir.join("model.safetensors"),
        )
        .unwrap();
        let mut config = serde_json::json!({
            "model_type": model.args.model_type,
            "hidden_size": model.args.hidden_size,
            "num_hidden_layers": model.args.num_hidden_layers,
            "intermediate_size": model.args.intermediate_size,
            "num_attention_heads": model.args.num_attention_heads,
            "num_key_value_heads": model.args.num_key_value_heads,
            "rms_norm_eps": model.args.rms_norm_eps,
            "vocab_size": model.args.vocab_size,
            "max_position_embeddings": model.args.max_position_embeddings,
            "rope_theta": model.args.rope_theta,
            "rope_traditional": model.args.rope_traditional,
            "head_dim": model.args.head_dim,
            "tie_word_embeddings": model.args.tie_word_embeddings,
            "attention_bias": model.args.attention_bias,
            "mlp_bias": model.args.mlp_bias
        });
        if let Some(window) = model
            .args
            .attention_schedule
            .get(0)
            .and_then(|policy| policy.window())
        {
            config["sliding_window"] = window.get().into();
        }
        fs::write(
            dir.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
    }

    fn assert_close(left: &Array, right: &Array) {
        let left = left.evaluated().unwrap();
        let right = right.evaluated().unwrap();
        assert_eq!(left.as_array().shape(), right.as_array().shape());
        for (left, right) in left.as_slice::<f32>().iter().zip(right.as_slice::<f32>()) {
            assert!((left - right).abs() <= 2e-5, "{left} != {right}");
        }
    }

    fn layer_reports(report: &ResidencyReport) -> Vec<&UnitResidencyReport> {
        report
            .units()
            .iter()
            .filter(|unit| unit.id().as_str().starts_with("llama.layer."))
            .collect()
    }

    fn run_parity(model_type: &str, tied: bool, sliding_window: Option<i32>, depth: usize) {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = gpu.stream();
        let mut reference =
            llama::ResidentModel::new(args(model_type, tied, sliding_window), stream).unwrap();
        initialize(&mut reference, stream);
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &reference);

        let mut fully_resident = load_llama_model(
            dir.path(),
            LlamaLoadOptions::fully_resident(),
            stream,
            cpu.stream(),
        )
        .unwrap();
        assert!(fully_resident.is_fully_resident());
        let resident_report = fully_resident.residency_report().unwrap().unwrap();
        assert!(layer_reports(&resident_report)
            .iter()
            .all(|unit| unit.device_resident() && !unit.host_resident()));
        let config = OffloadConfig::new(None, None, depth).unwrap();
        let mut offloaded = load_layerwise_llama(dir.path(), config, stream, cpu.stream()).unwrap();
        assert!(!offloaded.is_fully_resident());
        let initial = offloaded.residency_report().unwrap().unwrap();
        assert!(layer_reports(&initial)
            .iter()
            .all(|unit| unit.host_resident()));
        assert!(layer_reports(&initial)
            .iter()
            .all(|unit| !unit.device_resident()));

        let mut resident_cache = fully_resident.new_cache();
        let mut cache = offloaded.new_cache();
        for tokens in [
            Array::from_slice(&[1u32, 2], &[1, 2]),
            Array::from_slice(&[3u32], &[1, 1]),
            Array::from_slice(&[4u32], &[1, 1]),
            Array::from_slice(&[5u32], &[1, 1]),
        ] {
            let expected = fully_resident
                .forward(&tokens, &mut resident_cache, stream)
                .unwrap();
            let actual = offloaded.forward(&tokens, &mut cache, stream).unwrap();
            assert_close(&actual, &expected);
            let report = offloaded.residency_report().unwrap().unwrap();
            assert!(layer_reports(&report)
                .iter()
                .all(|unit| unit.host_resident()));
            assert!(
                layer_reports(&report)
                    .iter()
                    .filter(|unit| unit.device_resident())
                    .count()
                    <= depth
            );
        }

        let report = offloaded.residency_report().unwrap().unwrap();
        assert!(
            report
                .offload()
                .transfer(TransferDirection::HostToDevice)
                .count()
                >= 3
        );
        assert_eq!(offloaded.static_lease_count(), if tied { 2 } else { 3 });
        offloaded.clear_device_layer_window().unwrap();
        let cleared = offloaded.residency_report().unwrap().unwrap();
        assert!(layer_reports(&cleared)
            .iter()
            .all(|unit| !unit.device_resident()));
        assert!(cleared
            .units()
            .iter()
            .filter(|unit| unit.device_resident())
            .all(|unit| unit.policy() == ResidencyPolicy::Pinned));
    }

    #[test]
    fn llama_residency_dense_prefill_decode_parity() {
        run_parity("llama", true, None, 1);
        run_parity("llama", false, None, 2);
        run_parity("mistral", false, Some(4), 2);
    }

    #[test]
    fn arbitrary_llama_schedule_resident_layerwise_parity() {
        use crate::runtime::attention::{AttentionPolicy, LayerSchedule};

        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = gpu.stream();
        let mut model_args = args("mistral", false, None);
        model_args.attention_schedule = LayerSchedule::new(
            3,
            vec![
                AttentionPolicy::sliding(3).unwrap(),
                AttentionPolicy::Full,
                AttentionPolicy::sliding(5).unwrap(),
            ],
        )
        .unwrap();
        let mut resident = llama::ResidentModel::new(model_args.clone(), stream).unwrap();
        initialize(&mut resident, stream);
        let mut resident_cache = resident.new_cache();
        let mut layerwise_cache = LlamaCache::Device(resident.new_cache());
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &resident);
        let adapter =
            crate::architectures::llama::layerwise::LlamaLayerwiseAdapter::new(model_args, stream)
                .unwrap();
        let mut layerwise = load_safetensors_layerwise_model(
            dir.path(),
            adapter,
            LayerwiseLoadOptions::new(OffloadConfig::new(None, None, 2).unwrap()),
            stream,
            cpu.stream(),
        )
        .unwrap();

        for tokens in [
            Array::from_slice(&[1u32, 2, 3], &[1, 3]),
            Array::from_slice(&[4u32, 5], &[1, 2]),
            Array::from_slice(&[6u32], &[1, 1]),
        ] {
            let expected = resident
                .forward(
                    llama::ModelInput {
                        inputs: &tokens,
                        mask: None,
                        cache: &mut resident_cache,
                    },
                    stream,
                )
                .unwrap();
            let actual = layerwise
                .forward(
                    crate::architectures::llama::layerwise::LlamaAdapterInput {
                        inputs: &tokens,
                        mask: None,
                    },
                    &mut layerwise_cache,
                    stream,
                )
                .unwrap();
            assert_close(&actual, &expected);
        }
    }

    #[test]
    fn dense_stream_keeps_layers_cold_and_matches_cached_decode() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut reference =
            llama::ResidentModel::new(args("llama", true, None), gpu.stream()).unwrap();
        initialize(&mut reference, gpu.stream());
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &reference);

        let sizing = load_layerwise_llama(
            dir.path(),
            OffloadConfig::new(None, None, 1).unwrap(),
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let metadata = sizing.metadata();
        let device_budget = metadata
            .static_device_bytes()
            .checked_add(
                metadata
                    .maximum_device_layer_bytes()
                    .checked_mul(DENSE_TRANSFER_WINDOW as u64)
                    .unwrap(),
            )
            .unwrap();
        let host_budget = metadata.maximum_device_layer_bytes();
        drop(sizing);

        let options = DenseDiskStreamLoadOptions::new(device_budget, host_budget, 1, 1).unwrap();
        let mut streamed = load_llama_model(
            dir.path(),
            LlamaLoadOptions::dense_disk_stream(options),
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let initial = streamed.dense_stream_report().unwrap().unwrap();
        assert_ne!(
            initial.transfer_stream_index(),
            gpu.stream().get_index().unwrap()
        );
        assert_eq!(initial.planned_layer_count(), 3);
        assert_eq!(initial.host_layers().current_layer_count(), 0);
        assert_eq!(initial.device_layers().current_layer_count(), 0);
        assert_eq!(initial.execution_groups().len(), 1);
        assert_eq!(initial.execution_groups()[0].id(), "text_decoder");
        assert_eq!(initial.execution_groups()[0].planned_layers(), 3);
        assert_eq!(initial.execution_groups()[0].completed_executions(), 0);
        assert!(initial
            .residency()
            .units()
            .iter()
            .filter(|unit| unit.id().as_str().starts_with("llama.layer."))
            .all(|unit| {
                unit.planned_tier() == MemoryTier::Disk
                    && !unit.host_resident()
                    && !unit.device_resident()
            }));

        let mut resident = load_llama_model(
            dir.path(),
            LlamaLoadOptions::fully_resident(),
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let mut expected_cache = resident.new_cache();
        let mut actual_cache = streamed.new_cache();
        for tokens in [
            Array::from_slice(&[1u32, 2], &[1, 2]),
            Array::from_slice(&[3u32], &[1, 1]),
            Array::from_slice(&[4u32], &[1, 1]),
        ] {
            let expected = resident
                .forward(&tokens, &mut expected_cache, gpu.stream())
                .unwrap();
            let actual = streamed
                .forward(&tokens, &mut actual_cache, gpu.stream())
                .unwrap();
            assert_close(&actual, &expected);
            let report = streamed.dense_stream_report().unwrap().unwrap();
            assert!(
                report
                    .residency()
                    .offload()
                    .resident_bytes()
                    .get(MemoryTier::Host)
                    <= host_budget
            );
            assert!(
                report
                    .residency()
                    .offload()
                    .resident_bytes()
                    .get(MemoryTier::Device)
                    <= device_budget
            );
        }
        let report = streamed.dense_stream_report().unwrap().unwrap();
        assert!(report.background().submitted >= 3);
        assert!(report.host_layers().current_layer_count() <= 1);
        assert!(report.device_layers().current_layer_count() <= DENSE_TRANSFER_WINDOW);
        assert_eq!(report.host_layers().peak_layer_count(), 1);
        assert_eq!(
            report.device_layers().peak_layer_count(),
            DENSE_TRANSFER_WINDOW
        );
        assert!(report.host_layers().peak_layer_bytes() <= host_budget);
        assert!(
            report.device_layers().peak_layer_bytes()
                <= device_budget - report.pinned_static_device_bytes()
        );
        assert_eq!(
            report.host_layers().cache().requests(),
            report.host_layers().cache().hits() + report.host_layers().cache().misses()
        );
        assert_eq!(
            report.device_layers().cache().requests(),
            report.device_layers().cache().hits() + report.device_layers().cache().misses()
        );
        assert_eq!(report.prefill().forwards(), 1);
        assert_eq!(report.decode().forwards(), 2);
        assert_eq!(report.prefill().peak_host_layers(), 1);
        assert_eq!(report.prefill().peak_device_layers(), DENSE_TRANSFER_WINDOW);
        assert_eq!(report.decode().peak_host_layers(), 1);
        assert_eq!(report.decode().peak_device_layers(), DENSE_TRANSFER_WINDOW);
        assert!(report.prefill().peak_host_bytes() <= host_budget);
        assert!(report.prefill().peak_device_bytes() <= report.device_layers().peak_layer_bytes());
        assert!(report.prefill().host_cache().requests() >= 3);
        assert!(report.prefill().host_to_device_bytes() > 0);
        assert!(report.decode().host_cache().requests() >= 6);
        assert!(report.decode().host_to_device_bytes() > 0);
        let group = &report.execution_groups()[0];
        assert_eq!(group.completed_executions(), 3);
        assert_eq!(group.peak_host_layers(), 1);
        assert_eq!(group.peak_device_layers(), DENSE_TRANSFER_WINDOW);
        assert!(group.peak_host_bytes() <= host_budget);
        assert!(group.peak_device_bytes() <= report.device_layers().peak_layer_bytes());
        assert!(
            report
                .residency()
                .offload()
                .transfer(TransferDirection::DiskToHost)
                .count()
                >= 3
        );

        let direct_options = DenseDiskStreamLoadOptions::new(device_budget, 0, 0, 0).unwrap();
        let mut direct = load_llama_model(
            dir.path(),
            LlamaLoadOptions::dense_disk_stream(direct_options),
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let mut direct_cache = direct.new_cache();
        let tokens = Array::from_slice(&[6u32, 7], &[1, 2]);
        direct
            .forward(&tokens, &mut direct_cache, gpu.stream())
            .unwrap();
        let report = direct.dense_stream_report().unwrap().unwrap();
        assert_eq!(report.host_layers().peak_layer_count(), 0);
        assert_eq!(
            report.device_layers().peak_layer_count(),
            DENSE_TRANSFER_WINDOW
        );
        assert_eq!(report.prefill().forwards(), 1);
        assert!(report.prefill().disk_to_device_bytes() > 0);
        assert_eq!(report.prefill().host_to_device_bytes(), 0);
        assert_eq!(
            report
                .residency()
                .offload()
                .resident_bytes()
                .get(MemoryTier::Host),
            0
        );
        assert!(
            report
                .residency()
                .offload()
                .transfer(TransferDirection::DiskToDevice)
                .count()
                >= 3
        );
    }

    #[test]
    fn aborted_dense_forward_does_not_contaminate_completed_pass_telemetry() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut reference =
            llama::ResidentModel::new(args("llama", true, None), gpu.stream()).unwrap();
        initialize(&mut reference, gpu.stream());
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &reference);

        let sizing = load_layerwise_llama(
            dir.path(),
            OffloadConfig::new(None, None, 1).unwrap(),
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let metadata = sizing.metadata();
        let device_budget = metadata
            .static_device_bytes()
            .checked_add(
                metadata
                    .maximum_device_layer_bytes()
                    .checked_mul(DENSE_TRANSFER_WINDOW as u64)
                    .unwrap(),
            )
            .unwrap();
        let host_budget = metadata.maximum_device_layer_bytes();
        drop(sizing);

        let options = DenseDiskStreamLoadOptions::new(device_budget, host_budget, 1, 1).unwrap();
        let adapter = crate::architectures::llama::layerwise::LlamaLayerwiseAdapter::new(
            args("llama", true, None),
            gpu.stream(),
        )
        .unwrap();
        let mut streamed = load_safetensors_layerwise_model(
            dir.path(),
            adapter,
            options,
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();

        {
            let controller = streamed.dense_stream.as_ref().unwrap();
            let forward = controller.forward_guard(true, &streamed.residency).unwrap();
            let layer_group = &streamed.groups[0];
            let group = controller.group_guard(&streamed.residency, layer_group.id());
            {
                let _window = controller
                    .transfer_window(
                        &streamed.residency,
                        layer_group.id(),
                        layer_group.units(),
                        0..layer_group.units().len(),
                        true,
                    )
                    .unwrap();
            }
            drop(group);
            drop(forward);
        }

        let after_abort = streamed.dense_stream_report().unwrap().unwrap();
        assert_eq!(after_abort.prefill(), DensePassReport::default());
        assert_eq!(after_abort.decode(), DensePassReport::default());
        let successful_start = DenseCounterSnapshot::from_report(after_abort.residency().offload());

        let tokens = Array::from_slice(&[1u32], &[1, 1]);
        let mut cache = LlamaCache::Device(Vec::new());
        streamed
            .forward(
                crate::architectures::llama::layerwise::LlamaAdapterInput {
                    inputs: &tokens,
                    mask: None,
                },
                &mut cache,
                gpu.stream(),
            )
            .unwrap();

        let report = streamed.dense_stream_report().unwrap().unwrap();
        assert_eq!(report.prefill(), DensePassReport::default());
        assert_eq!(report.decode().forwards(), 1);
        assert_eq!(report.decode().peak_host_layers(), 1);
        assert_eq!(report.decode().peak_device_layers(), DENSE_TRANSFER_WINDOW);
        let expected =
            DenseCounterSnapshot::from_report(report.residency().offload()).delta(successful_start);
        assert_eq!(report.decode().host_cache(), expected.host_cache);
        assert_eq!(report.decode().device_cache(), expected.device_cache);
        assert_eq!(
            report.decode().disk_to_host_bytes(),
            expected.disk_to_host_bytes
        );
        assert_eq!(
            report.decode().disk_to_device_bytes(),
            expected.disk_to_device_bytes
        );
        assert_eq!(
            report.decode().host_to_device_bytes(),
            expected.host_to_device_bytes
        );
    }

    #[test]
    fn budget_and_cache_validation_are_structured() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut reference =
            llama::ResidentModel::new(args("llama", true, None), gpu.stream()).unwrap();
        initialize(&mut reference, gpu.stream());
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &reference);

        let host_error = load_layerwise_llama(
            dir.path(),
            OffloadConfig::new(None, Some(1), 1).unwrap(),
            gpu.stream(),
            cpu.stream(),
        )
        .err()
        .unwrap();
        assert!(host_error.to_string().contains("host budget"));

        let device_error = load_layerwise_llama(
            dir.path(),
            OffloadConfig::new(Some(1), None, 1).unwrap(),
            gpu.stream(),
            cpu.stream(),
        )
        .err()
        .unwrap();
        assert!(device_error.to_string().contains("device budget"));

        let mut model = load_layerwise_llama(
            dir.path(),
            OffloadConfig::new(None, None, 1).unwrap(),
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let mut bad_cache = LlamaCache::Device(vec![None]);
        let error = model
            .forward(
                &Array::from_slice(&[1u32], &[1, 1]),
                &mut bad_cache,
                gpu.stream(),
            )
            .unwrap_err();
        assert!(error.to_string().contains("cache has 1 layers"));
    }

    #[test]
    fn llama_residency_packed_affine_parity() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut quant_args = args("llama", false, None);
        quant_args.hidden_size = 32;
        quant_args.intermediate_size = 64;
        quant_args.num_attention_heads = 4;
        quant_args.num_key_value_heads = 2;
        quant_args.head_dim = 8;
        quant_args.vocab_size = 32;
        quant_args.num_hidden_layers = 2;
        let mut dense = llama::ResidentModel::new(quant_args, gpu.stream()).unwrap();
        initialize(&mut dense, gpu.stream());
        let source = tempfile::tempdir().unwrap();
        write_fixture(source.path(), &dense);

        let converted_root = tempfile::tempdir().unwrap();
        let converted = converted_root.path().join("affine");
        let options = crate::runtime::checkpoint::quantization::CheckpointQuantizationOptions {
            quantization: crate::runtime::checkpoint::quantization::AffineQuantization::new(32, 4)
                .unwrap()
                .into(),
            ..Default::default()
        };
        crate::runtime::checkpoint::quantization::quantize_checkpoint(
            source.path(),
            &converted,
            &options,
            gpu.stream(),
        )
        .unwrap();

        let mut resident = load_llama_model(
            &converted,
            LlamaLoadOptions::fully_resident(),
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let mut offloaded = load_layerwise_llama(
            &converted,
            OffloadConfig::new(None, None, 1).unwrap(),
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        assert!(offloaded.metadata().quantization().is_some());
        let mut resident_cache = resident.new_cache();
        let mut offloaded_cache = offloaded.new_cache();
        for tokens in [
            Array::from_slice(&[1u32, 2], &[1, 2]),
            Array::from_slice(&[3u32], &[1, 1]),
            Array::from_slice(&[4u32], &[1, 1]),
        ] {
            let expected = resident
                .forward(&tokens, &mut resident_cache, gpu.stream())
                .unwrap();
            let actual = offloaded
                .forward(&tokens, &mut offloaded_cache, gpu.stream())
                .unwrap();
            assert_close(&actual, &expected);
        }
    }
}
