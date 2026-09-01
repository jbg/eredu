//! Backend-neutral immutable-weight residency policy.

use std::collections::VecDeque;

use eredu_core::{
    residency::{CacheEvictionPolicy, OffloadConfig, OffloadError, OffloadUnitId},
    DEFAULT_MAX_CACHED_SHARDS,
};

use crate::WeightBinding;

/// Current plus next unit retained by dense streamed execution.
pub const DENSE_TRANSFER_WINDOW: usize = 2;

/// One pinned static module and its checkpoint bindings.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct StaticUnitBindings {
    id: OffloadUnitId,
    bindings: Vec<WeightBinding>,
}

impl StaticUnitBindings {
    /// Creates a pinned static unit definition.
    pub fn new(id: impl Into<String>, bindings: Vec<WeightBinding>) -> Result<Self, OffloadError> {
        Ok(Self {
            id: OffloadUnitId::new(id.into())?,
            bindings,
        })
    }

    /// Returns the stable residency identifier for this static module.
    pub const fn id(&self) -> &OffloadUnitId {
        &self.id
    }

    /// Returns the authoritative checkpoint bindings for this static module.
    pub fn bindings(&self) -> &[WeightBinding] {
        &self.bindings
    }

    /// Consumes the definition into its stable identifier and checkpoint bindings.
    pub fn into_parts(self) -> (OffloadUnitId, Vec<WeightBinding>) {
        (self.id, self.bindings)
    }
}

/// Ordered bounded transfer cursor used by dense streamed execution.
#[derive(Debug)]
pub struct DenseTransferSchedule<T> {
    capacity: usize,
    pending: VecDeque<usize>,
    ready: VecDeque<(usize, T)>,
}

impl<T> DenseTransferSchedule<T> {
    /// Creates a cursor over the remaining unit indices.
    pub fn new(
        pending: impl IntoIterator<Item = usize>,
        capacity: usize,
    ) -> Result<Self, DenseTransferScheduleError> {
        if capacity == 0 {
            return Err(DenseTransferScheduleError::ZeroCapacity);
        }
        let pending = pending.into_iter().collect::<VecDeque<_>>();
        let mut previous = None;
        for &index in &pending {
            if previous.is_some_and(|previous| previous >= index) {
                return Err(DenseTransferScheduleError::UnorderedPending {
                    previous: previous.expect("an invalid pair has a previous index"),
                    actual: index,
                });
            }
            previous = Some(index);
        }
        Ok(Self {
            capacity,
            pending,
            ready: VecDeque::new(),
        })
    }

    /// Returns whether at least one submitted transfer is ready for consumption.
    pub fn has_ready(&self) -> bool {
        !self.ready.is_empty()
    }

    /// Returns whether no ready or pending units remain.
    pub fn is_exhausted(&self) -> bool {
        self.ready.is_empty() && self.pending.is_empty()
    }

    /// Returns whether another transfer may be admitted into the bounded ready window.
    pub fn can_admit(&self) -> bool {
        self.ready.len() < self.capacity && !self.pending.is_empty()
    }

    /// Returns the next unit which must be submitted.
    pub fn next_pending(&self) -> Option<usize> {
        self.pending.front().copied()
    }

    /// Returns the current ready indices followed by future indices, truncated to lookahead.
    pub fn desired_indices(&self, lookahead: usize) -> Vec<usize> {
        self.ready
            .iter()
            .map(|(index, _)| *index)
            .chain(self.pending.iter().copied())
            .take(lookahead)
            .collect()
    }

    /// Commits one successfully submitted transfer in exact pending order.
    pub fn admit(&mut self, index: usize, transfer: T) -> Result<(), DenseTransferScheduleError> {
        if self.ready.len() >= self.capacity {
            return Err(DenseTransferScheduleError::CapacityExceeded {
                capacity: self.capacity,
            });
        }
        let expected = self
            .pending
            .front()
            .copied()
            .ok_or(DenseTransferScheduleError::NoPendingUnit)?;
        if expected != index {
            return Err(DenseTransferScheduleError::OutOfOrder {
                expected,
                actual: index,
            });
        }
        self.pending.pop_front();
        self.ready.push_back((index, transfer));
        Ok(())
    }

    /// Removes the oldest ready transfer for execution.
    pub fn pop_ready(&mut self) -> Option<(usize, T)> {
        self.ready.pop_front()
    }
}

/// Invalid transition in a bounded dense transfer schedule.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum DenseTransferScheduleError {
    /// A transfer window cannot have zero capacity.
    #[error("dense transfer window capacity must be nonzero")]
    ZeroCapacity,
    /// Pending units were not supplied in strictly increasing order.
    #[error("dense transfer pending unit {actual} does not follow {previous}")]
    UnorderedPending {
        /// Previous index.
        previous: usize,
        /// Invalid next index.
        actual: usize,
    },
    /// Admission was attempted while the ready window was full.
    #[error("dense transfer window exceeds its capacity of {capacity}")]
    CapacityExceeded {
        /// Configured ready capacity.
        capacity: usize,
    },
    /// Admission was attempted after all pending units were submitted.
    #[error("dense transfer schedule has no pending unit")]
    NoPendingUnit,
    /// A backend submitted transfers in a different order from the architecture sequence.
    #[error("dense transfer schedule expected unit {expected}, received {actual}")]
    OutOfOrder {
        /// Required next index.
        expected: usize,
        /// Submitted index.
        actual: usize,
    },
}

/// Loader controls for a host-backed layerwise execution engine.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct LayerwiseLoadOptions {
    /// Residency budgets and maximum device-unit window.
    offload: OffloadConfig,
    /// Maximum number of checkpoint payload shards or readers retained in cache.
    max_cached_shards: usize,
    /// Sample backend allocator memory when a forward pass completes.
    sample_backend_memory: bool,
    /// Sample process memory metrics when a forward pass completes.
    sample_process_memory: bool,
}

impl LayerwiseLoadOptions {
    /// Creates layerwise options with the default shard-cache bound.
    pub fn new(offload: OffloadConfig) -> Self {
        Self {
            offload,
            ..Self::default()
        }
    }

    /// Selects the checkpoint-reader cache bound.
    pub const fn with_max_cached_shards(mut self, maximum: usize) -> Self {
        self.max_cached_shards = maximum;
        self
    }
    /// Selects allocator and process memory sampling.
    pub const fn with_memory_sampling(mut self, backend: bool, process: bool) -> Self {
        self.sample_backend_memory = backend;
        self.sample_process_memory = process;
        self
    }
    /// Returns the exact offload limits.
    pub const fn offload(self) -> OffloadConfig {
        self.offload
    }
    /// Returns the checkpoint-reader cache bound.
    pub const fn max_cached_shards(self) -> usize {
        self.max_cached_shards
    }
    /// Returns whether backend allocator sampling is enabled.
    pub const fn samples_backend_memory(self) -> bool {
        self.sample_backend_memory
    }
    /// Returns whether process memory sampling is enabled.
    pub const fn samples_process_memory(self) -> bool {
        self.sample_process_memory
    }
}

impl Default for LayerwiseLoadOptions {
    fn default() -> Self {
        Self {
            offload: OffloadConfig::default(),
            max_cached_shards: DEFAULT_MAX_CACHED_SHARDS,
            sample_backend_memory: false,
            sample_process_memory: false,
        }
    }
}

/// Controls for bounded dense-unit streaming from checkpoint storage.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct DenseDiskStreamLoadOptions {
    /// Finite logical device parameter budget, including pinned static weights.
    device_budget_bytes: u64,
    /// Finite charged host-allocation budget. Zero selects direct disk-to-device loading.
    host_budget_bytes: u64,
    /// Number of current and imminent unit host copies protected from eviction.
    host_lookahead: usize,
    /// Maximum number of pending background host materializations.
    background_queue_capacity: usize,
    /// Deterministic ordering used when unprotected cached copies must be evicted.
    eviction_policy: CacheEvictionPolicy,
    /// Maximum number of checkpoint payload shards or readers retained in cache.
    max_cached_shards: usize,
    /// Sample backend allocator memory after a forward pass.
    sample_backend_memory: bool,
    /// Sample process memory and page-fault counters after a forward pass.
    sample_process_memory: bool,
}

impl DenseDiskStreamLoadOptions {
    /// Creates streaming options with finite tier budgets.
    pub fn new(
        device_budget_bytes: u64,
        host_budget_bytes: u64,
        host_lookahead: usize,
        background_queue_capacity: usize,
    ) -> Result<Self, WeightResidencyPolicyError> {
        let options = Self {
            device_budget_bytes,
            host_budget_bytes,
            host_lookahead,
            background_queue_capacity,
            eviction_policy: CacheEvictionPolicy::LeastRecentlyUsed,
            max_cached_shards: DEFAULT_MAX_CACHED_SHARDS,
            sample_backend_memory: false,
            sample_process_memory: false,
        };
        options.validate()?;
        Ok(options)
    }

    /// Revalidates the complete bounded policy.
    pub fn validate(self) -> Result<(), WeightResidencyPolicyError> {
        if self.host_budget_bytes == 0 {
            if self.host_lookahead != 0 || self.background_queue_capacity != 0 {
                return Err(WeightResidencyPolicyError::HostDisabledControls);
            }
        } else {
            if self.host_lookahead == 0 {
                return Err(WeightResidencyPolicyError::ZeroHostLookahead);
            }
            if self.background_queue_capacity == 0 {
                return Err(WeightResidencyPolicyError::ZeroQueueCapacity);
            }
        }
        Ok(())
    }

    /// Selects deterministic cache eviction.
    pub const fn with_eviction_policy(mut self, policy: CacheEvictionPolicy) -> Self {
        self.eviction_policy = policy;
        self
    }

    /// Selects the checkpoint-reader cache bound.
    pub const fn with_max_cached_shards(mut self, maximum: usize) -> Self {
        self.max_cached_shards = maximum;
        self
    }
    /// Selects allocator and process memory sampling.
    pub const fn with_memory_sampling(mut self, backend: bool, process: bool) -> Self {
        self.sample_backend_memory = backend;
        self.sample_process_memory = process;
        self
    }
    /// Returns the finite logical device budget.
    pub const fn device_budget_bytes(self) -> u64 {
        self.device_budget_bytes
    }
    /// Returns the finite charged host budget.
    pub const fn host_budget_bytes(self) -> u64 {
        self.host_budget_bytes
    }
    /// Returns the protected host lookahead.
    pub const fn host_lookahead(self) -> usize {
        self.host_lookahead
    }
    /// Returns the background materialization queue bound.
    pub const fn background_queue_capacity(self) -> usize {
        self.background_queue_capacity
    }
    /// Returns deterministic eviction ordering.
    pub const fn eviction_policy(self) -> CacheEvictionPolicy {
        self.eviction_policy
    }
    /// Returns the checkpoint-reader cache bound.
    pub const fn max_cached_shards(self) -> usize {
        self.max_cached_shards
    }
    /// Returns whether backend allocator sampling is enabled.
    pub const fn samples_backend_memory(self) -> bool {
        self.sample_backend_memory
    }
    /// Returns whether process memory sampling is enabled.
    pub const fn samples_process_memory(self) -> bool {
        self.sample_process_memory
    }
}

impl Default for DenseDiskStreamLoadOptions {
    fn default() -> Self {
        Self::new(4 << 30, 16 << 30, 2, 2).expect("default dense disk streaming controls are valid")
    }
}

/// Placement policy for ordinary architecture execution units.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Default)]
#[non_exhaustive]
pub enum LayerWeightResidency {
    /// Construct every rank-local module once and retain it on the execution device.
    #[default]
    FullyResident,
    /// Eagerly materialize units on a host stream and use a bounded device window.
    LayerwiseHost(LayerwiseLoadOptions),
    /// Leave units cold on disk and use finite host and device caches.
    DenseDiskStream(DenseDiskStreamLoadOptions),
}

/// Stable mechanism identity for one member of an independently addressable bank.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ParameterBankKey {
    unit: usize,
    member: usize,
}

impl ParameterBankKey {
    /// Creates one generic bank member identity after semantic translation.
    pub const fn new(unit: usize, member: usize) -> Self {
        Self { unit, member }
    }

    /// Returns the owning execution-unit ordinal.
    pub const fn unit(self) -> usize {
        self.unit
    }

    /// Returns the member ordinal within the architecture's global bank namespace.
    pub const fn member(self) -> usize {
        self.member
    }

    /// Returns the deterministic residency unit identifier.
    pub fn unit_id(self) -> OffloadUnitId {
        OffloadUnitId::new(format!(
            "bank.unit.{:05}.member.{:05}",
            self.unit, self.member
        ))
        .expect("parameter-bank unit identifier is non-empty")
    }
}

/// Execution-path classification for routed-expert telemetry and chunking.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum ExpertPass {
    /// Prompt processing with more than one input token.
    Prefill,
    /// Autoregressive processing of one input token.
    Decode,
}

/// Backend-neutral controls for independently addressable parameter-bank residency.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct ParameterBankLoadOptions {
    /// Independent host/device budgets and eviction policy for bank members.
    members: OffloadConfig,
    /// Hard maximum bytes for one materialized temporary compact bank.
    compact_bank_scratch_bytes: u64,
    /// Soft compact-bank target used to split multi-token prefill routing.
    prefill_compact_bank_target_bytes: u64,
}

impl ParameterBankLoadOptions {
    /// Creates strict independently addressable bank-caching options.
    pub fn new(
        members: OffloadConfig,
        compact_bank_scratch_bytes: u64,
        prefill_compact_bank_target_bytes: u64,
    ) -> Result<Self, WeightResidencyPolicyError> {
        let options = Self {
            members,
            compact_bank_scratch_bytes,
            prefill_compact_bank_target_bytes,
        };
        options.validate()?;
        Ok(options)
    }

    /// Revalidates the complete independently addressable residency policy.
    pub fn validate(self) -> Result<(), WeightResidencyPolicyError> {
        if self.compact_bank_scratch_bytes == 0 {
            return Err(WeightResidencyPolicyError::ZeroParameterBankScratchLimit);
        }
        if self.prefill_compact_bank_target_bytes == 0 {
            return Err(WeightResidencyPolicyError::ZeroParameterBankPrefillTarget);
        }
        if self.prefill_compact_bank_target_bytes > self.compact_bank_scratch_bytes {
            return Err(
                WeightResidencyPolicyError::ParameterBankPrefillTargetExceedsScratch {
                    target_bytes: self.prefill_compact_bank_target_bytes,
                    scratch_bytes: self.compact_bank_scratch_bytes,
                },
            );
        }
        Ok(())
    }

    /// Returns the independent bank offload limits.
    pub const fn offload(self) -> OffloadConfig {
        self.members
    }
    /// Returns the hard compact-bank scratch bound.
    pub const fn compact_bank_scratch_bytes(self) -> u64 {
        self.compact_bank_scratch_bytes
    }
    /// Returns the prefill compact-bank target.
    pub const fn prefill_compact_bank_target_bytes(self) -> u64 {
        self.prefill_compact_bank_target_bytes
    }
}

impl Default for ParameterBankLoadOptions {
    fn default() -> Self {
        Self {
            members: OffloadConfig::default(),
            compact_bank_scratch_bytes: u64::MAX,
            prefill_compact_bank_target_bytes: 1 << 30,
        }
    }
}

/// Placement of ordinary parameters beside independently addressable banks.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[non_exhaustive]
pub enum OrdinaryWeightResidency {
    /// Keep every ordinary parameter resident on the execution device.
    FullyResident,
    /// Eagerly materialize ordinary units on host behind a device window.
    LayerwiseHost(LayerwiseLoadOptions),
    /// Leave ordinary units cold on disk behind finite tier caches.
    DenseDiskStream(DenseDiskStreamLoadOptions),
}

impl OrdinaryWeightResidency {
    /// Returns the corresponding generalized layer policy.
    pub const fn layers(self) -> LayerWeightResidency {
        match self {
            Self::FullyResident => LayerWeightResidency::FullyResident,
            Self::LayerwiseHost(options) => LayerWeightResidency::LayerwiseHost(options),
            Self::DenseDiskStream(options) => LayerWeightResidency::DenseDiskStream(options),
        }
    }
}

impl From<OrdinaryWeightResidency> for LayerWeightResidency {
    fn from(value: OrdinaryWeightResidency) -> Self {
        value.layers()
    }
}

/// Independently addressable bank placement relative to ordinary execution units.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum ParameterBankResidency {
    /// Keep bank members in the ordinary unit residency allocation.
    #[default]
    WithLayer,
    /// Catalog bank members as independent atomic residency units.
    IndependentCache(ParameterBankLoadOptions),
}

/// Composable ordinary-unit and independently addressable bank placement.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[non_exhaustive]
pub enum WeightResidency {
    /// All parameters share the ordinary unit residency allocation.
    Layers(LayerWeightResidency),
    /// Addressable bank members are independent units beside bounded ordinary units.
    #[non_exhaustive]
    IndependentParameterBanks {
        /// Bounded ordinary-unit policy.
        ordinary: OrdinaryWeightResidency,
        /// Bank-member-granular cache controls.
        cache: ParameterBankLoadOptions,
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

    /// Keeps every owned parameter in its ordinary unit residency allocation.
    pub const fn with_layers(layers: LayerWeightResidency) -> Self {
        Self::Layers(layers)
    }

    /// Gives addressable parameter banks an independent cache beside ordinary units.
    pub const fn with_independent_parameter_banks(
        ordinary: OrdinaryWeightResidency,
        cache: ParameterBankLoadOptions,
    ) -> Self {
        Self::IndependentParameterBanks { ordinary, cache }
    }

    /// Returns ordinary-unit placement.
    pub const fn layers(self) -> LayerWeightResidency {
        match self {
            Self::Layers(layers) => layers,
            Self::IndependentParameterBanks { ordinary, .. } => ordinary.layers(),
        }
    }

    /// Returns independently addressable bank placement relative to ordinary units.
    pub const fn parameter_banks(self) -> ParameterBankResidency {
        match self {
            Self::Layers(_) => ParameterBankResidency::WithLayer,
            Self::IndependentParameterBanks { cache, .. } => {
                ParameterBankResidency::IndependentCache(cache)
            }
        }
    }

    /// Returns independently cached parameter-bank controls, when selected.
    pub const fn parameter_bank_cache(self) -> Option<ParameterBankLoadOptions> {
        match self {
            Self::Layers(_) => None,
            Self::IndependentParameterBanks { cache, .. } => Some(cache),
        }
    }

    /// Returns ordinary placement paired with an independent parameter-bank cache.
    pub const fn ordinary_residency(self) -> Option<OrdinaryWeightResidency> {
        match self {
            Self::Layers(_) => None,
            Self::IndependentParameterBanks { ordinary, .. } => Some(ordinary),
        }
    }

    /// Returns whether every ordinary parameter remains resident.
    pub const fn ordinary_is_fully_resident(self) -> bool {
        matches!(
            self,
            Self::Layers(LayerWeightResidency::FullyResident)
                | Self::IndependentParameterBanks {
                    ordinary: OrdinaryWeightResidency::FullyResident,
                    ..
                }
        )
    }

    /// Returns whether every ordinary unit remains resident.
    pub const fn is_fully_resident(self) -> bool {
        matches!(self, Self::Layers(LayerWeightResidency::FullyResident))
    }

    /// Returns the common checkpoint shard/reader cache bound.
    pub const fn max_cached_shards(self) -> usize {
        self.layers().max_cached_shards()
    }
}

impl Default for WeightResidency {
    fn default() -> Self {
        Self::fully_resident()
    }
}

impl LayerWeightResidency {
    /// Returns the checkpoint shard/reader cache bound carried by this policy.
    pub const fn max_cached_shards(self) -> usize {
        match self {
            Self::FullyResident => DEFAULT_MAX_CACHED_SHARDS,
            Self::LayerwiseHost(options) => options.max_cached_shards,
            Self::DenseDiskStream(options) => options.max_cached_shards,
        }
    }

    /// Returns whether backend allocator memory should be sampled.
    pub const fn sample_backend_memory(self) -> bool {
        match self {
            Self::FullyResident => false,
            Self::LayerwiseHost(options) => options.sample_backend_memory,
            Self::DenseDiskStream(options) => options.sample_backend_memory,
        }
    }

    /// Returns whether process memory should be sampled.
    pub const fn sample_process_memory(self) -> bool {
        match self {
            Self::FullyResident => false,
            Self::LayerwiseHost(options) => options.sample_process_memory,
            Self::DenseDiskStream(options) => options.sample_process_memory,
        }
    }

    /// Returns the maximum execution-device unit window.
    pub fn device_depth(self, unit_count: usize) -> usize {
        match self {
            Self::FullyResident => unit_count,
            Self::LayerwiseHost(options) => options.offload.prefetch_depth(),
            Self::DenseDiskStream(_) => unit_count.min(DENSE_TRANSFER_WINDOW),
        }
    }

    /// Resolves this policy into the shared offload configuration.
    pub fn offload(self) -> Result<OffloadConfig, WeightResidencyPolicyError> {
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

    /// Returns dense-stream controls when this policy streams from disk.
    pub const fn dense(self) -> Option<DenseDiskStreamLoadOptions> {
        match self {
            Self::DenseDiskStream(options) => Some(options),
            Self::FullyResident | Self::LayerwiseHost(_) => None,
        }
    }

    /// Returns whether every ordinary execution unit remains device-resident.
    pub const fn is_fully_resident(self) -> bool {
        matches!(self, Self::FullyResident)
    }

    /// Returns the stable physical execution classification.
    pub const fn execution_residency(self) -> ExecutionResidency {
        match self {
            Self::FullyResident => ExecutionResidency::FullyResident,
            Self::LayerwiseHost(_) => ExecutionResidency::LayerwiseHost,
            Self::DenseDiskStream(_) => ExecutionResidency::DenseDiskStream,
        }
    }
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

/// Static-parameter placement used by the generalized execution engine.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[non_exhaustive]
pub enum ExecutionResidency {
    /// Every module is constructed once and all rank-local parameters remain on device.
    FullyResident,
    /// Parameters remain on host behind bounded device windows.
    LayerwiseHost,
    /// Parameters are materialized through bounded disk, host, and device caches.
    DenseDiskStream,
}

/// Inspectable backend-neutral parameter-residency metadata for a layered model.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct LayerwiseModelMetadata {
    effective_model_type: String,
    quantization: Option<eredu_checkpoint::WeightQuantization>,
    layer_count: usize,
    static_device_bytes: u64,
    residency: ExecutionResidency,
    layer_parameter_bytes: u64,
    maximum_device_layer_bytes: u64,
    maximum_host_layer_bytes: u64,
    device_layer_capacity: usize,
    materialization: Option<crate::WeightMaterializationReport>,
}

impl LayerwiseModelMetadata {
    /// Creates one complete metadata snapshot before optional materialization telemetry.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        effective_model_type: impl Into<String>,
        quantization: Option<eredu_checkpoint::WeightQuantization>,
        layer_count: usize,
        static_device_bytes: u64,
        residency: ExecutionResidency,
        layer_parameter_bytes: u64,
        maximum_device_layer_bytes: u64,
        maximum_host_layer_bytes: u64,
        device_layer_capacity: usize,
    ) -> Self {
        Self {
            effective_model_type: effective_model_type.into(),
            quantization,
            layer_count,
            static_device_bytes,
            residency,
            layer_parameter_bytes,
            maximum_device_layer_bytes,
            maximum_host_layer_bytes,
            device_layer_capacity,
            materialization: None,
        }
    }

    /// Replaces the parsed implementation identity after checkpoint resolution.
    pub fn set_effective_model_type(&mut self, effective_model_type: impl Into<String>) {
        self.effective_model_type = effective_model_type.into();
    }

    /// Replaces checkpoint-native packed quantization metadata.
    pub fn set_quantization(&mut self, quantization: Option<eredu_checkpoint::WeightQuantization>) {
        self.quantization = quantization;
    }

    /// Replaces bounded load-time materialization telemetry.
    pub fn set_materialization(
        &mut self,
        materialization: Option<crate::WeightMaterializationReport>,
    ) {
        self.materialization = materialization;
    }

    /// Returns the parsed implementation or nested text-model type.
    pub fn effective_model_type(&self) -> &str {
        &self.effective_model_type
    }

    /// Returns checkpoint-native packed quantization metadata, if present.
    pub const fn quantization(&self) -> Option<eredu_checkpoint::WeightQuantization> {
        self.quantization
    }

    /// Returns the ordered execution-unit count.
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

    /// Returns complete rank-local execution-unit parameter bytes.
    pub const fn layer_parameter_bytes(&self) -> u64 {
        self.layer_parameter_bytes
    }

    /// Returns the largest possible device-resident execution-unit byte total.
    pub const fn maximum_device_layer_bytes(&self) -> u64 {
        self.maximum_device_layer_bytes
    }

    /// Returns the charged host-transfer capacity of the largest execution unit.
    pub const fn maximum_host_layer_bytes(&self) -> u64 {
        self.maximum_host_layer_bytes
    }

    /// Returns the maximum number of execution units retained on device.
    pub const fn device_layer_capacity(&self) -> usize {
        self.device_layer_capacity
    }

    /// Returns bounded load-time materialization telemetry.
    pub const fn materialization(&self) -> Option<&crate::WeightMaterializationReport> {
        self.materialization.as_ref()
    }
}

/// Invalid backend-neutral immutable-weight residency policy.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum WeightResidencyPolicyError {
    /// Enabled host caching needs a protected current unit.
    #[error("dense disk streaming host lookahead must be nonzero when the host budget is enabled")]
    ZeroHostLookahead,
    /// Enabled background work requires bounded capacity.
    #[error("dense disk streaming background queue capacity must be nonzero when host caching is enabled")]
    ZeroQueueCapacity,
    /// Direct-to-device mode cannot configure host-only controls.
    #[error("dense disk streaming with a zero host budget requires zero host lookahead and queue capacity")]
    HostDisabledControls,
    /// Parameter-bank scratch accounting was disabled with a zero limit.
    #[error("parameter-bank compact scratch limit must be nonzero")]
    ZeroParameterBankScratchLimit,
    /// Parameter-bank prefill chunking was disabled with a zero target.
    #[error("parameter-bank prefill target must be nonzero")]
    ZeroParameterBankPrefillTarget,
    /// The parameter-bank prefill target exceeded the hard scratch bound.
    #[error("parameter-bank prefill target {target_bytes} exceeds scratch limit {scratch_bytes}")]
    ParameterBankPrefillTargetExceedsScratch {
        /// Requested soft prefill target.
        target_bytes: u64,
        /// Configured hard scratch limit.
        scratch_bytes: u64,
    },
    /// The derived offload configuration was invalid.
    #[error(transparent)]
    Offload(#[from] OffloadError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_unit_bindings_are_runtime_owned_and_decomposable() {
        let binding = WeightBinding::new(
            "embedding",
            "model.embedding.weight",
            eredu_checkpoint::store::TensorSelection::Full,
            16,
        )
        .unwrap();
        let unit = StaticUnitBindings::new("static.embedding", vec![binding.clone()]).unwrap();

        assert_eq!(unit.id().as_str(), "static.embedding");
        assert_eq!(unit.bindings(), std::slice::from_ref(&binding));
        let (id, bindings) = unit.into_parts();
        assert_eq!(id.as_str(), "static.embedding");
        assert_eq!(bindings, vec![binding]);
    }

    #[test]
    fn dense_stream_controls_fail_closed_without_a_backend() {
        assert!(matches!(
            DenseDiskStreamLoadOptions::new(1, 1, 0, 1),
            Err(WeightResidencyPolicyError::ZeroHostLookahead)
        ));
        assert!(matches!(
            DenseDiskStreamLoadOptions::new(1, 1, 1, 0),
            Err(WeightResidencyPolicyError::ZeroQueueCapacity)
        ));
        assert!(matches!(
            DenseDiskStreamLoadOptions::new(1, 0, 1, 0),
            Err(WeightResidencyPolicyError::HostDisabledControls)
        ));
        assert!(DenseDiskStreamLoadOptions::new(1, 0, 0, 0).is_ok());
    }

    #[test]
    fn residency_policy_derives_finite_dense_windows() {
        let options = DenseDiskStreamLoadOptions::new(32, 64, 3, 2).unwrap();
        let policy = LayerWeightResidency::DenseDiskStream(options);
        assert_eq!(policy.device_depth(8), DENSE_TRANSFER_WINDOW);
        let offload = policy.offload().unwrap();
        assert_eq!(offload.device_budget_bytes(), Some(32));
        assert_eq!(offload.host_budget_bytes(), Some(64));
        assert_eq!(offload.prefetch_depth(), 3);
    }

    #[test]
    fn dense_transfer_schedule_preserves_order_and_bounded_lookahead() {
        let mut schedule = DenseTransferSchedule::new(3..7, 2).unwrap();
        assert_eq!(schedule.desired_indices(3), vec![3, 4, 5]);
        assert_eq!(
            schedule.admit(4, "wrong"),
            Err(DenseTransferScheduleError::OutOfOrder {
                expected: 3,
                actual: 4,
            })
        );
        schedule.admit(3, "three").unwrap();
        schedule.admit(4, "four").unwrap();
        assert!(!schedule.can_admit());
        assert_eq!(schedule.desired_indices(3), vec![3, 4, 5]);
        assert_eq!(schedule.pop_ready(), Some((3, "three")));
        schedule.admit(5, "five").unwrap();
        assert_eq!(schedule.desired_indices(4), vec![4, 5, 6]);
        assert_eq!(schedule.pop_ready(), Some((4, "four")));
        assert_eq!(schedule.pop_ready(), Some((5, "five")));
        schedule.admit(6, "six").unwrap();
        assert_eq!(schedule.pop_ready(), Some((6, "six")));
        assert!(schedule.is_exhausted());
    }

    #[test]
    fn expert_cache_controls_and_composite_placement_are_backend_neutral() {
        let experts = ParameterBankLoadOptions::new(OffloadConfig::default(), 64, 32).unwrap();
        let placement = WeightResidency::with_independent_parameter_banks(
            OrdinaryWeightResidency::FullyResident,
            experts,
        );
        assert_eq!(placement.parameter_bank_cache(), Some(experts));
        assert!(placement.ordinary_is_fully_resident());
        assert!(!placement.is_fully_resident());
        assert!(matches!(
            ParameterBankLoadOptions::new(OffloadConfig::default(), 64, 0),
            Err(WeightResidencyPolicyError::ZeroParameterBankPrefillTarget)
        ));
        assert_eq!(
            ParameterBankKey::new(3, 7).unit_id().as_str(),
            "bank.unit.00003.member.00007"
        );
    }

    #[test]
    fn layerwise_metadata_is_runtime_owned_and_updateable() {
        let mut metadata = LayerwiseModelMetadata::new(
            "generic",
            None,
            4,
            10,
            ExecutionResidency::LayerwiseHost,
            20,
            8,
            6,
            2,
        );
        metadata.set_effective_model_type("llama");
        metadata.set_quantization(Some(eredu_checkpoint::WeightQuantization::Affine(
            eredu_checkpoint::AffineQuantization::default(),
        )));

        assert_eq!(metadata.effective_model_type(), "llama");
        assert_eq!(metadata.layer_count(), 4);
        assert_eq!(metadata.static_device_bytes(), 10);
        assert_eq!(metadata.layer_parameter_bytes(), 20);
        assert_eq!(metadata.maximum_device_layer_bytes(), 8);
        assert_eq!(metadata.maximum_host_layer_bytes(), 6);
        assert_eq!(metadata.device_layer_capacity(), 2);
    }
}
