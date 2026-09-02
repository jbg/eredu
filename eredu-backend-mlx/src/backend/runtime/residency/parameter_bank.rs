//! MLX residency for independently addressable parameter-bank entries.
//!
//! Each logical entry is an atomic disk-planned residency unit. Selection ids are
//! inspected once per grouped block, validated before acquisition, coalesced in
//! deterministic global-id order, and rewritten to a temporary compact bank.

use eredu_checkpoint::{store::TensorSelection, WeightQuantization};
use eredu_nn::GroupedNeuralBackend;
use eredu_runtime::{
    AddressableGroupedBank, IndexedMovement, OffloadUnit, ParameterBankAccess,
    ParameterBankAcquisition, ParameterBankKey as NeutralParameterBankKey, ResidencyReport,
    WeightBinding, WeightMaterializationReport,
};

use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use safemlx::{
    ops::{concatenate_axis, indexing::TryIndexOp, r#where, segment_sum},
    transforms::eval,
    Array, Dtype, Stream,
};

#[cfg(test)]
use crate::backend::runtime::residency::manager::ResidentUnitLease;
use crate::MlxTensor;
use crate::{
    backend::error::Error,
    backend::nn::shared::MlxNeuralBackend,
    backend::runtime::checkpoint::bounded_quantization::{
        BoundedQuantizationPlan, BoundedQuantizationTarget, BoundedQuantizedWeightStore,
    },
    backend::runtime::residency::manager::{ResidencyError, ResidencyManager, ResidentTransfer},
};
use eredu_core::residency::{
    MemoryTier, OffloadConfig, OffloadPlan, OffloadUnitId, OffloadUnitSpec, ResidencyLedgerError,
    ResidencyPolicy,
};

/// Stable backend identity for one independently addressable bank entry.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ParameterBankKey {
    namespace: usize,
    index: usize,
}

impl ParameterBankKey {
    /// Creates an entry identity within a caller-defined namespace.
    pub const fn new(namespace: usize, index: usize) -> Self {
        Self { namespace, index }
    }

    /// Returns the caller-defined namespace.
    pub const fn namespace(self) -> usize {
        self.namespace
    }

    /// Returns the entry index within its namespace.
    pub const fn index(self) -> usize {
        self.index
    }

    /// Returns the deterministic residency-unit identifier.
    pub fn unit_id(self) -> OffloadUnitId {
        OffloadUnitId::new(format!(
            "parameter-bank.namespace.{:05}.entry.{:05}",
            self.namespace, self.index
        ))
        .expect("parameter-bank unit identifier is non-empty")
    }
}

/// Workload class used only for bank chunking and mechanism telemetry.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum BankAccessClass {
    /// A multi-row access that may be split to stay under a working-set target.
    Bulk,
    /// A latency-sensitive access that uses the hard working-set limit directly.
    Incremental,
}

/// Backend controls for an independently addressable parameter bank.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct ParameterBankOptions {
    storage: OffloadConfig,
    compact_bank_scratch_bytes: u64,
    bulk_compact_bank_target_bytes: u64,
}

impl ParameterBankOptions {
    /// Creates validated bank residency controls.
    pub fn new(
        storage: OffloadConfig,
        compact_bank_scratch_bytes: u64,
        bulk_compact_bank_target_bytes: u64,
    ) -> Result<Self, ParameterBankOptionsError> {
        let options = Self {
            storage,
            compact_bank_scratch_bytes,
            bulk_compact_bank_target_bytes,
        };
        options.validate()?;
        Ok(options)
    }

    /// Validates the bank working-set controls.
    pub fn validate(self) -> Result<(), ParameterBankOptionsError> {
        if self.compact_bank_scratch_bytes == 0 {
            return Err(ParameterBankOptionsError::ZeroScratchLimit);
        }
        if self.bulk_compact_bank_target_bytes == 0 {
            return Err(ParameterBankOptionsError::ZeroBulkBankTarget);
        }
        if self.bulk_compact_bank_target_bytes > self.compact_bank_scratch_bytes {
            return Err(ParameterBankOptionsError::BulkBankTargetExceedsScratch {
                target_bytes: self.bulk_compact_bank_target_bytes,
                scratch_bytes: self.compact_bank_scratch_bytes,
            });
        }
        Ok(())
    }
}

impl Default for ParameterBankOptions {
    fn default() -> Self {
        Self {
            storage: OffloadConfig::default(),
            compact_bank_scratch_bytes: u64::MAX,
            bulk_compact_bank_target_bytes: 1 << 30,
        }
    }
}

/// Invalid addressable-bank residency controls.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ParameterBankOptionsError {
    /// The hard temporary-bank limit was zero.
    #[error("parameter-bank scratch limit must be nonzero")]
    ZeroScratchLimit,
    /// The bulk-access working-set target was zero.
    #[error("parameter-bank bulk target must be nonzero")]
    ZeroBulkBankTarget,
    /// The soft bulk target exceeded the hard working-set limit.
    #[error("parameter-bank bulk target {target_bytes} exceeds scratch limit {scratch_bytes}")]
    BulkBankTargetExceedsScratch {
        /// Invalid soft target.
        target_bytes: u64,
        /// Configured hard limit.
        scratch_bytes: u64,
    },
}

/// One atomic entry definition supplied by a caller.
#[derive(Clone)]
pub struct ParameterBankEntry {
    identity: ParameterBankKey,
    unit: OffloadUnit,
    bytes: u64,
}

impl ParameterBankEntry {
    /// Creates one catalog entry and verifies its stable unit identity.
    pub fn new(
        identity: ParameterBankKey,
        unit: OffloadUnit,
        bytes: u64,
    ) -> Result<Self, AddressableParameterBankError> {
        if bytes == 0 {
            return Err(AddressableParameterBankError::ZeroSizedEntry { identity });
        }
        let expected = identity.unit_id();
        if unit.id() != &expected {
            return Err(AddressableParameterBankError::UnitIdentityMismatch {
                identity,
                expected,
                actual: unit.id().clone(),
            });
        }
        Ok(Self {
            identity,
            unit,
            bytes,
        })
    }

    /// Returns the logical identity.
    pub const fn identity(&self) -> ParameterBankKey {
        self.identity
    }

    /// Returns the atomic materialized byte length.
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    fn into_parts(self) -> (ParameterBankKey, OffloadUnit) {
        (self.identity, self.unit)
    }
}

/// Lowers generic selected storage members into MLX residency entries.
pub fn entries_from_selected_members(
    members: &[eredu_runtime::AddressableBankMember],
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<ParameterBankEntry>, Error> {
    members
        .iter()
        .map(|member| {
            let key = ParameterBankKey::new(member.key().unit(), member.key().member());
            let bindings = member
                .source()
                .bindings()
                .iter()
                .map(|binding| {
                    let mut recipe = binding.source_recipe();
                    if recipe.infer(store)?.dtype() == &eredu_checkpoint::recipe::RecipeDtype::F4 {
                        recipe = crate::backend::runtime::checkpoint::recipe::lower_mxfp4_recipe(
                            recipe, store,
                        )?;
                    }
                    let metadata = recipe.infer(store)?;
                    let mut selected =
                        WeightBinding::from_recipe(binding.name(), recipe, metadata.byte_len())?;
                    if let Some(target) = binding.logical_target() {
                        selected = selected.with_logical_target(target)?;
                    }
                    if let Some((scales, biases)) = binding.quantization_companions() {
                        selected = selected.with_quantization_companions(scales, biases)?;
                    }
                    Ok(selected)
                })
                .collect::<Result<Vec<_>, Error>>()?;
            let bytes = bindings.iter().try_fold(0u64, |total, binding| {
                total.checked_add(binding.expected_bytes()).ok_or_else(|| {
                    Error::ArchitectureModel(format!(
                        "addressable member {:?} source byte total overflowed",
                        member.key()
                    ))
                })
            })?;
            ParameterBankEntry::new(key, OffloadUnit::new(key.unit_id(), bindings)?, bytes)
                .map_err(Into::into)
        })
        .collect()
}

/// Result of replacing dense entry bindings with a disk-backed packed overlay.
pub(crate) struct QuantizedParameterBankCatalog {
    /// Store supplying synthetic packed bindings and delegating all other keys.
    pub(crate) store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    /// Entry units rebuilt against the packed store.
    pub(crate) entries: Vec<ParameterBankEntry>,
    /// Deterministic bounded-materialisation telemetry.
    pub(crate) report: WeightMaterializationReport,
}

/// Quantizes every floating entry projection through its authoritative
/// rank-local semantic recipe and rebuilds the catalog against packed keys.
pub(crate) fn quantize_entry_catalog(
    source: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    entries: Vec<ParameterBankEntry>,
    quantization: WeightQuantization,
    max_working_set_bytes: u64,
    source_stream: &Stream,
) -> Result<QuantizedParameterBankCatalog, Error> {
    let mut units = Vec::with_capacity(entries.len());
    let mut targets = Vec::new();
    let mut target_by_binding = BTreeMap::new();
    let mut packed_catalog_bytes = 0u64;
    for entry in entries {
        let (identity, unit) = entry.into_parts();
        for binding in unit.bindings() {
            let Some((scales_binding, biases_binding)) = binding.quantization_companions() else {
                continue;
            };
            let recipe = binding.source_recipe();
            let metadata = recipe.infer(source.as_ref())?;
            let target_name = format!(
                "__eredu.entry.namespace.{:05}.global.{:05}.{}.weight",
                identity.namespace,
                identity.index,
                binding.name()
            );
            let target_prefix = target_name
                .strip_suffix(".weight")
                .expect("synthetic entry target has a weight suffix");
            let target = BoundedQuantizationTarget::from_recipe(
                target_name.clone(),
                format!("{target_prefix}.scales"),
                Some(format!("{target_prefix}.biases")),
                recipe,
            )?;
            packed_catalog_bytes = packed_catalog_bytes
                .checked_add(packed_projection_bytes(metadata.shape(), quantization)?)
                .ok_or_else(|| {
                    Error::Quantization("packed entry catalog size overflowed".into())
                })?;
            target_by_binding.insert(
                (identity, binding.name().to_string()),
                (
                    target.clone(),
                    scales_binding.to_owned(),
                    biases_binding.to_owned(),
                ),
            );
            targets.push(target);
        }
        units.push((identity, unit));
    }
    if targets.is_empty() {
        return Err(Error::Quantization(
            "entry catalog contains no floating projection bindings to quantize".into(),
        ));
    }
    let plan = BoundedQuantizationPlan::new(
        quantization,
        max_working_set_bytes.min(packed_catalog_bytes),
        targets,
    )?;
    let transformed = Arc::new(BoundedQuantizedWeightStore::create(
        Arc::clone(&source),
        plan,
        source_stream,
    )?);
    let report = transformed.report().clone();
    let store: Arc<dyn eredu_checkpoint::store::CheckpointSource> = transformed;
    let mut rebuilt = Vec::with_capacity(units.len());
    for (identity, unit) in units {
        let mut bindings = Vec::new();
        for binding in unit.bindings() {
            let Some((target, scales_name, biases_name)) =
                target_by_binding.get(&(identity, binding.name().to_string()))
            else {
                bindings.push(binding.clone());
                continue;
            };
            bindings.push(packed_binding(
                binding.name(),
                target.weight_name(),
                store.as_ref(),
            )?);
            bindings.push(packed_binding(
                scales_name,
                target.scales_name(),
                store.as_ref(),
            )?);
            if quantization.has_biases() {
                bindings.push(packed_binding(
                    biases_name,
                    target
                        .biases_name()
                        .expect("affine entry target declared a bias identity"),
                    store.as_ref(),
                )?);
            }
        }
        let bytes = bindings.iter().try_fold(0u64, |total, binding| {
            total.checked_add(binding.expected_bytes()).ok_or_else(|| {
                Error::Quantization("quantized entry catalog byte total overflowed".into())
            })
        })?;
        rebuilt.push(ParameterBankEntry::new(
            identity,
            OffloadUnit::new(identity.unit_id(), bindings)?,
            bytes,
        )?);
    }
    Ok(QuantizedParameterBankCatalog {
        store,
        entries: rebuilt,
        report,
    })
}

fn packed_projection_bytes(
    shape: &[usize],
    quantization: WeightQuantization,
) -> Result<u64, Error> {
    let (&columns, rows) = shape
        .split_last()
        .ok_or_else(|| Error::Quantization("entry projection has no input dimension".into()))?;
    let rows = rows.iter().try_fold(1u64, |count, dimension| {
        count
            .checked_mul(*dimension as u64)
            .ok_or_else(|| Error::Quantization("entry projection row count overflowed".into()))
    })?;
    let packed = (columns as u64)
        .checked_mul(quantization.bits() as u64)
        .and_then(|bits| bits.checked_div(8))
        .ok_or_else(|| Error::Quantization("packed entry row size overflowed".into()))?;
    let groups = columns
        .checked_div(quantization.group_size() as usize)
        .ok_or_else(|| Error::Quantization("entry group geometry is invalid".into()))?
        as u64;
    let scale_bytes = if matches!(quantization, WeightQuantization::MxFp4) {
        groups
    } else {
        groups
            .checked_mul(4)
            .ok_or_else(|| Error::Quantization("entry scale row size overflowed".into()))?
    };
    let bias_bytes = if quantization.has_biases() {
        groups
            .checked_mul(4)
            .ok_or_else(|| Error::Quantization("entry bias row size overflowed".into()))?
    } else {
        0
    };
    let row_bytes = packed
        .checked_add(scale_bytes)
        .and_then(|bytes| bytes.checked_add(bias_bytes))
        .ok_or_else(|| Error::Quantization("packed entry row total overflowed".into()))?;
    rows.checked_mul(row_bytes)
        .ok_or_else(|| Error::Quantization("packed entry projection size overflowed".into()))
}

fn packed_binding(
    local_name: &str,
    checkpoint_key: &str,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<WeightBinding, Error> {
    let metadata = store.source_metadata(checkpoint_key)?;
    Ok(WeightBinding::new(
        local_name,
        checkpoint_key,
        TensorSelection::Full,
        metadata.encoded_byte_len,
    )?)
}

/// Tier-local cache request counters.
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq)]
pub struct BankTierStatistics {
    /// Logical entry acquisition requests after duplicate coalescing.
    requests: u64,
    /// Requests served by an already resident copy.
    hits: u64,
    /// Requests that materialized or promoted a copy.
    misses: u64,
    /// Copies evicted while satisfying cache requests.
    evictions: u64,
    /// Bytes evicted while satisfying cache requests.
    eviction_bytes: u64,
}

impl BankTierStatistics {
    /// Returns logical acquisition requests after coalescing.
    pub const fn requests(&self) -> u64 {
        self.requests
    }
    /// Returns requests served by a resident copy.
    pub const fn hits(&self) -> u64 {
        self.hits
    }
    /// Returns requests requiring materialization or promotion.
    pub const fn misses(&self) -> u64 {
        self.misses
    }
    /// Returns evicted copies.
    pub const fn evictions(&self) -> u64 {
        self.evictions
    }
    /// Returns bytes removed by eviction.
    pub const fn eviction_bytes(&self) -> u64 {
        self.eviction_bytes
    }
}

/// Cumulative statistics for one public execution-path class.
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq)]
pub struct BankPassStatistics {
    /// Selection rows requested by the selector, including duplicates.
    requested_selections: u64,
    /// Distinct logical entries requested after coalescing.
    distinct_entries: u64,
    /// Duplicate requests eliminated before materialization.
    coalesced_duplicates: u64,
    /// Temporary compact banks built.
    compact_banks: u64,
    /// Cumulative compact-bank bytes.
    compact_bank_bytes: u64,
    /// Peak temporary compact-bank bytes.
    peak_compact_bank_bytes: u64,
    /// Cumulative compact-bank construction time.
    compact_bank_time: Duration,
    /// Time preparing and reserving entry materialization or promotion.
    ///
    /// Deferred device completion is charged to the dependent entry output,
    /// not this counter.
    materialization_wait: Duration,
    /// Host-tier cache activity.
    host: BankTierStatistics,
    /// Device-tier cache activity.
    device: BankTierStatistics,
}

impl BankPassStatistics {
    /// Returns requested selection rows, including duplicates.
    pub const fn requested_selections(&self) -> u64 {
        self.requested_selections
    }
    /// Returns distinct entries after coalescing.
    pub const fn distinct_entries(&self) -> u64 {
        self.distinct_entries
    }
    /// Returns duplicates eliminated before materialization.
    pub const fn coalesced_duplicates(&self) -> u64 {
        self.coalesced_duplicates
    }
    /// Returns temporary compact banks built.
    pub const fn compact_banks(&self) -> u64 {
        self.compact_banks
    }
    /// Returns cumulative compact-bank bytes.
    pub const fn compact_bank_bytes(&self) -> u64 {
        self.compact_bank_bytes
    }
    /// Returns the peak compact-bank byte size.
    pub const fn peak_compact_bank_bytes(&self) -> u64 {
        self.peak_compact_bank_bytes
    }
    /// Returns cumulative compact-bank construction time.
    pub const fn compact_bank_time(&self) -> Duration {
        self.compact_bank_time
    }
    /// Returns time spent preparing entry materialization or promotion.
    pub const fn materialization_wait(&self) -> Duration {
        self.materialization_wait
    }
    /// Returns host-tier activity.
    pub const fn host(&self) -> &BankTierStatistics {
        &self.host
    }
    /// Returns device-tier activity.
    pub const fn device(&self) -> &BankTierStatistics {
        &self.device
    }
}

/// Point-in-time entry residency and execution report.
pub struct ParameterBankResidencyReport {
    /// Packed encoding used by load-time transformed entry bindings.
    weight_quantization: Option<WeightQuantization>,
    /// Owned logical entry count.
    owned_entries: usize,
    /// Owned logical entry bytes, including cold checkpoint-only entries.
    owned_bytes: u64,
    /// Current host-resident entry count.
    host_resident_entries: usize,
    /// Current device-resident entry count.
    device_resident_entries: usize,
    /// Current physical capacity of host-resident entry allocations.
    host_resident_bytes: u64,
    /// Current device-resident entry bytes.
    device_resident_bytes: u64,
    /// Peak physical capacity of host-resident entry allocations.
    peak_host_resident_bytes: u64,
    /// Peak device-resident entry bytes.
    peak_device_resident_bytes: u64,
    /// Prompt-processing statistics.
    bulk: BankPassStatistics,
    /// Autoregressive incremental statistics.
    incremental: BankPassStatistics,
    /// Underlying logical transfer and checkpoint diagnostics.
    residency: ResidencyReport,
    /// Bounded load-time entry materialisation telemetry, when the catalog
    /// was transformed from floating checkpoint weights.
    materialization: Option<WeightMaterializationReport>,
}

impl ParameterBankResidencyReport {
    /// Returns the optional packed load-time weight encoding.
    pub const fn weight_quantization(&self) -> Option<WeightQuantization> {
        self.weight_quantization
    }
    /// Returns the number of owned entries.
    pub const fn owned_entries(&self) -> usize {
        self.owned_entries
    }
    /// Returns total owned bytes, including cold entries.
    pub const fn owned_bytes(&self) -> u64 {
        self.owned_bytes
    }
    /// Returns host-resident entry count.
    pub const fn host_resident_entries(&self) -> usize {
        self.host_resident_entries
    }
    /// Returns device-resident entry count.
    pub const fn device_resident_entries(&self) -> usize {
        self.device_resident_entries
    }
    /// Returns current host-resident capacity.
    pub const fn host_resident_bytes(&self) -> u64 {
        self.host_resident_bytes
    }
    /// Returns current device-resident bytes.
    pub const fn device_resident_bytes(&self) -> u64 {
        self.device_resident_bytes
    }
    /// Returns peak host-resident capacity.
    pub const fn peak_host_resident_bytes(&self) -> u64 {
        self.peak_host_resident_bytes
    }
    /// Returns peak device-resident bytes.
    pub const fn peak_device_resident_bytes(&self) -> u64 {
        self.peak_device_resident_bytes
    }
    /// Returns bulk-access statistics.
    pub const fn bulk(&self) -> &BankPassStatistics {
        &self.bulk
    }
    /// Returns incremental-access statistics.
    pub const fn incremental(&self) -> &BankPassStatistics {
        &self.incremental
    }
    /// Returns underlying residency diagnostics.
    pub const fn residency(&self) -> &ResidencyReport {
        &self.residency
    }
    /// Returns bounded load-time materialization telemetry, when present.
    pub const fn materialization(&self) -> Option<&WeightMaterializationReport> {
        self.materialization.as_ref()
    }
}

#[derive(Default)]
struct ParameterBankStatistics {
    bulk: BankPassStatistics,
    incremental: BankPassStatistics,
}

impl ParameterBankStatistics {
    fn pass_mut(&mut self, pass: BankAccessClass) -> &mut BankPassStatistics {
        match pass {
            BankAccessClass::Bulk => &mut self.bulk,
            BankAccessClass::Incremental => &mut self.incremental,
        }
    }
}

/// Shared entry catalog, scheduler, residency manager, and telemetry.
pub struct AddressableParameterBank {
    manager: ResidencyManager,
    catalog: BTreeMap<ParameterBankKey, u64>,
    #[cfg(test)]
    namespace_entry_counts: BTreeMap<usize, usize>,
    namespace_global_spans: BTreeMap<usize, usize>,
    host_budget: Option<u64>,
    scratch_limit: u64,
    bulk_bank_target: u64,
    statistics: Mutex<ParameterBankStatistics>,
    weight_quantization: Option<WeightQuantization>,
    materialization: Option<WeightMaterializationReport>,
}

impl AddressableParameterBank {
    /// Creates a disk-planned cache over exactly the supplied owned entries.
    pub fn new<S, O>(
        store: Arc<S>,
        entries: impl IntoIterator<Item = ParameterBankEntry>,
        options: O,
        source_stream: Stream,
        device_stream: Stream,
    ) -> Result<Self, AddressableParameterBankError>
    where
        S: eredu_checkpoint::store::CheckpointSource + 'static,
        O: Into<ParameterBankOptions>,
    {
        let store: Arc<dyn eredu_checkpoint::store::CheckpointSource> = store;
        Self::new_shared(store, entries, options.into(), source_stream, device_stream)
    }

    /// Creates a cache from an already type-erased checkpoint store.
    pub fn new_shared<O>(
        store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
        entries: impl IntoIterator<Item = ParameterBankEntry>,
        options: O,
        source_stream: Stream,
        device_stream: Stream,
    ) -> Result<Self, AddressableParameterBankError>
    where
        O: Into<ParameterBankOptions>,
    {
        Self::new_shared_with_policy(
            store,
            entries,
            options.into(),
            ResidencyPolicy::Cacheable,
            MemoryTier::Disk,
            source_stream,
            device_stream,
            None,
            None,
        )
    }

    /// Creates a cache after boundedly transforming its rank-local semantic
    /// entry recipes into packed checkpoint bindings.
    pub fn new_quantized_shared<O>(
        store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
        entries: Vec<ParameterBankEntry>,
        options: O,
        quantization: WeightQuantization,
        source_stream: Stream,
        device_stream: Stream,
    ) -> Result<Self, AddressableParameterBankError>
    where
        O: Into<ParameterBankOptions>,
    {
        let options = options.into();
        let transformed = quantize_entry_catalog(
            store,
            entries,
            quantization,
            options.compact_bank_scratch_bytes,
            &source_stream,
        )
        .map_err(|source| AddressableParameterBankError::Transformation {
            source: Box::new(source),
        })?;
        let report = transformed.report;
        Self::new_shared_with_policy(
            transformed.store,
            transformed.entries,
            options,
            ResidencyPolicy::Cacheable,
            MemoryTier::Disk,
            source_stream,
            device_stream,
            Some(quantization),
            Some(report),
        )
    }

    /// Creates a fully resident store over exactly the supplied owned entries.
    ///
    /// Every entry is pinned on the execution device during construction. The
    /// same selection compaction and backend-neutral binding machinery is used
    /// by sparse and resident execution, but resident entries cannot be evicted
    /// and never trigger checkpoint reads during a forward pass.
    pub fn new_resident_shared(
        store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
        entries: impl IntoIterator<Item = ParameterBankEntry>,
        source_stream: Stream,
        device_stream: Stream,
    ) -> Result<Self, AddressableParameterBankError> {
        Self::new_shared_with_policy(
            store,
            entries,
            ParameterBankOptions::default(),
            ResidencyPolicy::Pinned,
            MemoryTier::Device,
            source_stream,
            device_stream,
            None,
            None,
        )
    }

    /// Creates a fully resident cache after boundedly transforming the exact
    /// rank-local entry catalog into its requested packed representation.
    pub fn new_quantized_resident_shared(
        store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
        entries: Vec<ParameterBankEntry>,
        quantization: WeightQuantization,
        source_stream: Stream,
        device_stream: Stream,
    ) -> Result<Self, AddressableParameterBankError> {
        let options = ParameterBankOptions::default();
        let transformed = quantize_entry_catalog(
            store,
            entries,
            quantization,
            options.compact_bank_scratch_bytes,
            &source_stream,
        )
        .map_err(|source| AddressableParameterBankError::Transformation {
            source: Box::new(source),
        })?;
        let report = transformed.report;
        Self::new_shared_with_policy(
            transformed.store,
            transformed.entries,
            options,
            ResidencyPolicy::Pinned,
            MemoryTier::Device,
            source_stream,
            device_stream,
            Some(quantization),
            Some(report),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_shared_with_policy(
        store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
        entries: impl IntoIterator<Item = ParameterBankEntry>,
        options: ParameterBankOptions,
        policy: ResidencyPolicy,
        initial_tier: MemoryTier,
        source_stream: Stream,
        device_stream: Stream,
        weight_quantization: Option<WeightQuantization>,
        materialization: Option<WeightMaterializationReport>,
    ) -> Result<Self, AddressableParameterBankError> {
        options.validate()?;
        let mut catalog = BTreeMap::new();
        let mut definitions = Vec::new();
        let mut specs = Vec::new();
        #[cfg(test)]
        let mut namespace_entry_counts = BTreeMap::new();
        let mut namespace_global_spans = BTreeMap::new();
        for entry in entries {
            if catalog.insert(entry.identity, entry.bytes).is_some() {
                return Err(AddressableParameterBankError::DuplicateEntry {
                    identity: entry.identity,
                });
            }
            #[cfg(test)]
            {
                *namespace_entry_counts
                    .entry(entry.identity.namespace)
                    .or_insert(0) += 1;
            }
            namespace_global_spans
                .entry(entry.identity.namespace)
                .and_modify(|span: &mut usize| *span = (*span).max(entry.identity.index + 1))
                .or_insert(entry.identity.index + 1);
            specs.push(OffloadUnitSpec::new(
                entry.identity.unit_id(),
                entry.bytes,
                policy,
                initial_tier,
            )?);
            definitions.push(entry.unit);
        }
        if catalog.is_empty() {
            return Err(AddressableParameterBankError::EmptyCatalog);
        }
        let plan = OffloadPlan::new(options.storage, specs)?;
        let manager =
            ResidencyManager::new_shared(store, plan, definitions, source_stream, device_stream)?;
        manager.initialize()?;
        Ok(Self {
            manager,
            catalog,
            #[cfg(test)]
            namespace_entry_counts,
            namespace_global_spans,
            host_budget: if initial_tier == MemoryTier::Device {
                Some(0)
            } else {
                options.storage.host_budget_bytes()
            },
            scratch_limit: options.compact_bank_scratch_bytes,
            bulk_bank_target: options.bulk_compact_bank_target_bytes,
            statistics: Mutex::new(ParameterBankStatistics::default()),
            weight_quantization,
            materialization,
        })
    }

    /// Returns the load-time encoding of transformed entry bindings.
    pub const fn weight_quantization(&self) -> Option<WeightQuantization> {
        self.weight_quantization
    }

    /// Returns the underlying reusable residency manager.
    pub const fn residency_manager(&self) -> &ResidencyManager {
        &self.manager
    }

    /// Returns the largest source member in the generic storage catalog.
    pub(crate) fn maximum_member_bytes(&self) -> u64 {
        self.catalog.values().copied().max().unwrap_or(0)
    }

    /// Returns the hard compact-bank byte limit.
    pub(crate) const fn compact_bank_scratch_bytes(&self) -> u64 {
        self.scratch_limit
    }

    /// Returns the bulk compact-bank working-set target.
    pub(crate) const fn bulk_compact_bank_target_bytes(&self) -> u64 {
        self.bulk_bank_target
    }

    /// Returns the admitted global member span for a caller-owned namespace.
    pub(crate) fn namespace_global_span(&self, namespace: usize) -> Option<usize> {
        self.namespace_global_spans.get(&namespace).copied()
    }

    /// Completes one generic acquisition after its dependent output is evaluated.
    pub(crate) fn complete_acquisition(
        &self,
        mut acquisition: AcquiredParameterGroups,
        output: &Array,
    ) -> Result<(), AddressableParameterBankError> {
        eval([output])?;
        acquisition.transfer.synchronize()?;
        Ok(())
    }

    /// Acquires a caller-provided selection table while preserving its exact shape and order.
    #[cfg(test)]
    pub fn acquire_selection_slice(
        &self,
        namespace: usize,
        grouped_ids: &[i32],
        selection_shape: &[i32],
        pass: BankAccessClass,
        stream: &Stream,
    ) -> Result<AcquiredParameterGroups, AddressableParameterBankError> {
        let expected_elements = selection_shape
            .iter()
            .try_fold(1usize, |count, dimension| {
                let dimension = usize::try_from(*dimension).map_err(|_| {
                    AddressableParameterBankError::InvalidSelectionShape(selection_shape.to_vec())
                })?;
                count.checked_mul(dimension).ok_or_else(|| {
                    AddressableParameterBankError::InvalidSelectionShape(selection_shape.to_vec())
                })
            })?;
        if expected_elements != grouped_ids.len() {
            return Err(AddressableParameterBankError::SelectionShapeMismatch {
                shape: selection_shape.to_vec(),
                elements: grouped_ids.len(),
            });
        }
        let namespace_count = self
            .namespace_entry_counts
            .get(&namespace)
            .copied()
            .ok_or(AddressableParameterBankError::UnknownNamespace { namespace })?;
        let mut demand = BTreeMap::<ParameterBankKey, u64>::new();
        for id in grouped_ids {
            let global_entry = usize::try_from(*id).map_err(|_| {
                AddressableParameterBankError::InvalidEntryId {
                    namespace,
                    entry: i64::from(*id),
                    known_owned_entries: namespace_count,
                }
            })?;
            let identity = ParameterBankKey::new(namespace, global_entry);
            if !self.catalog.contains_key(&identity) {
                return Err(AddressableParameterBankError::MissingOwnedEntry { identity });
            }
            let count = demand.entry(identity).or_insert(0);
            *count = count.saturating_add(1);
        }

        let compact_ids = demand.keys().copied().collect::<Vec<_>>();
        self.acquire_demand(demand, compact_ids, grouped_ids.len() as u64, pass, stream)
    }

    fn acquire_demand(
        &self,
        demand: BTreeMap<ParameterBankKey, u64>,
        compact_ids: Vec<ParameterBankKey>,
        selection_count: u64,
        pass: BankAccessClass,
        stream: &Stream,
    ) -> Result<AcquiredParameterGroups, AddressableParameterBankError> {
        let scratch_bytes = demand.keys().try_fold(0u64, |total, identity| {
            total
                .checked_add(self.catalog[identity])
                .ok_or(AddressableParameterBankError::ByteOverflow)
        })?;
        if scratch_bytes > self.scratch_limit {
            return Err(AddressableParameterBankError::ScratchLimitExceeded {
                required_bytes: scratch_bytes,
                limit_bytes: self.scratch_limit,
                distinct_entries: demand.len(),
            });
        }
        let before = self.resident_snapshot()?;
        let started = Instant::now();
        let mut host_hits = 0u64;
        let mut host_misses = 0u64;
        let mut device_hits = 0u64;
        let mut device_misses = 0u64;
        let mut requests = Vec::with_capacity(compact_ids.len());
        let mut host_requests = Vec::new();
        for identity in &compact_ids {
            let unit = identity.unit_id();
            let selection_demand = demand[identity];
            let host_hit = self.manager.is_resident(&unit, MemoryTier::Host)?;
            let device_hit = self.manager.is_resident(&unit, MemoryTier::Device)?;
            if host_hit {
                host_hits = host_hits.saturating_add(1);
            } else {
                host_misses = host_misses.saturating_add(1);
            }
            if device_hit {
                device_hits = device_hits.saturating_add(1);
            } else {
                device_misses = device_misses.saturating_add(1);
            }
            if host_hit || (!device_hit && self.host_budget != Some(0)) {
                host_requests.push((unit.clone(), selection_demand));
            }
            requests.push((unit, selection_demand));
        }
        if !host_requests.is_empty() {
            match self
                .manager
                .acquire_many_with_demand(&host_requests, MemoryTier::Host)
            {
                Ok(host) => drop(host),
                Err(ResidencyError::Ledger(ResidencyLedgerError::BudgetExhausted {
                    tier: MemoryTier::Host,
                    ..
                })) => {}
                Err(error) => return Err(error.into()),
            }
        }
        let transfer = self
            .manager
            .acquire_many_with_transfer(&requests, MemoryTier::Device)?;
        transfer.order_after(stream)?;
        let wait = started.elapsed();
        let after = self.resident_snapshot()?;
        let (host_evictions, host_eviction_bytes) = before.evicted(&after, MemoryTier::Host);
        let (device_evictions, device_eviction_bytes) = before.evicted(&after, MemoryTier::Device);

        let mut statistics = self
            .statistics
            .lock()
            .map_err(|_| AddressableParameterBankError::StatisticsPoisoned)?;
        let stats = statistics.pass_mut(pass);
        let distinct = compact_ids.len() as u64;
        stats.requested_selections = stats.requested_selections.saturating_add(selection_count);
        stats.distinct_entries = stats.distinct_entries.saturating_add(distinct);
        stats.coalesced_duplicates = stats
            .coalesced_duplicates
            .saturating_add(selection_count.saturating_sub(distinct));
        stats.materialization_wait = stats.materialization_wait.saturating_add(wait);
        stats.host.requests = stats.host.requests.saturating_add(distinct);
        stats.host.hits = stats.host.hits.saturating_add(host_hits);
        stats.host.misses = stats.host.misses.saturating_add(host_misses);
        stats.host.evictions = stats.host.evictions.saturating_add(host_evictions);
        stats.host.eviction_bytes = stats
            .host
            .eviction_bytes
            .saturating_add(host_eviction_bytes);
        stats.device.requests = stats.device.requests.saturating_add(distinct);
        stats.device.hits = stats.device.hits.saturating_add(device_hits);
        stats.device.misses = stats.device.misses.saturating_add(device_misses);
        stats.device.evictions = stats.device.evictions.saturating_add(device_evictions);
        stats.device.eviction_bytes = stats
            .device
            .eviction_bytes
            .saturating_add(device_eviction_bytes);
        drop(statistics);

        Ok(AcquiredParameterGroups {
            identities: compact_ids,
            demand: demand.into_values().collect(),
            scratch_bytes,
            pass,
            transfer,
        })
    }

    /// Acquires a deterministic, already coalesced generic entry demand.
    pub fn acquire_entry_demand(
        &self,
        entries: &[(ParameterBankKey, u64)],
        pass: BankAccessClass,
        stream: &Stream,
    ) -> Result<AcquiredParameterGroups, AddressableParameterBankError> {
        if entries.is_empty() {
            return Err(AddressableParameterBankError::EmptyDemand);
        }
        let mut demand = BTreeMap::new();
        let mut selection_count = 0u64;
        for &(identity, count) in entries {
            if count == 0 {
                return Err(AddressableParameterBankError::ZeroDemand { identity });
            }
            if !self.catalog.contains_key(&identity) {
                return Err(AddressableParameterBankError::MissingOwnedEntry { identity });
            }
            if demand.insert(identity, count).is_some() {
                return Err(AddressableParameterBankError::DuplicateDemand { identity });
            }
            selection_count = selection_count
                .checked_add(count)
                .ok_or(AddressableParameterBankError::ByteOverflow)?;
        }
        let compact_ids = demand.keys().copied().collect::<Vec<_>>();
        self.acquire_demand(demand, compact_ids, selection_count, pass, stream)
    }

    /// Records a completed compact-bank construction.
    pub fn record_compact_bank(
        &self,
        pass: BankAccessClass,
        bytes: u64,
        duration: Duration,
    ) -> Result<(), AddressableParameterBankError> {
        if bytes > self.scratch_limit {
            return Err(AddressableParameterBankError::ScratchLimitExceeded {
                required_bytes: bytes,
                limit_bytes: self.scratch_limit,
                distinct_entries: 0,
            });
        }
        let mut statistics = self
            .statistics
            .lock()
            .map_err(|_| AddressableParameterBankError::StatisticsPoisoned)?;
        let stats = statistics.pass_mut(pass);
        stats.compact_banks = stats.compact_banks.saturating_add(1);
        stats.compact_bank_bytes = stats.compact_bank_bytes.saturating_add(bytes);
        stats.peak_compact_bank_bytes = stats.peak_compact_bank_bytes.max(bytes);
        stats.compact_bank_time = stats.compact_bank_time.saturating_add(duration);
        Ok(())
    }

    /// Returns current entry residency, transfer, storage, and pass statistics.
    pub fn report(&self) -> Result<ParameterBankResidencyReport, AddressableParameterBankError> {
        let residency = self.manager.report()?;
        let mut host_resident_entries = 0;
        let mut device_resident_entries = 0;
        let mut host_resident_bytes = 0u64;
        let mut device_resident_bytes = 0u64;
        for unit in residency.units() {
            if unit.host_resident() {
                host_resident_entries += 1;
                host_resident_bytes =
                    host_resident_bytes.saturating_add(unit.host_allocated_bytes());
            }
            if unit.device_resident() {
                device_resident_entries += 1;
                device_resident_bytes =
                    device_resident_bytes.saturating_add(unit.device_allocated_bytes());
            }
        }
        let statistics = self
            .statistics
            .lock()
            .map_err(|_| AddressableParameterBankError::StatisticsPoisoned)?;
        Ok(ParameterBankResidencyReport {
            weight_quantization: self.weight_quantization,
            owned_entries: self.catalog.len(),
            owned_bytes: self.catalog.values().copied().sum(),
            host_resident_entries,
            device_resident_entries,
            host_resident_bytes,
            device_resident_bytes,
            peak_host_resident_bytes: residency
                .offload()
                .peak_resident_bytes()
                .get(MemoryTier::Host),
            peak_device_resident_bytes: residency
                .offload()
                .peak_resident_bytes()
                .get(MemoryTier::Device),
            bulk: statistics.bulk,
            incremental: statistics.incremental,
            residency,
            materialization: self.materialization.clone(),
        })
    }

    fn resident_snapshot(&self) -> Result<ResidentSnapshot, AddressableParameterBankError> {
        let report = self.manager.report()?;
        Ok(ResidentSnapshot {
            host: report
                .units()
                .iter()
                .filter(|unit| unit.host_resident())
                .map(|unit| (unit.id().clone(), unit.host_allocated_bytes()))
                .collect(),
            device: report
                .units()
                .iter()
                .filter(|unit| unit.device_resident())
                .map(|unit| (unit.id().clone(), unit.device_allocated_bytes()))
                .collect(),
        })
    }
}

struct ResidentSnapshot {
    host: BTreeMap<OffloadUnitId, u64>,
    device: BTreeMap<OffloadUnitId, u64>,
}

impl ResidentSnapshot {
    fn evicted(&self, after: &Self, tier: MemoryTier) -> (u64, u64) {
        let (before, after) = match tier {
            MemoryTier::Host => (&self.host, &after.host),
            MemoryTier::Device => (&self.device, &after.device),
            MemoryTier::Disk => return (0, 0),
        };
        before
            .iter()
            .filter(|(id, _)| !after.contains_key(*id))
            .fold((0u64, 0u64), |(count, bytes), (_, size)| {
                (count.saturating_add(1), bytes.saturating_add(*size))
            })
    }
}

/// A deterministic compact selection table and the leases protecting its sources.
pub struct AcquiredParameterGroups {
    identities: Vec<ParameterBankKey>,
    demand: Vec<u64>,
    scratch_bytes: u64,
    pass: BankAccessClass,
    transfer: ResidentTransfer,
}

impl AcquiredParameterGroups {
    /// Returns selected entries in compact-bank order.
    pub fn identities(&self) -> &[ParameterBankKey] {
        &self.identities
    }

    /// Returns duplicate-preserving demand counts in compact-bank order.
    pub fn demand(&self) -> &[u64] {
        &self.demand
    }

    /// Returns the conservatively reserved compact-bank byte count.
    pub const fn scratch_bytes(&self) -> u64 {
        self.scratch_bytes
    }

    /// Returns the execution-path classification used for telemetry.
    pub const fn pass(&self) -> BankAccessClass {
        self.pass
    }

    /// Returns source leases in the same order as [`Self::identities`].
    #[cfg(test)]
    pub fn leases(&self) -> &[ResidentUnitLease] {
        self.transfer.leases()
    }

    /// Concatenates one required per-entry binding along its leading axis.
    pub fn compact_binding(
        &self,
        name: &str,
        stream: &Stream,
    ) -> Result<Array, AddressableParameterBankError> {
        let values = self
            .transfer
            .leases()
            .iter()
            .map(|lease| lease.device_value(name).cloned())
            .collect::<Result<Vec<_>, _>>()?;
        if values.is_empty() {
            return Err(AddressableParameterBankError::EmptyCompactBinding {
                name: name.to_string(),
            });
        }
        Ok(concatenate_axis(&values, 0, stream)?)
    }

    /// Concatenates an optional companion binding when every entry provides it.
    pub fn optional_compact_binding(
        &self,
        name: &str,
        stream: &Stream,
    ) -> Result<Option<Array>, AddressableParameterBankError> {
        let present = self
            .transfer
            .leases()
            .iter()
            .map(|lease| lease.binding_names().any(|binding| binding == name))
            .collect::<Vec<_>>();
        if present.iter().all(|value| !value) {
            return Ok(None);
        }
        if present.iter().any(|value| !value) {
            return Err(AddressableParameterBankError::InconsistentCompanion {
                name: name.to_string(),
            });
        }
        self.compact_binding(name, stream).map(Some)
    }

    /// Returns whether no grouped entries were selected.
    pub fn is_empty(&self) -> bool {
        self.identities.is_empty()
    }
}

/// MLX integer discovery and device-side indexed movement mechanism.
#[derive(Debug, Default, Clone, Copy)]
pub struct MlxIndexedMovement;

impl IndexedMovement<MlxNeuralBackend> for MlxIndexedMovement {
    type Error = Error;

    fn index_demands(
        &mut self,
        indices: &MlxTensor,
        upper_bound: usize,
        stream: &Stream,
    ) -> Result<Vec<(usize, u64)>, Self::Error> {
        if !matches!(
            indices.as_array().dtype(),
            Dtype::Int32 | Dtype::Uint32 | Dtype::Int64 | Dtype::Uint64
        ) {
            return Err(AddressableParameterBankError::InvalidSelectionDtype {
                actual: indices.as_array().dtype(),
            }
            .into());
        }
        let upper = i32::try_from(upper_bound).map_err(|_| {
            Error::ArchitectureModel("indexed movement upper bound exceeds MLX i32 indexing".into())
        })?;
        if upper == 0 {
            return Err(Error::ArchitectureModel(
                "indexed movement upper bound must be nonzero".into(),
            ));
        }
        let flat = indices.as_array().reshape(&[-1], stream)?;
        let below = flat.lt(Array::from_int(upper), stream)?;
        let valid = if matches!(flat.dtype(), Dtype::Uint32 | Dtype::Uint64) {
            below
        } else {
            flat.ge(Array::from_int(0), stream)?
                .logical_and(below, stream)?
        };
        let invalid =
            crate::backend::compaction::count_nonzero(&valid.logical_not(stream)?, stream)?;
        let flat_i32 = if flat.dtype() == Dtype::Int32 {
            flat
        } else {
            flat.as_dtype(Dtype::Int32, stream)?
        };
        let safe = r#where(
            &valid,
            flat_i32,
            Array::zeros::<i32>(&[indices.as_array().size() as i32], stream)?,
            stream,
        )?;
        let ones = Array::ones::<i32>(&[safe.size() as i32], stream)?;
        let histogram = segment_sum(&ones, &safe, upper, 0, stream)?;
        eval([&histogram, &invalid])?;
        let invalid_count = invalid.evaluated()?.as_slice::<i32>()[0];
        if invalid_count != 0 {
            return Err(AddressableParameterBankError::InvalidSelectionSet {
                namespace: 0,
                invalid_count: invalid_count as usize,
                global_span: upper_bound,
            }
            .into());
        }
        Ok(histogram
            .evaluated()?
            .as_slice::<i32>()
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, count)| *count != 0)
            .map(|(index, count)| (index, count as u64))
            .collect())
    }

    fn remap_indices(
        &mut self,
        indices: &MlxTensor,
        mapping: &[(usize, usize)],
        stream: &Stream,
    ) -> Result<MlxTensor, Self::Error> {
        let span = mapping
            .iter()
            .map(|(source, _)| source.saturating_add(1))
            .max()
            .ok_or_else(|| Error::ArchitectureModel("indexed remapping is empty".into()))?;
        let mut lookup = vec![-1i32; span];
        for &(source, destination) in mapping {
            let destination = i32::try_from(destination).map_err(|_| {
                Error::ArchitectureModel("compact index exceeds MLX i32 indexing".into())
            })?;
            if source >= span || lookup[source] != -1 {
                return Err(Error::ArchitectureModel(
                    "indexed remapping contains a duplicate source".into(),
                ));
            }
            lookup[source] = destination;
        }
        let lookup = Array::from_slice(&lookup, &[span as i32]).copy(stream)?;
        let normalized = if indices.as_array().dtype() == Dtype::Int32 {
            indices.as_array().clone()
        } else {
            indices.as_array().as_dtype(Dtype::Int32, stream)?
        };
        Ok(MlxTensor::from_array(lookup.take(&normalized, stream)?))
    }

    fn select_rows(
        &mut self,
        value: &MlxTensor,
        start: usize,
        end: usize,
        stream: &Stream,
    ) -> Result<MlxTensor, Self::Error> {
        let start = i32::try_from(start)
            .map_err(|_| Error::ArchitectureModel("row start exceeds MLX indexing".into()))?;
        let end = i32::try_from(end)
            .map_err(|_| Error::ArchitectureModel("row end exceeds MLX indexing".into()))?;
        Ok(MlxTensor::from_array(
            value.as_array().try_index_device(start..end, stream)?,
        ))
    }

    fn concatenate_rows(
        &mut self,
        values: &[MlxTensor],
        stream: &Stream,
    ) -> Result<MlxTensor, Self::Error> {
        if values.is_empty() {
            return Err(Error::ArchitectureModel(
                "row concatenation requires at least one partition".into(),
            ));
        }
        let values = values.iter().map(MlxTensor::as_array).collect::<Vec<_>>();
        Ok(MlxTensor::from_array(concatenate_axis(&values, 0, stream)?))
    }
}

impl AddressableGroupedBank<MlxNeuralBackend> for AddressableParameterBank {
    type Acquisition = AcquiredParameterGroups;
    type Report = ParameterBankResidencyReport;
    type Error = Error;

    fn member_bytes(&self, key: NeutralParameterBankKey) -> Option<u64> {
        self.catalog
            .get(&ParameterBankKey::new(key.unit(), key.member()))
            .copied()
    }

    fn acquire(
        &mut self,
        request: ParameterBankAcquisition<'_>,
        stream: &Stream,
    ) -> Result<Self::Acquisition, Self::Error> {
        let entries = request
            .entries()
            .iter()
            .map(|(key, count)| (ParameterBankKey::new(key.unit(), key.member()), *count))
            .collect::<Vec<_>>();
        let pass = match request.access() {
            ParameterBankAccess::Bulk => BankAccessClass::Bulk,
            ParameterBankAccess::Incremental => BankAccessClass::Incremental,
            _ => {
                return Err(Error::ArchitectureModel(
                    "unsupported addressable storage access class".into(),
                ))
            }
        };
        self.acquire_entry_demand(&entries, pass, stream)
            .map_err(Into::into)
    }

    fn gated_product_groups(
        &mut self,
        acquisition: &Self::Acquisition,
        spec: &eredu_nn::GroupedGatedProductSpec,
        stream: &Stream,
    ) -> Result<<MlxNeuralBackend as GroupedNeuralBackend>::GatedProductGroups, Self::Error> {
        let started = Instant::now();
        let mut groups = MlxNeuralBackend::grouped_gated_product(spec.clone(), stream)?;
        let bindings = groups
            .local_parameter_names()
            .into_iter()
            .map(|name| {
                acquisition
                    .compact_binding(&name, stream)
                    .map(|value| (name, value))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        groups.bind_local_parameters(bindings)?;
        self.record_compact_bank(
            acquisition.pass(),
            acquisition.scratch_bytes(),
            started.elapsed(),
        )?;
        Ok(groups)
    }

    fn relu2_groups(
        &mut self,
        acquisition: &Self::Acquisition,
        spec: &eredu_nn::GroupedRelu2Spec,
        stream: &Stream,
    ) -> Result<<MlxNeuralBackend as GroupedNeuralBackend>::Relu2Groups, Self::Error> {
        let started = Instant::now();
        let mut groups = MlxNeuralBackend::grouped_relu2(spec.clone(), stream)?;
        let bindings = groups
            .local_parameter_names()
            .into_iter()
            .map(|name| {
                acquisition
                    .compact_binding(&name, stream)
                    .map(|value| (name, value))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        groups.bind_local_parameters(bindings)?;
        self.record_compact_bank(
            acquisition.pass(),
            acquisition.scratch_bytes(),
            started.elapsed(),
        )?;
        Ok(groups)
    }

    fn complete(
        &mut self,
        mut acquisition: Self::Acquisition,
        output: &MlxTensor,
        _: &Stream,
    ) -> Result<(), Self::Error> {
        eval([output.as_array()])?;
        acquisition.transfer.synchronize()?;
        Ok(())
    }

    fn report(&self) -> Result<Self::Report, Self::Error> {
        AddressableParameterBank::report(self).map_err(Into::into)
    }
}

/// Structured sparse entry cache failures.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum AddressableParameterBankError {
    /// Grouped-entry placement controls were invalid.
    #[error(transparent)]
    Policy(#[from] ParameterBankOptionsError),
    /// Bounded entry materialisation failed before cache construction.
    #[error("bounded entry materialisation failed: {source}")]
    Transformation {
        /// Shared checkpoint transformation failure.
        #[source]
        source: Box<Error>,
    },
    /// No entry definitions were supplied.
    #[error("sparse entry cache requires at least one owned entry")]
    EmptyCatalog,
    /// A generic acquisition contained no entries.
    #[error("addressable entry demand must not be empty")]
    EmptyDemand,
    /// A generic acquisition assigned no uses to one entry.
    #[error("addressable entry {identity:?} has zero demand")]
    ZeroDemand {
        /// Entry with invalid demand.
        identity: ParameterBankKey,
    },
    /// A generic acquisition repeated one entry identity.
    #[error("addressable entry demand repeats {identity:?}")]
    DuplicateDemand {
        /// Repeated entry.
        identity: ParameterBankKey,
    },
    /// One logical entry declared no materialized bytes.
    #[error("entry {identity:?} must contain at least one byte")]
    ZeroSizedEntry {
        /// Invalid logical identity.
        identity: ParameterBankKey,
    },
    /// Two catalog entries used the same namespace/global identity.
    #[error("duplicate sparse entry catalog entry {identity:?}")]
    DuplicateEntry {
        /// Duplicated logical identity.
        identity: ParameterBankKey,
    },
    /// The caller used a noncanonical residency unit id.
    #[error("entry {identity:?} requires unit id {expected}, got {actual}")]
    UnitIdentityMismatch {
        /// Logical catalog identity.
        identity: ParameterBankKey,
        /// Required stable unit id.
        expected: OffloadUnitId,
        /// Adapter-supplied unit id.
        actual: OffloadUnitId,
    },
    /// No owned entry catalog exists for this namespace.
    #[error("sparse entry cache has no catalog for namespace {namespace}")]
    UnknownNamespace {
        /// Missing namespace identity.
        namespace: usize,
    },
    /// A namespace's global entry span cannot be represented by MLX indexing.
    #[error("namespace {namespace} global entry span {global_span} exceeds MLX i32 indexing")]
    EntryCountOverflow {
        /// Namespace identity.
        namespace: usize,
        /// Required global entry span.
        global_span: usize,
    },
    /// Device-side validation found one or more out-of-range selections.
    #[error("namespace {namespace} contains {invalid_count} grouped ids outside 0..{global_span}")]
    InvalidSelectionSet {
        /// Incrementalr namespace identity.
        namespace: usize,
        /// Number of invalid selection rows.
        invalid_count: usize,
        /// Valid global entry span.
        global_span: usize,
    },
    /// A selection id was negative or otherwise invalid.
    #[error("invalid grouped entry id {entry} for namespace {namespace}; this rank catalogs {known_owned_entries} owned entries")]
    InvalidEntryId {
        /// Namespace containing the selection.
        namespace: usize,
        /// Invalid signed selection value.
        entry: i64,
        /// Owned entries cataloged for diagnostics.
        known_owned_entries: usize,
    },
    /// A valid global selection referred to an entry this cache does not own.
    #[error("grouped entry {identity:?} is not owned by this cache")]
    MissingOwnedEntry {
        /// Requested non-owned global identity.
        identity: ParameterBankKey,
    },
    /// Selectionr ids used an unsupported scalar type.
    #[error("grouped entry ids must use an integer dtype, got {actual:?}")]
    InvalidSelectionDtype {
        /// Unsupported selector-id scalar type.
        actual: Dtype,
    },
    /// A supplied selection shape had a negative dimension or overflowed.
    #[error("invalid grouped entry shape {0:?}")]
    InvalidSelectionShape(Vec<i32>),
    /// Selection shape and host values disagreed.
    #[error("grouped entry shape {shape:?} does not describe {elements} values")]
    SelectionShapeMismatch {
        /// Declared selection shape.
        shape: Vec<i32>,
        /// Supplied host value count.
        elements: usize,
    },
    /// Hidden rows, selection rows, and selection-weight rows did not align.
    #[error("grouped entry batch shapes do not align: hidden {hidden:?}, selections {selections:?}, weights {weights:?}")]
    GroupedBatchShapeMismatch {
        /// Grouped hidden-state shape.
        hidden: Vec<i32>,
        /// Grouped entry-id shape.
        selections: Vec<i32>,
        /// Grouped entry-weight shape.
        weights: Vec<i32>,
    },
    /// A caller-supplied compact bank returned the wrong row count.
    #[error("compact entry bank returned shape {actual:?}, expected {expected_rows} rows")]
    CompactBankOutputShapeMismatch {
        /// Required output rows.
        expected_rows: i32,
        /// Returned output shape.
        actual: Vec<i32>,
    },
    /// Selected entries exceed the configured temporary compact-bank allowance.
    #[error("compact entry bank for {distinct_entries} entries requires {required_bytes} bytes, exceeding the {limit_bytes}-byte scratch limit")]
    ScratchLimitExceeded {
        /// Required compact-bank bytes.
        required_bytes: u64,
        /// Configured compact-bank byte limit.
        limit_bytes: u64,
        /// Selected unique entry count.
        distinct_entries: usize,
    },
    /// Entry byte arithmetic overflowed.
    #[error("sparse entry byte accounting overflowed")]
    ByteOverflow,
    /// Cache statistics mutex was poisoned by a panic.
    #[error("sparse entry cache statistics are unavailable after a panic")]
    StatisticsPoisoned,
    /// A required compact binding had no selected source entries.
    #[error("compact entry binding {name:?} has no source arrays")]
    EmptyCompactBinding {
        /// Required binding name.
        name: String,
    },
    /// Optional companion presence differed across selected entries.
    #[error("compact entry companion {name:?} is missing from only part of the selected bank")]
    InconsistentCompanion {
        /// Inconsistent optional binding name.
        name: String,
    },
    /// Invalid offload plan configuration.
    #[error(transparent)]
    Offload(#[from] eredu_core::residency::OffloadError),
    /// Residency validation or materialization failed.
    #[error(transparent)]
    Residency(#[from] ResidencyError),
    /// MLX evaluation, synchronization, or transfer failed.
    #[error(transparent)]
    Mlx(#[from] safemlx::error::Exception),
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use eredu_checkpoint::{
        recipe::DerivedWeightRecipe,
        store::{CheckpointSource, SafetensorsWeightStore},
    };
    use safemlx::{
        host_transfer_capacity_upper_bound, Device, DeviceType, HostTransferPolicy,
        HostTransferStorageKind,
    };
    use safetensors::tensor::{serialize_to_file, Dtype as StoredDtype, TensorView};

    use super::*;
    use crate::composition::grouped_provider::ParameterBankSelection;
    use eredu_core::residency::CacheEvictionPolicy;
    use eredu_runtime::WeightBinding;

    fn stream() -> Stream {
        Stream::new_with_device(&Device::new(DeviceType::Cpu, 0))
    }

    fn fixture() -> (tempfile::TempDir, Arc<SafetensorsWeightStore>) {
        let dir = tempfile::tempdir().unwrap();
        let values = [[1i32, 2], [3, 4], [5, 6]]
            .into_iter()
            .map(|values| {
                values
                    .into_iter()
                    .flat_map(i32::to_le_bytes)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        serialize_to_file(
            values.iter().enumerate().map(|(entry, bytes)| {
                (
                    format!("entry.{entry}"),
                    TensorView::new(StoredDtype::I32, vec![1, 2], bytes).unwrap(),
                )
            }),
            None,
            &dir.path().join("model.safetensors"),
        )
        .unwrap();
        let store = Arc::new(SafetensorsWeightStore::open(dir.path()).unwrap());
        (dir, store)
    }

    fn entries() -> Vec<ParameterBankEntry> {
        (0..3)
            .map(|entry| {
                let identity = ParameterBankKey::new(2, entry);
                let bindings = [
                    WeightBinding::new(
                        "weight",
                        format!("entry.{entry}"),
                        TensorSelection::Full,
                        8,
                    )
                    .unwrap(),
                    WeightBinding::new("scale", format!("entry.{entry}"), TensorSelection::Full, 8)
                        .unwrap(),
                ];
                let unit = OffloadUnit::new(identity.unit_id(), bindings).unwrap();
                ParameterBankEntry::new(identity, unit, 16).unwrap()
            })
            .collect()
    }

    fn cache(
        store: Arc<SafetensorsWeightStore>,
        device: u64,
        host: u64,
        scratch: u64,
        eviction: CacheEvictionPolicy,
    ) -> AddressableParameterBank {
        cache_with_target(store, device, host, scratch, scratch, eviction)
    }

    fn cache_with_target(
        store: Arc<SafetensorsWeightStore>,
        device: u64,
        host: u64,
        scratch: u64,
        bulk_target: u64,
        eviction: CacheEvictionPolicy,
    ) -> AddressableParameterBank {
        let binding_capacity =
            host_transfer_capacity_upper_bound(8, HostTransferPolicy::Transfer).unwrap() as u64;
        let physical_host = (host / 8).checked_mul(binding_capacity).unwrap();
        let storage = OffloadConfig::new(Some(device), Some(physical_host), 1)
            .unwrap()
            .with_eviction_policy(eviction);
        AddressableParameterBank::new(
            store,
            entries(),
            ParameterBankOptions::new(storage, scratch, bulk_target).unwrap(),
            stream(),
            stream(),
        )
        .unwrap()
    }

    #[test]
    fn quantized_cache_materializes_only_rank_local_entry_and_tp_recipes() {
        let dir = tempfile::tempdir().unwrap();
        let gate = (0..2 * 4 * 64)
            .map(|index| (index as f32 - 127.0) / 32.0)
            .collect::<Vec<_>>();
        let down = gate.iter().map(|value| value * 0.5).collect::<Vec<_>>();
        let gate_bytes = gate
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let down_bytes = down
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        serialize_to_file(
            [
                (
                    "entries.gate",
                    TensorView::new(StoredDtype::F32, vec![2, 4, 64], &gate_bytes).unwrap(),
                ),
                (
                    "entries.down",
                    TensorView::new(StoredDtype::F32, vec![2, 4, 64], &down_bytes).unwrap(),
                ),
            ],
            None,
            &dir.path().join("model.safetensors"),
        )
        .unwrap();
        let store: Arc<dyn eredu_checkpoint::store::CheckpointSource> =
            Arc::new(SafetensorsWeightStore::open(dir.path()).unwrap());
        let entries = (0..2)
            .map(|entry| {
                let identity = ParameterBankKey::new(3, entry);
                let bindings = ["gate", "down"].map(|projection| {
                    let owned = DerivedWeightRecipe::source(
                        format!("entries.{projection}"),
                        TensorSelection::Range {
                            axis: 0,
                            start: entry,
                            end: entry + 1,
                        },
                    );
                    let tp = DerivedWeightRecipe::Select {
                        input: Box::new(owned),
                        selection: TensorSelection::Range {
                            axis: 1,
                            start: entry * 2,
                            end: entry * 2 + 2,
                        },
                    };
                    WeightBinding::from_recipe(format!("{projection}_proj"), tp, 512)
                        .unwrap()
                        .with_quantization_companions(
                            format!("{projection}_proj_scales"),
                            format!("{projection}_proj_biases"),
                        )
                        .unwrap()
                });
                let unit = OffloadUnit::new(identity.unit_id(), bindings).unwrap();
                ParameterBankEntry::new(identity, unit, 1_024).unwrap()
            })
            .collect::<Vec<_>>();
        let options =
            ParameterBankOptions::new(OffloadConfig::new(None, None, 1).unwrap(), 1_024, 1_024)
                .unwrap();
        let cache = AddressableParameterBank::new_quantized_shared(
            store,
            entries,
            options,
            WeightQuantization::Affine(Default::default()),
            stream(),
            stream(),
        )
        .unwrap();

        assert_eq!(
            cache.weight_quantization(),
            Some(WeightQuantization::Affine(Default::default()))
        );
        let report = cache.report().unwrap();
        assert_eq!(report.owned_entries, 2);
        assert_eq!(report.owned_bytes, 320);
        assert_eq!(
            report.materialization,
            Some(WeightMaterializationReport {
                admitted_working_set_bytes: 320,
                transformed_weights: 4,
                source_tiles: 8,
                peak_in_flight_tiles: 1,
                source_bytes_read: 2_048,
                output_bytes: 320,
                peak_planned_working_set_bytes: 296,
                largest_source_tile_bytes: 256,
                largest_output_tile_bytes: 40,
            })
        );
        let acquired = cache
            .acquire_selection_slice(3, &[1], &[1, 1], BankAccessClass::Incremental, &stream())
            .unwrap();
        assert_eq!(
            acquired
                .compact_binding("gate_proj", &stream())
                .unwrap()
                .shape(),
            &[1, 2, 8]
        );
        assert_eq!(
            acquired
                .compact_binding("gate_proj_scales", &stream())
                .unwrap()
                .shape(),
            &[1, 2, 1]
        );
    }

    #[test]
    fn entry_quantization_uses_only_declared_roles_and_companion_names() {
        let dir = tempfile::tempdir().unwrap();
        let values = (0..8 * 64)
            .map(|index| index as f32 / 16.0)
            .collect::<Vec<_>>();
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        serialize_to_file(
            [
                (
                    "legitimate.bias_scale_blocks",
                    TensorView::new(StoredDtype::F32, vec![8, 64], &bytes).unwrap(),
                ),
                (
                    "preserved.matrix",
                    TensorView::new(StoredDtype::F32, vec![8, 64], &bytes).unwrap(),
                ),
            ],
            None,
            &dir.path().join("model.safetensors"),
        )
        .unwrap();
        let store: Arc<dyn CheckpointSource> =
            Arc::new(SafetensorsWeightStore::open(dir.path()).unwrap());
        let identity = ParameterBankKey::new(0, 0);
        let quantized = WeightBinding::from_recipe(
            "legitimate.bias_scale_blocks",
            DerivedWeightRecipe::source("legitimate.bias_scale_blocks", TensorSelection::Full),
            bytes.len() as u64,
        )
        .unwrap()
        .with_quantization_companions("declared.scale", "declared.bias")
        .unwrap();
        let preserved = WeightBinding::from_recipe(
            "ordinary_matrix",
            DerivedWeightRecipe::source("preserved.matrix", TensorSelection::Full),
            bytes.len() as u64,
        )
        .unwrap();
        let entry = ParameterBankEntry::new(
            identity,
            OffloadUnit::new(identity.unit_id(), [quantized, preserved]).unwrap(),
            (bytes.len() * 2) as u64,
        )
        .unwrap();

        let transformed = quantize_entry_catalog(
            store,
            vec![entry],
            WeightQuantization::Affine(Default::default()),
            1_024,
            &stream(),
        )
        .unwrap();

        assert_eq!(transformed.report.transformed_weights, 1);
        assert_eq!(
            transformed.entries[0]
                .unit
                .bindings()
                .iter()
                .map(WeightBinding::name)
                .collect::<Vec<_>>(),
            [
                "declared.bias",
                "declared.scale",
                "legitimate.bias_scale_blocks",
                "ordinary_matrix",
            ]
        );
    }

    #[test]
    fn coalesces_selections_in_global_order_and_separates_pass_counters() {
        let (_dir, store) = fixture();
        let cache = cache(store, 32, 32, 32, CacheEvictionPolicy::LeastRecentlyUsed);
        let first = cache
            .acquire_selection_slice(2, &[2, 0, 2, 0], &[2, 2], BankAccessClass::Bulk, &stream())
            .unwrap();
        assert_eq!(
            first.identities(),
            &[ParameterBankKey::new(2, 0), ParameterBankKey::new(2, 2)]
        );
        assert_eq!(first.demand(), &[2, 2]);
        drop(first);

        let host = cache
            .manager
            .acquire(&ParameterBankKey::new(2, 0).unit_id(), MemoryTier::Host)
            .unwrap();
        assert!(matches!(
            host.host_value("weight").unwrap().storage_kind().unwrap(),
            HostTransferStorageKind::Cpu
                | HostTransferStorageKind::MetalShared
                | HostTransferStorageKind::CudaPinned
        ));
        assert!(matches!(
            host.device_value("weight"),
            Err(ResidencyError::HostBindingIsNotArray { .. })
        ));
        drop(host);

        let second = cache
            .acquire_selection_slice(2, &[0, 2], &[1, 2], BankAccessClass::Incremental, &stream())
            .unwrap();
        drop(second);
        let report = cache.report().unwrap();
        assert_eq!(report.bulk.requested_selections, 4);
        assert_eq!(report.bulk.distinct_entries, 2);
        assert_eq!(report.bulk.coalesced_duplicates, 2);
        assert_eq!(report.bulk.device.misses, 2);
        assert_eq!(report.incremental.requested_selections, 2);
        assert_eq!(report.incremental.device.hits, 2);
        assert_eq!(report.owned_entries, 3);
        assert_eq!(report.owned_bytes, 48);
    }

    #[test]
    fn resident_store_pins_every_entry_and_never_rereads_checkpoint_weights() {
        let (_dir, store) = fixture();
        let cache = AddressableParameterBank::new_resident_shared(
            store.clone(),
            entries(),
            stream(),
            stream(),
        )
        .unwrap();
        let initialized = cache.report().unwrap();
        assert_eq!(initialized.owned_entries, 3);
        assert_eq!(initialized.device_resident_entries, 3);
        assert_eq!(initialized.device_resident_bytes, initialized.owned_bytes);
        assert_eq!(initialized.host_resident_entries, 0);

        let reads_after_load = store.source_diagnostics().unwrap().physical_reads;
        let acquired = cache
            .acquire_selection_slice(
                2,
                &[2, 0, 2],
                &[3, 1],
                BankAccessClass::Incremental,
                &stream(),
            )
            .unwrap();
        let compact = acquired.compact_binding("weight", &stream()).unwrap();
        eval([&compact]).unwrap();

        assert_eq!(
            store.source_diagnostics().unwrap().physical_reads,
            reads_after_load
        );
        let executed = cache.report().unwrap();
        assert_eq!(executed.incremental.device.hits, 2);
        assert_eq!(executed.incremental.device.misses, 0);
        assert_eq!(executed.incremental.device.evictions, 0);
    }

    #[test]
    fn bulk_target_is_required_and_cannot_exceed_scratch() {
        let storage = OffloadConfig::new(Some(48), Some(0), 1).unwrap();
        assert!(matches!(
            ParameterBankOptions::new(storage, 64, 0),
            Err(ParameterBankOptionsError::ZeroBulkBankTarget)
        ));
        assert!(matches!(
            ParameterBankOptions::new(storage, 64, 65),
            Err(ParameterBankOptionsError::BulkBankTargetExceedsScratch { .. })
        ));
    }

    #[test]
    fn bounded_execution_chunks_bulk_but_not_incremental_and_preserves_row_order() {
        let (_dir, store) = fixture();
        let cache = cache_with_target(store, 48, 0, 48, 32, CacheEvictionPolicy::LeastRecentlyUsed);
        let execution = stream();
        let hidden = Array::from_slice(&[1f32, 2., 3., 4., 5., 6.], &[3, 2]);
        let selections = Array::from_slice(&[0i32, 1, 1, 2, 2, 0], &[3, 2]);
        let weights = Array::from_slice(&[0.5f32; 6], &[3, 2]);
        let mut bulk_banks = 0;
        let output = crate::composition::grouped_provider::execute_selections_bounded(
            &cache,
            ParameterBankSelection::new(2, &hidden, &selections, &weights, BankAccessClass::Bulk),
            &execution,
            |hidden, _acquired, _compact, _weights, _stream| {
                bulk_banks += 1;
                Ok(hidden.clone())
            },
        )
        .unwrap();
        assert_eq!(bulk_banks, 3);
        assert_eq!(
            output.evaluated().unwrap().as_slice::<f32>(),
            hidden.evaluated().unwrap().as_slice::<f32>()
        );

        let mut incremental_banks = 0;
        crate::composition::grouped_provider::execute_selections_bounded(
            &cache,
            ParameterBankSelection::new(
                2,
                &hidden,
                &selections,
                &weights,
                BankAccessClass::Incremental,
            ),
            &execution,
            |hidden, _acquired, _compact, _weights, _stream| {
                incremental_banks += 1;
                Ok(hidden.clone())
            },
        )
        .unwrap();
        assert_eq!(incremental_banks, 1);

        let distributed_selections = Array::from_slice(&[0i32, 1, 2], &[3]);
        let distributed_weights = Array::from_slice(&[1f32; 3], &[3]);
        let mut distributed_banks = 0;
        crate::composition::grouped_provider::execute_selections_bounded(
            &cache,
            ParameterBankSelection::new(
                2,
                &hidden,
                &distributed_selections,
                &distributed_weights,
                BankAccessClass::Bulk,
            ),
            &execution,
            |hidden, _acquired, _compact, _weights, _stream| {
                distributed_banks += 1;
                Ok(hidden.clone())
            },
        )
        .unwrap();
        assert_eq!(distributed_banks, 2);
    }

    #[test]
    fn indexed_movement_validates_before_loading_and_bank_coalesces_demand() {
        let (_dir, store) = fixture();
        let cache = cache(store, 32, 32, 32, CacheEvictionPolicy::LeastRecentlyUsed);
        let execution = stream();
        let acquired = cache
            .acquire_selection_slice(2, &[2, 0, 2, 0], &[2, 2], BankAccessClass::Bulk, &execution)
            .unwrap();
        assert_eq!(
            acquired.identities(),
            &[ParameterBankKey::new(2, 0), ParameterBankKey::new(2, 2)]
        );
        assert_eq!(acquired.demand(), &[2, 2]);
        drop(acquired);

        let mut movement = MlxIndexedMovement;
        let invalid = MlxTensor::from_array(Array::from_slice(&[-1i32, 0], &[2]));
        assert!(matches!(
            movement.index_demands(&invalid, 3, &execution),
            Err(Error::AddressableParameterBank(
                AddressableParameterBankError::InvalidSelectionSet {
                    invalid_count: 1,
                    ..
                }
            ))
        ));
        let report = cache.report().unwrap();
        assert_eq!(report.incremental.requested_selections, 0);

        let narrowing_alias = MlxTensor::from_array(Array::from_slice(&[1u64 << 32], &[1]));
        assert!(matches!(
            movement.index_demands(&narrowing_alias, 3, &execution),
            Err(Error::AddressableParameterBank(
                AddressableParameterBankError::InvalidSelectionSet {
                    invalid_count: 1,
                    ..
                }
            ))
        ));
    }

    #[test]
    fn rejects_invalid_missing_and_over_scratch_selections_before_loading() {
        let (_dir, store) = fixture();
        let cache = cache(store, 48, 0, 16, CacheEvictionPolicy::LeastRecentlyUsed);
        assert!(matches!(
            cache.acquire_selection_slice(2, &[-1], &[1], BankAccessClass::Incremental, &stream()),
            Err(AddressableParameterBankError::InvalidEntryId { .. })
        ));
        assert!(matches!(
            cache.acquire_selection_slice(2, &[3], &[1], BankAccessClass::Incremental, &stream()),
            Err(AddressableParameterBankError::MissingOwnedEntry { .. })
        ));
        assert!(matches!(
            cache.acquire_selection_slice(2, &[0, 1], &[2], BankAccessClass::Bulk, &stream()),
            Err(AddressableParameterBankError::ScratchLimitExceeded { .. })
        ));
        let report = cache.report().unwrap();
        assert_eq!(report.device_resident_entries, 0);
        assert_eq!(report.bulk.requested_selections, 0);
        assert_eq!(report.incremental.requested_selections, 0);
    }

    #[test]
    fn empty_selections_do_not_materialize_or_build_a_bank() {
        let (_dir, store) = fixture();
        let cache = cache(store, 16, 16, 16, CacheEvictionPolicy::LeastRecentlyUsed);
        let acquired = cache
            .acquire_selection_slice(2, &[], &[0, 2], BankAccessClass::Incremental, &stream())
            .unwrap();
        assert!(acquired.is_empty());
        assert_eq!(acquired.scratch_bytes(), 0);
        drop(acquired);
        let report = cache.report().unwrap();
        assert_eq!(report.host_resident_entries, 0);
        assert_eq!(report.device_resident_entries, 0);
        assert_eq!(report.incremental.compact_banks, 0);
    }

    #[test]
    fn lfu_uses_duplicate_selection_demand_and_deterministic_recency_ties() {
        let (_dir, store) = fixture();
        let cache = cache(store, 32, 0, 32, CacheEvictionPolicy::LeastFrequentlyUsed);
        drop(
            cache
                .acquire_selection_slice(
                    2,
                    &[0, 0, 0],
                    &[3],
                    BankAccessClass::Incremental,
                    &stream(),
                )
                .unwrap(),
        );
        drop(
            cache
                .acquire_selection_slice(2, &[1], &[1], BankAccessClass::Incremental, &stream())
                .unwrap(),
        );
        drop(
            cache
                .acquire_selection_slice(2, &[2], &[1], BankAccessClass::Incremental, &stream())
                .unwrap(),
        );
        let report = cache.report().unwrap();
        let resident = report
            .residency
            .units()
            .iter()
            .filter(|unit| unit.device_resident())
            .map(|unit| unit.id().as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            resident,
            vec![
                ParameterBankKey::new(2, 0).unit_id().as_str(),
                ParameterBankKey::new(2, 2).unit_id().as_str()
            ]
        );
        assert_eq!(report.incremental.device.evictions, 1);
    }
}
