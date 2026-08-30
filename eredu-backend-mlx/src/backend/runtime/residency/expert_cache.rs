//! MLX sparse routed-expert caching over backend-neutral residency plans.

//!
//! Each logical expert is an atomic disk-planned residency unit. Route ids are
//! inspected once per routed block, validated before acquisition, coalesced in
//! deterministic global-id order, and rewritten to a temporary compact bank.

use eredu_checkpoint::{store::TensorSelection, WeightQuantization};
use eredu_runtime::{
    ExpertCacheLoadOptions, ExpertIdentity, ExpertPass, OffloadUnit, ResidencyReport,
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
use crate::{
    backend::error::Error,
    backend::runtime::checkpoint::bounded_quantization::{
        BoundedQuantizationPlan, BoundedQuantizationTarget, BoundedQuantizedWeightStore,
    },
    backend::runtime::residency::manager::{ResidencyError, ResidencyManager, ResidentTransfer},
};
use eredu_core::residency::{
    MemoryTier, OffloadPlan, OffloadUnitId, OffloadUnitSpec, ResidencyLedgerError, ResidencyPolicy,
};

#[cfg(test)]
use eredu_core::residency::OffloadConfig;

/// One routed hidden-state batch and its layer-local expert assignments.
#[derive(Debug, Clone, Copy)]
pub struct ExpertRouteBatch<'a> {
    layer: usize,
    hidden: &'a Array,
    expert_ids: &'a Array,
    weights: &'a Array,
    pass: ExpertPass,
}

impl<'a> ExpertRouteBatch<'a> {
    /// Binds routed activations, expert ids, and weights to one decoder layer.
    pub const fn new(
        layer: usize,
        hidden: &'a Array,
        expert_ids: &'a Array,
        weights: &'a Array,
        pass: ExpertPass,
    ) -> Self {
        Self {
            layer,
            hidden,
            expert_ids,
            weights,
            pass,
        }
    }
}

/// One atomic expert definition supplied by an architecture adapter.
#[derive(Clone)]
pub struct ExpertCatalogEntry {
    identity: ExpertIdentity,
    unit: OffloadUnit,
    bytes: u64,
}

impl ExpertCatalogEntry {
    /// Creates one catalog entry and verifies its stable unit identity.
    pub fn new(
        identity: ExpertIdentity,
        unit: OffloadUnit,
        bytes: u64,
    ) -> Result<Self, ExpertCacheError> {
        if bytes == 0 {
            return Err(ExpertCacheError::ZeroSizedExpert { identity });
        }
        let expected = identity.unit_id();
        if unit.id() != &expected {
            return Err(ExpertCacheError::UnitIdentityMismatch {
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
    pub const fn identity(&self) -> ExpertIdentity {
        self.identity
    }

    /// Returns the atomic materialized byte length.
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    fn into_parts(self) -> (ExpertIdentity, OffloadUnit) {
        (self.identity, self.unit)
    }
}

/// Result of replacing dense expert bindings with a disk-backed packed overlay.
pub struct QuantizedExpertCatalog {
    /// Store supplying synthetic packed bindings and delegating all other keys.
    pub store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    /// Expert units rebuilt against the packed store.
    pub entries: Vec<ExpertCatalogEntry>,
    /// Deterministic bounded-materialisation telemetry.
    pub report: WeightMaterializationReport,
}

/// Quantizes every floating expert projection through its authoritative
/// rank-local semantic recipe and rebuilds the catalog against packed keys.
pub fn quantize_expert_catalog(
    source: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    entries: Vec<ExpertCatalogEntry>,
    quantization: WeightQuantization,
    max_working_set_bytes: u64,
    source_stream: &Stream,
) -> Result<QuantizedExpertCatalog, Error> {
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
                "__eredu.expert.layer.{:05}.global.{:05}.{}.weight",
                identity.layer,
                identity.global_expert,
                binding.name()
            );
            let target_prefix = target_name
                .strip_suffix(".weight")
                .expect("synthetic expert target has a weight suffix");
            let target = BoundedQuantizationTarget::from_recipe(
                target_name.clone(),
                format!("{target_prefix}.scales"),
                Some(format!("{target_prefix}.biases")),
                recipe,
            )?;
            packed_catalog_bytes = packed_catalog_bytes
                .checked_add(packed_projection_bytes(metadata.shape(), quantization)?)
                .ok_or_else(|| {
                    Error::Quantization("packed expert catalog size overflowed".into())
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
            "expert catalog contains no floating projection bindings to quantize".into(),
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
                        .expect("affine expert target declared a bias identity"),
                    store.as_ref(),
                )?);
            }
        }
        let bytes = bindings.iter().try_fold(0u64, |total, binding| {
            total.checked_add(binding.expected_bytes()).ok_or_else(|| {
                Error::Quantization("quantized expert catalog byte total overflowed".into())
            })
        })?;
        rebuilt.push(ExpertCatalogEntry::new(
            identity,
            OffloadUnit::new(identity.unit_id(), bindings)?,
            bytes,
        )?);
    }
    Ok(QuantizedExpertCatalog {
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
        .ok_or_else(|| Error::Quantization("expert projection has no input dimension".into()))?;
    let rows = rows.iter().try_fold(1u64, |count, dimension| {
        count
            .checked_mul(*dimension as u64)
            .ok_or_else(|| Error::Quantization("expert projection row count overflowed".into()))
    })?;
    let packed = (columns as u64)
        .checked_mul(quantization.bits() as u64)
        .and_then(|bits| bits.checked_div(8))
        .ok_or_else(|| Error::Quantization("packed expert row size overflowed".into()))?;
    let groups = columns
        .checked_div(quantization.group_size() as usize)
        .ok_or_else(|| Error::Quantization("expert group geometry is invalid".into()))?
        as u64;
    let scale_bytes = if matches!(quantization, WeightQuantization::MxFp4) {
        groups
    } else {
        groups
            .checked_mul(4)
            .ok_or_else(|| Error::Quantization("expert scale row size overflowed".into()))?
    };
    let bias_bytes = if quantization.has_biases() {
        groups
            .checked_mul(4)
            .ok_or_else(|| Error::Quantization("expert bias row size overflowed".into()))?
    } else {
        0
    };
    let row_bytes = packed
        .checked_add(scale_bytes)
        .and_then(|bytes| bytes.checked_add(bias_bytes))
        .ok_or_else(|| Error::Quantization("packed expert row total overflowed".into()))?;
    rows.checked_mul(row_bytes)
        .ok_or_else(|| Error::Quantization("packed expert projection size overflowed".into()))
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
pub struct ExpertTierStatistics {
    /// Logical expert acquisition requests after duplicate coalescing.
    pub requests: u64,
    /// Requests served by an already resident copy.
    pub hits: u64,
    /// Requests that materialized or promoted a copy.
    pub misses: u64,
    /// Copies evicted while satisfying cache requests.
    pub evictions: u64,
    /// Bytes evicted while satisfying cache requests.
    pub eviction_bytes: u64,
}

/// Cumulative statistics for one public execution-path class.
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq)]
pub struct ExpertPassStatistics {
    /// Route rows requested by the router, including duplicates.
    pub requested_routes: u64,
    /// Distinct logical experts requested after coalescing.
    pub distinct_experts: u64,
    /// Duplicate requests eliminated before materialization.
    pub coalesced_duplicates: u64,
    /// Temporary compact banks built.
    pub compact_banks: u64,
    /// Cumulative compact-bank bytes.
    pub compact_bank_bytes: u64,
    /// Peak temporary compact-bank bytes.
    pub peak_compact_bank_bytes: u64,
    /// Cumulative compact-bank construction time.
    pub compact_bank_time: Duration,
    /// Time preparing and reserving expert materialization or promotion.
    ///
    /// Deferred device completion is charged to the dependent expert output,
    /// not this counter.
    pub materialization_wait: Duration,
    /// Host-tier cache activity.
    pub host: ExpertTierStatistics,
    /// Device-tier cache activity.
    pub device: ExpertTierStatistics,
}

/// Point-in-time expert residency and execution report.
pub struct ExpertCacheReport {
    /// Packed encoding used by load-time transformed expert bindings.
    pub weight_quantization: Option<WeightQuantization>,
    /// Owned logical expert count.
    pub owned_experts: usize,
    /// Owned logical expert bytes, including cold checkpoint-only experts.
    pub owned_bytes: u64,
    /// Current host-resident expert count.
    pub host_resident_experts: usize,
    /// Current device-resident expert count.
    pub device_resident_experts: usize,
    /// Current physical capacity of host-resident expert allocations.
    pub host_resident_bytes: u64,
    /// Current device-resident expert bytes.
    pub device_resident_bytes: u64,
    /// Peak physical capacity of host-resident expert allocations.
    pub peak_host_resident_bytes: u64,
    /// Peak device-resident expert bytes.
    pub peak_device_resident_bytes: u64,
    /// Prompt-processing statistics.
    pub prefill: ExpertPassStatistics,
    /// Autoregressive decode statistics.
    pub decode: ExpertPassStatistics,
    /// Underlying logical transfer and checkpoint diagnostics.
    pub residency: ResidencyReport,
    /// Bounded load-time expert materialisation telemetry, when the catalog
    /// was transformed from floating checkpoint weights.
    pub materialization: Option<WeightMaterializationReport>,
}

#[derive(Default)]
struct ExpertStatistics {
    prefill: ExpertPassStatistics,
    decode: ExpertPassStatistics,
}

impl ExpertStatistics {
    fn pass_mut(&mut self, pass: ExpertPass) -> &mut ExpertPassStatistics {
        match pass {
            ExpertPass::Prefill => &mut self.prefill,
            ExpertPass::Decode => &mut self.decode,
        }
    }
}

/// Shared expert catalog, scheduler, residency manager, and telemetry.
pub struct ExpertCache {
    manager: ResidencyManager,
    catalog: BTreeMap<ExpertIdentity, u64>,
    #[cfg(test)]
    layer_expert_counts: BTreeMap<usize, usize>,
    layer_global_spans: BTreeMap<usize, usize>,
    host_budget: Option<u64>,
    scratch_limit: u64,
    prefill_bank_target: u64,
    statistics: Mutex<ExpertStatistics>,
    weight_quantization: Option<WeightQuantization>,
    materialization: Option<WeightMaterializationReport>,
}

impl ExpertCache {
    /// Creates a disk-planned cache over exactly the supplied owned experts.
    pub fn new<S>(
        store: Arc<S>,
        entries: impl IntoIterator<Item = ExpertCatalogEntry>,
        options: ExpertCacheLoadOptions,
        source_stream: Stream,
        device_stream: Stream,
    ) -> Result<Self, ExpertCacheError>
    where
        S: eredu_checkpoint::store::CheckpointSource + 'static,
    {
        let store: Arc<dyn eredu_checkpoint::store::CheckpointSource> = store;
        Self::new_shared(store, entries, options, source_stream, device_stream)
    }

    /// Creates a cache from an already type-erased checkpoint store.
    pub fn new_shared(
        store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
        entries: impl IntoIterator<Item = ExpertCatalogEntry>,
        options: ExpertCacheLoadOptions,
        source_stream: Stream,
        device_stream: Stream,
    ) -> Result<Self, ExpertCacheError> {
        Self::new_shared_with_policy(
            store,
            entries,
            options,
            ResidencyPolicy::Cacheable,
            MemoryTier::Disk,
            source_stream,
            device_stream,
            None,
            None,
        )
    }

    /// Creates a cache after boundedly transforming its rank-local semantic
    /// expert recipes into packed checkpoint bindings.
    pub fn new_quantized_shared(
        store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
        entries: Vec<ExpertCatalogEntry>,
        options: ExpertCacheLoadOptions,
        quantization: WeightQuantization,
        source_stream: Stream,
        device_stream: Stream,
    ) -> Result<Self, ExpertCacheError> {
        let transformed = quantize_expert_catalog(
            store,
            entries,
            quantization,
            options.compact_bank_scratch_bytes,
            &source_stream,
        )
        .map_err(|source| ExpertCacheError::Transformation {
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

    /// Creates a fully resident store over exactly the supplied owned experts.
    ///
    /// Every expert is pinned on the execution device during construction. The
    /// same route compaction and architecture-neutral binding machinery is used
    /// by sparse and resident execution, but resident experts cannot be evicted
    /// and never trigger checkpoint reads during a forward pass.
    pub fn new_resident_shared(
        store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
        entries: impl IntoIterator<Item = ExpertCatalogEntry>,
        source_stream: Stream,
        device_stream: Stream,
    ) -> Result<Self, ExpertCacheError> {
        Self::new_shared_with_policy(
            store,
            entries,
            ExpertCacheLoadOptions::default(),
            ResidencyPolicy::Pinned,
            MemoryTier::Device,
            source_stream,
            device_stream,
            None,
            None,
        )
    }

    /// Creates a fully resident cache after boundedly transforming the exact
    /// rank-local expert catalog into its requested packed representation.
    pub fn new_quantized_resident_shared(
        store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
        entries: Vec<ExpertCatalogEntry>,
        quantization: WeightQuantization,
        source_stream: Stream,
        device_stream: Stream,
    ) -> Result<Self, ExpertCacheError> {
        let options = ExpertCacheLoadOptions::default();
        let transformed = quantize_expert_catalog(
            store,
            entries,
            quantization,
            options.compact_bank_scratch_bytes,
            &source_stream,
        )
        .map_err(|source| ExpertCacheError::Transformation {
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
        entries: impl IntoIterator<Item = ExpertCatalogEntry>,
        options: ExpertCacheLoadOptions,
        policy: ResidencyPolicy,
        initial_tier: MemoryTier,
        source_stream: Stream,
        device_stream: Stream,
        weight_quantization: Option<WeightQuantization>,
        materialization: Option<WeightMaterializationReport>,
    ) -> Result<Self, ExpertCacheError> {
        options.validate()?;
        let mut catalog = BTreeMap::new();
        let mut definitions = Vec::new();
        let mut specs = Vec::new();
        #[cfg(test)]
        let mut layer_expert_counts = BTreeMap::new();
        let mut layer_global_spans = BTreeMap::new();
        for entry in entries {
            if catalog.insert(entry.identity, entry.bytes).is_some() {
                return Err(ExpertCacheError::DuplicateExpert {
                    identity: entry.identity,
                });
            }
            #[cfg(test)]
            {
                *layer_expert_counts.entry(entry.identity.layer).or_insert(0) += 1;
            }
            layer_global_spans
                .entry(entry.identity.layer)
                .and_modify(|span: &mut usize| {
                    *span = (*span).max(entry.identity.global_expert + 1)
                })
                .or_insert(entry.identity.global_expert + 1);
            specs.push(OffloadUnitSpec::new(
                entry.identity.unit_id(),
                entry.bytes,
                policy,
                initial_tier,
            )?);
            definitions.push(entry.unit);
        }
        if catalog.is_empty() {
            return Err(ExpertCacheError::EmptyCatalog);
        }
        let plan = OffloadPlan::new(options.experts, specs)?;
        let manager =
            ResidencyManager::new_shared(store, plan, definitions, source_stream, device_stream)?;
        manager.initialize()?;
        Ok(Self {
            manager,
            catalog,
            #[cfg(test)]
            layer_expert_counts,
            layer_global_spans,
            host_budget: if initial_tier == MemoryTier::Device {
                Some(0)
            } else {
                options.experts.host_budget_bytes()
            },
            scratch_limit: options.compact_bank_scratch_bytes,
            prefill_bank_target: options.prefill_compact_bank_target_bytes,
            statistics: Mutex::new(ExpertStatistics::default()),
            weight_quantization,
            materialization,
        })
    }

    /// Returns the load-time encoding of transformed expert bindings.
    pub const fn weight_quantization(&self) -> Option<WeightQuantization> {
        self.weight_quantization
    }

    /// Returns the underlying reusable residency manager.
    pub const fn residency_manager(&self) -> &ResidencyManager {
        &self.manager
    }

    /// Returns a conservative row count whose worst-case distinct routes fit
    /// the configured prefill compact-bank target.
    fn route_chunk_rows(&self, routes_per_row: usize) -> usize {
        if routes_per_row == 0 {
            return 1;
        }
        let max_expert_bytes = self.catalog.values().copied().max().unwrap_or(1);
        let bytes_per_row =
            max_expert_bytes.saturating_mul(u64::try_from(routes_per_row).unwrap_or(u64::MAX));
        let budget = self.scratch_limit.min(self.prefill_bank_target).max(1);
        let rows = budget.checked_div(bytes_per_row).unwrap_or(0).max(1);
        usize::try_from(rows).unwrap_or(usize::MAX)
    }

    /// Executes routed experts through bounded compact banks.
    ///
    /// Prefill rows are split conservatively from the catalog's largest expert
    /// and the configured target. Decode remains a single bank. The callback
    /// constructs and executes one architecture-specific compact bank; this
    /// method owns acquisition, output evaluation, lease completion, and
    /// concatenation in original row order.
    pub fn execute_routes_bounded<F>(
        &self,
        batch: ExpertRouteBatch<'_>,
        stream: &Stream,
        mut execute_bank: F,
    ) -> Result<Array, ExpertCacheError>
    where
        F: FnMut(&Array, &AcquiredExperts, &Array, &Stream) -> Result<Array, ExpertCacheError>,
    {
        let ExpertRouteBatch {
            layer,
            hidden: routed_hidden,
            expert_ids: routed_ids,
            weights: route_weights,
            pass,
        } = batch;
        if routed_hidden.ndim() == 0
            || routed_ids.ndim() == 0
            || route_weights.ndim() == 0
            || routed_hidden.dim(0) != routed_ids.dim(0)
            || routed_hidden.dim(0) != route_weights.dim(0)
        {
            return Err(ExpertCacheError::RoutedBatchShapeMismatch {
                hidden: routed_hidden.shape().to_vec(),
                routes: routed_ids.shape().to_vec(),
                weights: route_weights.shape().to_vec(),
            });
        }
        let routes_per_row = routed_ids.shape()[1..]
            .iter()
            .try_fold(1usize, |total, dimension| {
                usize::try_from(*dimension)
                    .ok()
                    .and_then(|dimension| total.checked_mul(dimension))
            })
            .ok_or_else(|| ExpertCacheError::InvalidRouteShape(routed_ids.shape().to_vec()))?;
        let row_count = routed_hidden.dim(0);
        let chunk_rows = if pass == ExpertPass::Prefill {
            i32::try_from(self.route_chunk_rows(routes_per_row)).unwrap_or(i32::MAX)
        } else {
            row_count.max(1)
        };
        let mut outputs = Vec::new();
        let mut start = 0;
        while start < row_count {
            let end = (start + chunk_rows).min(row_count);
            let hidden = routed_hidden.try_index_device(start..end, stream)?;
            let routes = routed_ids.try_index_device(start..end, stream)?;
            let weights = route_weights.try_index_device(start..end, stream)?;
            let mut acquired = self.acquire_routes(layer, &routes, pass, stream)?;
            let output = execute_bank(&hidden, &acquired, &weights, stream)?;
            if output.ndim() == 0 || output.dim(0) != end - start {
                return Err(ExpertCacheError::CompactBankOutputShapeMismatch {
                    expected_rows: end - start,
                    actual: output.shape().to_vec(),
                });
            }
            eval([&output])?;
            acquired.transfer.synchronize()?;
            outputs.push(output);
            start = end;
        }
        if outputs.is_empty() {
            let mut acquired = self.acquire_routes(layer, routed_ids, pass, stream)?;
            let output = execute_bank(routed_hidden, &acquired, route_weights, stream)?;
            if output.ndim() == 0 || output.dim(0) != row_count {
                return Err(ExpertCacheError::CompactBankOutputShapeMismatch {
                    expected_rows: row_count,
                    actual: output.shape().to_vec(),
                });
            }
            eval([&output])?;
            acquired.transfer.synchronize()?;
            return Ok(output);
        }
        Ok(concatenate_axis(&outputs, 0, stream)?)
    }

    /// Discovers, validates, coalesces, and acquires routed experts.
    ///
    /// A device-side demand histogram bounds host readback by the layer's
    /// global expert count. Original routes remain on-device and are rewritten
    /// through a compact-id lookup table after validation.
    fn acquire_routes(
        &self,
        layer: usize,
        routed_ids: &Array,
        pass: ExpertPass,
        stream: &Stream,
    ) -> Result<AcquiredExperts, ExpertCacheError> {
        if !matches!(
            routed_ids.dtype(),
            Dtype::Int32 | Dtype::Uint32 | Dtype::Int64 | Dtype::Uint64
        ) {
            return Err(ExpertCacheError::InvalidRouteDtype {
                actual: routed_ids.dtype(),
            });
        }
        let global_span = self
            .layer_global_spans
            .get(&layer)
            .copied()
            .ok_or(ExpertCacheError::UnknownLayer { layer })?;
        let global_span_i32 = i32::try_from(global_span)
            .map_err(|_| ExpertCacheError::ExpertCountOverflow { layer, global_span })?;
        let flat_routes = routed_ids.reshape(&[-1], stream)?;
        let below_span = flat_routes.lt(Array::from_int(global_span_i32), stream)?;
        let valid = if matches!(routed_ids.dtype(), Dtype::Uint32 | Dtype::Uint64) {
            below_span
        } else {
            flat_routes
                .ge(Array::from_int(0), stream)?
                .logical_and(below_span, stream)?
        };
        let invalid =
            crate::backend::compaction::count_nonzero(&valid.logical_not(stream)?, stream)?;
        let flat = if routed_ids.dtype() == Dtype::Int32 {
            flat_routes
        } else {
            flat_routes.as_dtype(Dtype::Int32, stream)?
        };
        let safe_ids = r#where(
            &valid,
            flat.clone(),
            Array::zeros::<i32>(&[flat.size() as i32], stream)?,
            stream,
        )?;
        let demand_values = Array::ones::<i32>(&[flat.size() as i32], stream)?;
        let histogram = segment_sum(&demand_values, &safe_ids, global_span_i32, 0, stream)?;
        eval([&histogram, &invalid])?;
        let invalid_count = invalid.evaluated()?.as_slice::<i32>()[0];
        if invalid_count != 0 {
            return Err(ExpertCacheError::InvalidRouteSet {
                layer,
                invalid_count: invalid_count as usize,
                global_span,
            });
        }
        let histogram = histogram.evaluated()?;
        let mut demand = BTreeMap::new();
        for (global_expert, count) in histogram.as_slice::<i32>().iter().copied().enumerate() {
            if count == 0 {
                continue;
            }
            let identity = ExpertIdentity::new(layer, global_expert);
            if !self.catalog.contains_key(&identity) {
                return Err(ExpertCacheError::MissingOwnedExpert { identity });
            }
            demand.insert(identity, count as u64);
        }
        let compact_ids = demand.keys().copied().collect::<Vec<_>>();
        let mut lookup = vec![-1i32; global_span];
        for (compact, identity) in compact_ids.iter().enumerate() {
            lookup[identity.global_expert] = compact as i32;
        }
        let lookup = Array::from_slice(&lookup, &[global_span_i32]).copy(stream)?;
        let normalized = flat.reshape(routed_ids.shape(), stream)?;
        let compact_routes = lookup.take(&normalized, stream)?;
        self.acquire_demand(
            demand,
            compact_ids,
            compact_routes,
            routed_ids.size() as u64,
            pass,
            stream,
        )
    }

    /// Acquires a caller-provided route table while preserving its exact shape and order.
    #[cfg(test)]
    pub fn acquire_route_slice(
        &self,
        layer: usize,
        routed_ids: &[i32],
        route_shape: &[i32],
        pass: ExpertPass,
        stream: &Stream,
    ) -> Result<AcquiredExperts, ExpertCacheError> {
        let expected_elements = route_shape.iter().try_fold(1usize, |count, dimension| {
            let dimension = usize::try_from(*dimension)
                .map_err(|_| ExpertCacheError::InvalidRouteShape(route_shape.to_vec()))?;
            count
                .checked_mul(dimension)
                .ok_or_else(|| ExpertCacheError::InvalidRouteShape(route_shape.to_vec()))
        })?;
        if expected_elements != routed_ids.len() {
            return Err(ExpertCacheError::RouteShapeMismatch {
                shape: route_shape.to_vec(),
                elements: routed_ids.len(),
            });
        }
        let layer_count = self
            .layer_expert_counts
            .get(&layer)
            .copied()
            .ok_or(ExpertCacheError::UnknownLayer { layer })?;
        let mut demand = BTreeMap::<ExpertIdentity, u64>::new();
        for id in routed_ids {
            let global_expert =
                usize::try_from(*id).map_err(|_| ExpertCacheError::InvalidExpertId {
                    layer,
                    expert: i64::from(*id),
                    known_owned_experts: layer_count,
                })?;
            let identity = ExpertIdentity::new(layer, global_expert);
            if !self.catalog.contains_key(&identity) {
                return Err(ExpertCacheError::MissingOwnedExpert { identity });
            }
            let count = demand.entry(identity).or_insert(0);
            *count = count.saturating_add(1);
        }

        let compact_ids = demand.keys().copied().collect::<Vec<_>>();
        let translations = compact_ids
            .iter()
            .enumerate()
            .map(|(compact, identity)| (*identity, compact as i32))
            .collect::<BTreeMap<_, _>>();
        let compact_values = routed_ids
            .iter()
            .map(|id| translations[&ExpertIdentity::new(layer, *id as usize)])
            .collect::<Vec<_>>();
        let compact_routes = Array::from_slice(&compact_values, route_shape).copy(stream)?;
        self.acquire_demand(
            demand,
            compact_ids,
            compact_routes,
            routed_ids.len() as u64,
            pass,
            stream,
        )
    }

    fn acquire_demand(
        &self,
        demand: BTreeMap<ExpertIdentity, u64>,
        compact_ids: Vec<ExpertIdentity>,
        compact_routes: Array,
        route_count: u64,
        pass: ExpertPass,
        stream: &Stream,
    ) -> Result<AcquiredExperts, ExpertCacheError> {
        let scratch_bytes = demand.keys().try_fold(0u64, |total, identity| {
            total
                .checked_add(self.catalog[identity])
                .ok_or(ExpertCacheError::ByteOverflow)
        })?;
        if scratch_bytes > self.scratch_limit {
            return Err(ExpertCacheError::ScratchLimitExceeded {
                required_bytes: scratch_bytes,
                limit_bytes: self.scratch_limit,
                distinct_experts: demand.len(),
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
            let route_demand = demand[identity];
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
                host_requests.push((unit.clone(), route_demand));
            }
            requests.push((unit, route_demand));
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
            .map_err(|_| ExpertCacheError::StatisticsPoisoned)?;
        let stats = statistics.pass_mut(pass);
        let distinct = compact_ids.len() as u64;
        stats.requested_routes = stats.requested_routes.saturating_add(route_count);
        stats.distinct_experts = stats.distinct_experts.saturating_add(distinct);
        stats.coalesced_duplicates = stats
            .coalesced_duplicates
            .saturating_add(route_count.saturating_sub(distinct));
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

        Ok(AcquiredExperts {
            identities: compact_ids,
            demand: demand.into_values().collect(),
            compact_routes,
            scratch_bytes,
            pass,
            transfer,
        })
    }

    /// Records a completed compact-bank construction.
    pub fn record_compact_bank(
        &self,
        pass: ExpertPass,
        bytes: u64,
        duration: Duration,
    ) -> Result<(), ExpertCacheError> {
        if bytes > self.scratch_limit {
            return Err(ExpertCacheError::ScratchLimitExceeded {
                required_bytes: bytes,
                limit_bytes: self.scratch_limit,
                distinct_experts: 0,
            });
        }
        let mut statistics = self
            .statistics
            .lock()
            .map_err(|_| ExpertCacheError::StatisticsPoisoned)?;
        let stats = statistics.pass_mut(pass);
        stats.compact_banks = stats.compact_banks.saturating_add(1);
        stats.compact_bank_bytes = stats.compact_bank_bytes.saturating_add(bytes);
        stats.peak_compact_bank_bytes = stats.peak_compact_bank_bytes.max(bytes);
        stats.compact_bank_time = stats.compact_bank_time.saturating_add(duration);
        Ok(())
    }

    /// Returns current expert residency, transfer, storage, and pass statistics.
    pub fn report(&self) -> Result<ExpertCacheReport, ExpertCacheError> {
        let residency = self.manager.report()?;
        let mut host_resident_experts = 0;
        let mut device_resident_experts = 0;
        let mut host_resident_bytes = 0u64;
        let mut device_resident_bytes = 0u64;
        for unit in residency.units() {
            if unit.host_resident() {
                host_resident_experts += 1;
                host_resident_bytes =
                    host_resident_bytes.saturating_add(unit.host_allocated_bytes());
            }
            if unit.device_resident() {
                device_resident_experts += 1;
                device_resident_bytes =
                    device_resident_bytes.saturating_add(unit.device_allocated_bytes());
            }
        }
        let statistics = self
            .statistics
            .lock()
            .map_err(|_| ExpertCacheError::StatisticsPoisoned)?;
        Ok(ExpertCacheReport {
            weight_quantization: self.weight_quantization,
            owned_experts: self.catalog.len(),
            owned_bytes: self.catalog.values().copied().sum(),
            host_resident_experts,
            device_resident_experts,
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
            prefill: statistics.prefill,
            decode: statistics.decode,
            residency,
            materialization: self.materialization.clone(),
        })
    }

    fn resident_snapshot(&self) -> Result<ResidentSnapshot, ExpertCacheError> {
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

/// A deterministic compact route table and the leases protecting its sources.
pub struct AcquiredExperts {
    identities: Vec<ExpertIdentity>,
    demand: Vec<u64>,
    compact_routes: Array,
    scratch_bytes: u64,
    pass: ExpertPass,
    transfer: ResidentTransfer,
}

impl AcquiredExperts {
    /// Returns selected experts in compact-bank order.
    pub fn identities(&self) -> &[ExpertIdentity] {
        &self.identities
    }

    /// Returns duplicate-preserving demand counts in compact-bank order.
    pub fn demand(&self) -> &[u64] {
        &self.demand
    }

    /// Returns routes rewritten bijectively to compact-bank ids.
    pub const fn compact_routes(&self) -> &Array {
        &self.compact_routes
    }

    /// Returns the conservatively reserved compact-bank byte count.
    pub const fn scratch_bytes(&self) -> u64 {
        self.scratch_bytes
    }

    /// Returns the execution-path classification used for telemetry.
    pub const fn pass(&self) -> ExpertPass {
        self.pass
    }

    /// Returns source leases in the same order as [`Self::identities`].
    #[cfg(test)]
    pub fn leases(&self) -> &[ResidentUnitLease] {
        self.transfer.leases()
    }

    /// Concatenates one required per-expert binding along its leading axis.
    pub fn compact_binding(&self, name: &str, stream: &Stream) -> Result<Array, ExpertCacheError> {
        let values = self
            .transfer
            .leases()
            .iter()
            .map(|lease| lease.device_value(name).cloned())
            .collect::<Result<Vec<_>, _>>()?;
        if values.is_empty() {
            return Err(ExpertCacheError::EmptyCompactBinding {
                name: name.to_string(),
            });
        }
        Ok(concatenate_axis(&values, 0, stream)?)
    }

    /// Concatenates an optional companion binding when every expert provides it.
    pub fn optional_compact_binding(
        &self,
        name: &str,
        stream: &Stream,
    ) -> Result<Option<Array>, ExpertCacheError> {
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
            return Err(ExpertCacheError::InconsistentCompanion {
                name: name.to_string(),
            });
        }
        self.compact_binding(name, stream).map(Some)
    }

    /// Returns whether no routed experts were selected.
    pub fn is_empty(&self) -> bool {
        self.identities.is_empty()
    }
}

/// Structured sparse expert cache failures.
#[derive(Debug, thiserror::Error)]
pub enum ExpertCacheError {
    /// Backend-neutral routed-expert placement controls were invalid.
    #[error(transparent)]
    Policy(#[from] eredu_runtime::WeightResidencyPolicyError),
    /// Bounded expert materialisation failed before cache construction.
    #[error("bounded expert materialisation failed: {source}")]
    Transformation {
        /// Shared checkpoint transformation failure.
        #[source]
        source: Box<Error>,
    },
    /// No expert definitions were supplied.
    #[error("sparse expert cache requires at least one owned expert")]
    EmptyCatalog,
    /// An architecture attempted cached execution without an initialized cache.
    #[error("{architecture} sparse expert cache was not initialized")]
    CacheUnavailable {
        /// Model family reporting the missing cache.
        architecture: &'static str,
    },
    /// A non-empty routed block selected no experts.
    #[error("{architecture} router selected no experts for a non-empty routed block")]
    EmptyRoutedBank {
        /// Model family reporting the invalid router output.
        architecture: &'static str,
    },
    /// One logical expert declared no materialized bytes.
    #[error("expert {identity:?} must contain at least one byte")]
    ZeroSizedExpert {
        /// Invalid logical identity.
        identity: ExpertIdentity,
    },
    /// Two catalog entries used the same layer/global identity.
    #[error("duplicate sparse expert catalog entry {identity:?}")]
    DuplicateExpert {
        /// Duplicated logical identity.
        identity: ExpertIdentity,
    },
    /// The architecture adapter used a noncanonical residency unit id.
    #[error("expert {identity:?} requires unit id {expected}, got {actual}")]
    UnitIdentityMismatch {
        /// Logical catalog identity.
        identity: ExpertIdentity,
        /// Required stable unit id.
        expected: OffloadUnitId,
        /// Adapter-supplied unit id.
        actual: OffloadUnitId,
    },
    /// No owned expert catalog exists for this decoder layer.
    #[error("sparse expert cache has no catalog for layer {layer}")]
    UnknownLayer {
        /// Missing decoder layer identity.
        layer: usize,
    },
    /// A layer's global expert span cannot be represented by MLX indexing.
    #[error("layer {layer} global expert span {global_span} exceeds MLX i32 indexing")]
    ExpertCountOverflow {
        /// Decoder layer identity.
        layer: usize,
        /// Required global expert span.
        global_span: usize,
    },
    /// Device-side validation found one or more out-of-range routes.
    #[error("layer {layer} contains {invalid_count} routed ids outside 0..{global_span}")]
    InvalidRouteSet {
        /// Decoder layer identity.
        layer: usize,
        /// Number of invalid route rows.
        invalid_count: usize,
        /// Valid global expert span.
        global_span: usize,
    },
    /// A route id was negative or otherwise invalid.
    #[error("invalid routed expert id {expert} for layer {layer}; this rank catalogs {known_owned_experts} owned experts")]
    InvalidExpertId {
        /// Decoder layer containing the route.
        layer: usize,
        /// Invalid signed route value.
        expert: i64,
        /// Owned experts cataloged for diagnostics.
        known_owned_experts: usize,
    },
    /// A valid global route referred to an expert this cache does not own.
    #[error("routed expert {identity:?} is not owned by this cache")]
    MissingOwnedExpert {
        /// Requested non-owned global identity.
        identity: ExpertIdentity,
    },
    /// Router ids used an unsupported scalar type.
    #[error("routed expert ids must use an integer dtype, got {actual:?}")]
    InvalidRouteDtype {
        /// Unsupported router-id scalar type.
        actual: Dtype,
    },
    /// A supplied route shape had a negative dimension or overflowed.
    #[error("invalid routed expert shape {0:?}")]
    InvalidRouteShape(Vec<i32>),
    /// Route shape and host values disagreed.
    #[error("routed expert shape {shape:?} does not describe {elements} values")]
    RouteShapeMismatch {
        /// Declared route shape.
        shape: Vec<i32>,
        /// Supplied host value count.
        elements: usize,
    },
    /// Hidden rows, route rows, and route-weight rows did not align.
    #[error("routed expert batch shapes do not align: hidden {hidden:?}, routes {routes:?}, weights {weights:?}")]
    RoutedBatchShapeMismatch {
        /// Routed hidden-state shape.
        hidden: Vec<i32>,
        /// Routed expert-id shape.
        routes: Vec<i32>,
        /// Routed expert-weight shape.
        weights: Vec<i32>,
    },
    /// An architecture-specific compact bank returned the wrong row count.
    #[error("compact expert bank returned shape {actual:?}, expected {expected_rows} rows")]
    CompactBankOutputShapeMismatch {
        /// Required output rows.
        expected_rows: i32,
        /// Returned output shape.
        actual: Vec<i32>,
    },
    /// Selected experts exceed the configured temporary compact-bank allowance.
    #[error("compact expert bank for {distinct_experts} experts requires {required_bytes} bytes, exceeding the {limit_bytes}-byte scratch limit")]
    ScratchLimitExceeded {
        /// Required compact-bank bytes.
        required_bytes: u64,
        /// Configured compact-bank byte limit.
        limit_bytes: u64,
        /// Selected unique expert count.
        distinct_experts: usize,
    },
    /// Expert byte arithmetic overflowed.
    #[error("sparse expert byte accounting overflowed")]
    ByteOverflow,
    /// Cache statistics mutex was poisoned by a panic.
    #[error("sparse expert cache statistics are unavailable after a panic")]
    StatisticsPoisoned,
    /// A required compact binding had no selected source experts.
    #[error("compact expert binding {name:?} has no source arrays")]
    EmptyCompactBinding {
        /// Required binding name.
        name: String,
    },
    /// Optional companion presence differed across selected experts.
    #[error("compact expert companion {name:?} is missing from only part of the selected bank")]
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
            values.iter().enumerate().map(|(expert, bytes)| {
                (
                    format!("expert.{expert}"),
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

    fn entries() -> Vec<ExpertCatalogEntry> {
        (0..3)
            .map(|expert| {
                let identity = ExpertIdentity::new(2, expert);
                let bindings = [
                    WeightBinding::new(
                        "weight",
                        format!("expert.{expert}"),
                        TensorSelection::Full,
                        8,
                    )
                    .unwrap(),
                    WeightBinding::new(
                        "scale",
                        format!("expert.{expert}"),
                        TensorSelection::Full,
                        8,
                    )
                    .unwrap(),
                ];
                let unit = OffloadUnit::new(identity.unit_id(), bindings).unwrap();
                ExpertCatalogEntry::new(identity, unit, 16).unwrap()
            })
            .collect()
    }

    fn cache(
        store: Arc<SafetensorsWeightStore>,
        device: u64,
        host: u64,
        scratch: u64,
        eviction: CacheEvictionPolicy,
    ) -> ExpertCache {
        cache_with_target(store, device, host, scratch, scratch, eviction)
    }

    fn cache_with_target(
        store: Arc<SafetensorsWeightStore>,
        device: u64,
        host: u64,
        scratch: u64,
        prefill_target: u64,
        eviction: CacheEvictionPolicy,
    ) -> ExpertCache {
        let binding_capacity =
            host_transfer_capacity_upper_bound(8, HostTransferPolicy::Transfer).unwrap() as u64;
        let physical_host = (host / 8).checked_mul(binding_capacity).unwrap();
        let experts = OffloadConfig::new(Some(device), Some(physical_host), 1)
            .unwrap()
            .with_eviction_policy(eviction);
        ExpertCache::new(
            store,
            entries(),
            ExpertCacheLoadOptions::new(experts, scratch, prefill_target).unwrap(),
            stream(),
            stream(),
        )
        .unwrap()
    }

    #[test]
    fn quantized_cache_materializes_only_rank_local_expert_and_tp_recipes() {
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
                    "experts.gate",
                    TensorView::new(StoredDtype::F32, vec![2, 4, 64], &gate_bytes).unwrap(),
                ),
                (
                    "experts.down",
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
            .map(|expert| {
                let identity = ExpertIdentity::new(3, expert);
                let bindings = ["gate", "down"].map(|projection| {
                    let owned = DerivedWeightRecipe::source(
                        format!("experts.{projection}"),
                        TensorSelection::Range {
                            axis: 0,
                            start: expert,
                            end: expert + 1,
                        },
                    );
                    let tp = DerivedWeightRecipe::Select {
                        input: Box::new(owned),
                        selection: TensorSelection::Range {
                            axis: 1,
                            start: expert * 2,
                            end: expert * 2 + 2,
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
                ExpertCatalogEntry::new(identity, unit, 1_024).unwrap()
            })
            .collect::<Vec<_>>();
        let options =
            ExpertCacheLoadOptions::new(OffloadConfig::new(None, None, 1).unwrap(), 1_024, 1_024)
                .unwrap();
        let cache = ExpertCache::new_quantized_shared(
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
        assert_eq!(report.owned_experts, 2);
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
            .acquire_route_slice(3, &[1], &[1, 1], ExpertPass::Decode, &stream())
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
    fn expert_quantization_uses_only_declared_roles_and_companion_names() {
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
        let identity = ExpertIdentity::new(0, 0);
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
        let entry = ExpertCatalogEntry::new(
            identity,
            OffloadUnit::new(identity.unit_id(), [quantized, preserved]).unwrap(),
            (bytes.len() * 2) as u64,
        )
        .unwrap();

        let transformed = quantize_expert_catalog(
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
    fn coalesces_routes_in_global_order_and_separates_pass_counters() {
        let (_dir, store) = fixture();
        let cache = cache(store, 32, 32, 32, CacheEvictionPolicy::LeastRecentlyUsed);
        let first = cache
            .acquire_route_slice(2, &[2, 0, 2, 0], &[2, 2], ExpertPass::Prefill, &stream())
            .unwrap();
        assert_eq!(
            first.identities(),
            &[ExpertIdentity::new(2, 0), ExpertIdentity::new(2, 2)]
        );
        assert_eq!(first.demand(), &[2, 2]);
        assert_eq!(
            first
                .compact_routes()
                .evaluated()
                .unwrap()
                .as_slice::<i32>(),
            &[1, 0, 1, 0]
        );
        drop(first);

        let host = cache
            .manager
            .acquire(&ExpertIdentity::new(2, 0).unit_id(), MemoryTier::Host)
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
            .acquire_route_slice(2, &[0, 2], &[1, 2], ExpertPass::Decode, &stream())
            .unwrap();
        drop(second);
        let report = cache.report().unwrap();
        assert_eq!(report.prefill.requested_routes, 4);
        assert_eq!(report.prefill.distinct_experts, 2);
        assert_eq!(report.prefill.coalesced_duplicates, 2);
        assert_eq!(report.prefill.device.misses, 2);
        assert_eq!(report.decode.requested_routes, 2);
        assert_eq!(report.decode.device.hits, 2);
        assert_eq!(report.owned_experts, 3);
        assert_eq!(report.owned_bytes, 48);
    }

    #[test]
    fn resident_store_pins_every_expert_and_never_rereads_checkpoint_weights() {
        let (_dir, store) = fixture();
        let cache =
            ExpertCache::new_resident_shared(store.clone(), entries(), stream(), stream()).unwrap();
        let initialized = cache.report().unwrap();
        assert_eq!(initialized.owned_experts, 3);
        assert_eq!(initialized.device_resident_experts, 3);
        assert_eq!(initialized.device_resident_bytes, initialized.owned_bytes);
        assert_eq!(initialized.host_resident_experts, 0);

        let reads_after_load = store.source_diagnostics().unwrap().physical_reads;
        let acquired = cache
            .acquire_route_slice(2, &[2, 0, 2], &[3, 1], ExpertPass::Decode, &stream())
            .unwrap();
        let compact = acquired.compact_binding("weight", &stream()).unwrap();
        eval([&compact]).unwrap();

        assert_eq!(
            store.source_diagnostics().unwrap().physical_reads,
            reads_after_load
        );
        let executed = cache.report().unwrap();
        assert_eq!(executed.decode.device.hits, 2);
        assert_eq!(executed.decode.device.misses, 0);
        assert_eq!(executed.decode.device.evictions, 0);
    }

    #[test]
    fn route_chunk_size_respects_scratch_target_and_worst_case_expert_bytes() {
        let (_dir, store) = fixture();
        let bounded = cache(
            Arc::clone(&store),
            48,
            0,
            64,
            CacheEvictionPolicy::LeastRecentlyUsed,
        );
        assert_eq!(bounded.route_chunk_rows(2), 2);
        assert_eq!(bounded.route_chunk_rows(0), 1);
        let small = cache(store, 48, 0, 16, CacheEvictionPolicy::LeastRecentlyUsed);
        assert_eq!(small.route_chunk_rows(2), 1);
    }

    #[test]
    fn prefill_target_is_required_and_cannot_exceed_scratch() {
        let experts = OffloadConfig::new(Some(48), Some(0), 1).unwrap();
        assert!(matches!(
            ExpertCacheLoadOptions::new(experts, 64, 0),
            Err(eredu_runtime::WeightResidencyPolicyError::ZeroExpertPrefillBankTarget)
        ));
        assert!(matches!(
            ExpertCacheLoadOptions::new(experts, 64, 65),
            Err(
                eredu_runtime::WeightResidencyPolicyError::ExpertPrefillBankTargetExceedsScratch { .. }
            )
        ));
    }

    #[test]
    fn bounded_execution_chunks_prefill_but_not_decode_and_preserves_row_order() {
        let (_dir, store) = fixture();
        let cache = cache_with_target(store, 48, 0, 48, 32, CacheEvictionPolicy::LeastRecentlyUsed);
        let execution = stream();
        let hidden = Array::from_slice(&[1f32, 2., 3., 4., 5., 6.], &[3, 2]);
        let routes = Array::from_slice(&[0i32, 1, 1, 2, 2, 0], &[3, 2]);
        let weights = Array::from_slice(&[0.5f32; 6], &[3, 2]);
        let mut prefill_banks = 0;
        let output = cache
            .execute_routes_bounded(
                ExpertRouteBatch::new(2, &hidden, &routes, &weights, ExpertPass::Prefill),
                &execution,
                |hidden, _acquired, _weights, _stream| {
                    prefill_banks += 1;
                    Ok(hidden.clone())
                },
            )
            .unwrap();
        assert_eq!(prefill_banks, 3);
        assert_eq!(
            output.evaluated().unwrap().as_slice::<f32>(),
            hidden.evaluated().unwrap().as_slice::<f32>()
        );

        let mut decode_banks = 0;
        cache
            .execute_routes_bounded(
                ExpertRouteBatch::new(2, &hidden, &routes, &weights, ExpertPass::Decode),
                &execution,
                |hidden, _acquired, _weights, _stream| {
                    decode_banks += 1;
                    Ok(hidden.clone())
                },
            )
            .unwrap();
        assert_eq!(decode_banks, 1);

        let distributed_routes = Array::from_slice(&[0i32, 1, 2], &[3]);
        let distributed_weights = Array::from_slice(&[1f32; 3], &[3]);
        let mut distributed_banks = 0;
        cache
            .execute_routes_bounded(
                ExpertRouteBatch::new(
                    2,
                    &hidden,
                    &distributed_routes,
                    &distributed_weights,
                    ExpertPass::Prefill,
                ),
                &execution,
                |hidden, _acquired, _weights, _stream| {
                    distributed_banks += 1;
                    Ok(hidden.clone())
                },
            )
            .unwrap();
        assert_eq!(distributed_banks, 2);
    }

    #[test]
    fn device_histogram_preserves_duplicate_routes_and_validates_before_loading() {
        let (_dir, store) = fixture();
        let cache = cache(store, 32, 32, 32, CacheEvictionPolicy::LeastRecentlyUsed);
        let execution = stream();
        let routes = Array::from_slice(&[2i32, 0, 2, 0], &[2, 2]);
        let acquired = cache
            .acquire_routes(2, &routes, ExpertPass::Prefill, &execution)
            .unwrap();
        assert_eq!(
            acquired.identities(),
            &[ExpertIdentity::new(2, 0), ExpertIdentity::new(2, 2)]
        );
        assert_eq!(acquired.demand(), &[2, 2]);
        assert_eq!(
            acquired
                .compact_routes()
                .evaluated()
                .unwrap()
                .as_slice::<i32>(),
            &[1, 0, 1, 0]
        );
        drop(acquired);

        let invalid = Array::from_slice(&[-1i32, 0], &[2]);
        assert!(matches!(
            cache.acquire_routes(2, &invalid, ExpertPass::Decode, &execution),
            Err(ExpertCacheError::InvalidRouteSet {
                invalid_count: 1,
                ..
            })
        ));
        let report = cache.report().unwrap();
        assert_eq!(report.decode.requested_routes, 0);

        let narrowing_alias = Array::from_slice(&[1u64 << 32], &[1]);
        assert!(matches!(
            cache.acquire_routes(2, &narrowing_alias, ExpertPass::Decode, &execution),
            Err(ExpertCacheError::InvalidRouteSet {
                invalid_count: 1,
                ..
            })
        ));
    }

    #[test]
    fn rejects_invalid_missing_and_over_scratch_routes_before_loading() {
        let (_dir, store) = fixture();
        let cache = cache(store, 48, 0, 16, CacheEvictionPolicy::LeastRecentlyUsed);
        assert!(matches!(
            cache.acquire_route_slice(2, &[-1], &[1], ExpertPass::Decode, &stream()),
            Err(ExpertCacheError::InvalidExpertId { .. })
        ));
        assert!(matches!(
            cache.acquire_route_slice(2, &[3], &[1], ExpertPass::Decode, &stream()),
            Err(ExpertCacheError::MissingOwnedExpert { .. })
        ));
        assert!(matches!(
            cache.acquire_route_slice(2, &[0, 1], &[2], ExpertPass::Prefill, &stream()),
            Err(ExpertCacheError::ScratchLimitExceeded { .. })
        ));
        let report = cache.report().unwrap();
        assert_eq!(report.device_resident_experts, 0);
        assert_eq!(report.prefill.requested_routes, 0);
        assert_eq!(report.decode.requested_routes, 0);
    }

    #[test]
    fn empty_routes_do_not_materialize_or_build_a_bank() {
        let (_dir, store) = fixture();
        let cache = cache(store, 16, 16, 16, CacheEvictionPolicy::LeastRecentlyUsed);
        let acquired = cache
            .acquire_route_slice(2, &[], &[0, 2], ExpertPass::Decode, &stream())
            .unwrap();
        assert!(acquired.is_empty());
        assert_eq!(acquired.scratch_bytes(), 0);
        assert_eq!(acquired.compact_routes().shape(), &[0, 2]);
        drop(acquired);
        let report = cache.report().unwrap();
        assert_eq!(report.host_resident_experts, 0);
        assert_eq!(report.device_resident_experts, 0);
        assert_eq!(report.decode.compact_banks, 0);
    }

    #[test]
    fn lfu_uses_duplicate_route_demand_and_deterministic_recency_ties() {
        let (_dir, store) = fixture();
        let cache = cache(store, 32, 0, 32, CacheEvictionPolicy::LeastFrequentlyUsed);
        drop(
            cache
                .acquire_route_slice(2, &[0, 0, 0], &[3], ExpertPass::Decode, &stream())
                .unwrap(),
        );
        drop(
            cache
                .acquire_route_slice(2, &[1], &[1], ExpertPass::Decode, &stream())
                .unwrap(),
        );
        drop(
            cache
                .acquire_route_slice(2, &[2], &[1], ExpertPass::Decode, &stream())
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
                ExpertIdentity::new(2, 0).unit_id().as_str(),
                ExpertIdentity::new(2, 2).unit_id().as_str()
            ]
        );
        assert_eq!(report.decode.device.evictions, 1);
    }
}
