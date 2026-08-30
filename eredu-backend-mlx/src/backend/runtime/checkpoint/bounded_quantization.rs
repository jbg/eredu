//! Bounded load-time weight quantization into resident memory.

//!
//! A [`BoundedQuantizedWeightStore`](crate::backend::runtime::checkpoint::bounded_quantization::BoundedQuantizedWeightStore)
//! overlays packed tensors on an existing
//! checkpoint store. Source matrices are selected in row tiles, quantized on
//! explicit conversion streams, and written directly into final in-memory
//! encoded tensors. A fixed two-slot completion window spans tensor boundaries and
//! overlaps the next tile with the prior tile's host copy when both slots fit
//! the admitted working set, and otherwise falls back to one slot. The
//! process never requires a complete dense matrix in active memory: tile size is
//! admitted against an explicit byte bound, while final packed storage remains
//! resident for the subsequent model load. The
//! ordinary residency machinery subsequently sees only the final packed tensor
//! geometry.

use eredu_checkpoint::store::{
    CheckpointLease, CheckpointSource, MemoryWeightStore, StoreError, TensorReadRequest,
    TensorSelection, WeightStoreDiagnostics,
};
use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, RecipeDtype},
    WeightQuantization,
};
use eredu_runtime::WeightMaterializationReport;

use std::{
    collections::{BTreeSet, VecDeque},
    path::PathBuf,
    sync::Arc,
};

use safemlx::{memory, transforms::async_eval_with_event, Array, Dtype, Event, Stream};
use safetensors::tensor::Dtype as SafeDtype;

use crate::{
    backend::error::Error,
    backend::runtime::checkpoint::{
        quantization::quantize_tensor,
        recipe::MlxWeightRecipeExt,
        store::{MlxParameterMaterializationContext, PendingWeightMaterialization},
    },
};

const BOUNDED_QUANTIZATION_TILE_BUFFERS: usize = 2;
const BOUNDED_QUANTIZATION_MAX_CACHE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_QUANTIZATION_SUBMISSION_ELEMENTS: usize = i32::MAX as usize;

/// One dense semantic weight that will be replaced by its packed representation.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct BoundedQuantizationTarget {
    weight_name: String,
    scales_name: String,
    biases_name: Option<String>,
    source: DerivedWeightRecipe,
    affine_companion_dtype: RecipeDtype,
}

impl BoundedQuantizationTarget {
    /// Quantizes a checkpoint tensor in place under exact output identities.
    pub fn direct(
        weight_name: impl Into<String>,
        scales_name: impl Into<String>,
        biases_name: Option<impl Into<String>>,
    ) -> Result<Self, Error> {
        let weight_name = weight_name.into();
        Self::from_recipe(
            weight_name.clone(),
            scales_name,
            biases_name,
            DerivedWeightRecipe::source(weight_name, TensorSelection::Full),
        )
    }

    /// Quantizes a semantic recipe under exact packed runtime identities.
    pub fn from_recipe(
        weight_name: impl Into<String>,
        scales_name: impl Into<String>,
        biases_name: Option<impl Into<String>>,
        source: DerivedWeightRecipe,
    ) -> Result<Self, Error> {
        let weight_name = weight_name.into();
        let scales_name = scales_name.into();
        let biases_name = biases_name.map(Into::into);
        validate_output_names(&weight_name, &scales_name, biases_name.as_deref())?;
        Ok(Self {
            weight_name,
            scales_name,
            biases_name,
            source,
            affine_companion_dtype: RecipeDtype::F32,
        })
    }

    /// Selects the runtime dtype required for affine scales and biases.
    pub fn with_affine_companion_dtype(mut self, dtype: RecipeDtype) -> Result<Self, Error> {
        if !matches!(
            dtype,
            RecipeDtype::F16 | RecipeDtype::BF16 | RecipeDtype::F32
        ) {
            return Err(quantization_error(format!(
                "bounded affine companions require F16, BF16, or F32, got {dtype:?}"
            )));
        }
        self.affine_companion_dtype = dtype;
        Ok(self)
    }

    /// Returns the packed weight's canonical logical name.
    pub fn weight_name(&self) -> &str {
        &self.weight_name
    }

    /// Returns the semantic source recipe.
    pub fn source(&self) -> &DerivedWeightRecipe {
        &self.source
    }

    /// Returns the scalar width selected for affine scale and bias outputs.
    pub fn affine_companion_bytes(&self) -> u64 {
        match self.affine_companion_dtype {
            RecipeDtype::F16 | RecipeDtype::BF16 => 2,
            RecipeDtype::F32 => 4,
            _ => unreachable!("validated bounded affine companion dtype"),
        }
    }

    /// Returns the exact packed scale tensor name.
    pub fn scales_name(&self) -> &str {
        &self.scales_name
    }

    /// Returns the exact packed affine-bias tensor name, when declared.
    pub fn biases_name(&self) -> Option<&str> {
        self.biases_name.as_deref()
    }
}

/// A validated collection of source-bounded, memory-resident transformations.
#[derive(Debug, Clone)]
pub struct BoundedQuantizationPlan {
    quantization: WeightQuantization,
    max_working_set_bytes: u64,
    targets: Vec<BoundedQuantizationTarget>,
}

impl BoundedQuantizationPlan {
    /// Creates a non-empty plan with an explicit conversion working-set bound.
    pub fn new(
        quantization: impl Into<WeightQuantization>,
        max_working_set_bytes: u64,
        targets: impl IntoIterator<Item = BoundedQuantizationTarget>,
    ) -> Result<Self, Error> {
        let quantization = quantization.into();
        quantization.validate()?;
        if quantization.gguf_iquant().is_some() {
            return Err(quantization_error(
                "checkpoint-native GGUF IQ encodings cannot be produced by load-time quantization",
            ));
        }
        if max_working_set_bytes == 0 {
            return Err(quantization_error(
                "bounded quantization working-set bytes must be nonzero",
            ));
        }
        let mut targets = targets.into_iter().collect::<Vec<_>>();
        if targets.is_empty() {
            return Err(quantization_error(
                "bounded quantization requires at least one target",
            ));
        }
        targets.sort_by(|left, right| left.weight_name.cmp(&right.weight_name));
        let mut output_names = BTreeSet::new();
        for target in &targets {
            if quantization.has_biases() && target.biases_name.is_none() {
                return Err(quantization_error(format!(
                    "bounded affine quantization target {:?} has no affine-bias identity",
                    target.weight_name
                )));
            }
            for name in output_names_for(target, quantization)? {
                if !output_names.insert(name.clone()) {
                    return Err(quantization_error(format!(
                        "bounded quantization output {name:?} is produced more than once"
                    )));
                }
            }
        }
        Ok(Self {
            quantization,
            max_working_set_bytes,
            targets,
        })
    }

    /// Returns the final packed encoding.
    pub const fn quantization(&self) -> WeightQuantization {
        self.quantization
    }

    /// Returns the admitted conversion working-set bound.
    pub const fn max_working_set_bytes(&self) -> u64 {
        self.max_working_set_bytes
    }

    /// Returns targets in deterministic logical-name order.
    pub fn targets(&self) -> &[BoundedQuantizationTarget] {
        &self.targets
    }
}

/// A source checkpoint overlaid with memory-backed, load-time-quantized weights.
///
/// The packed store lives as long as this value. Runtime acquisitions of
/// transformed keys use bounded in-memory leases; all other keys
/// delegate to the original store.
pub struct BoundedQuantizedWeightStore {
    source: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    transformed: MemoryWeightStore,
    transformed_keys: BTreeSet<String>,
    materialized_source_keys: BTreeSet<String>,
    materialized_source_shards: BTreeSet<PathBuf>,
    report: WeightMaterializationReport,
}

impl std::fmt::Debug for BoundedQuantizedWeightStore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("BoundedQuantizedWeightStore")
            .field(
                "backend",
                &self
                    .source
                    .source_diagnostics()
                    .map(|report| report.backend),
            )
            .field("transformed_keys", &self.transformed_keys)
            .field("report", &self.report)
            .finish_non_exhaustive()
    }
}

impl BoundedQuantizedWeightStore {
    /// Executes `plan` without materializing a complete dense source matrix.
    ///
    /// Conversion runs on the supplied stream's device, allowing model loads
    /// to use the accelerator quantizer while retaining CPU fallback. Source
    /// and packed tile storage remain covered by the admitted working set.
    ///
    /// A second same-device stream is used only when two minimum tiles fit the
    /// admitted bound.
    pub fn create(
        source: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
        plan: BoundedQuantizationPlan,
        conversion_stream: &Stream,
    ) -> Result<Self, Error> {
        if !cfg!(target_endian = "little") {
            return Err(quantization_error(
                "bounded SafeTensors quantization requires a little-endian host",
            ));
        }
        let device = conversion_stream.get_device()?;
        let tile_streams = [conversion_stream.clone(), Stream::new_with_device(&device)];
        let tile_contexts = tile_streams
            .each_ref()
            .map(|stream| MlxParameterMaterializationContext::new(stream, stream));

        preflight_source_collisions(source.as_ref(), &plan)?;
        let materialized_source_keys = plan
            .targets
            .iter()
            .flat_map(|target| target.source.source_keys())
            .map(ToString::to_string)
            .collect::<BTreeSet<_>>();
        let mut materialized_source_shards = source
            .materialized_source_shards()
            .into_iter()
            .collect::<BTreeSet<_>>();
        for key in &materialized_source_keys {
            if let Some(path) = source.source_metadata(key)?.backing_shard {
                materialized_source_shards.insert(path);
            }
        }
        let mut transformed_keys = BTreeSet::new();
        let mut report = WeightMaterializationReport {
            admitted_working_set_bytes: plan.max_working_set_bytes,
            ..WeightMaterializationReport::default()
        };
        let mut output_shards = Vec::with_capacity(plan.targets.len());
        let mut allocator_cache = BoundedAllocatorCache::new(plan.max_working_set_bytes);
        let mut pending_tiles = VecDeque::with_capacity(BOUNDED_QUANTIZATION_TILE_BUFFERS);
        allocator_cache.begin()?;

        for target in &plan.targets {
            transform_target(
                source.as_ref(),
                target,
                &plan,
                &tile_contexts,
                &mut output_shards,
                &mut pending_tiles,
                &mut allocator_cache,
                &mut report,
            )?;
            for name in output_names_for(target, plan.quantization)? {
                transformed_keys.insert(name);
            }
        }
        while !pending_tiles.is_empty() {
            write_oldest_tile(&mut pending_tiles, &mut output_shards, &mut allocator_cache)?;
        }
        debug_assert!(output_shards
            .iter()
            .all(|shard| shard.sealed && shard.pending_tiles == 0));
        allocator_cache.finish()?;

        let transformed =
            MemoryWeightStore::from_safetensors(output_shards.into_iter().flat_map(|shard| {
                shard
                    .layouts
                    .into_iter()
                    .zip(shard.buffers)
                    .map(|(layout, bytes)| (layout.name, layout.dtype, layout.shape, bytes))
            }))?;
        Ok(Self {
            source,
            transformed,
            transformed_keys,
            materialized_source_keys,
            materialized_source_shards,
            report,
        })
    }

    /// Returns conversion telemetry captured before runtime materialization.
    pub const fn report(&self) -> &WeightMaterializationReport {
        &self.report
    }

    /// Returns whether `key` is supplied by the packed overlay.
    pub fn is_transformed(&self, key: &str) -> bool {
        self.transformed_keys.contains(key)
    }
}

impl CheckpointSource for BoundedQuantizedWeightStore {
    fn source_keys(&self) -> Vec<String> {
        self.source
            .source_keys()
            .into_iter()
            .chain(self.transformed_keys.iter().cloned())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    fn materialized_source_keys(&self) -> Vec<String> {
        self.source
            .materialized_source_keys()
            .into_iter()
            .chain(self.materialized_source_keys.iter().cloned())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    fn materialized_source_shards(&self) -> Vec<PathBuf> {
        self.materialized_source_shards.iter().cloned().collect()
    }

    fn unclaimed_checkpoint_keys(&self) -> Vec<String> {
        self.source.unclaimed_checkpoint_keys()
    }

    fn is_authoritative_materialized_key(&self, key: &str) -> bool {
        self.is_transformed(key)
    }

    fn is_checkpoint_contract_resolved(&self) -> bool {
        self.source.is_checkpoint_contract_resolved()
    }

    fn source_metadata(
        &self,
        key: &str,
    ) -> Result<eredu_checkpoint::store::TensorMetadata, StoreError> {
        if self.is_transformed(key) {
            CheckpointSource::source_metadata(&self.transformed, key)
        } else {
            self.source.source_metadata(key)
        }
    }

    fn acquire_lease(&self, request: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
        if self.is_transformed(&request.key) {
            CheckpointSource::acquire_lease(&self.transformed, request)
        } else {
            self.source.acquire_lease(request)
        }
    }

    fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        let source = self.source.source_diagnostics()?;
        let transformed = CheckpointSource::source_diagnostics(&self.transformed)?;
        let mut touched = source.touched_shard_paths;
        touched.extend(transformed.touched_shard_paths);
        touched.sort();
        touched.dedup();
        let mut payloads = source.payload_shard_paths;
        payloads.extend(transformed.payload_shard_paths);
        payloads.sort();
        payloads.dedup();
        Ok(WeightStoreDiagnostics {
            backend: source.backend,
            mapping_hits: source.mapping_hits.saturating_add(transformed.mapping_hits),
            mapping_misses: source
                .mapping_misses
                .saturating_add(transformed.mapping_misses),
            evictions: source.evictions.saturating_add(transformed.evictions),
            currently_mapped_shards: source
                .currently_mapped_shards
                .saturating_add(transformed.currently_mapped_shards),
            touched_shard_paths: touched,
            payload_shard_paths: payloads,
            physical_reads: source
                .physical_reads
                .saturating_add(transformed.physical_reads),
            physical_read_bytes: source
                .physical_read_bytes
                .saturating_add(transformed.physical_read_bytes),
            coalesced_group_hits: source
                .coalesced_group_hits
                .saturating_add(transformed.coalesced_group_hits),
        })
    }
}

#[derive(Debug, Clone)]
struct OutputLayout {
    name: String,
    dtype: SafeDtype,
    shape: Vec<usize>,
    row_bytes: u64,
    byte_len: u64,
}

struct OutputShard {
    layouts: Vec<OutputLayout>,
    buffers: Vec<Vec<u8>>,
    pending_tiles: usize,
    sealed: bool,
}

struct BoundedAllocatorCache {
    working_set_limit_bytes: u64,
    retained_limit_bytes: u64,
    finished: bool,
}

impl BoundedAllocatorCache {
    fn new(working_set_limit_bytes: u64) -> Self {
        Self {
            working_set_limit_bytes,
            retained_limit_bytes: (working_set_limit_bytes / 4)
                .min(BOUNDED_QUANTIZATION_MAX_CACHE_BYTES),
            finished: false,
        }
    }

    fn begin(&mut self) -> Result<(), Error> {
        memory::clear_cache()?;
        Ok(())
    }

    fn prepare_submission(
        &mut self,
        queued_working_set_bytes: u64,
        incoming_tile_bytes: u64,
    ) -> Result<(), Error> {
        let planned_bytes = queued_working_set_bytes
            .checked_add(incoming_tile_bytes)
            .ok_or_else(|| quantization_error("conversion submission working-set overflow"))?;
        if planned_bytes > self.working_set_limit_bytes {
            return Err(quantization_error(format!(
                "conversion submission requires {planned_bytes} working-set bytes, but the plan permits {}",
                self.working_set_limit_bytes
            )));
        }
        self.clear_if_needed(planned_bytes)
    }

    fn tile_completed(&mut self, queued_working_set_bytes: u64) -> Result<(), Error> {
        if queued_working_set_bytes > self.working_set_limit_bytes {
            return Err(quantization_error(format!(
                "queued conversion tiles require {queued_working_set_bytes} working-set bytes, but the plan permits {}",
                self.working_set_limit_bytes
            )));
        }
        self.clear_if_needed(queued_working_set_bytes)
    }

    fn clear_if_needed(&mut self, active_working_set_bytes: u64) -> Result<(), Error> {
        let cached_bytes = u64::try_from(memory::cache_memory()?)
            .map_err(|_| quantization_error("allocator-cache bytes are not representable"))?;
        let available_cache_bytes = self
            .working_set_limit_bytes
            .checked_sub(active_working_set_bytes)
            .expect("validated active conversion working set");
        if allocator_cache_requires_clear(
            cached_bytes,
            self.retained_limit_bytes,
            available_cache_bytes,
        ) {
            memory::clear_cache()?;
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<(), Error> {
        memory::clear_cache()?;
        self.finished = true;
        Ok(())
    }
}

const fn allocator_cache_requires_clear(
    cached_bytes: u64,
    retained_limit_bytes: u64,
    available_cache_bytes: u64,
) -> bool {
    cached_bytes > retained_limit_bytes || cached_bytes > available_cache_bytes
}

impl Drop for BoundedAllocatorCache {
    fn drop(&mut self) {
        if !self.finished {
            let _ = memory::clear_cache();
        }
    }
}

impl OutputShard {
    fn tile_submitted(&mut self) {
        debug_assert!(!self.sealed);
        self.pending_tiles = self
            .pending_tiles
            .checked_add(1)
            .expect("output shard pending-tile count overflowed");
    }

    fn tile_completed(&mut self) -> Result<(), Error> {
        self.pending_tiles = self
            .pending_tiles
            .checked_sub(1)
            .expect("completed output tile was previously submitted");
        self.close_if_complete()
    }

    fn seal(&mut self) -> Result<(), Error> {
        self.sealed = true;
        self.close_if_complete()
    }

    fn close_if_complete(&mut self) -> Result<(), Error> {
        Ok(())
    }
}

fn transform_target(
    source: &dyn eredu_checkpoint::store::CheckpointSource,
    target: &BoundedQuantizationTarget,
    plan: &BoundedQuantizationPlan,
    tile_contexts: &[MlxParameterMaterializationContext; BOUNDED_QUANTIZATION_TILE_BUFFERS],
    output_shards: &mut Vec<OutputShard>,
    pending_tiles: &mut VecDeque<SubmittedQuantizationTile>,
    allocator_cache: &mut BoundedAllocatorCache,
    report: &mut WeightMaterializationReport,
) -> Result<(), Error> {
    let metadata = target.source.infer(source)?;
    if metadata.shape().len() < 2
        || !matches!(
            metadata.dtype(),
            RecipeDtype::F16 | RecipeDtype::BF16 | RecipeDtype::F32
        )
    {
        return Err(quantization_error(format!(
            "bounded quantization target {:?} must produce a floating-point matrix, got shape {:?} and dtype {:?}",
            target.weight_name,
            metadata.shape(),
            metadata.dtype()
        )));
    }
    let row_axis = metadata.shape().len() - 2;
    let leading = checked_product(&metadata.shape()[..row_axis], "leading target dimensions")?;
    if leading == 0 {
        return Err(quantization_error(format!(
            "bounded quantization target {:?} must contain at least one leading matrix",
            target.weight_name
        )));
    }
    let rows = metadata.shape()[row_axis];
    let columns = metadata.shape()[row_axis + 1];
    if rows == 0 {
        return Err(quantization_error(format!(
            "bounded quantization target {:?} must contain at least one row",
            target.weight_name
        )));
    }
    let group_size = usize::try_from(plan.quantization.group_size())
        .map_err(|_| quantization_error("quantization group size is not representable"))?;
    let bits = usize::try_from(plan.quantization.bits())
        .map_err(|_| quantization_error("quantization bit width is not representable"))?;
    if columns % group_size != 0 || columns % 32 != 0 {
        return Err(quantization_error(format!(
            "bounded quantization target {:?} input dimension {} must be divisible by group_size {} and 32",
            target.weight_name, columns, group_size
        )));
    }

    let layouts = output_layouts(
        target,
        plan.quantization,
        metadata.shape(),
        rows,
        columns,
        bits,
    )?;
    let output_row_bytes = layouts.iter().try_fold(0u64, |total, layout| {
        total
            .checked_add(layout.row_bytes)
            .ok_or_else(|| quantization_error("quantized output row size overflow"))
    })?;
    let mut one_row_source_bytes = 0u64;
    let mut one_row_peak = 0u64;
    for matrix in 0..leading {
        let one_row = target
            .source
            .select_bounded_matrix_rows(source, matrix, 0, 1)?;
        one_row.preflight_bounded(source)?;
        one_row_source_bytes = one_row_source_bytes.max(one_row.infer(source)?.byte_len());
        one_row_peak = one_row_peak.max(
            one_row
                .peak_materialization_bytes(source)?
                .checked_add(output_row_bytes)
                .ok_or_else(|| quantization_error("one-row conversion working-set overflow"))?,
        );
    }
    if one_row_peak > plan.max_working_set_bytes {
        return Err(quantization_error(format!(
            "bounded quantization target {:?} requires at least {} working-set bytes for one row, but the plan permits {}",
            target.weight_name, one_row_peak, plan.max_working_set_bytes
        )));
    }
    let tile_buffers = if one_row_peak
        .checked_mul(BOUNDED_QUANTIZATION_TILE_BUFFERS as u64)
        .is_some_and(|bytes| bytes <= plan.max_working_set_bytes)
    {
        BOUNDED_QUANTIZATION_TILE_BUFFERS
    } else {
        1
    };
    if tile_buffers == 1 {
        while !pending_tiles.is_empty() {
            write_oldest_tile(pending_tiles, output_shards, allocator_cache)?;
        }
    }
    let tile_budget = plan.max_working_set_bytes / tile_buffers as u64;
    let output_bytes = layouts.iter().try_fold(0u64, |total, layout| {
        total
            .checked_add(layout.byte_len)
            .ok_or_else(|| quantization_error("quantized output telemetry overflow"))
    })?;
    let output_shard = output_shards.len();
    let buffers = layouts
        .iter()
        .map(|layout| allocate_output_buffer(&layout.name, layout.byte_len))
        .collect::<Result<Vec<_>, _>>()?;
    output_shards.push(OutputShard {
        layouts,
        buffers,
        pending_tiles: 0,
        sealed: false,
    });
    let complete_rows = leading
        .checked_mul(rows)
        .ok_or_else(|| quantization_error("complete target row count overflow"))?;
    let complete_elements = complete_rows
        .checked_mul(columns)
        .ok_or_else(|| quantization_error("complete target element count overflow"))?;
    let complete_peak = target
        .source
        .peak_materialization_bytes(source)?
        .checked_add(output_bytes)
        .ok_or_else(|| quantization_error("complete target working-set overflow"))?;
    let leading_batch_admissible = if row_axis == 1 {
        let one_matrix = target.source.select_bounded(
            source,
            TensorSelection::Range {
                axis: 0,
                start: 0,
                end: 1,
            },
        )?;
        let one_matrix_output = output_row_bytes
            .checked_mul(rows as u64)
            .ok_or_else(|| quantization_error("one-matrix output size overflow"))?;
        one_matrix
            .peak_materialization_bytes(source)?
            .checked_add(one_matrix_output)
            .is_some_and(|peak| peak <= tile_budget)
            && rows
                .checked_mul(columns)
                .is_some_and(|elements| elements <= MAX_QUANTIZATION_SUBMISSION_ELEMENTS)
    } else {
        false
    };
    if complete_peak <= tile_budget && complete_elements <= MAX_QUANTIZATION_SUBMISSION_ELEMENTS {
        target.source.preflight_bounded(source)?;
        submit_quantization_tile(
            source,
            &target.source,
            target,
            plan.quantization,
            tile_contexts,
            tile_buffers,
            output_shard,
            0,
            complete_rows,
            complete_peak,
            output_bytes,
            output_shards,
            pending_tiles,
            allocator_cache,
            report,
        )?;
    } else if leading_batch_admissible {
        let mut matrix_start = 0usize;
        while matrix_start < leading {
            let mut matrix_end = matrix_start + 1;
            let mut rejected_end = leading.saturating_add(1);
            while matrix_end + 1 < rejected_end {
                let candidate_end = matrix_end + (rejected_end - matrix_end) / 2;
                let candidate = target.source.select_bounded(
                    source,
                    TensorSelection::Range {
                        axis: 0,
                        start: matrix_start,
                        end: candidate_end,
                    },
                )?;
                candidate.preflight_bounded(source)?;
                let candidate_rows = (candidate_end - matrix_start)
                    .checked_mul(rows)
                    .ok_or_else(|| quantization_error("expert batch row count overflow"))?;
                let candidate_elements = candidate_rows
                    .checked_mul(columns)
                    .ok_or_else(|| quantization_error("expert batch element count overflow"))?;
                let candidate_output_bytes = output_row_bytes
                    .checked_mul(candidate_rows as u64)
                    .ok_or_else(|| quantization_error("expert batch output size overflow"))?;
                let candidate_peak = candidate
                    .peak_materialization_bytes(source)?
                    .checked_add(candidate_output_bytes)
                    .ok_or_else(|| quantization_error("expert batch working-set overflow"))?;
                if candidate_elements <= MAX_QUANTIZATION_SUBMISSION_ELEMENTS
                    && candidate_peak <= tile_budget
                {
                    matrix_end = candidate_end;
                } else {
                    rejected_end = candidate_end;
                }
            }
            let recipe = target.source.select_bounded(
                source,
                TensorSelection::Range {
                    axis: 0,
                    start: matrix_start,
                    end: matrix_end,
                },
            )?;
            recipe.preflight_bounded(source)?;
            let batch_rows = (matrix_end - matrix_start)
                .checked_mul(rows)
                .ok_or_else(|| quantization_error("expert batch row count overflow"))?;
            let batch_output_bytes = output_row_bytes
                .checked_mul(batch_rows as u64)
                .ok_or_else(|| quantization_error("expert batch output size overflow"))?;
            let batch_peak = recipe
                .peak_materialization_bytes(source)?
                .checked_add(batch_output_bytes)
                .ok_or_else(|| quantization_error("expert batch working-set overflow"))?;
            if batch_peak > tile_budget {
                return Err(quantization_error(format!(
                    "bounded quantization target {:?} cannot admit one leading matrix within the {}-byte tile slot",
                    target.weight_name, tile_budget
                )));
            }
            submit_quantization_tile(
                source,
                &recipe,
                target,
                plan.quantization,
                tile_contexts,
                tile_buffers,
                output_shard,
                matrix_start * rows,
                batch_rows,
                batch_peak,
                batch_output_bytes,
                output_shards,
                pending_tiles,
                allocator_cache,
                report,
            )?;
            matrix_start = matrix_end;
        }
    } else {
        for matrix in 0..leading {
            let mut start = 0usize;
            while start < rows {
                // TP segmented placements can have discontinuities in their compact
                // row space. Admit each tile at its actual start so no tile crosses a
                // semantic segment boundary with a larger peak than the first tile.
                let mut end = start + 1;
                let mut rejected_end = rows.saturating_add(1);
                while end + 1 < rejected_end {
                    let candidate_end = end + (rejected_end - end) / 2;
                    let candidate = target.source.select_bounded_matrix_rows(
                        source,
                        matrix,
                        start,
                        candidate_end,
                    )?;
                    candidate.preflight_bounded(source)?;
                    let candidate_rows = candidate_end - start;
                    let output_bytes = output_row_bytes
                        .checked_mul(candidate_rows as u64)
                        .ok_or_else(|| quantization_error("candidate tile output size overflow"))?;
                    let peak = candidate
                        .peak_materialization_bytes(source)?
                        .checked_add(output_bytes)
                        .ok_or_else(|| quantization_error("candidate tile working-set overflow"))?;
                    if peak <= tile_budget {
                        end = candidate_end;
                    } else {
                        rejected_end = candidate_end;
                    }
                }
                let tile_rows = end - start;
                let tile_recipe = target
                    .source
                    .select_bounded_matrix_rows(source, matrix, start, end)?;
                tile_recipe.preflight_bounded(source)?;
                let tile_output_bytes = output_row_bytes
                    .checked_mul(tile_rows as u64)
                    .ok_or_else(|| quantization_error("quantized tile output size overflow"))?;
                let tile_peak = tile_recipe
                    .peak_materialization_bytes(source)?
                    .checked_add(tile_output_bytes)
                    .ok_or_else(|| quantization_error("conversion tile working-set overflow"))?;
                if tile_peak > tile_budget {
                    return Err(quantization_error(format!(
                    "bounded quantization planner admitted {} rows for {:?}, but their {}-byte working set exceeds the {}-byte tile slot",
                    tile_rows, target.weight_name, tile_peak, tile_budget
                )));
                }
                let output_start = matrix
                    .checked_mul(rows)
                    .and_then(|offset| offset.checked_add(start))
                    .ok_or_else(|| quantization_error("quantized output row offset overflow"))?;
                submit_quantization_tile(
                    source,
                    &tile_recipe,
                    target,
                    plan.quantization,
                    tile_contexts,
                    tile_buffers,
                    output_shard,
                    output_start,
                    tile_rows,
                    tile_peak,
                    tile_output_bytes,
                    output_shards,
                    pending_tiles,
                    allocator_cache,
                    report,
                )?;
                start = end;
            }
        }
    }
    output_shards[output_shard].seal()?;
    report.transformed_weights = report.transformed_weights.saturating_add(1);
    report.output_bytes = report
        .output_bytes
        .checked_add(output_bytes)
        .ok_or_else(|| quantization_error("quantized output telemetry overflow"))?;
    debug_assert_eq!(
        metadata.byte_len(),
        report_source_bytes(
            leading
                .checked_mul(rows)
                .ok_or_else(|| quantization_error("source row count overflow"))?,
            one_row_source_bytes
        )?
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn submit_quantization_tile(
    source: &dyn CheckpointSource,
    recipe: &DerivedWeightRecipe,
    target: &BoundedQuantizationTarget,
    quantization: WeightQuantization,
    tile_contexts: &[MlxParameterMaterializationContext; BOUNDED_QUANTIZATION_TILE_BUFFERS],
    tile_buffers: usize,
    output_shard: usize,
    output_start: usize,
    rows: usize,
    planned_working_set_bytes: u64,
    output_bytes: u64,
    output_shards: &mut [OutputShard],
    pending_tiles: &mut VecDeque<SubmittedQuantizationTile>,
    allocator_cache: &mut BoundedAllocatorCache,
    report: &mut WeightMaterializationReport,
) -> Result<(), Error> {
    allocator_cache.prepare_submission(
        queued_working_set_bytes(pending_tiles)?,
        planned_working_set_bytes,
    )?;
    let metadata = recipe.infer(source)?;
    let tile_context = &tile_contexts[report.source_tiles % tile_buffers];
    let tile_stream = tile_context.source_stream();
    let pending = recipe.prepare_borrowed_materialization(source, tile_context)?;
    let (dense, source_mappings) = pending.into_parts();
    let outputs = quantize_tile_outputs(&dense, quantization, target, tile_stream)?;
    let completion = async_eval_with_event(outputs.iter())?;
    output_shards[output_shard].tile_submitted();
    pending_tiles.push_back(SubmittedQuantizationTile {
        outputs,
        _dense: dense,
        source_mappings,
        completion: Some(completion),
        output_start,
        rows,
        planned_working_set_bytes,
        output_shard,
    });
    report.peak_in_flight_tiles = report.peak_in_flight_tiles.max(pending_tiles.len());
    report.source_tiles = report.source_tiles.saturating_add(1);
    report.source_bytes_read = report
        .source_bytes_read
        .checked_add(metadata.byte_len())
        .ok_or_else(|| quantization_error("source-read telemetry overflow"))?;
    report.peak_planned_working_set_bytes = report
        .peak_planned_working_set_bytes
        .max(queued_working_set_bytes(pending_tiles)?);
    report.largest_source_tile_bytes = report.largest_source_tile_bytes.max(metadata.byte_len());
    report.largest_output_tile_bytes = report.largest_output_tile_bytes.max(output_bytes);
    if pending_tiles.len() == tile_buffers {
        write_oldest_tile(pending_tiles, output_shards, allocator_cache)?;
    }
    Ok(())
}

fn quantize_tile_outputs(
    dense: &Array,
    quantization: WeightQuantization,
    target: &BoundedQuantizationTarget,
    stream: &Stream,
) -> Result<Vec<Array>, Error> {
    let quantized = quantize_tensor(dense, quantization, stream)?;
    let companion_dtype = match target.affine_companion_dtype {
        RecipeDtype::F16 => Dtype::Float16,
        RecipeDtype::BF16 => Dtype::Bfloat16,
        RecipeDtype::F32 => Dtype::Float32,
        _ => unreachable!("validated bounded companion dtype"),
    };
    let scales = if matches!(quantization, WeightQuantization::MxFp4) {
        quantized.scales
    } else {
        quantized.scales.as_dtype(companion_dtype, stream)?
    };
    let mut outputs = vec![quantized.weight, scales];
    if let Some(biases) = quantized.biases {
        outputs.push(biases.as_dtype(companion_dtype, stream)?);
    }
    Ok(outputs)
}

fn output_layouts(
    target: &BoundedQuantizationTarget,
    quantization: WeightQuantization,
    source_shape: &[usize],
    rows: usize,
    columns: usize,
    bits: usize,
) -> Result<Vec<OutputLayout>, Error> {
    let scales_name = target.scales_name.clone();
    let biases_name = target.biases_name.clone();
    let packed_columns = columns
        .checked_mul(bits)
        .and_then(|bits| bits.checked_div(32))
        .ok_or_else(|| quantization_error("packed column count overflow"))?;
    let scale_columns = columns
        .checked_div(quantization.group_size() as usize)
        .ok_or_else(|| quantization_error("scale column count overflow"))?;
    let mut prefix = source_shape[..source_shape.len() - 2].to_vec();
    prefix.push(rows);

    let mut layouts = Vec::with_capacity(if quantization.has_biases() { 3 } else { 2 });
    let (affine_dtype, affine_scalar_bytes) = match target.affine_companion_dtype {
        RecipeDtype::F16 => (SafeDtype::F16, 2),
        RecipeDtype::BF16 => (SafeDtype::BF16, 2),
        RecipeDtype::F32 => (SafeDtype::F32, 4),
        _ => {
            return Err(quantization_error(format!(
                "bounded affine companion output has invalid dtype {:?}",
                target.affine_companion_dtype
            )))
        }
    };
    layouts.push(layout(
        target.weight_name.clone(),
        SafeDtype::U32,
        prefix.clone(),
        packed_columns,
        4,
    )?);
    layouts.push(layout(
        scales_name,
        if matches!(quantization, WeightQuantization::MxFp4) {
            SafeDtype::U8
        } else {
            affine_dtype
        },
        prefix.clone(),
        scale_columns,
        if matches!(quantization, WeightQuantization::MxFp4) {
            1
        } else {
            affine_scalar_bytes
        },
    )?);
    if quantization.has_biases() {
        layouts.push(layout(
            biases_name.expect("bounded plan validated affine-bias identity"),
            affine_dtype,
            prefix,
            scale_columns,
            affine_scalar_bytes,
        )?);
    }

    Ok(layouts)
}

fn layout(
    name: String,
    dtype: SafeDtype,
    mut prefix: Vec<usize>,
    columns: usize,
    scalar_bytes: u64,
) -> Result<OutputLayout, Error> {
    prefix.push(columns);
    let row_bytes = (columns as u64)
        .checked_mul(scalar_bytes)
        .ok_or_else(|| quantization_error("quantized row size overflow"))?;
    let rows = checked_product(&prefix[..prefix.len() - 1], "quantized output rows")?;
    let byte_len = (rows as u64)
        .checked_mul(row_bytes)
        .ok_or_else(|| quantization_error("quantized tensor size overflow"))?;
    Ok(OutputLayout {
        name,
        dtype,
        shape: prefix,
        row_bytes,
        byte_len,
    })
}

fn allocate_output_buffer(name: &str, byte_len: u64) -> Result<Vec<u8>, Error> {
    let byte_len = usize::try_from(byte_len)
        .map_err(|_| quantization_error(format!("output {name:?} is too large for memory")))?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(byte_len).map_err(|error| {
        quantization_error(format!(
            "cannot allocate {byte_len} bytes for in-memory quantized output {name:?}: {error}"
        ))
    })?;
    bytes.resize(byte_len, 0);
    Ok(bytes)
}

struct SubmittedQuantizationTile {
    outputs: Vec<Array>,
    _dense: Array,
    source_mappings: Vec<PendingWeightMaterialization>,
    completion: Option<Event>,
    output_start: usize,
    rows: usize,
    planned_working_set_bytes: u64,
    output_shard: usize,
}

impl SubmittedQuantizationTile {
    fn write(mut self, shard: &mut OutputShard) -> Result<(), Error> {
        self.complete()?;
        for ((layout, output), buffer) in shard
            .layouts
            .iter()
            .zip(&self.outputs)
            .zip(&mut shard.buffers)
        {
            write_tile(buffer, layout, self.output_start, self.rows, output)?;
        }
        Ok(())
    }

    fn complete(&mut self) -> Result<(), Error> {
        let result = self
            .completion
            .take()
            .expect("submitted tile retains its completion")
            .synchronize();
        for mapping in self.source_mappings.drain(..) {
            mapping.complete();
        }
        result.map_err(Error::from)
    }
}

fn write_oldest_tile(
    pending_tiles: &mut VecDeque<SubmittedQuantizationTile>,
    output_shards: &mut [OutputShard],
    allocator_cache: &mut BoundedAllocatorCache,
) -> Result<(), Error> {
    let tile = pending_tiles
        .pop_front()
        .expect("non-empty tile window has a front");
    let output_shard = tile.output_shard;
    let shard = output_shards
        .get_mut(output_shard)
        .expect("submitted tile references an existing output shard");
    tile.write(shard)?;
    shard.tile_completed()?;
    allocator_cache.tile_completed(queued_working_set_bytes(pending_tiles)?)?;
    Ok(())
}

fn queued_working_set_bytes(
    pending_tiles: &VecDeque<SubmittedQuantizationTile>,
) -> Result<u64, Error> {
    pending_tiles.iter().try_fold(0u64, |total, tile| {
        total
            .checked_add(tile.planned_working_set_bytes)
            .ok_or_else(|| quantization_error("double-buffered working-set overflow"))
    })
}

impl Drop for SubmittedQuantizationTile {
    fn drop(&mut self) {
        if self.completion.is_some() {
            let _ = self.complete();
        }
    }
}

fn write_tile(
    buffer: &mut [u8],
    layout: &OutputLayout,
    start_row: usize,
    rows: usize,
    output: &Array,
) -> Result<(), Error> {
    let expected_bytes = layout
        .row_bytes
        .checked_mul(rows as u64)
        .ok_or_else(|| quantization_error("quantized tile byte count overflow"))?;
    if output.nbytes() as u64 != expected_bytes {
        return Err(quantization_error(format!(
            "quantized output {:?} produced {} bytes for rows {}..{}, expected {}",
            layout.name,
            output.nbytes(),
            start_row,
            start_row + rows,
            expected_bytes
        )));
    }
    let offset = layout
        .row_bytes
        .checked_mul(start_row as u64)
        .ok_or_else(|| quantization_error("quantized tile memory offset overflow"))?;
    let offset = usize::try_from(offset)
        .map_err(|_| quantization_error("quantized tile memory offset is not representable"))?;
    let expected_bytes = usize::try_from(expected_bytes)
        .map_err(|_| quantization_error("quantized tile byte count is not representable"))?;
    let destination = buffer
        .get_mut(offset..offset + expected_bytes)
        .ok_or_else(|| {
            quantization_error(format!(
                "quantized output {:?} tile rows {}..{} exceed its in-memory buffer",
                layout.name,
                start_row,
                start_row + rows
            ))
        })?;
    let evaluated = output.evaluated()?;
    let source = match output.dtype() {
        Dtype::Uint32 => native_slice_bytes(evaluated.as_slice::<u32>()),
        Dtype::Float16 => native_slice_bytes(evaluated.as_slice::<half::f16>()),
        Dtype::Bfloat16 => native_slice_bytes(evaluated.as_slice::<half::bf16>()),
        Dtype::Float32 => native_slice_bytes(evaluated.as_slice::<f32>()),
        Dtype::Uint8 => evaluated.as_slice::<u8>(),
        dtype => {
            return Err(quantization_error(format!(
                "quantized output {:?} has unsupported dtype {dtype:?}",
                layout.name
            )))
        }
    };
    destination.copy_from_slice(source);
    Ok(())
}

fn native_slice_bytes<T>(values: &[T]) -> &[u8] {
    unsafe {
        // SAFETY: `values` is a valid initialized slice, and its byte view has
        // the identical lifetime. The public constructor rejects big-endian
        // hosts because SafeTensors payloads are little-endian.
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn preflight_source_collisions(
    source: &dyn eredu_checkpoint::store::CheckpointSource,
    plan: &BoundedQuantizationPlan,
) -> Result<(), Error> {
    let source_keys = source.source_keys().into_iter().collect::<BTreeSet<_>>();
    for target in &plan.targets {
        if source_keys.contains(&target.scales_name)
            || target
                .biases_name
                .as_ref()
                .is_some_and(|name| source_keys.contains(name))
        {
            return Err(quantization_error(format!(
                "bounded quantization target {:?} already has checkpoint quantization companions; implicit transcoding is unsupported",
                target.weight_name
            )));
        }
    }
    Ok(())
}

fn output_names_for(
    target: &BoundedQuantizationTarget,
    quantization: WeightQuantization,
) -> Result<Vec<String>, Error> {
    let mut names = vec![target.weight_name.clone(), target.scales_name.clone()];
    if quantization.has_biases() {
        names.push(
            target
                .biases_name
                .clone()
                .expect("bounded plan validated affine-bias identity"),
        );
    }
    Ok(names)
}

fn validate_output_names(
    weight_name: &str,
    scales_name: &str,
    biases_name: Option<&str>,
) -> Result<(), Error> {
    if weight_name.trim().is_empty()
        || scales_name.trim().is_empty()
        || biases_name.is_some_and(|name| name.trim().is_empty())
    {
        return Err(quantization_error(
            "bounded quantization output identities must not be empty",
        ));
    }
    if weight_name == scales_name
        || biases_name.is_some_and(|name| name == weight_name || name == scales_name)
    {
        return Err(quantization_error(
            "bounded quantization output identities must be distinct",
        ));
    }
    Ok(())
}

fn checked_product(dimensions: &[usize], context: &'static str) -> Result<usize, Error> {
    dimensions.iter().try_fold(1usize, |product, dimension| {
        product
            .checked_mul(*dimension)
            .ok_or_else(|| quantization_error(format!("{context} overflow")))
    })
}

fn report_source_bytes(rows: usize, one_row_bytes: u64) -> Result<u64, Error> {
    one_row_bytes
        .checked_mul(rows as u64)
        .ok_or_else(|| quantization_error("source byte count overflow"))
}

fn quantization_error(message: impl Into<String>) -> Error {
    Error::Quantization(message.into())
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use crate::{backend::runtime::checkpoint::gguf::GgufCheckpoint, backend::ExecutionContext};
    use eredu_checkpoint::{
        store::{ReadPolicy as WeightReadPolicy, SafetensorsWeightStore, WeightStoreBackend},
        AffineQuantization,
    };
    use safemlx::{Device, DeviceType};
    use safetensors::tensor::{serialize_to_file, TensorView};
    use tempfile::TempDir;

    use super::*;
    use crate::backend::runtime::{
        checkpoint::store::open_gguf_checkpoint_source_for_test,
        residency::manager::{host_capacity_upper_bound_for_bindings, ResidencyManager},
    };
    use crate::test_utils::SyntheticGguf;
    use eredu_core::residency::{
        MemoryTier, OffloadConfig, OffloadPlan, OffloadUnitId, OffloadUnitSpec, ResidencyPolicy,
    };
    use eredu_runtime::{OffloadUnit, WeightBinding};

    fn cpu_context() -> ExecutionContext {
        ExecutionContext::new(Device::new(DeviceType::Cpu, 0))
    }

    fn test_target(weight_name: &str, source: DerivedWeightRecipe) -> BoundedQuantizationTarget {
        let (scales, biases) = weight_name.strip_suffix(".weight").map_or_else(
            || {
                (
                    format!("{weight_name}_scales"),
                    format!("{weight_name}_biases"),
                )
            },
            |prefix| (format!("{prefix}.scales"), format!("{prefix}.biases")),
        );
        BoundedQuantizationTarget::from_recipe(weight_name, scales, Some(biases), source).unwrap()
    }

    fn direct_test_target(weight_name: &str) -> BoundedQuantizationTarget {
        test_target(
            weight_name,
            DerivedWeightRecipe::source(weight_name, TensorSelection::Full),
        )
    }

    fn float_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn matrix_values(matrices: usize, rows: usize, columns: usize) -> Vec<f32> {
        (0..matrices * rows * columns)
            .map(|index| (index as f32 - 255.5) / 64.0)
            .collect()
    }

    #[test]
    fn allocator_cache_reuse_respects_retained_and_working_set_bounds() {
        assert!(!allocator_cache_requires_clear(64, 64, 64));
        assert!(allocator_cache_requires_clear(65, 64, 128));
        assert!(allocator_cache_requires_clear(65, 128, 64));
    }

    fn direct_fixture() -> (TempDir, Arc<SafetensorsWeightStore>, Vec<f32>) {
        let directory = tempfile::tempdir().unwrap();
        let values = matrix_values(1, 8, 64);
        let bytes = float_bytes(&values);
        serialize_to_file(
            [(
                "model.proj.weight",
                TensorView::new(SafeDtype::F32, vec![8, 64], &bytes).unwrap(),
            )],
            None,
            &directory.path().join("model.safetensors"),
        )
        .unwrap();
        let store = Arc::new(SafetensorsWeightStore::open(directory.path()).unwrap());
        (directory, store, values)
    }

    fn two_target_fixture() -> (TempDir, Arc<SafetensorsWeightStore>, [Vec<f32>; 2]) {
        let directory = tempfile::tempdir().unwrap();
        let values = [matrix_values(1, 8, 64), matrix_values(1, 8, 64)];
        let bytes = values.each_ref().map(|values| float_bytes(values));
        serialize_to_file(
            [
                (
                    "model.first.weight",
                    TensorView::new(SafeDtype::F32, vec![8, 64], &bytes[0]).unwrap(),
                ),
                (
                    "model.second.weight",
                    TensorView::new(SafeDtype::F32, vec![8, 64], &bytes[1]).unwrap(),
                ),
            ],
            None,
            &directory.path().join("model.safetensors"),
        )
        .unwrap();
        let store = Arc::new(SafetensorsWeightStore::open(directory.path()).unwrap());
        (directory, store, values)
    }

    fn materialize(
        store: &dyn eredu_checkpoint::store::CheckpointSource,
        name: &str,
        stream: &Stream,
    ) -> Array {
        let lease = store
            .acquire_lease(TensorReadRequest {
                key: name.into(),
                selection: TensorSelection::Full,
                policy: WeightReadPolicy::RequireBounded,
            })
            .unwrap();
        MlxParameterMaterializationContext::new(stream, stream)
            .weight_lease(lease)
            .unwrap()
            .materialize(stream, stream)
            .unwrap()
            .synchronize()
            .unwrap()
    }

    fn assert_affine_outputs_match_reference(
        transformed: &BoundedQuantizedWeightStore,
        values: &[f32],
        quantization: AffineQuantization,
        context: &ExecutionContext,
    ) {
        let dense = Array::from_slice(values, &[8, 64]);
        let expected = quantize_tensor(&dense, quantization, context.stream()).unwrap();
        let weight = materialize(transformed, "model.proj.weight", context.stream());
        let scales = materialize(transformed, "model.proj.scales", context.stream());
        let biases = materialize(transformed, "model.proj.biases", context.stream());
        assert_eq!(
            weight.evaluated().unwrap().as_slice::<u32>(),
            expected.weight.evaluated().unwrap().as_slice::<u32>()
        );
        assert_eq!(
            scales.evaluated().unwrap().as_slice::<f32>(),
            expected.scales.evaluated().unwrap().as_slice::<f32>()
        );
        assert_eq!(
            biases.evaluated().unwrap().as_slice::<f32>(),
            expected
                .biases
                .unwrap()
                .evaluated()
                .unwrap()
                .as_slice::<f32>()
        );
    }

    fn assert_mxfp4_outputs_match_reference(
        transformed: &BoundedQuantizedWeightStore,
        values: &[f32],
        context: &ExecutionContext,
    ) {
        let dense = Array::from_slice(values, &[8, 64]);
        let expected =
            quantize_tensor(&dense, WeightQuantization::MxFp4, context.stream()).unwrap();
        let weight = materialize(transformed, "model.proj.weight", context.stream());
        let scales = materialize(transformed, "model.proj.scales", context.stream());
        assert_eq!(
            weight.evaluated().unwrap().as_slice::<u32>(),
            expected.weight.evaluated().unwrap().as_slice::<u32>()
        );
        assert_eq!(
            scales.evaluated().unwrap().as_slice::<u8>(),
            expected.scales.evaluated().unwrap().as_slice::<u8>()
        );
    }

    #[test]
    fn affine_conversion_is_row_bounded_and_matches_the_canonical_quantizer() {
        let (_directory, source, values) = direct_fixture();
        let context = cpu_context();
        let quantization = AffineQuantization::default();
        let target = direct_test_target("model.proj.weight");
        // One f32 source row (256 bytes) plus one packed affine output row
        // (32-byte weight, 4-byte scale, and 4-byte bias). The complete
        // quantized matrix is 320 bytes, so that runtime-sized budget is also
        // sufficient to convert the 2,048-byte dense source.
        let plan = BoundedQuantizationPlan::new(quantization, 320, [target]).unwrap();
        let transformed =
            BoundedQuantizedWeightStore::create(source.clone(), plan, context.stream()).unwrap();

        assert_eq!(
            transformed.report(),
            &WeightMaterializationReport {
                admitted_working_set_bytes: 320,
                transformed_weights: 1,
                source_tiles: 8,
                peak_in_flight_tiles: 1,
                source_bytes_read: 2_048,
                output_bytes: 320,
                peak_planned_working_set_bytes: 296,
                largest_source_tile_bytes: 256,
                largest_output_tile_bytes: 40,
            }
        );
        assert!(
            transformed.report().peak_planned_working_set_bytes
                <= transformed.report().output_bytes
        );
        assert_eq!(
            transformed
                .source_metadata("model.proj.weight")
                .unwrap()
                .encoded_byte_len,
            256
        );
        assert_eq!(
            transformed
                .source_metadata("model.proj.scales")
                .unwrap()
                .encoded_byte_len,
            32
        );
        assert_eq!(
            transformed
                .source_metadata("model.proj.biases")
                .unwrap()
                .encoded_byte_len,
            32
        );

        assert_affine_outputs_match_reference(&transformed, &values, quantization, &context);
    }

    #[test]
    fn affine_conversion_double_buffers_two_cpu_tiles_within_the_bound() {
        let (_directory, source, values) = direct_fixture();
        let context = cpu_context();
        let quantization = AffineQuantization::default();
        let plan = BoundedQuantizationPlan::new(
            quantization,
            640,
            [direct_test_target("model.proj.weight")],
        )
        .unwrap();
        let transformed =
            BoundedQuantizedWeightStore::create(source, plan, context.stream()).unwrap();

        let report = transformed.report();
        assert_eq!(report.source_tiles, 8);
        assert_eq!(report.peak_in_flight_tiles, 2);
        assert_eq!(report.peak_planned_working_set_bytes, 592);
        assert!(report.peak_planned_working_set_bytes <= report.admitted_working_set_bytes);

        assert_affine_outputs_match_reference(&transformed, &values, quantization, &context);
    }

    #[test]
    fn mxfp4_conversion_double_buffers_across_target_boundaries() {
        let (_directory, source, values) = two_target_fixture();
        let context = cpu_context();
        let quantization = WeightQuantization::MxFp4;
        // Each complete target needs 2,320 bytes: 2,048 source bytes plus
        // 272 packed output bytes. Both targets therefore fit as one tile in
        // the two-slot window, making cross-target overlap observable.
        let plan = BoundedQuantizationPlan::new(
            quantization,
            4_640,
            [
                direct_test_target("model.first.weight"),
                direct_test_target("model.second.weight"),
            ],
        )
        .unwrap();
        let transformed =
            BoundedQuantizedWeightStore::create(source, plan, context.stream()).unwrap();

        let report = transformed.report();
        assert_eq!(report.transformed_weights, 2);
        assert_eq!(report.source_tiles, 2);
        assert_eq!(report.peak_in_flight_tiles, 2);
        assert_eq!(report.peak_planned_working_set_bytes, 4_640);
        assert_eq!(report.source_bytes_read, 4_096);
        assert_eq!(report.output_bytes, 544);

        for (name, values) in [("model.first", &values[0]), ("model.second", &values[1])] {
            let dense = Array::from_slice(values, &[8, 64]);
            let expected = quantize_tensor(&dense, quantization, context.stream()).unwrap();
            let weight = materialize(&transformed, &format!("{name}.weight"), context.stream());
            let scales = materialize(&transformed, &format!("{name}.scales"), context.stream());
            assert_eq!(
                weight.evaluated().unwrap().as_slice::<u32>(),
                expected.weight.evaluated().unwrap().as_slice::<u32>()
            );
            assert_eq!(
                scales.evaluated().unwrap().as_slice::<u8>(),
                expected.scales.evaluated().unwrap().as_slice::<u8>()
            );
            assert!(!transformed
                .source_keys()
                .contains(&format!("{name}.biases")));
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn affine_gpu_conversion_matches_the_canonical_gpu_quantizer() {
        let (_directory, source, values) = direct_fixture();
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let quantization = AffineQuantization::default();
        let plan = BoundedQuantizationPlan::new(
            quantization,
            640,
            [direct_test_target("model.proj.weight")],
        )
        .unwrap();
        let transformed =
            BoundedQuantizedWeightStore::create(source, plan, context.stream()).unwrap();

        assert_affine_outputs_match_reference(&transformed, &values, quantization, &context);
    }

    #[test]
    fn insufficient_bound_fails_before_a_source_array_is_materialized() {
        let (_directory, source, _values) = direct_fixture();
        let context = cpu_context();
        let plan = BoundedQuantizationPlan::new(
            AffineQuantization::default(),
            295,
            [direct_test_target("model.proj.weight")],
        )
        .unwrap();
        let error = BoundedQuantizedWeightStore::create(source.clone(), plan, context.stream())
            .unwrap_err();
        assert!(error.to_string().contains(
            "requires at least 296 working-set bytes for one row, but the plan permits 295"
        ));
        let diagnostics = source.source_diagnostics().unwrap();
        // Metadata/preflight may map the shard, but no selected payload was
        // converted and no GGUF physical read was issued.
        assert_eq!(diagnostics.physical_reads, 0);
    }

    #[test]
    fn semantic_expert_recipe_is_quantized_under_its_local_target_name() {
        let directory = tempfile::tempdir().unwrap();
        let values = matrix_values(1, 4, 64);
        let bytes = float_bytes(&values);
        serialize_to_file(
            [(
                "checkpoint.expert_1.weight",
                TensorView::new(SafeDtype::F32, vec![4, 64], &bytes).unwrap(),
            )],
            None,
            &directory.path().join("model.safetensors"),
        )
        .unwrap();
        let source = Arc::new(SafetensorsWeightStore::open(directory.path()).unwrap());
        let context = cpu_context();
        let recipe =
            DerivedWeightRecipe::source("checkpoint.expert_1.weight", TensorSelection::Full);
        let target = test_target("local.expert.weight", recipe);
        let plan =
            BoundedQuantizationPlan::new(AffineQuantization::default(), 296, [target]).unwrap();
        let transformed =
            BoundedQuantizedWeightStore::create(source, plan, context.stream()).unwrap();
        assert_eq!(transformed.report().source_tiles, 4);
        assert_eq!(transformed.report().source_bytes_read, 1_024);

        let dense = Array::from_slice(&values, &[4, 64]);
        let expected =
            quantize_tensor(&dense, AffineQuantization::default(), context.stream()).unwrap();
        let actual = materialize(&transformed, "local.expert.weight", context.stream());
        assert_eq!(
            actual.evaluated().unwrap().as_slice::<u32>(),
            expected.weight.evaluated().unwrap().as_slice::<u32>()
        );
    }

    #[test]
    fn expert_ownership_and_tp_row_tile_compose_into_bounded_reads() {
        let directory = tempfile::tempdir().unwrap();
        let values = matrix_values(2, 4, 64);
        let bytes = float_bytes(&values);
        serialize_to_file(
            [(
                "checkpoint.experts.weight",
                TensorView::new(SafeDtype::F32, vec![2, 4, 64], &bytes).unwrap(),
            )],
            None,
            &directory.path().join("model.safetensors"),
        )
        .unwrap();
        let source = Arc::new(SafetensorsWeightStore::open(directory.path()).unwrap());
        let context = cpu_context();
        let recipe = DerivedWeightRecipe::Select {
            input: Box::new(DerivedWeightRecipe::source(
                "checkpoint.experts.weight",
                TensorSelection::Range {
                    axis: 0,
                    start: 1,
                    end: 2,
                },
            )),
            selection: TensorSelection::Range {
                axis: 1,
                start: 1,
                end: 3,
            },
        };
        let target = test_target("rank.expert.weight", recipe);
        let plan =
            BoundedQuantizationPlan::new(AffineQuantization::default(), 296, [target]).unwrap();
        let transformed =
            BoundedQuantizedWeightStore::create(source, plan, context.stream()).unwrap();

        assert_eq!(transformed.report().source_tiles, 2);
        assert_eq!(transformed.report().source_bytes_read, 512);
        assert_eq!(transformed.report().output_bytes, 80);
        assert_eq!(transformed.report().peak_planned_working_set_bytes, 296);

        let selected = &values[(4 + 1) * 64..(4 + 3) * 64];
        let dense = Array::from_slice(selected, &[1, 2, 64]);
        let expected =
            quantize_tensor(&dense, AffineQuantization::default(), context.stream()).unwrap();
        let actual = materialize(&transformed, "rank.expert.weight", context.stream());
        assert_eq!(actual.shape(), &[1, 2, 8]);
        assert_eq!(
            actual.evaluated().unwrap().as_slice::<u32>(),
            expected.weight.evaluated().unwrap().as_slice::<u32>()
        );
    }

    #[test]
    fn complete_rank_three_expert_bank_is_tiled_without_flattening_its_output() {
        let directory = tempfile::tempdir().unwrap();
        let values = matrix_values(2, 4, 64);
        let bytes = float_bytes(&values);
        serialize_to_file(
            [(
                "model.experts.weight",
                TensorView::new(SafeDtype::F32, vec![2, 4, 64], &bytes).unwrap(),
            )],
            None,
            &directory.path().join("model.safetensors"),
        )
        .unwrap();
        let source = Arc::new(SafetensorsWeightStore::open(directory.path()).unwrap());
        let context = cpu_context();
        let plan = BoundedQuantizationPlan::new(
            AffineQuantization::default(),
            296,
            [direct_test_target("model.experts.weight")],
        )
        .unwrap();
        let transformed =
            BoundedQuantizedWeightStore::create(source, plan, context.stream()).unwrap();

        assert_eq!(transformed.report().source_tiles, 8);
        assert_eq!(transformed.report().source_bytes_read, 2_048);
        assert_eq!(transformed.report().output_bytes, 320);
        assert_eq!(
            transformed
                .source_metadata("model.experts.weight")
                .unwrap()
                .logical_shape,
            vec![2, 4, 8]
        );
        let expected = quantize_tensor(
            &Array::from_slice(&values, &[2, 4, 64]),
            AffineQuantization::default(),
            context.stream(),
        )
        .unwrap();
        let actual = materialize(&transformed, "model.experts.weight", context.stream());
        assert_eq!(actual.shape(), &[2, 4, 8]);
        assert_eq!(
            actual.evaluated().unwrap().as_slice::<u32>(),
            expected.weight.evaluated().unwrap().as_slice::<u32>()
        );
    }

    #[test]
    fn complete_rank_three_expert_bank_uses_one_submission_when_admitted() {
        let directory = tempfile::tempdir().unwrap();
        let values = matrix_values(2, 4, 64);
        let bytes = float_bytes(&values);
        serialize_to_file(
            [(
                "model.experts.weight",
                TensorView::new(SafeDtype::F32, vec![2, 4, 64], &bytes).unwrap(),
            )],
            None,
            &directory.path().join("model.safetensors"),
        )
        .unwrap();
        let source = Arc::new(SafetensorsWeightStore::open(directory.path()).unwrap());
        let context = cpu_context();
        let plan = BoundedQuantizationPlan::new(
            AffineQuantization::default(),
            5_000,
            [direct_test_target("model.experts.weight")],
        )
        .unwrap();
        let transformed =
            BoundedQuantizedWeightStore::create(source, plan, context.stream()).unwrap();

        assert_eq!(transformed.report().source_tiles, 1);
        assert_eq!(transformed.report().source_bytes_read, 2_048);
        let expected = quantize_tensor(
            &Array::from_slice(&values, &[2, 4, 64]),
            AffineQuantization::default(),
            context.stream(),
        )
        .unwrap();
        let actual = materialize(&transformed, "model.experts.weight", context.stream());
        assert_eq!(actual.shape(), &[2, 4, 8]);
        assert_eq!(
            actual.evaluated().unwrap().as_slice::<u32>(),
            expected.weight.evaluated().unwrap().as_slice::<u32>()
        );
    }

    #[test]
    fn complete_rank_three_bank_preserves_exact_runtime_companion_names() {
        let values = matrix_values(2, 4, 64);
        let dense = Array::from_slice(&values, &[2, 4, 64]);
        let fixture = SyntheticGguf::dense(
            &HashMap::from([("checkpoint.experts.weight".to_string(), dense)]),
            &HashMap::new(),
        );
        let source = Arc::new(
            open_gguf_checkpoint_source_for_test(
                GgufCheckpoint::open(fixture.path()).unwrap(),
                |name| name.to_string(),
            )
            .unwrap(),
        );
        let context = cpu_context();
        let target = BoundedQuantizationTarget::from_recipe(
            "model.experts.down_proj",
            "architecture.scale-table",
            Some("architecture.zero-points"),
            DerivedWeightRecipe::source("checkpoint.experts.weight", TensorSelection::Full),
        )
        .unwrap();
        assert_eq!(target.scales_name(), "architecture.scale-table");
        assert_eq!(target.biases_name(), Some("architecture.zero-points"));
        let plan =
            BoundedQuantizationPlan::new(AffineQuantization::default(), 296, [target]).unwrap();
        let transformed =
            BoundedQuantizedWeightStore::create(source.clone(), plan, context.stream()).unwrap();

        assert_eq!(
            transformed
                .source_metadata("model.experts.down_proj")
                .unwrap()
                .logical_shape,
            vec![2, 4, 8]
        );
        assert_eq!(
            transformed
                .source_metadata("architecture.scale-table")
                .unwrap()
                .logical_shape,
            vec![2, 4, 1]
        );
        assert_eq!(
            transformed
                .source_metadata("architecture.zero-points")
                .unwrap()
                .logical_shape,
            vec![2, 4, 1]
        );
        let diagnostics = source.source_diagnostics().unwrap();
        assert_eq!(diagnostics.physical_reads, 8);
        assert_eq!(diagnostics.physical_read_bytes, 2_048);
    }

    #[test]
    fn dense_gguf_expert_and_row_selections_compose_without_full_bank_reads() {
        let values = matrix_values(2, 4, 64);
        let dense = Array::from_slice(&values, &[2, 4, 64]);
        let fixture = SyntheticGguf::dense(
            &HashMap::from([("checkpoint.experts.weight".to_string(), dense)]),
            &HashMap::new(),
        );
        let source = Arc::new(
            open_gguf_checkpoint_source_for_test(
                GgufCheckpoint::open(fixture.path()).unwrap(),
                |name| name.to_string(),
            )
            .unwrap(),
        );
        let context = cpu_context();
        let recipe = DerivedWeightRecipe::Select {
            input: Box::new(DerivedWeightRecipe::source(
                "checkpoint.experts.weight",
                TensorSelection::Range {
                    axis: 0,
                    start: 1,
                    end: 2,
                },
            )),
            selection: TensorSelection::Range {
                axis: 1,
                start: 1,
                end: 3,
            },
        };
        let target = test_target("rank.expert.weight", recipe);
        let plan =
            BoundedQuantizationPlan::new(AffineQuantization::default(), 296, [target]).unwrap();
        let transformed =
            BoundedQuantizedWeightStore::create(source.clone(), plan, context.stream()).unwrap();

        assert_eq!(transformed.report().source_tiles, 2);
        assert_eq!(transformed.report().source_bytes_read, 512);
        assert_eq!(transformed.report().output_bytes, 80);
        assert_eq!(transformed.report().peak_planned_working_set_bytes, 296);
        let diagnostics = source.source_diagnostics().unwrap();
        assert_eq!(diagnostics.physical_reads, 2);
        assert_eq!(diagnostics.physical_read_bytes, 512);

        let selected = &values[(4 + 1) * 64..(4 + 3) * 64];
        let expected = quantize_tensor(
            &Array::from_slice(selected, &[1, 2, 64]),
            AffineQuantization::default(),
            context.stream(),
        )
        .unwrap();
        let actual = materialize(&transformed, "rank.expert.weight", context.stream());
        assert_eq!(actual.shape(), &[1, 2, 8]);
        assert_eq!(
            actual.evaluated().unwrap().as_slice::<u32>(),
            expected.weight.evaluated().unwrap().as_slice::<u32>()
        );
    }

    #[test]
    fn complete_rank_three_dense_gguf_bank_is_read_one_matrix_row_at_a_time() {
        let values = matrix_values(2, 4, 64);
        let dense = Array::from_slice(&values, &[2, 4, 64]);
        let fixture = SyntheticGguf::dense(
            &HashMap::from([("model.experts.weight".to_string(), dense.clone())]),
            &HashMap::new(),
        );
        let source = Arc::new(
            open_gguf_checkpoint_source_for_test(
                GgufCheckpoint::open(fixture.path()).unwrap(),
                |name| name.to_string(),
            )
            .unwrap(),
        );
        let context = cpu_context();
        let plan = BoundedQuantizationPlan::new(
            AffineQuantization::default(),
            296,
            [direct_test_target("model.experts.weight")],
        )
        .unwrap();
        let transformed =
            BoundedQuantizedWeightStore::create(source.clone(), plan, context.stream()).unwrap();

        assert_eq!(transformed.report().source_tiles, 8);
        assert_eq!(transformed.report().source_bytes_read, 2_048);
        assert_eq!(transformed.report().output_bytes, 320);
        let diagnostics = source.source_diagnostics().unwrap();
        assert_eq!(diagnostics.physical_reads, 8);
        assert_eq!(diagnostics.physical_read_bytes, 2_048);
        assert_eq!(
            transformed
                .source_metadata("model.experts.weight")
                .unwrap()
                .logical_shape,
            vec![2, 4, 8]
        );
        let expected =
            quantize_tensor(&dense, AffineQuantization::default(), context.stream()).unwrap();
        let actual = materialize(&transformed, "model.experts.weight", context.stream());
        assert_eq!(
            actual.evaluated().unwrap().as_slice::<u32>(),
            expected.weight.evaluated().unwrap().as_slice::<u32>()
        );
    }

    #[test]
    fn mxfp4_layout_has_byte_scales_and_no_biases() {
        let (_directory, source, values) = direct_fixture();
        let context = cpu_context();
        let plan = BoundedQuantizationPlan::new(
            WeightQuantization::MxFp4,
            290,
            [direct_test_target("model.proj.weight")],
        )
        .unwrap();
        let transformed =
            BoundedQuantizedWeightStore::create(source, plan, context.stream()).unwrap();
        assert_eq!(transformed.report().source_tiles, 8);
        assert_eq!(transformed.report().output_bytes, 272);
        assert_eq!(
            transformed
                .source_metadata("model.proj.scales")
                .unwrap()
                .stored_dtype,
            eredu_checkpoint::StoredDtype::U8
        );
        assert!(!transformed
            .source_keys()
            .contains(&"model.proj.biases".into()));

        assert_mxfp4_outputs_match_reference(&transformed, &values, &context);
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn mxfp4_gpu_conversion_matches_the_canonical_gpu_quantizer() {
        let (_directory, source, values) = direct_fixture();
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let plan = BoundedQuantizationPlan::new(
            WeightQuantization::MxFp4,
            580,
            [direct_test_target("model.proj.weight")],
        )
        .unwrap();
        let transformed =
            BoundedQuantizedWeightStore::create(source, plan, context.stream()).unwrap();

        assert_mxfp4_outputs_match_reference(&transformed, &values, &context);
    }

    #[test]
    fn dense_gguf_source_is_read_and_quantized_in_bounded_rows() {
        let values = matrix_values(1, 8, 64);
        let dense = Array::from_slice(&values, &[8, 64]);
        let fixture = SyntheticGguf::dense(
            &HashMap::from([("model.proj.weight".to_string(), dense.clone())]),
            &HashMap::new(),
        );
        let source = Arc::new(
            open_gguf_checkpoint_source_for_test(
                GgufCheckpoint::open(fixture.path()).unwrap(),
                |name| name.to_string(),
            )
            .unwrap(),
        );
        let context = cpu_context();
        let plan = BoundedQuantizationPlan::new(
            AffineQuantization::default(),
            320,
            [direct_test_target("model.proj.weight")],
        )
        .unwrap();
        let transformed =
            BoundedQuantizedWeightStore::create(source, plan, context.stream()).unwrap();
        assert_eq!(
            transformed.source_diagnostics().unwrap().backend,
            WeightStoreBackend::Gguf
        );
        assert_eq!(transformed.report().source_tiles, 8);
        assert_eq!(transformed.report().source_bytes_read, 2_048);
        let diagnostics = transformed.source_diagnostics().unwrap();
        assert_eq!(diagnostics.physical_reads, 8);
        assert_eq!(diagnostics.physical_read_bytes, 2_048);

        let expected =
            quantize_tensor(&dense, AffineQuantization::default(), context.stream()).unwrap();
        let actual = materialize(&transformed, "model.proj.weight", context.stream());
        assert_eq!(
            actual.evaluated().unwrap().as_slice::<u32>(),
            expected.weight.evaluated().unwrap().as_slice::<u32>()
        );
    }

    #[test]
    fn residency_budgets_and_arrays_use_only_packed_bytes() {
        let (_directory, source, _values) = direct_fixture();
        let conversion_context = cpu_context();
        let plan = BoundedQuantizationPlan::new(
            AffineQuantization::default(),
            296,
            [direct_test_target("model.proj.weight")],
        )
        .unwrap();
        let transformed = Arc::new(
            BoundedQuantizedWeightStore::create(source, plan, conversion_context.stream()).unwrap(),
        );
        let id = OffloadUnitId::new("projection").unwrap();
        let bindings = [
            ("weight", "model.proj.weight", 256),
            ("scales", "model.proj.scales", 32),
            ("biases", "model.proj.biases", 32),
        ]
        .into_iter()
        .map(|(name, key, bytes)| {
            WeightBinding::new(name, key, TensorSelection::Full, bytes).unwrap()
        })
        .collect::<Vec<_>>();
        let host_capacity = host_capacity_upper_bound_for_bindings(&bindings).unwrap();
        let unit = OffloadUnit::new(id.clone(), bindings).unwrap();
        let spec = OffloadUnitSpec::new(id.clone(), 320, ResidencyPolicy::Pinned, MemoryTier::Host)
            .unwrap();
        let offload = OffloadPlan::new(
            OffloadConfig::new(None, Some(host_capacity), 1).unwrap(),
            [spec],
        )
        .unwrap();
        let source_context = cpu_context();
        let device_context = cpu_context();
        let manager = ResidencyManager::new(
            transformed,
            offload,
            [unit],
            source_context.stream().clone(),
            device_context.stream().clone(),
        )
        .unwrap();
        manager.initialize().unwrap();
        let report = manager.report().unwrap();
        assert_eq!(report.offload().planned_bytes().get(MemoryTier::Host), 320);
        assert_eq!(
            report.offload().resident_bytes().get(MemoryTier::Host),
            host_capacity
        );
        let lease = manager.acquire(&id, MemoryTier::Host).unwrap();
        let resident_bytes = lease
            .binding_names()
            .map(|name| lease.host_value(name).unwrap().nbytes().unwrap())
            .sum::<usize>();
        assert_eq!(resident_bytes, 320);
    }
}
