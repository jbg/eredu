//! Bounded, out-of-core load-time weight quantization.
//!
//! A [`BoundedQuantizedWeightStore`](crate::runtime::checkpoint::bounded_quantization::BoundedQuantizedWeightStore)
//! overlays packed tensors on an existing
//! checkpoint store. Source matrices are selected in row tiles, quantized on
//! explicit conversion streams, and written directly into temporary SafeTensors
//! shards. A fixed two-slot completion window spans tensor boundaries and
//! overlaps the next tile with the prior tile's host write when both slots fit
//! the admitted working set, and otherwise falls back to one slot. The
//! process never requires a complete dense matrix or a complete packed matrix
//! in active memory: tile size is admitted against an explicit byte bound. The
//! ordinary residency machinery subsequently sees only the final packed tensor
//! geometry.

use std::{
    any::Any,
    collections::{BTreeMap, BTreeSet, VecDeque},
    fs::{File, OpenOptions},
    io::{Seek, SeekFrom, Write},
    path::Path,
    sync::Arc,
};

use safemlx::{memory, transforms::async_eval_with_event, Array, Dtype, Event, Stream};
use safetensors::tensor::{Dtype as SafeDtype, TensorInfo};
use serde::Serialize;
use tempfile::TempDir;

use crate::{
    error::Error,
    runtime::checkpoint::{
        quantization::{quantize_tensor, WeightQuantization},
        recipe::{DerivedWeightRecipe, RecipeDtype},
        store::{
            PendingWeightMaterialization, SafetensorsWeightStore, TensorSelection, WeightLease,
            WeightMetadata, WeightReadPolicy, WeightStore, WeightStoreBackend,
            WeightStoreDiagnostics, WeightStoreError,
        },
    },
};

const BOUNDED_QUANTIZATION_TILE_BUFFERS: usize = 2;
const BOUNDED_QUANTIZATION_MAX_CACHE_BYTES: u64 = 64 * 1024 * 1024;

/// One dense semantic weight that will be replaced by its packed representation.
///
/// Scale and affine-bias names are derived from the required `.weight` suffix,
/// so a target cannot describe mismatched companion tensors.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct BoundedQuantizationTarget {
    weight_name: String,
    scales_name: String,
    biases_name: String,
    source: DerivedWeightRecipe,
    affine_companion_dtype: RecipeDtype,
}

impl BoundedQuantizationTarget {
    /// Quantizes a checkpoint tensor in place under the same logical name.
    pub fn direct(weight_name: impl Into<String>) -> Result<Self, Error> {
        let weight_name = weight_name.into();
        Self::from_recipe(
            weight_name.clone(),
            DerivedWeightRecipe::source(weight_name, TensorSelection::Full),
        )
    }

    /// Quantizes a semantic recipe under its packed runtime weight name.
    ///
    /// Distributed planners can use a bounded selection recipe here to encode
    /// rank-local TP or EP ownership before the conversion runs. Conventional
    /// module weights use `.weight`, `.scales`, and `.biases`; packed matrix
    /// banks use the same runtime weight name plus `_scales` and `_biases`.
    /// Deriving the complete output triplet here prevents a target from
    /// describing an incoherent mixture of the two layouts.
    pub fn from_recipe(
        weight_name: impl Into<String>,
        source: DerivedWeightRecipe,
    ) -> Result<Self, Error> {
        let weight_name = weight_name.into();
        let (scales_name, biases_name) = companion_names(&weight_name)?;
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
    pub(crate) fn affine_companion_bytes(&self) -> u64 {
        match self.affine_companion_dtype {
            RecipeDtype::F16 | RecipeDtype::BF16 => 2,
            RecipeDtype::F32 => 4,
            _ => unreachable!("validated bounded affine companion dtype"),
        }
    }

    /// Returns the packed scale tensor name derived from this target.
    pub fn scales_name(&self) -> String {
        self.scales_name.clone()
    }

    /// Returns the packed affine-bias tensor name derived from this target.
    pub fn biases_name(&self) -> String {
        self.biases_name.clone()
    }
}

/// A validated collection of out-of-core weight transformations.
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

/// Deterministic telemetry from one bounded transformation pass.
#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct BoundedQuantizationReport {
    /// Planner ceiling for simultaneously live conversion data. This is the
    /// larger of the final packed output and the smallest legal semantic row
    /// tile when a synthetic fixture is smaller than one quantization row.
    pub admitted_working_set_bytes: u64,
    /// Number of dense semantic matrices transformed.
    pub transformed_weights: usize,
    /// Number of independently evaluated source row tiles.
    pub source_tiles: usize,
    /// Largest number of submitted tile completions retained simultaneously.
    pub peak_in_flight_tiles: usize,
    /// Total logical dense bytes selected from the source store.
    pub source_bytes_read: u64,
    /// Total packed weight, scale, and bias bytes written to disk.
    pub output_bytes: u64,
    /// Largest conservative conversion working set admitted for one tile.
    pub peak_planned_working_set_bytes: u64,
    /// Largest dense recipe output tile.
    pub largest_source_tile_bytes: u64,
    /// Largest packed weight, scale, and bias tile written together.
    pub largest_output_tile_bytes: u64,
}

/// A source checkpoint overlaid with disk-backed, load-time-quantized weights.
///
/// The temporary packed store lives as long as this value. Runtime acquisitions
/// of transformed keys use ordinary bounded SafeTensors leases; all other keys
/// delegate to the original store.
pub struct BoundedQuantizedWeightStore {
    source: Arc<dyn WeightStore + Send + Sync>,
    transformed: SafetensorsWeightStore,
    transformed_keys: BTreeSet<String>,
    materialized_source_keys: BTreeSet<String>,
    report: BoundedQuantizationReport,
    _directory: TempDir,
}

impl std::fmt::Debug for BoundedQuantizedWeightStore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("BoundedQuantizedWeightStore")
            .field("backend", &self.source.backend())
            .field("transformed_keys", &self.transformed_keys)
            .field("report", &self.report)
            .finish_non_exhaustive()
    }
}

impl BoundedQuantizedWeightStore {
    /// Executes `plan` without materializing a complete source or destination matrix.
    ///
    /// Conversion runs on the supplied stream's device, allowing model loads
    /// to use the accelerator quantizer while retaining CPU fallback. Source
    /// and packed tile storage remain covered by the admitted working set.
    ///
    /// A second same-device stream is used only when two minimum tiles fit the
    /// admitted bound.
    pub fn create(
        source: Arc<dyn WeightStore + Send + Sync>,
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

        preflight_source_collisions(source.as_ref(), &plan)?;
        let materialized_source_keys = plan
            .targets
            .iter()
            .flat_map(|target| target.source.source_keys())
            .map(ToString::to_string)
            .collect();
        let directory = tempfile::Builder::new()
            .prefix("safemlx-bounded-quantization-")
            .tempdir()?;
        let mut index = BTreeMap::<String, String>::new();
        let mut report = BoundedQuantizationReport {
            admitted_working_set_bytes: plan.max_working_set_bytes,
            ..BoundedQuantizationReport::default()
        };
        let mut output_shards = Vec::with_capacity(plan.targets.len());
        let mut allocator_cache = BoundedAllocatorCache::new(plan.max_working_set_bytes);
        let mut pending_tiles = VecDeque::with_capacity(BOUNDED_QUANTIZATION_TILE_BUFFERS);
        allocator_cache.begin()?;

        for (ordinal, target) in plan.targets.iter().enumerate() {
            let shard_name = format!("quantized-{ordinal:05}.safetensors");
            transform_target(
                source.as_ref(),
                target,
                &plan,
                &tile_streams,
                &directory.path().join(&shard_name),
                &mut output_shards,
                &mut pending_tiles,
                &mut allocator_cache,
                &mut report,
            )?;
            for name in output_names_for(target, plan.quantization)? {
                index.insert(name, shard_name.clone());
            }
        }
        while !pending_tiles.is_empty() {
            write_oldest_tile(&mut pending_tiles, &mut output_shards, &mut allocator_cache)?;
        }
        debug_assert!(output_shards.iter().all(|shard| shard.file.is_none()));
        allocator_cache.finish()?;

        write_index(directory.path(), &index, report.output_bytes)?;
        let transformed = SafetensorsWeightStore::open_with_max_mapped_shards(
            directory.path(),
            index.len().max(1),
        )?;
        let transformed_keys = index.into_keys().collect();
        Ok(Self {
            source,
            transformed,
            transformed_keys,
            materialized_source_keys,
            report,
            _directory: directory,
        })
    }

    /// Returns conversion telemetry captured before runtime materialization.
    pub const fn report(&self) -> &BoundedQuantizationReport {
        &self.report
    }

    /// Returns whether `key` is supplied by the packed overlay.
    pub fn is_transformed(&self, key: &str) -> bool {
        self.transformed_keys.contains(key)
    }
}

impl WeightStore for BoundedQuantizedWeightStore {
    fn backend(&self) -> WeightStoreBackend {
        self.source.backend()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn keys(&self) -> Vec<String> {
        self.source
            .keys()
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

    fn is_authoritative_materialized_key(&self, key: &str) -> bool {
        self.is_transformed(key)
    }

    fn metadata(&self, key: &str) -> Result<WeightMetadata, WeightStoreError> {
        if self.is_transformed(key) {
            self.transformed.metadata(key)
        } else {
            self.source.metadata(key)
        }
    }

    fn acquire_with_policy(
        &self,
        key: &str,
        selection: TensorSelection,
        policy: WeightReadPolicy,
    ) -> Result<WeightLease, WeightStoreError> {
        if self.is_transformed(key) {
            self.transformed.acquire_with_policy(key, selection, policy)
        } else {
            self.source.acquire_with_policy(key, selection, policy)
        }
    }

    fn diagnostics(&self) -> Result<WeightStoreDiagnostics, WeightStoreError> {
        let source = self.source.diagnostics()?;
        let transformed = self.transformed.diagnostics()?;
        let mut touched = source.touched_shard_paths;
        touched.extend(transformed.touched_shard_paths);
        touched.sort();
        touched.dedup();
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
    data_offset: u64,
}

struct OutputShard {
    file: Option<File>,
    payload_offset: u64,
    layouts: Vec<OutputLayout>,
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
        if self.sealed && self.pending_tiles == 0 {
            // These shards are process-local scratch data consumed immediately
            // below. Per-shard durability would serialize unnecessary disk syncs.
            self.file
                .take()
                .expect("an incomplete output shard retains its file")
                .flush()?;
        }
        Ok(())
    }
}

fn transform_target(
    source: &dyn WeightStore,
    target: &BoundedQuantizationTarget,
    plan: &BoundedQuantizationPlan,
    tile_streams: &[Stream; BOUNDED_QUANTIZATION_TILE_BUFFERS],
    path: &Path,
    output_shards: &mut Vec<OutputShard>,
    pending_tiles: &mut VecDeque<SubmittedQuantizationTile>,
    allocator_cache: &mut BoundedAllocatorCache,
    report: &mut BoundedQuantizationReport,
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
    let payload_offset = create_shard(path, &layouts)?;
    let output_bytes = layouts.iter().try_fold(0u64, |total, layout| {
        total
            .checked_add(layout.byte_len)
            .ok_or_else(|| quantization_error("quantized output telemetry overflow"))
    })?;
    let output_shard = output_shards.len();
    output_shards.push(OutputShard {
        file: Some(OpenOptions::new().write(true).open(path)?),
        payload_offset,
        layouts,
        pending_tiles: 0,
        sealed: false,
    });
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
            let tile_metadata = tile_recipe.infer(source)?;
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
            allocator_cache
                .prepare_submission(queued_working_set_bytes(pending_tiles)?, tile_peak)?;

            let tile_stream = &tile_streams[report.source_tiles % tile_buffers];
            let pending = tile_recipe.prepare_borrowed_materialization(source, tile_stream)?;
            let (dense, source_mappings) = pending.into_parts();
            let quantized = quantize_tensor(&dense, plan.quantization, tile_stream)?;
            let companion_dtype = match target.affine_companion_dtype {
                RecipeDtype::F16 => Dtype::Float16,
                RecipeDtype::BF16 => Dtype::Bfloat16,
                RecipeDtype::F32 => Dtype::Float32,
                _ => unreachable!("validated bounded companion dtype"),
            };
            let scales = if matches!(plan.quantization, WeightQuantization::MxFp4) {
                quantized.scales
            } else {
                quantized.scales.as_dtype(companion_dtype, tile_stream)?
            };
            let mut outputs = vec![quantized.weight, scales];
            if let Some(biases) = quantized.biases {
                outputs.push(biases.as_dtype(companion_dtype, tile_stream)?);
            }
            let completion = async_eval_with_event(outputs.iter())?;
            let output_start = matrix
                .checked_mul(rows)
                .and_then(|offset| offset.checked_add(start))
                .ok_or_else(|| quantization_error("quantized output row offset overflow"))?;
            let submitted = SubmittedQuantizationTile {
                outputs,
                _dense: dense,
                source_mappings,
                completion: Some(completion),
                output_start,
                rows: tile_rows,
                planned_working_set_bytes: tile_peak,
                output_shard,
            };
            output_shards[output_shard].tile_submitted();
            pending_tiles.push_back(submitted);
            report.peak_in_flight_tiles = report.peak_in_flight_tiles.max(pending_tiles.len());

            report.source_tiles = report.source_tiles.saturating_add(1);
            report.source_bytes_read = report
                .source_bytes_read
                .checked_add(tile_metadata.byte_len())
                .ok_or_else(|| quantization_error("source-read telemetry overflow"))?;
            let queued_working_set = queued_working_set_bytes(pending_tiles)?;
            report.peak_planned_working_set_bytes = report
                .peak_planned_working_set_bytes
                .max(queued_working_set);
            report.largest_source_tile_bytes = report
                .largest_source_tile_bytes
                .max(tile_metadata.byte_len());
            report.largest_output_tile_bytes =
                report.largest_output_tile_bytes.max(tile_output_bytes);

            if pending_tiles.len() == tile_buffers {
                write_oldest_tile(pending_tiles, output_shards, allocator_cache)?;
            }
            start = end;
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
            biases_name,
            affine_dtype,
            prefix,
            scale_columns,
            affine_scalar_bytes,
        )?);
    }

    let mut offset = 0u64;
    for layout in &mut layouts {
        layout.data_offset = offset;
        offset = offset
            .checked_add(layout.byte_len)
            .ok_or_else(|| quantization_error("SafeTensors payload offset overflow"))?;
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
        data_offset: 0,
    })
}

fn create_shard(path: &Path, layouts: &[OutputLayout]) -> Result<u64, Error> {
    let metadata = layouts
        .iter()
        .map(|layout| {
            let start = usize::try_from(layout.data_offset)
                .map_err(|_| quantization_error("SafeTensors offset is not representable"))?;
            let end = usize::try_from(
                layout
                    .data_offset
                    .checked_add(layout.byte_len)
                    .ok_or_else(|| quantization_error("SafeTensors offset overflow"))?,
            )
            .map_err(|_| quantization_error("SafeTensors offset is not representable"))?;
            Ok((
                layout.name.clone(),
                TensorInfo {
                    dtype: layout.dtype,
                    shape: layout.shape.clone(),
                    data_offsets: (start, end),
                },
            ))
        })
        .collect::<Result<BTreeMap<_, _>, Error>>()?;
    let mut header = serde_json::to_vec(&metadata)?;
    let aligned = header
        .len()
        .checked_add(7)
        .map(|length| length & !7)
        .ok_or_else(|| quantization_error("SafeTensors header size overflow"))?;
    header.resize(aligned, b' ');
    let payload_offset = 8u64
        .checked_add(header.len() as u64)
        .ok_or_else(|| quantization_error("SafeTensors payload offset overflow"))?;
    let payload_bytes = layouts.iter().try_fold(0u64, |total, layout| {
        total
            .checked_add(layout.byte_len)
            .ok_or_else(|| quantization_error("SafeTensors payload size overflow"))
    })?;
    let file_len = payload_offset
        .checked_add(payload_bytes)
        .ok_or_else(|| quantization_error("SafeTensors shard size overflow"))?;
    let mut file = File::create(path)?;
    file.write_all(&(header.len() as u64).to_le_bytes())?;
    file.write_all(&header)?;
    file.set_len(file_len)?;
    file.flush()?;
    Ok(payload_offset)
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
        for (layout, output) in shard.layouts.iter().zip(&self.outputs) {
            write_tile(
                shard
                    .file
                    .as_mut()
                    .expect("a pending output tile retains its shard file"),
                shard.payload_offset,
                layout,
                self.output_start,
                self.rows,
                output,
            )?;
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
    file: &mut File,
    payload_offset: u64,
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
    let offset = payload_offset
        .checked_add(layout.data_offset)
        .and_then(|offset| offset.checked_add(layout.row_bytes.checked_mul(start_row as u64)?))
        .ok_or_else(|| quantization_error("quantized tile file offset overflow"))?;
    file.seek(SeekFrom::Start(offset))?;
    let evaluated = output.evaluated()?;
    match output.dtype() {
        Dtype::Uint32 => write_native_slice(file, evaluated.as_slice::<u32>())?,
        Dtype::Float16 => write_native_slice(file, evaluated.as_slice::<half::f16>())?,
        Dtype::Bfloat16 => write_native_slice(file, evaluated.as_slice::<half::bf16>())?,
        Dtype::Float32 => write_native_slice(file, evaluated.as_slice::<f32>())?,
        Dtype::Uint8 => file.write_all(evaluated.as_slice::<u8>())?,
        dtype => {
            return Err(quantization_error(format!(
                "quantized output {:?} has unsupported dtype {dtype:?}",
                layout.name
            )))
        }
    }
    Ok(())
}

fn write_native_slice<T>(file: &mut File, values: &[T]) -> Result<(), std::io::Error> {
    let bytes = unsafe {
        // SAFETY: `values` is a valid initialized slice, and its byte view has
        // the identical lifetime. The public constructor rejects big-endian
        // hosts because SafeTensors payloads are little-endian.
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    };
    file.write_all(bytes)
}

#[derive(Serialize)]
struct SafetensorsIndex<'a> {
    metadata: IndexMetadata,
    weight_map: &'a BTreeMap<String, String>,
}

#[derive(Serialize)]
struct IndexMetadata {
    total_size: u64,
}

fn write_index(
    directory: &Path,
    weight_map: &BTreeMap<String, String>,
    total_size: u64,
) -> Result<(), Error> {
    let index = SafetensorsIndex {
        metadata: IndexMetadata { total_size },
        weight_map,
    };
    let file = File::create(directory.join("model.safetensors.index.json"))?;
    serde_json::to_writer(file, &index)?;
    Ok(())
}

fn preflight_source_collisions(
    source: &dyn WeightStore,
    plan: &BoundedQuantizationPlan,
) -> Result<(), Error> {
    let source_keys = source.keys().into_iter().collect::<BTreeSet<_>>();
    for target in &plan.targets {
        if source_keys.contains(&target.scales_name) || source_keys.contains(&target.biases_name) {
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
        names.push(target.biases_name.clone());
    }
    Ok(names)
}

fn companion_names(weight_name: &str) -> Result<(String, String), Error> {
    if weight_name.trim().is_empty() {
        return Err(quantization_error(
            "bounded quantization target name must not be empty",
        ));
    }
    if weight_name == ".weight" {
        return Err(quantization_error(
            "bounded quantization target prefix must not be empty",
        ));
    }
    Ok(weight_name.strip_suffix(".weight").map_or_else(
        || {
            (
                format!("{weight_name}_scales"),
                format!("{weight_name}_biases"),
            )
        },
        |prefix| (format!("{prefix}.scales"), format!("{prefix}.biases")),
    ))
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

    use safemlx::{ops::GgufCheckpoint, Device, DeviceType, ExecutionContext};
    use safetensors::tensor::{serialize_to_file, TensorView};

    use super::*;
    use crate::runtime::{
        checkpoint::{quantization::AffineQuantization, store::GgufWeightStore},
        residency::{
            manager::{
                host_capacity_upper_bound_for_bindings, OffloadUnit, ResidencyManager,
                WeightBinding,
            },
            policy::{
                MemoryTier, OffloadConfig, OffloadPlan, OffloadUnitId, OffloadUnitSpec,
                ResidencyPolicy,
            },
        },
    };
    use crate::test_utils::SyntheticGguf;

    fn cpu_context() -> ExecutionContext {
        ExecutionContext::new(Device::new(DeviceType::Cpu, 0))
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

    fn materialize(store: &dyn WeightStore, name: &str, stream: &Stream) -> Array {
        store
            .acquire_with_policy(
                name,
                TensorSelection::Full,
                WeightReadPolicy::RequireBounded,
            )
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
        let target = BoundedQuantizationTarget::direct("model.proj.weight").unwrap();
        // One f32 source row (256 bytes) plus one packed affine output row
        // (32-byte weight, 4-byte scale, and 4-byte bias). The complete
        // quantized matrix is 320 bytes, so that runtime-sized budget is also
        // sufficient to convert the 2,048-byte dense source.
        let plan = BoundedQuantizationPlan::new(quantization, 320, [target]).unwrap();
        let transformed =
            BoundedQuantizedWeightStore::create(source.clone(), plan, context.stream()).unwrap();

        assert_eq!(
            transformed.report(),
            &BoundedQuantizationReport {
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
                .metadata("model.proj.weight")
                .unwrap()
                .logical_byte_len,
            256
        );
        assert_eq!(
            transformed
                .metadata("model.proj.scales")
                .unwrap()
                .logical_byte_len,
            32
        );
        assert_eq!(
            transformed
                .metadata("model.proj.biases")
                .unwrap()
                .logical_byte_len,
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
            [BoundedQuantizationTarget::direct("model.proj.weight").unwrap()],
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
                BoundedQuantizationTarget::direct("model.first.weight").unwrap(),
                BoundedQuantizationTarget::direct("model.second.weight").unwrap(),
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
            assert!(!transformed.keys().contains(&format!("{name}.biases")));
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn affine_gpu_conversion_matches_the_canonical_gpu_quantizer() {
        let (_directory, source, values) = direct_fixture();
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let quantization = AffineQuantization::default();
        let plan = BoundedQuantizationPlan::new(
            quantization,
            640,
            [BoundedQuantizationTarget::direct("model.proj.weight").unwrap()],
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
            [BoundedQuantizationTarget::direct("model.proj.weight").unwrap()],
        )
        .unwrap();
        let error = BoundedQuantizedWeightStore::create(source.clone(), plan, context.stream())
            .unwrap_err();
        assert!(error.to_string().contains(
            "requires at least 296 working-set bytes for one row, but the plan permits 295"
        ));
        let diagnostics = source.diagnostics().unwrap();
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
        let target = BoundedQuantizationTarget::from_recipe("local.expert.weight", recipe).unwrap();
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
        let target = BoundedQuantizationTarget::from_recipe("rank.expert.weight", recipe).unwrap();
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
            [BoundedQuantizationTarget::direct("model.experts.weight").unwrap()],
        )
        .unwrap();
        let transformed =
            BoundedQuantizedWeightStore::create(source, plan, context.stream()).unwrap();

        assert_eq!(transformed.report().source_tiles, 8);
        assert_eq!(transformed.report().source_bytes_read, 2_048);
        assert_eq!(transformed.report().output_bytes, 320);
        assert_eq!(
            transformed.metadata("model.experts.weight").unwrap().shape,
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
    fn complete_rank_three_bank_uses_atomic_runtime_companion_names() {
        let values = matrix_values(2, 4, 64);
        let dense = Array::from_slice(&values, &[2, 4, 64]);
        let fixture = SyntheticGguf::dense(
            &HashMap::from([("checkpoint.experts.weight".to_string(), dense)]),
            &HashMap::new(),
        );
        let source = Arc::new(
            GgufWeightStore::new(GgufCheckpoint::open(fixture.path()).unwrap(), |name| {
                name.to_string()
            })
            .unwrap(),
        );
        let context = cpu_context();
        let target = BoundedQuantizationTarget::from_recipe(
            "model.experts.down_proj",
            DerivedWeightRecipe::source("checkpoint.experts.weight", TensorSelection::Full),
        )
        .unwrap();
        assert_eq!(target.scales_name(), "model.experts.down_proj_scales");
        assert_eq!(target.biases_name(), "model.experts.down_proj_biases");
        let plan =
            BoundedQuantizationPlan::new(AffineQuantization::default(), 296, [target]).unwrap();
        let transformed =
            BoundedQuantizedWeightStore::create(source.clone(), plan, context.stream()).unwrap();

        assert_eq!(
            transformed
                .metadata("model.experts.down_proj")
                .unwrap()
                .shape,
            vec![2, 4, 8]
        );
        assert_eq!(
            transformed
                .metadata("model.experts.down_proj_scales")
                .unwrap()
                .shape,
            vec![2, 4, 1]
        );
        assert_eq!(
            transformed
                .metadata("model.experts.down_proj_biases")
                .unwrap()
                .shape,
            vec![2, 4, 1]
        );
        let diagnostics = source.diagnostics().unwrap();
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
            GgufWeightStore::new(GgufCheckpoint::open(fixture.path()).unwrap(), |name| {
                name.to_string()
            })
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
        let target = BoundedQuantizationTarget::from_recipe("rank.expert.weight", recipe).unwrap();
        let plan =
            BoundedQuantizationPlan::new(AffineQuantization::default(), 296, [target]).unwrap();
        let transformed =
            BoundedQuantizedWeightStore::create(source.clone(), plan, context.stream()).unwrap();

        assert_eq!(transformed.report().source_tiles, 2);
        assert_eq!(transformed.report().source_bytes_read, 512);
        assert_eq!(transformed.report().output_bytes, 80);
        assert_eq!(transformed.report().peak_planned_working_set_bytes, 296);
        let diagnostics = source.diagnostics().unwrap();
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
            GgufWeightStore::new(GgufCheckpoint::open(fixture.path()).unwrap(), |name| {
                name.to_string()
            })
            .unwrap(),
        );
        let context = cpu_context();
        let plan = BoundedQuantizationPlan::new(
            AffineQuantization::default(),
            296,
            [BoundedQuantizationTarget::direct("model.experts.weight").unwrap()],
        )
        .unwrap();
        let transformed =
            BoundedQuantizedWeightStore::create(source.clone(), plan, context.stream()).unwrap();

        assert_eq!(transformed.report().source_tiles, 8);
        assert_eq!(transformed.report().source_bytes_read, 2_048);
        assert_eq!(transformed.report().output_bytes, 320);
        let diagnostics = source.diagnostics().unwrap();
        assert_eq!(diagnostics.physical_reads, 8);
        assert_eq!(diagnostics.physical_read_bytes, 2_048);
        assert_eq!(
            transformed.metadata("model.experts.weight").unwrap().shape,
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
            [BoundedQuantizationTarget::direct("model.proj.weight").unwrap()],
        )
        .unwrap();
        let transformed =
            BoundedQuantizedWeightStore::create(source, plan, context.stream()).unwrap();
        assert_eq!(transformed.report().source_tiles, 8);
        assert_eq!(transformed.report().output_bytes, 272);
        assert_eq!(
            transformed
                .metadata("model.proj.scales")
                .unwrap()
                .stored_dtype,
            crate::runtime::checkpoint::store::StoredDtype::U8
        );
        assert!(!transformed.keys().contains(&"model.proj.biases".into()));

        assert_mxfp4_outputs_match_reference(&transformed, &values, &context);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn mxfp4_gpu_conversion_matches_the_canonical_gpu_quantizer() {
        let (_directory, source, values) = direct_fixture();
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let plan = BoundedQuantizationPlan::new(
            WeightQuantization::MxFp4,
            580,
            [BoundedQuantizationTarget::direct("model.proj.weight").unwrap()],
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
            GgufWeightStore::new(GgufCheckpoint::open(fixture.path()).unwrap(), |name| {
                name.to_string()
            })
            .unwrap(),
        );
        let context = cpu_context();
        let plan = BoundedQuantizationPlan::new(
            AffineQuantization::default(),
            320,
            [BoundedQuantizationTarget::direct("model.proj.weight").unwrap()],
        )
        .unwrap();
        let transformed =
            BoundedQuantizedWeightStore::create(source, plan, context.stream()).unwrap();
        assert_eq!(transformed.backend(), WeightStoreBackend::Gguf);
        assert_eq!(transformed.report().source_tiles, 8);
        assert_eq!(transformed.report().source_bytes_read, 2_048);
        let diagnostics = transformed.diagnostics().unwrap();
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
            [BoundedQuantizationTarget::direct("model.proj.weight").unwrap()],
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
            .map(|name| lease.host_buffer(name).unwrap().nbytes().unwrap())
            .sum::<usize>();
        assert_eq!(resident_bytes, 320);
    }
}
