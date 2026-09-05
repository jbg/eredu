use super::*;
use super::{
    layout::{
        allocate_output_buffer, output_layouts, BoundedAllocatorCache, OutputLayout, OutputShard,
    },
    preflight::{
        checked_product, output_names_for, preflight_source_collisions, quantization_error,
        report_source_bytes,
    },
};

/// A source checkpoint overlaid with memory-backed, load-time-quantized weights.
///
/// The packed store lives as long as this value. Runtime acquisitions of
/// transformed keys use bounded in-memory leases; all other keys delegate to
/// the original store.
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
            cache_hits: source.cache_hits.saturating_add(transformed.cache_hits),
            cache_misses: source.cache_misses.saturating_add(transformed.cache_misses),
            evictions: source.evictions.saturating_add(transformed.evictions),
            currently_cached_shards: source
                .currently_cached_shards
                .saturating_add(transformed.currently_cached_shards),
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
    for matrix in 0..leading {
        target
            .source
            .select_bounded_matrix_rows(source, matrix, 0, 1)?
            .preflight_bounded(source)?;
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
                    .ok_or_else(|| quantization_error("leading batch row count overflow"))?;
                let candidate_elements = candidate_rows
                    .checked_mul(columns)
                    .ok_or_else(|| quantization_error("leading batch element count overflow"))?;
                let candidate_output_bytes = output_row_bytes
                    .checked_mul(candidate_rows as u64)
                    .ok_or_else(|| quantization_error("leading batch output size overflow"))?;
                let candidate_peak = candidate
                    .peak_materialization_bytes(source)?
                    .checked_add(candidate_output_bytes)
                    .ok_or_else(|| quantization_error("leading batch working-set overflow"))?;
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
                .ok_or_else(|| quantization_error("leading batch row count overflow"))?;
            let batch_output_bytes = output_row_bytes
                .checked_mul(batch_rows as u64)
                .ok_or_else(|| quantization_error("leading batch output size overflow"))?;
            let batch_peak = recipe
                .peak_materialization_bytes(source)?
                .checked_add(batch_output_bytes)
                .ok_or_else(|| quantization_error("leading batch working-set overflow"))?;
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
    let (dense, source_leases) = pending.into_parts();
    let outputs = quantize_tile_outputs(&dense, quantization, target, tile_stream)?;
    let completion = async_eval_with_event(outputs.iter())?;
    output_shards[output_shard].tile_submitted();
    pending_tiles.push_back(SubmittedQuantizationTile {
        outputs,
        _dense: dense,
        source_leases,
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

struct SubmittedQuantizationTile {
    outputs: Vec<Array>,
    _dense: Array,
    source_leases: Vec<PendingWeightMaterialization>,
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
        for mapping in self.source_leases.drain(..) {
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
    match output.dtype() {
        Dtype::Uint32 | Dtype::Float16 | Dtype::Bfloat16 | Dtype::Float32 | Dtype::Uint8 => {}
        dtype => {
            return Err(quantization_error(format!(
                "quantized output {:?} has unsupported dtype {dtype:?}",
                layout.name
            )))
        }
    }
    destination.copy_from_slice(&evaluated.to_native_bytes());
    Ok(())
}
