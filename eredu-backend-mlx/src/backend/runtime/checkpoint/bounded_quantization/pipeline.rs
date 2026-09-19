use super::*;
use super::{
    layout::{BoundedAllocatorCache, OutputLayout, OutputShard},
    preflight::{quantization_error, report_source_bytes},
};

/// A source checkpoint overlaid with memory-backed, load-time-quantized weights.
///
/// The packed store lives as long as this value. Runtime acquisitions of
/// transformed keys use bounded in-memory leases; all other keys delegate to
/// the original store.
pub struct BoundedQuantizedWeightStore {
    source: eredu_checkpoint::store::RetainedCheckpointSource,
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
        source: impl Into<eredu_checkpoint::store::RetainedCheckpointSource>,
        plan: BoundedQuantizationPlan,
        conversion_stream: &Stream,
    ) -> Result<Self, Error> {
        super::preparation::ColdQuantization::prepare(source.into(), plan)?
            .allocate_ordinary()?
            .materialize(conversion_stream)
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

impl super::preparation::PreparedQuantization {
    pub(crate) fn materialize(
        self,
        conversion_stream: &Stream,
    ) -> Result<BoundedQuantizedWeightStore, Error> {
        self.materialize_with_plan(conversion_stream)
            .map(|(store, _)| store)
    }

    /// Retain the actual plan for later adoption by typed model construction.
    pub(crate) fn materialize_handoff(
        self,
        conversion_stream: &Stream,
    ) -> Result<ConvertedQuantization, Error> {
        let source = self.source.clone();
        let (store, plan) = self.materialize_with_plan(conversion_stream)?;
        Ok(ConvertedQuantization::new(source, plan, store))
    }

    fn materialize_with_plan(
        self,
        conversion_stream: &Stream,
    ) -> Result<(BoundedQuantizedWeightStore, BoundedQuantizationPlan), Error> {
        let Self {
            source,
            plan,
            mut output_shards,
            transformed_keys,
            materialized_source_keys,
            materialized_source_shards,
        } = self;
        let device = conversion_stream.get_device()?;
        let tile_streams = [conversion_stream.clone(), Stream::new_with_device(&device)];
        let tile_contexts = tile_streams
            .each_ref()
            .map(|stream| MlxParameterMaterializationContext::new(stream, stream));

        let mut report = WeightMaterializationReport {
            admitted_working_set_bytes: plan.max_working_set_bytes,
            ..WeightMaterializationReport::default()
        };
        let mut allocator_cache = BoundedAllocatorCache::new(plan.max_working_set_bytes);
        let mut pending_tiles = VecDeque::with_capacity(BOUNDED_QUANTIZATION_TILE_BUFFERS);
        allocator_cache.begin()?;

        for (index, target) in plan.targets.iter().enumerate() {
            transform_target(
                source.as_ref(),
                target,
                &plan,
                index,
                &tile_contexts,
                &mut output_shards,
                &mut pending_tiles,
                &mut allocator_cache,
                &mut report,
            )?;
        }
        while !pending_tiles.is_empty() {
            write_oldest_tile(&mut pending_tiles, &mut output_shards, &mut allocator_cache)?;
        }
        debug_assert!(output_shards
            .iter()
            .all(|shard| shard.sealed && shard.pending_tiles == 0));
        allocator_cache.finish()?;

        let transformed = MemoryWeightStore::from_buffers(
            output_shards.into_iter().flat_map(|shard| shard.buffers),
        )
        .map_err(|cause| Error::Other(Box::new(cause)))?;
        Ok((
            BoundedQuantizedWeightStore {
                source,
                transformed,
                transformed_keys,
                materialized_source_keys,
                materialized_source_shards,
                report,
            },
            plan,
        ))
    }
}

impl CheckpointSource for BoundedQuantizedWeightStore {
    fn prepare_encoded_read(
        &self,
        keys: &[String],
    ) -> Result<Option<eredu_checkpoint::store::EncodedReadBatch>, StoreError> {
        // A batch has one exact retained source, matching the existing composed
        // source contract. Per-binding materialization reaches one of these.
        if keys.iter().all(|key| self.is_transformed(key)) {
            self.transformed.prepare_encoded_read(keys)
        } else if keys.iter().all(|key| !self.is_transformed(key)) {
            self.source.prepare_encoded_read(keys)
        } else {
            Ok(None)
        }
    }

    fn source_lease_controls<'a>(
        &'a self,
        key: &'a str,
    ) -> Result<
        eredu_checkpoint::store::SourceLeaseControls<'a>,
        eredu_checkpoint::store::LeaseControlBorrowError<'a>,
    > {
        if self.is_transformed(key) {
            self.transformed.source_lease_controls(key)
        } else {
            self.source.source_lease_controls(key)
        }
    }

    fn source_metadata_borrowed(
        &self,
        key: &str,
    ) -> eredu_checkpoint::store::SourceMetadataLoan<'_> {
        if self.is_transformed(key) {
            self.transformed.source_metadata_borrowed(key)
        } else {
            self.source.source_metadata_borrowed(key)
        }
    }
    fn source_key_authority_borrowed(
        &self,
        key: &str,
    ) -> Result<
        eredu_checkpoint::store::SourceKeyAuthority,
        eredu_checkpoint::store::SourceMetadataBorrowError<'_>,
    > {
        use eredu_checkpoint::store::SourceKeyAuthority;
        Ok(if self.is_transformed(key) {
            SourceKeyAuthority::Materialized
        } else {
            SourceKeyAuthority::Ordinary
        })
    }

    fn source_storage_slot_bound(&self) -> Result<Option<usize>, StoreError> {
        match (
            self.source.source_storage_slot_bound()?,
            self.transformed.source_storage_slot_bound()?,
        ) {
            (Some(source), Some(transformed)) => source
                .checked_add(transformed)
                .map(Some)
                .ok_or_else(|| StoreError::Overflow {
                    context: "transformed source storage slots".into(),
                }),
            _ => Ok(None),
        }
    }

    fn visit_source_storage(
        &self,
        visitor: &mut dyn FnMut(eredu_checkpoint::store::SourceStorageRef<'_>),
    ) -> Result<bool, StoreError> {
        // Hidden original owners remain live beneath the overlay. Visit both
        // physical branches even when one reports incomplete coverage.
        let source = self.source.visit_source_storage(visitor)?;
        let transformed = self.transformed.visit_source_storage(visitor)?;
        Ok(source && transformed)
    }

    fn source_storage(&self) -> Result<Option<eredu_checkpoint::store::SourceStorage>, StoreError> {
        eredu_checkpoint::store::SourceStorage::collect([
            self.source.as_ref(),
            &self.transformed as &dyn CheckpointSource,
        ])
    }

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

    fn source_provenance(
        &self,
        key: &str,
    ) -> Result<eredu_checkpoint::store::TensorSourceProvenance, StoreError> {
        if self.is_transformed(key) {
            self.transformed.source_provenance(key)
        } else {
            self.source.source_provenance(key)
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
    output_shard: usize,
    tile_contexts: &[MlxParameterMaterializationContext; BOUNDED_QUANTIZATION_TILE_BUFFERS],
    output_shards: &mut Vec<OutputShard>,
    pending_tiles: &mut VecDeque<SubmittedQuantizationTile>,
    allocator_cache: &mut BoundedAllocatorCache,
    report: &mut WeightMaterializationReport,
) -> Result<(), Error> {
    let super::preparation::ConversionGeometry {
        leading,
        rows,
        columns,
        output_row_bytes,
        output_bytes,
        one_row_source_bytes,
        source_bytes,
        tile_buffers,
        tile_budget,
        complete_rows,
        complete_peak,
        complete_admissible,
        leading_batch_admissible,
    } = output_shards[output_shard].geometry;
    if tile_buffers == 1 {
        while !pending_tiles.is_empty() {
            write_oldest_tile(pending_tiles, output_shards, allocator_cache)?;
        }
    }
    if complete_admissible {
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
        source_bytes,
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
    // Verification acquires real payload leases. Only the selected tile may
    // read them, after its peak fits alongside the still-pending submissions.
    recipe.preflight_bounded(source)?;
    let metadata = recipe.infer(source)?;
    let tile_context = &tile_contexts[report.source_tiles % tile_buffers];
    let tile_stream = tile_context.source_stream();
    let pending = recipe.prepare_borrowed_materialization(source, tile_context)?;
    let (dense, source_leases) = pending.into_parts();
    let prepared = WeightMaterialization::prepare_retained(vec![dense], source_leases)?;
    let outputs = quantize_tile_outputs(&prepared.inputs()[0], quantization, target, tile_stream)?;
    let completion = prepared.submit_outputs(outputs)?;
    output_shards[output_shard].tile_submitted();
    pending_tiles.push_back(SubmittedQuantizationTile {
        completion,
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
    completion: WeightMaterialization,
    output_start: usize,
    rows: usize,
    planned_working_set_bytes: u64,
    output_shard: usize,
}

impl SubmittedQuantizationTile {
    fn write(self, shard: &mut OutputShard) -> Result<(), Error> {
        self.completion.wait()?;
        for ((layout, output), buffer) in shard
            .layouts
            .iter()
            .zip(self.completion.outputs())
            .zip(&mut shard.buffers)
        {
            write_tile(
                buffer.bytes_mut(),
                layout,
                self.output_start,
                self.rows,
                output,
            )?;
        }
        Ok(())
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
            )));
        }
    }
    evaluated
        .try_copy_native_bytes_into(destination)
        .map_err(|cause| Error::Other(Box::new(cause)))?;
    Ok(())
}

#[cfg(test)]
mod provenance_tests {
    use super::*;
    #[test]
    fn borrowed_overlay_metadata_keeps_real_materialized_authority_through_resolved_view() {
        use eredu_checkpoint::{
            schema::{
                CatalogPolicy, SafetensorsCheckpointPlan, SafetensorsTensorConstraint,
                StoredDtypeConstraint,
            },
            store::{ResolvedCheckpointSource, SourceKeyAuthority, SourceMetadataBorrowError},
            validation::resolve_safetensors_plan,
            StoredDtype,
        };
        let source: eredu_checkpoint::store::RetainedCheckpointSource = (Arc::new(
            MemoryWeightStore::from_safetensors([
                (
                    "weight".into(),
                    SafeDtype::F32,
                    vec![1],
                    0.75f32.to_le_bytes().to_vec(),
                ),
                (
                    "unchanged".into(),
                    SafeDtype::F32,
                    vec![1],
                    (-2.5f32).to_le_bytes().to_vec(),
                ),
            ])
            .unwrap(),
        ))
        .into();
        let plan = SafetensorsCheckpointPlan::new(
            "selected",
            vec![SafetensorsTensorConstraint::required(
                "unchanged",
                vec![1],
                StoredDtypeConstraint::Exact(StoredDtype::F32),
            )],
            Vec::new(),
            CatalogPolicy::non_strict(),
        )
        .unwrap();
        let contract = resolve_safetensors_plan(source.as_ref(), &plan).unwrap();
        let overlay = Arc::new(BoundedQuantizedWeightStore {
            source,
            transformed: MemoryWeightStore::from_safetensors([(
                "packed".into(),
                SafeDtype::U32,
                vec![1],
                0x76543210u32.to_le_bytes().to_vec(),
            )])
            .unwrap(),
            transformed_keys: BTreeSet::from(["packed".into()]),
            materialized_source_keys: BTreeSet::new(),
            materialized_source_shards: BTreeSet::new(),
            report: WeightMaterializationReport::default(),
        });
        let resolved = ResolvedCheckpointSource::new(overlay.clone(), contract);
        // This exact transformed memory source must qualify the same original
        // manager reader; no file descriptor or ordinary lease is substituted.
        let packed_read = DerivedWeightRecipe::source("packed", TensorSelection::Full)
            .prepare_encoded_read(&resolved)
            .unwrap()
            .unwrap();
        let mut packed_bytes = [0; 4];
        eredu_checkpoint::recipe::EncodedRecipeRead::read_many_borrowed_into(
            std::iter::once(&packed_read),
            &mut [&mut packed_bytes],
        )
        .unwrap();
        assert_eq!(packed_bytes, 0x76543210u32.to_le_bytes());
        let unchanged_read = DerivedWeightRecipe::source("unchanged", TensorSelection::Full)
            .prepare_encoded_read(&resolved)
            .unwrap()
            .unwrap();
        let mut unchanged_bytes = [0; 4];
        eredu_checkpoint::recipe::EncodedRecipeRead::read_many_borrowed_into(
            std::iter::once(&unchanged_read),
            &mut [&mut unchanged_bytes],
        )
        .unwrap();
        assert_eq!(unchanged_bytes, (-2.5f32).to_le_bytes());
        let packed = overlay
            .transformed
            .source_metadata_borrowed("packed")
            .unwrap();
        assert!(std::ptr::eq(
            packed,
            resolved.source_metadata_borrowed("packed").unwrap()
        ));
        let packed_loan = resolved.source_lease_controls("packed").unwrap();
        assert!(
            packed_loan.same_entry(&overlay.transformed.source_lease_controls("packed").unwrap())
        );
        assert!(!packed_loan.same_entry(&overlay.source.source_lease_controls("weight").unwrap()));
        assert!(resolved
            .source_lease_controls("unchanged")
            .unwrap()
            .same_entry(&overlay.source.source_lease_controls("unchanged").unwrap()));
        assert!(matches!(
            resolved.source_lease_controls("weight"),
            Err(eredu_checkpoint::store::LeaseControlBorrowError::Source(
                SourceMetadataBorrowError::UnauthorizedTensor
            ))
        ));
        assert_eq!(packed.stored_dtype, StoredDtype::U32);
        assert_eq!(resolved.source_metadata("packed").unwrap(), *packed);
        assert_eq!(
            resolved.source_key_authority_borrowed("packed").unwrap(),
            SourceKeyAuthority::Materialized
        );
        let unchanged = overlay
            .source
            .source_metadata_borrowed("unchanged")
            .unwrap();
        assert!(std::ptr::eq(
            unchanged,
            resolved.source_metadata_borrowed("unchanged").unwrap()
        ));
        for key in ["weight", "absent"] {
            assert!(matches!(
                resolved.source_metadata_borrowed(key),
                Err(SourceMetadataBorrowError::UnauthorizedTensor)
            ));
            assert!(matches!(
                resolved.source_metadata(key),
                Err(StoreError::UnauthorizedTensor { .. })
            ));
        }
    }

    struct Renamed(MemoryWeightStore);
    impl CheckpointSource for Renamed {
        fn source_keys(&self) -> Vec<String> {
            self.0.source_keys()
        }
        fn source_metadata(
            &self,
            key: &str,
        ) -> Result<eredu_checkpoint::store::TensorMetadata, StoreError> {
            self.0.source_metadata(key)
        }
        fn acquire_lease(&self, request: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
            self.0.acquire_lease(request)
        }
        fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
            self.0.source_diagnostics()
        }
        fn source_provenance(
            &self,
            key: &str,
        ) -> Result<eredu_checkpoint::store::TensorSourceProvenance, StoreError> {
            let mut provenance = self.0.source_provenance(key)?;
            provenance.physical_tensor = format!("checkpoint.prefix.{key}");
            provenance.output = format!("selected.{key}");
            Ok(provenance)
        }
    }
    #[test]
    fn bounded_overlay_preserves_original_passthrough_and_actual_output_provenance() {
        let source: eredu_checkpoint::store::RetainedCheckpointSource = (Arc::new(Renamed(
            MemoryWeightStore::from_safetensors(["weight", "unchanged"].map(|key| {
                (
                    key.to_owned(),
                    SafeDtype::F32,
                    vec![1],
                    0.75f32.to_le_bytes().to_vec(),
                )
            }))
            .unwrap(),
        )))
        .into();
        let transformed = MemoryWeightStore::from_safetensors([(
            "weight".to_owned(),
            SafeDtype::U32,
            vec![1],
            0x76543210u32.to_le_bytes().to_vec(),
        )])
        .unwrap();
        let expected_output = transformed.source_provenance("weight").unwrap();
        let expected_passthrough = source.source_provenance("unchanged").unwrap();
        let overlay = BoundedQuantizedWeightStore {
            source,
            transformed,
            transformed_keys: BTreeSet::from(["weight".to_owned()]),
            materialized_source_keys: BTreeSet::new(),
            materialized_source_shards: BTreeSet::new(),
            report: WeightMaterializationReport::default(),
        };
        assert_eq!(
            overlay.source_provenance("unchanged").unwrap(),
            expected_passthrough
        );
        assert_eq!(
            overlay.source_provenance("weight").unwrap(),
            expected_output
        );
        assert_eq!(
            overlay.source_provenance("weight").unwrap().source_encoding,
            eredu_checkpoint::SourceTensorEncoding::Safetensors(eredu_checkpoint::StoredDtype::U32)
        );
    }
}
