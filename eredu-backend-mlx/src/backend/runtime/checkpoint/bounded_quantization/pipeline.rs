use super::*;
use super::{
    layout::{BoundedAllocatorCache, OutputLayout, OutputShard},
    preflight::{quantization_error, report_source_bytes},
};
pub(super) mod controls;
mod producer;
use controls::{PipelineAdmissionError, TileWindow};
use producer::{OrdinaryTileProducer, TileCompletion, TileProducer};

/// A completed quantization result and its conversion telemetry.
#[derive(Debug)]
pub struct QuantizedCheckpoint {
    source: eredu_checkpoint::store::MaterializedCheckpointSource,
    report: WeightMaterializationReport,
}
impl QuantizedCheckpoint {
    /// Execute bounded conversion on the supplied stream's device. A second
    /// same-device stream is used when two minimum tiles fit the working set.
    pub fn create(
        source: impl Into<eredu_checkpoint::store::RetainedCheckpointSource>,
        plan: BoundedQuantizationPlan,
        conversion_stream: &Stream,
    ) -> Result<Self, Error> {
        super::preparation::ColdQuantization::prepare(source.into(), plan)?
            .allocate_ordinary(conversion_stream)?
            .materialize(conversion_stream)
    }
    /// The backend-neutral completed checkpoint source.
    pub fn source(&self) -> &eredu_checkpoint::store::MaterializedCheckpointSource {
        &self.source
    }
    /// Conversion telemetry recorded by the shared tile driver.
    pub const fn report(&self) -> &WeightMaterializationReport {
        &self.report
    }
    /// Move the completed checkpoint and telemetry into model preparation.
    pub fn into_parts(
        self,
    ) -> (
        eredu_checkpoint::store::MaterializedCheckpointSource,
        WeightMaterializationReport,
    ) {
        (self.source, self.report)
    }
}

impl super::preparation::PreparedQuantization {
    pub(crate) fn materialize(
        self,
        conversion_stream: &Stream,
    ) -> Result<QuantizedCheckpoint, Error> {
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
        Ok(ConvertedQuantization::new(source, plan, store, None))
    }

    fn materialize_with_plan(
        self,
        conversion_stream: &Stream,
    ) -> Result<(QuantizedCheckpoint, BoundedQuantizationPlan), Error> {
        let device = conversion_stream.get_device()?;
        let device_type = device.get_type()?;
        if self.workspace
            != super::workspace::QuantizerWorkspace::selected(self.plan.quantization, device_type)
        {
            return Err(quantization_error(
                "quantization destinations require preparation for the selected stream",
            ));
        }
        let tile_streams = [conversion_stream.clone(), Stream::new_with_device(&device)];
        let tile_contexts = tile_streams
            .each_ref()
            .map(|stream| MlxParameterMaterializationContext::new(stream, stream));
        self.materialize_with_producer(device_type, &mut OrdinaryTileProducer(tile_contexts))
    }

    /// Shared tile selection, overlap, writeback and telemetry. The producer
    /// supplies its selected device fact and independently owned completions;
    /// it must fund its inputs, native work and retained resources. This driver
    /// creates no additional device or stream wrapper.
    pub(super) fn materialize_with_producer<P: TileProducer>(
        self,
        device_type: safemlx::DeviceType,
        producer: &mut P,
    ) -> Result<(QuantizedCheckpoint, BoundedQuantizationPlan), P::Error> {
        self.validate_workspace(device_type)?;
        let allocator_cache = BoundedAllocatorCache::new(self.plan.max_working_set_bytes);
        self.materialize_with_controls(producer, allocator_cache)
    }

    /// Admits the fixed tile-window controls and actual deferred cache cleanup
    /// node before entering the shared driver. Source/recipe/overlay metadata,
    /// output buffers, native resources and each submission need their own owners.
    pub(super) fn materialize_with_admitted_producer<P: TileProducer>(
        self,
        pool: &eredu_runtime::working_memory::WorkingMemoryPool,
        device_type: safemlx::DeviceType,
        producer: &mut P,
    ) -> Result<(QuantizedCheckpoint, BoundedQuantizationPlan), PipelineAdmissionError<P::Error>>
    {
        self.validate_workspace(device_type)
            .map_err(|cause| PipelineAdmissionError::Producer(P::Error::from(cause)))?;
        let controls =
            controls::required_bytes::<P::Completion>().map_err(PipelineAdmissionError::Policy)?;
        let allocator_cache = BoundedAllocatorCache::prepare_original(
            pool,
            self.plan.max_working_set_bytes,
            controls,
        )
        .map_err(PipelineAdmissionError::Admission)?;
        self.materialize_with_controls(producer, allocator_cache)
            .map_err(PipelineAdmissionError::Producer)
    }

    fn validate_workspace(&self, device_type: safemlx::DeviceType) -> Result<(), Error> {
        if self.workspace
            != super::workspace::QuantizerWorkspace::selected(self.plan.quantization, device_type)
        {
            return Err(quantization_error(
                "quantization destinations require preparation for the selected stream",
            ));
        }
        Ok(())
    }

    fn materialize_with_controls<P: TileProducer>(
        self,
        producer: &mut P,
        mut allocator_cache: BoundedAllocatorCache,
    ) -> Result<(QuantizedCheckpoint, BoundedQuantizationPlan), P::Error> {
        let Self {
            workspace: _,
            source,
            plan,
            mut output_shards,
            materialized_source_keys,
            materialized_source_shards,
        } = self;

        let mut report = WeightMaterializationReport {
            admitted_working_set_bytes: plan.max_working_set_bytes,
            ..WeightMaterializationReport::default()
        };
        let mut pending_tiles = TileWindow::new();
        allocator_cache.begin()?;

        for (index, target) in plan.targets.iter().enumerate() {
            transform_target(
                &source,
                target,
                &plan,
                index,
                producer,
                &mut output_shards,
                &mut pending_tiles,
                &mut allocator_cache,
                &mut report,
            )?;
        }
        while !pending_tiles.is_empty() {
            write_oldest_tile(&mut pending_tiles, &mut output_shards, &mut allocator_cache)?;
        }
        debug_assert!(
            output_shards
                .iter()
                .all(|shard| shard.sealed && shard.pending_tiles == 0)
        );
        allocator_cache.finish()?;

        let transformed = MemoryWeightStore::from_buffers(
            output_shards.into_iter().flat_map(|shard| shard.buffers),
        )
        .map_err(|cause| Error::Other(Box::new(cause)))?;
        Ok((
            QuantizedCheckpoint {
                source: eredu_checkpoint::store::MaterializedCheckpointSource::new(
                    source,
                    transformed,
                    materialized_source_keys,
                    materialized_source_shards,
                ),
                report,
            },
            plan,
        ))
    }
}

fn transform_target<P: TileProducer>(
    source: &eredu_checkpoint::store::RetainedCheckpointSource,
    target: &BoundedQuantizationTarget,
    plan: &BoundedQuantizationPlan,
    output_shard: usize,
    producer: &mut P,
    output_shards: &mut Vec<OutputShard>,
    pending_tiles: &mut TileWindow<P::Completion>,
    allocator_cache: &mut BoundedAllocatorCache,
    report: &mut WeightMaterializationReport,
) -> Result<(), P::Error> {
    let catalog = eredu_checkpoint::recipe::UncachedRecipeCatalog::new(source.as_ref());
    let super::preparation::ConversionGeometry {
        leading,
        rows,
        columns,
        output_row_bytes,
        live_output_row_bytes,
        fixed_working_set_bytes,
        maximum_submission_elements,
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
            producer,
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
                    &catalog,
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
                let candidate_output_bytes = live_output_row_bytes
                    .checked_mul(candidate_rows as u64)
                    .ok_or_else(|| quantization_error("leading batch output size overflow"))?;
                let candidate_peak = candidate
                    .peak_materialization_bytes(&catalog)?
                    .checked_add(candidate_output_bytes)
                    .and_then(|bytes| bytes.checked_add(fixed_working_set_bytes))
                    .ok_or_else(|| quantization_error("leading batch working-set overflow"))?;
                if candidate_elements <= maximum_submission_elements
                    && candidate_peak <= tile_budget
                {
                    matrix_end = candidate_end;
                } else {
                    rejected_end = candidate_end;
                }
            }
            let recipe = target.source.select_bounded(
                &catalog,
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
                .peak_materialization_bytes(&catalog)?
                .checked_add(
                    live_output_row_bytes
                        .checked_mul(batch_rows as u64)
                        .ok_or_else(|| {
                            quantization_error("leading batch live output size overflow")
                        })?,
                )
                .and_then(|bytes| bytes.checked_add(fixed_working_set_bytes))
                .ok_or_else(|| quantization_error("leading batch working-set overflow"))?;
            if batch_peak > tile_budget {
                return Err(quantization_error(format!(
                    "bounded quantization target {:?} cannot admit one leading matrix within the {}-byte tile slot",
                    target.weight_name, tile_budget
                )).into());
            }
            submit_quantization_tile(
                source,
                &recipe,
                target,
                plan.quantization,
                producer,
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
                        &catalog,
                        matrix,
                        start,
                        candidate_end,
                    )?;
                    let candidate_rows = candidate_end - start;
                    let output_bytes = live_output_row_bytes
                        .checked_mul(candidate_rows as u64)
                        .ok_or_else(|| quantization_error("candidate tile output size overflow"))?;
                    let peak = candidate
                        .peak_materialization_bytes(&catalog)?
                        .checked_add(output_bytes)
                        .and_then(|bytes| bytes.checked_add(fixed_working_set_bytes))
                        .ok_or_else(|| quantization_error("candidate tile working-set overflow"))?;
                    if peak <= tile_budget
                        && candidate_rows
                            .checked_mul(columns)
                            .is_some_and(|elements| elements <= maximum_submission_elements)
                    {
                        end = candidate_end;
                    } else {
                        rejected_end = candidate_end;
                    }
                }
                let tile_rows = end - start;
                let tile_recipe = target
                    .source
                    .select_bounded_matrix_rows(&catalog, matrix, start, end)?;
                let tile_output_bytes = output_row_bytes
                    .checked_mul(tile_rows as u64)
                    .ok_or_else(|| quantization_error("quantized tile output size overflow"))?;
                let tile_peak = tile_recipe
                    .peak_materialization_bytes(&catalog)?
                    .checked_add(
                        live_output_row_bytes
                            .checked_mul(tile_rows as u64)
                            .ok_or_else(|| quantization_error("tile live output size overflow"))?,
                    )
                    .and_then(|bytes| bytes.checked_add(fixed_working_set_bytes))
                    .ok_or_else(|| quantization_error("conversion tile working-set overflow"))?;
                if tile_peak > tile_budget {
                    return Err(quantization_error(format!(
                        "bounded quantization planner admitted {} rows for {:?}, but their {}-byte working set exceeds the {}-byte tile slot",
                        tile_rows, target.weight_name, tile_peak, tile_budget
                    )).into());
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
                    producer,
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
fn submit_quantization_tile<P: TileProducer>(
    source: &eredu_checkpoint::store::RetainedCheckpointSource,
    recipe: &DerivedWeightRecipe,
    target: &BoundedQuantizationTarget,
    quantization: WeightQuantization,
    producer: &mut P,
    tile_buffers: usize,
    output_shard: usize,
    output_start: usize,
    rows: usize,
    planned_working_set_bytes: u64,
    output_bytes: u64,
    output_shards: &mut [OutputShard],
    pending_tiles: &mut TileWindow<P::Completion>,
    allocator_cache: &mut BoundedAllocatorCache,
    report: &mut WeightMaterializationReport,
) -> Result<(), P::Error> {
    allocator_cache.prepare_submission(
        queued_working_set_bytes(pending_tiles)?,
        planned_working_set_bytes,
    )?;
    let (completion, source_bytes) = producer.submit(
        source,
        recipe,
        target,
        quantization,
        report.source_tiles % tile_buffers,
    )?;
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
        .checked_add(source_bytes)
        .ok_or_else(|| quantization_error("source-read telemetry overflow"))?;
    report.peak_planned_working_set_bytes = report
        .peak_planned_working_set_bytes
        .max(queued_working_set_bytes(pending_tiles)?);
    report.largest_source_tile_bytes = report.largest_source_tile_bytes.max(source_bytes);
    report.largest_output_tile_bytes = report.largest_output_tile_bytes.max(output_bytes);
    if pending_tiles.len() == tile_buffers {
        write_oldest_tile(pending_tiles, output_shards, allocator_cache)?;
    }
    Ok(())
}

fn prepare_quantized_outputs(
    prepared: &mut WeightMaterialization,
    quantization: WeightQuantization,
    target: &BoundedQuantizationTarget,
    stream: &Stream,
) -> Result<(), Error> {
    prepare_quantized_outputs_with(
        prepared,
        quantization,
        target,
        stream,
        None,
        |array, dtype, stream| array.as_dtype(dtype, stream),
    )
}

fn prepare_quantized_outputs_with(
    prepared: &mut WeightMaterialization,
    quantization: WeightQuantization,
    target: &BoundedQuantizationTarget,
    stream: &Stream,
    original: Option<(
        safemlx::CpuAffineQuantizeSubmissionLayout,
        &safemlx::OriginalScopeObserver,
    )>,
    mut convert: impl FnMut(&Array, Dtype, &Stream) -> Result<Array, safemlx::error::Exception>,
) -> Result<(), Error> {
    prepared.prepare_output_capacity(if matches!(quantization, WeightQuantization::MxFp4) {
        2
    } else {
        3
    })?;
    use crate::backend::runtime::checkpoint::store::CheckpointMaterializationError;
    let construction = original
        .map(|(layout, observer)| {
            safemlx::OperationEvent::prepare_affine_quantize_graph(layout.construction(), observer)
                .map_err(CheckpointMaterializationError::OriginalNative)
        })
        .transpose()?;
    let quantized = quantize_tensor(&prepared.inputs()[0], quantization, stream)?;
    drop(construction);
    prepared.retain_output(quantized.weight)?;
    prepared.retain_output(quantized.scales)?;
    if let Some(biases) = quantized.biases {
        prepared.retain_output(biases)?;
    }
    let companion_dtype = match target.affine_companion_dtype {
        RecipeDtype::F16 => Dtype::Float16,
        RecipeDtype::BF16 => Dtype::Bfloat16,
        RecipeDtype::F32 => Dtype::Float32,
        _ => unreachable!("validated bounded companion dtype"),
    };
    if !matches!(quantization, WeightQuantization::MxFp4) {
        let construction = original
            .and_then(|(layout, observer)| {
                layout
                    .companion_construction()
                    .map(|layout| (layout, observer))
            })
            .map(|(layout, observer)| {
                safemlx::OperationEvent::prepare_resident_graph(layout, observer)
                    .map_err(CheckpointMaterializationError::OriginalNative)
            })
            .transpose()?;
        for index in 1..prepared.outputs().len() {
            let converted = convert(&prepared.outputs()[index], companion_dtype, stream)?;
            prepared.replace_output(index, converted);
        }
        drop(construction);
    }
    Ok(())
}

mod original_affine;
pub(crate) use original_affine::submit_original_affine_tile;

#[cfg(test)]
mod output_tests;

struct SubmittedQuantizationTile<C> {
    completion: C,
    output_start: usize,
    rows: usize,
    planned_working_set_bytes: u64,
    output_shard: usize,
}

impl<C: TileCompletion> SubmittedQuantizationTile<C> {
    fn write(self, shard: &mut OutputShard) -> Result<(), Error> {
        let materialization = self.completion.materialization();
        materialization.wait()?;
        for ((layout, output), buffer) in shard
            .layouts
            .iter()
            .zip(materialization.completed_outputs())
            .zip(&mut shard.buffers)
        {
            write_tile(
                buffer.bytes_mut(),
                layout,
                self.output_start,
                self.rows,
                &output?,
            )?;
        }
        self.completion.retire()
    }
}

fn write_oldest_tile<C: TileCompletion>(
    pending_tiles: &mut TileWindow<C>,
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

fn queued_working_set_bytes<C>(pending_tiles: &TileWindow<C>) -> Result<u64, Error> {
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
    evaluated: &safemlx::EvaluatedArray<'_>,
) -> Result<(), Error> {
    let output = evaluated.as_array();
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

pub(crate) use producer::{ColdConversion, CpuTileResources};
