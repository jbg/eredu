use eredu_checkpoint::{
    prepared_cache::PreparedTensorLayout,
    quantization::{BoundedQuantizationPlan, BoundedQuantizationTarget},
    recipe::DerivedWeightRecipe,
    store::{safetensors_dtype_from_stored, CheckpointSource, TensorSelection},
    StoredDtype,
};
use eredu_core::{
    checkpoint_mechanisms::bounded_quantization::QuantizationTileLimits, BackendFailure,
};

use super::{add_bytes, invalid};

pub(super) struct Output {
    pub(super) layout: PreparedTensorLayout,
    pub(super) dtype: StoredDtype,
    pub(super) row_bytes: u64,
}

pub(super) struct Target<'a> {
    pub(super) definition: &'a BoundedQuantizationTarget,
    pub(super) outputs: Vec<Output>,
    pub(super) output_bytes: u64,
    pub(super) minimum_bytes: u64,
    pub(super) slots: usize,
    matrices: usize,
    rows: usize,
    columns: usize,
    row_axis: usize,
    output_row_bytes: u64,
    slot_bytes: u64,
    max_elements: usize,
}

pub(super) struct Tile {
    pub(super) recipe: DerivedWeightRecipe,
    pub(super) start_row: usize,
    pub(super) rows: usize,
    pub(super) source_bytes: u64,
    pub(super) output_bytes: u64,
    pub(super) working_set_bytes: u64,
}

impl<'a> Target<'a> {
    pub(super) fn admit(
        source: &dyn CheckpointSource,
        definition: &'a BoundedQuantizationTarget,
        plan: &BoundedQuantizationPlan,
        limits: &QuantizationTileLimits,
    ) -> Result<Self, BackendFailure> {
        let metadata = definition
            .source()
            .infer(source)
            .map_err(BackendFailure::from_error)?;
        let shape = metadata.shape();
        let mut output_metadata = definition
            .output_metadata(plan.quantization(), shape)
            .map_err(BackendFailure::from_error)?;
        let row_axis = shape.len() - 2;
        let matrices = shape[..row_axis].iter().try_fold(1usize, |n, &size| {
            n.checked_mul(size)
                .ok_or_else(|| invalid("matrix count overflow"))
        })?;
        let rows = shape[row_axis];
        let columns = shape[row_axis + 1];
        let total_rows = matrices
            .checked_mul(rows)
            .ok_or_else(|| invalid("row count overflow"))?;
        let mut output_bytes = 0;
        let mut outputs = Vec::new();
        for name in definition
            .output_names(plan.quantization())
            .map_err(BackendFailure::from_error)?
        {
            let tensor = output_metadata
                .remove(&name)
                .ok_or_else(|| invalid("missing encoded output geometry"))?;
            let dtype = safetensors_dtype_from_stored(&tensor.stored_dtype)
                .ok_or_else(|| invalid("unsupported encoded output dtype"))?;
            output_bytes = add_bytes(output_bytes, tensor.encoded_byte_len)?;
            outputs.push(Output {
                layout: PreparedTensorLayout {
                    name,
                    dtype,
                    shape: tensor.physical_shape,
                },
                dtype: tensor.stored_dtype,
                row_bytes: tensor.encoded_byte_len / total_rows as u64,
            });
        }
        let output_row_bytes = output_bytes / total_rows as u64;
        let mut minimum_bytes = 0;
        for matrix in 0..matrices {
            let recipe = definition
                .source()
                .select_bounded_matrix_rows(source, matrix, 0, 1)
                .map_err(BackendFailure::from_error)?;
            recipe
                .preflight_bounded(source)
                .map_err(BackendFailure::from_error)?;
            let peak = recipe
                .peak_materialization_bytes(source)
                .map_err(BackendFailure::from_error)?;
            minimum_bytes = minimum_bytes.max(add_bytes(peak, output_row_bytes)?);
        }
        if columns > limits.elements.get() || minimum_bytes > plan.max_working_set_bytes() {
            return Err(invalid(format!(
                "one row of {:?} exceeds the native element limit or the {}-byte working set",
                definition.weight_name(),
                plan.max_working_set_bytes(),
            )));
        }
        let slots = limits.slots.get().min(
            usize::try_from(plan.max_working_set_bytes() / minimum_bytes).unwrap_or(usize::MAX),
        );
        Ok(Self {
            definition,
            outputs,
            output_bytes,
            minimum_bytes,
            slots,
            matrices,
            rows,
            columns,
            row_axis,
            output_row_bytes,
            slot_bytes: plan.max_working_set_bytes() / slots as u64,
            max_elements: limits.elements.get(),
        })
    }

    pub(super) fn for_each_tile(
        &self,
        source: &dyn CheckpointSource,
        mut submit: impl FnMut(Tile) -> Result<(), BackendFailure>,
    ) -> Result<(), BackendFailure> {
        let full = self.definition.source();
        let complete_rows = self.matrices * self.rows;
        let max_rows = self.max_elements / self.columns;
        if complete_rows <= max_rows {
            let tile = self.tile(source, full.clone(), 0, complete_rows)?;
            if tile.working_set_bytes <= self.slot_bytes {
                return submit(tile);
            }
        }

        if self.row_axis == 1 && self.rows <= max_rows {
            let select = |start, end| {
                full.select_bounded(
                    source,
                    TensorSelection::Range {
                        axis: 0,
                        start,
                        end,
                    },
                )
                .map_err(BackendFailure::from_error)
            };
            let first = self.tile(source, select(0, 1)?, 0, self.rows)?;
            if first.working_set_bytes <= self.slot_bytes {
                let mut start = 0;
                while start < self.matrices {
                    let limit = self
                        .matrices
                        .min(start.saturating_add(max_rows / self.rows));
                    let (end, tile) = self.largest_tile(start, limit, |end| {
                        self.tile(
                            source,
                            select(start, end)?,
                            start * self.rows,
                            (end - start) * self.rows,
                        )
                    })?;
                    submit(tile)?;
                    start = end;
                }
                return Ok(());
            }
        }

        for matrix in 0..self.matrices {
            let mut start = 0;
            while start < self.rows {
                let limit = self.rows.min(start.saturating_add(max_rows));
                let (end, tile) = self.largest_tile(start, limit, |end| {
                    let recipe = full
                        .select_bounded_matrix_rows(source, matrix, start, end)
                        .map_err(BackendFailure::from_error)?;
                    self.tile(source, recipe, matrix * self.rows + start, end - start)
                })?;
                submit(tile)?;
                start = end;
            }
        }
        Ok(())
    }

    fn largest_tile(
        &self,
        start: usize,
        limit: usize,
        select: impl Fn(usize) -> Result<Tile, BackendFailure>,
    ) -> Result<(usize, Tile), BackendFailure> {
        let mut accepted = start + 1;
        let mut best = select(accepted)?;
        if best.working_set_bytes > self.slot_bytes {
            return Err(invalid(format!(
                "minimum tile of {:?} at row {} needs {} bytes, but the slot allows {}",
                self.definition.weight_name(),
                best.start_row,
                best.working_set_bytes,
                self.slot_bytes,
            )));
        }
        let mut high = limit;
        while accepted < high {
            let end = accepted + (high - accepted).div_ceil(2);
            let candidate = select(end)?;
            if candidate.working_set_bytes <= self.slot_bytes {
                accepted = end;
                best = candidate;
            } else {
                high = end - 1;
            }
        }
        Ok((accepted, best))
    }

    fn tile(
        &self,
        source: &dyn CheckpointSource,
        recipe: DerivedWeightRecipe,
        start_row: usize,
        rows: usize,
    ) -> Result<Tile, BackendFailure> {
        let source_bytes = recipe
            .infer(source)
            .map_err(BackendFailure::from_error)?
            .byte_len();
        let peak = recipe
            .peak_materialization_bytes(source)
            .map_err(BackendFailure::from_error)?;
        let output_bytes = self
            .output_row_bytes
            .checked_mul(rows as u64)
            .ok_or_else(|| invalid("encoded tile size overflow"))?;
        Ok(Tile {
            recipe,
            start_row,
            rows,
            source_bytes,
            output_bytes,
            working_set_bytes: add_bytes(peak, output_bytes)?,
        })
    }
}
