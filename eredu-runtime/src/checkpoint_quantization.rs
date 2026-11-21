//! Portable scheduling and persistence for bounded checkpoint transformations.

use std::collections::VecDeque;

use eredu_checkpoint::{
    prepared_cache::{
        prepared_checkpoint_recipe_key, PreparedCheckpointEntry, PreparedCheckpointWriter,
    },
    quantization::{BoundedQuantizationPlan, WeightMaterializationReport},
    store::{CheckpointSource, SharedCheckpointSource, StoreError},
};
use eredu_core::{
    checkpoint_mechanisms::bounded_quantization::{
        BoundedQuantizationBackend, QuantizationTileCompletion,
    },
    BackendFailure,
};

mod tiles;
use tiles::Target;

pub(crate) fn realize_prepared_quantization<'a>(
    source: &SharedCheckpointSource,
    open_quantizer: impl FnOnce() -> Result<Box<dyn BoundedQuantizationBackend + 'a>, BackendFailure>,
) -> Result<Option<WeightMaterializationReport>, BackendFailure> {
    let mut pending = Vec::new();
    let mut input = source.clone();
    while let Some((dependency, plan)) = input.preparation() {
        pending.push((input, plan));
        input = dependency;
    }
    if pending.is_empty() {
        return Ok(None);
    }

    let mut encoder = open_quantizer()?;
    let mut report = None;
    for (destination, plan) in pending.into_iter().rev() {
        let (output, completed) = encode(input.as_ref(), &plan, encoder.as_mut())?;
        destination
            .publish_preparation(output)
            .map_err(BackendFailure::from_error)?;
        input = destination;
        report = Some(completed);
    }
    Ok(report)
}

fn encode(
    source: &dyn CheckpointSource,
    plan: &BoundedQuantizationPlan,
    encoder: &mut dyn BoundedQuantizationBackend,
) -> Result<(SharedCheckpointSource, WeightMaterializationReport), BackendFailure> {
    let cache = plan
        .prepared_cache()
        .or_else(|| source.prepared_cache_policy())
        .ok_or_else(|| BackendFailure::from_error(StoreError::PreparedCacheRequired))?;
    let limits = encoder.limits();
    let targets = plan
        .targets()
        .iter()
        .map(|target| Target::admit(source, target, plan, &limits))
        .collect::<Result<Vec<_>, _>>()?;
    let layouts = targets
        .iter()
        .flat_map(|t| t.outputs.iter().map(|o| o.layout.clone()))
        .collect::<Vec<_>>();
    let fingerprint_bound = plan
        .max_working_set_bytes()
        .min(8 * 1024 * 1024)
        .max(targets.iter().map(|t| t.minimum_bytes).max().unwrap_or(0));
    let identity = format!(
        "{}:{:?}:{:?}",
        encoder.cache_identity()?,
        plan.quantization(),
        plan.targets()
    );
    let recipes = plan
        .targets()
        .iter()
        .map(|target| target.source().clone())
        .collect::<Vec<_>>();
    let key =
        prepared_checkpoint_recipe_key(source, &recipes, identity.as_bytes(), fingerprint_bound)
            .map_err(BackendFailure::from_error)?;
    let mut report = WeightMaterializationReport {
        admitted_working_set_bytes: plan.max_working_set_bytes(),
        transformed_weights: targets.len(),
        output_bytes: targets
            .iter()
            .try_fold(0, |sum, target| add_bytes(sum, target.output_bytes))?,
        ..Default::default()
    };
    let mut writer = match cache
        .prepare(&key, &layouts, plan.max_working_set_bytes())
        .map_err(BackendFailure::from_error)?
    {
        PreparedCheckpointEntry::Cached(output) => return Ok((output, report)),
        PreparedCheckpointEntry::Vacant(writer) => writer,
    };

    let slots = targets
        .iter()
        .map(|t| t.slots)
        .max()
        .expect("admitted plan has targets");
    encoder.begin(slots, plan.max_working_set_bytes())?;
    let mut window = Window {
        encoder,
        writer: &mut writer,
        budget: plan.max_working_set_bytes(),
        pending: VecDeque::with_capacity(slots),
        live_bytes: 0,
    };
    let mut window_slots = slots;
    for target in &targets {
        // Pending tiles may have been admitted with a different per-slot budget.
        if target.slots != window_slots {
            while !window.pending.is_empty() {
                window.write_oldest()?;
            }
        }
        window_slots = target.slots;
        target.for_each_tile(source, |tile| {
            window.submit(source, target, tile, plan.quantization(), &mut report)
        })?;
    }
    while !window.pending.is_empty() {
        window.write_oldest()?;
    }
    window.encoder.finish()?;
    Ok((writer.finish().map_err(BackendFailure::from_error)?, report))
}

struct Pending<'a> {
    completion: Box<dyn QuantizationTileCompletion>,
    target: &'a Target<'a>,
    slot: usize,
    start_row: usize,
    rows: usize,
    working_set_bytes: u64,
}

struct Window<'a, 'targets> {
    encoder: &'a mut dyn BoundedQuantizationBackend,
    writer: &'a mut PreparedCheckpointWriter,
    budget: u64,
    pending: VecDeque<Pending<'targets>>,
    live_bytes: u64,
}

impl<'targets> Window<'_, 'targets> {
    fn submit(
        &mut self,
        source: &dyn CheckpointSource,
        target: &'targets Target<'targets>,
        tile: tiles::Tile,
        encoding: eredu_checkpoint::WeightQuantization,
        report: &mut WeightMaterializationReport,
    ) -> Result<(), BackendFailure> {
        tile.recipe
            .preflight_bounded(source)
            .map_err(BackendFailure::from_error)?;
        let slot = (0..target.slots)
            .find(|&slot| self.pending.iter().all(|pending| pending.slot != slot))
            .ok_or_else(|| invalid("no free slot for an admitted quantization tile"))?;
        let live_bytes = add_bytes(self.live_bytes, tile.working_set_bytes)?;
        let available = self
            .budget
            .checked_sub(live_bytes)
            .ok_or_else(|| invalid("quantization window exceeds the working-set budget"))?;
        self.encoder.trim(available)?;
        let completion = self.encoder.submit(
            slot,
            source,
            &tile.recipe,
            encoding,
            target.definition.affine_companion_dtype().clone(),
        )?;
        self.pending.push_back(Pending {
            completion,
            target,
            slot,
            start_row: tile.start_row,
            rows: tile.rows,
            working_set_bytes: tile.working_set_bytes,
        });
        self.live_bytes = live_bytes;
        report.peak_in_flight_tiles = report.peak_in_flight_tiles.max(self.pending.len());
        report.source_tiles += 1;
        report.source_bytes_read = add_bytes(report.source_bytes_read, tile.source_bytes)?;
        report.peak_planned_working_set_bytes =
            report.peak_planned_working_set_bytes.max(live_bytes);
        report.largest_source_tile_bytes = report.largest_source_tile_bytes.max(tile.source_bytes);
        report.largest_output_tile_bytes = report.largest_output_tile_bytes.max(tile.output_bytes);
        if self.pending.len() == target.slots {
            self.write_oldest()?;
        }
        Ok(())
    }

    fn write_oldest(&mut self) -> Result<(), BackendFailure> {
        let tile = self
            .pending
            .pop_front()
            .expect("non-empty quantization window");
        let packed = tile.completion.complete()?;
        if packed.output_count() != tile.target.outputs.len() {
            return Err(invalid(
                "quantization tile returned the wrong number of outputs",
            ));
        }
        for (index, output) in tile.target.outputs.iter().enumerate() {
            let (dtype, bytes) = packed.output(index)?;
            if dtype != output.dtype || bytes.len() as u64 != output.row_bytes * tile.rows as u64 {
                return Err(invalid(format!(
                    "encoded tile differs from the geometry of {:?}",
                    output.layout.name
                )));
            }
            self.writer
                .write(
                    &output.layout.name,
                    output.row_bytes * tile.start_row as u64,
                    bytes,
                )
                .map_err(BackendFailure::from_error)?;
        }
        drop(packed);
        self.live_bytes -= tile.working_set_bytes;
        self.encoder.trim(self.budget - self.live_bytes)
    }
}

fn invalid(message: impl Into<String>) -> BackendFailure {
    BackendFailure::from_error(StoreError::InvalidPreparation(message.into()))
}

fn add_bytes(left: u64, right: u64) -> Result<u64, BackendFailure> {
    left.checked_add(right)
        .ok_or_else(|| invalid("quantization byte count overflow"))
}
