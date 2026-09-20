//! Cold source and geometry preflight and final output allocation before conversion.
use super::layout::{output_layouts, OutputLayout, OutputShard};
use super::preflight::{
    checked_product, output_names_for, preflight_source_collisions, quantization_error,
};
use super::workspace::QuantizerWorkspace;
use super::*;

#[derive(Clone, Copy)]
pub(super) struct ConversionGeometry {
    pub(super) leading: usize,
    pub(super) rows: usize,
    pub(super) columns: usize,
    pub(super) output_row_bytes: u64,
    pub(super) live_output_row_bytes: u64,
    pub(super) fixed_working_set_bytes: u64,
    pub(super) maximum_submission_elements: usize,
    pub(super) output_bytes: u64,
    pub(super) one_row_source_bytes: u64,
    pub(super) source_bytes: u64,
    pub(super) tile_buffers: usize,
    pub(super) tile_budget: u64,
    pub(super) complete_rows: usize,
    pub(super) complete_peak: u64,
    pub(super) complete_admissible: bool,
    pub(super) leading_batch_admissible: bool,
}

struct TargetOutputs {
    layouts: Vec<OutputLayout>,
    geometry: ConversionGeometry,
}

/// Retains the exact source and validated metadata without native state or payloads.
pub(crate) struct ColdQuantization {
    workspace: QuantizerWorkspace,
    source: eredu_checkpoint::store::RetainedCheckpointSource,
    plan: BoundedQuantizationPlan,
    targets: Vec<TargetOutputs>,
    transformed_keys: BTreeSet<String>,
    materialized_source_keys: BTreeSet<String>,
    materialized_source_shards: BTreeSet<PathBuf>,
}

/// Owns the final destinations before the conversion streams are constructed.
pub(crate) struct PreparedQuantization {
    pub(super) workspace: QuantizerWorkspace,
    pub(super) source: eredu_checkpoint::store::RetainedCheckpointSource,
    pub(super) plan: BoundedQuantizationPlan,
    pub(super) output_shards: Vec<OutputShard>,
    pub(super) transformed_keys: BTreeSet<String>,
    pub(super) materialized_source_keys: BTreeSet<String>,
    pub(super) materialized_source_shards: BTreeSet<PathBuf>,
}

impl ColdQuantization {
    /// Authenticate a completed conversion against this source and exact plan.
    pub(crate) fn validate_conversion(
        &self,
        converted: &ConvertedQuantization,
    ) -> Result<(), Error> {
        converted.validate(&self.source, &self.plan)
    }

    /// Common metadata preflight. Allocation qualifies the selected stream;
    /// materialization checks that profile again before constructing conversion
    /// streams or reading payloads.
    pub(crate) fn prepare(
        source: eredu_checkpoint::store::RetainedCheckpointSource,
        plan: BoundedQuantizationPlan,
    ) -> Result<Self, Error> {
        Self::prepare_with_workspace(source, plan, QuantizerWorkspace::Direct)
    }

    /// Qualifies the actual quantizer before allocating final output buffers.
    fn for_stream(self, stream: &Stream) -> Result<Self, Error> {
        let workspace =
            QuantizerWorkspace::selected(self.plan.quantization, stream.get_device()?.get_type()?);
        if self.workspace == workspace {
            return Ok(self);
        }
        Self::prepare_with_workspace(self.source, self.plan, workspace)
    }

    fn prepare_with_workspace(
        source: eredu_checkpoint::store::RetainedCheckpointSource,
        plan: BoundedQuantizationPlan,
        workspace: QuantizerWorkspace,
    ) -> Result<Self, Error> {
        if !cfg!(target_endian = "little") {
            return Err(quantization_error(
                "bounded SafeTensors quantization requires a little-endian host",
            ));
        }
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
        let mut targets = Vec::with_capacity(plan.targets.len());
        for target in &plan.targets {
            targets.push(prepare_target(source.as_ref(), target, &plan, workspace)?);
            transformed_keys.extend(output_names_for(target, plan.quantization)?);
        }
        Ok(Self {
            workspace,
            source,
            plan,
            targets,
            transformed_keys,
            materialized_source_keys,
            materialized_source_shards,
        })
    }

    /// Compares all encoded outputs with the actual selected destination slots.
    pub(crate) fn validate_destinations(
        &self,
        destinations: &std::collections::BTreeMap<String, eredu_runtime::ParameterBindingTarget>,
    ) -> Result<(), Error> {
        for layout in self.targets.iter().flat_map(|target| &target.layouts) {
            let destination = destinations.get(&layout.name).ok_or_else(|| {
                quantization_error(format!(
                    "quantized output {:?} has no selected destination",
                    layout.name
                ))
            })?;
            let dtype = match layout.dtype {
                SafeDtype::U32 => RecipeDtype::U32,
                SafeDtype::U8 => RecipeDtype::U8,
                SafeDtype::F16 => RecipeDtype::F16,
                SafeDtype::BF16 => RecipeDtype::BF16,
                SafeDtype::F32 => RecipeDtype::F32,
                other => {
                    return Err(quantization_error(format!(
                        "unsupported quantized output dtype {other:?}"
                    )))
                }
            };
            if destination.shape != layout.shape
                || (destination.dtype != dtype
                    && !destination.permitted_source_dtypes.contains(&dtype))
            {
                return Err(quantization_error(format!(
                    "quantized output {:?} has shape {:?} and dtype {:?}, incompatible with selected destination {:?}",
                    layout.name, layout.shape, dtype, destination,
                )));
            }
        }
        Ok(())
    }

    pub(crate) fn allocate_ordinary(self, stream: &Stream) -> Result<PreparedQuantization, Error> {
        self.allocate(stream, |layout| {
            eredu_checkpoint::store::MemoryTensorBuffer::allocate(
                &layout.name,
                layout.dtype,
                &layout.shape,
                layout.byte_len,
            )
            .map_err(|cause| Error::Other(Box::new(cause)))
        })
    }

    /// Qualify all targets for the stream before invoking any output allocator.
    /// The allocator must establish custody before creating each destination.
    /// Neither preflight nor this callback boundary grants pool admission.
    pub(super) fn allocate(
        self,
        stream: &Stream,
        mut allocate: impl FnMut(
            &OutputLayout,
        ) -> Result<eredu_checkpoint::store::MemoryTensorBuffer, Error>,
    ) -> Result<PreparedQuantization, Error> {
        let qualified = self.for_stream(stream)?;
        let output_shards = qualified
            .targets
            .into_iter()
            .map(|target| {
                let buffers = target
                    .layouts
                    .iter()
                    .map(&mut allocate)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(OutputShard {
                    layouts: target.layouts,
                    geometry: target.geometry,
                    buffers,
                    pending_tiles: 0,
                    sealed: false,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(PreparedQuantization {
            workspace: qualified.workspace,
            source: qualified.source,
            plan: qualified.plan,
            output_shards,
            transformed_keys: qualified.transformed_keys,
            materialized_source_keys: qualified.materialized_source_keys,
            materialized_source_shards: qualified.materialized_source_shards,
        })
    }

    /// Creates final payloads under the pool's original source constructor.
    /// Plan/overlay metadata and native conversion work remain separately funded.
    pub(crate) fn allocate_original(
        self,
        pool: &eredu_runtime::working_memory::WorkingMemoryPool,
        metadata_policy: eredu_runtime::working_memory::DependencyMemoryPolicy,
        stream: &Stream,
    ) -> Result<PreparedQuantization, Error> {
        self.allocate(stream, |layout| {
            pool.allocate_memory_tensor_buffer(
                &layout.name,
                layout.dtype,
                &layout.shape,
                layout.byte_len,
                metadata_policy,
            )
            .map_err(|cause| Error::Other(Box::new(cause)))
        })
    }
}

fn prepare_target(
    source: &dyn CheckpointSource,
    target: &BoundedQuantizationTarget,
    plan: &BoundedQuantizationPlan,
    workspace: QuantizerWorkspace,
) -> Result<TargetOutputs, Error> {
    let catalog = eredu_checkpoint::recipe::UncachedRecipeCatalog::new(source);
    let metadata = target.source.infer(&catalog)?;
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
    let temporary = workspace.payload(metadata.dtype(), columns)?;
    let fixed_working_set_bytes = temporary.fixed_bytes;
    let maximum_submission_elements = workspace.maximum_submission_elements();
    if columns > maximum_submission_elements {
        return Err(quantization_error(
            "one quantization row exceeds the native submission element limit",
        ));
    }
    let live_output_row_bytes = output_row_bytes
        .checked_add(target.companion_cast_source_row_bytes(
            plan.quantization,
            metadata.dtype(),
            columns,
        )?)
        .and_then(|bytes| bytes.checked_add(temporary.row_bytes))
        .ok_or_else(|| quantization_error("live quantized output row size overflow"))?;
    let mut one_row_source_bytes = 0u64;
    let mut one_row_peak = 0u64;
    for matrix in 0..leading {
        let one_row = target
            .source
            .select_bounded_matrix_rows(&catalog, matrix, 0, 1)?;
        one_row_source_bytes = one_row_source_bytes.max(one_row.infer(&catalog)?.byte_len());
        one_row_peak = one_row_peak.max(
            one_row
                .peak_materialization_bytes(&catalog)?
                .checked_add(live_output_row_bytes)
                .and_then(|bytes| bytes.checked_add(fixed_working_set_bytes))
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
    let tile_budget = plan.max_working_set_bytes / tile_buffers as u64;
    let output_bytes = layouts.iter().try_fold(0u64, |total, layout| {
        total
            .checked_add(layout.byte_len)
            .ok_or_else(|| quantization_error("quantized output telemetry overflow"))
    })?;
    let complete_rows = leading
        .checked_mul(rows)
        .ok_or_else(|| quantization_error("complete target row count overflow"))?;
    let complete_elements = complete_rows
        .checked_mul(columns)
        .ok_or_else(|| quantization_error("complete target element count overflow"))?;
    let complete_peak = target
        .source
        .peak_materialization_bytes(&catalog)?
        .checked_add(
            live_output_row_bytes
                .checked_mul(complete_rows as u64)
                .ok_or_else(|| quantization_error("complete live output size overflow"))?,
        )
        .and_then(|bytes| bytes.checked_add(fixed_working_set_bytes))
        .ok_or_else(|| quantization_error("complete target working-set overflow"))?;
    let leading_batch_admissible = if row_axis == 1 {
        let one_matrix = target.source.select_bounded(
            &catalog,
            TensorSelection::Range {
                axis: 0,
                start: 0,
                end: 1,
            },
        )?;
        let one_matrix_output = live_output_row_bytes
            .checked_mul(rows as u64)
            .ok_or_else(|| quantization_error("one-matrix output size overflow"))?;
        one_matrix
            .peak_materialization_bytes(&catalog)?
            .checked_add(one_matrix_output)
            .and_then(|bytes| bytes.checked_add(fixed_working_set_bytes))
            .is_some_and(|peak| peak <= tile_budget)
            && rows
                .checked_mul(columns)
                .is_some_and(|elements| elements <= maximum_submission_elements)
    } else {
        false
    };
    let source_bytes = metadata.byte_len();
    let complete_admissible =
        complete_peak <= tile_budget && complete_elements <= maximum_submission_elements;
    Ok(TargetOutputs {
        layouts,
        geometry: ConversionGeometry {
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
        },
    })
}
