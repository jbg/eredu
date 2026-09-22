//! Borrowed common constructor inputs; no source or native owner is created here.
use super::*;
mod materialized;
pub(super) use materialized::{MaterializedReadPlan, PreparedMaterializedRead, PreparedRecipeLeaf};
mod value;
pub(super) use value::{EncodedLeaves, ReadValue};
pub(super) struct ReadSourcePlan<'a> {
    pub(super) units: &'a [OffloadUnit],
    pub(super) reads: &'a [PlannedRead],
    pub(super) catalogs: &'a source::OriginalHostCatalogPlan,
}
impl<'a> ReadSourcePlan<'a> {
    pub(super) fn encoded_leaves(&self) -> Option<EncodedLeaves<'a>> {
        EncodedLeaves::new(self.reads)
    }
    pub(super) fn leaf_range(
        &self,
        rows: std::ops::Range<usize>,
    ) -> Option<std::ops::Range<usize>> {
        let start = EncodedLeaves::new(self.reads.get(..rows.start)?)?.len();
        let count = EncodedLeaves::new(self.reads.get(rows)?)?.len();
        Some(start..start.checked_add(count)?)
    }
    pub(super) fn read_range(&self, unit: usize) -> std::ops::Range<usize> {
        let start = self.reads.partition_point(|row| row.unit < unit);
        start..self.reads.partition_point(|row| row.unit <= unit)
    }
}

/// Both source modes retain the exact existing recipe reader; unsupported read
/// construction remains a typed absence before source admission.
pub(super) fn prepare_reads<'a>(
    units: &[OffloadUnit],
    source: impl Fn(&OffloadUnitId) -> &'a dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Option<Vec<PlannedRead>>, WeightRecipeError> {
    prepare_reads_impl(units, source, |_, _| Ok(None))
}

pub(super) fn prepare_host_reads<'a>(
    units: &[OffloadUnit],
    source: impl Fn(&OffloadUnitId) -> &'a eredu_checkpoint::store::RetainedCheckpointSource,
    pool: &MemoryLedger,
    stream: &Stream,
) -> Result<Option<Vec<PlannedRead>>, WeightRecipeError> {
    prepare_reads_impl(
        units,
        |id| source(id).as_ref(),
        |recipe, id| {
            #[cfg(target_vendor = "apple")]
            {
                use crate::backend::nn::workspace::{
                    MlxMetalWorkspaceMechanisms, ResidentExecutionMechanisms,
                };
                let funding = pool.prepare_construction_metadata().map_err(|cause| {
                    WeightRecipeError::Workspace(
                        eredu_nn::workspace::WorkspaceMetadataError::Funding(cause).into(),
                    )
                })?;
                let mechanism = ResidentExecutionMechanisms::from_stream(
                    MlxMetalWorkspaceMechanisms::current_host()?.original_storage(),
                    stream,
                    funding.funding(),
                )?;
                MaterializedReadPlan::prepare(recipe, source(id), pool, funding, mechanism)
                    .map(Some)
            }
            #[cfg(not(target_vendor = "apple"))]
            {
                let _ = (recipe, id, pool, stream);
                Ok(None)
            }
        },
    )
}

fn prepare_reads_impl<'a>(
    units: &[OffloadUnit],
    source: impl Fn(&OffloadUnitId) -> &'a dyn eredu_checkpoint::store::CheckpointSource,
    mut materialize: impl FnMut(
        &eredu_checkpoint::recipe::DerivedWeightRecipe,
        &OffloadUnitId,
    ) -> Result<Option<PreparedMaterializedRead>, WeightRecipeError>,
) -> Result<Option<Vec<PlannedRead>>, WeightRecipeError> {
    let mut reads = Vec::new();
    for (unit, definition) in units.iter().enumerate() {
        for (binding, value) in definition
            .bindings()
            .iter()
            .enumerate()
            .filter(|(_, value)| !value.is_alias())
        {
            let direct;
            let recipe = match value.recipe() {
                Some(recipe) => recipe,
                None => {
                    direct = value.source_recipe();
                    &direct
                }
            };
            let read = match DirectRecipeRead::prepare(recipe, source(definition.id()))? {
                Some(read) => ReadValue::Direct(read),
                None => match materialize(recipe, definition.id())? {
                    Some(read) => ReadValue::Materialized(read),
                    None => return Ok(None),
                },
            };
            reads.push(PlannedRead {
                unit,
                binding,
                read,
            });
        }
    }
    Ok(Some(reads))
}
