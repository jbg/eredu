//! Borrowed common constructor inputs; no source or native owner is created here.
use super::*;
pub(super) struct ReadSourcePlan<'a> {
    pub(super) units: &'a [OffloadUnit],
    pub(super) reads: &'a [PlannedRead],
    pub(super) catalogs: &'a source::OriginalHostCatalogPlan,
}
impl ReadSourcePlan<'_> {
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
    let mut reads = Vec::new();
    for (unit, definition) in units.iter().enumerate() {
        for (binding, value) in definition
            .bindings()
            .iter()
            .enumerate()
            .filter(|(_, value)| !value.is_alias())
        {
            let recipe = value.source_recipe();
            let Some(read) = DirectRecipeRead::prepare(&recipe, source(definition.id()))? else {
                return Ok(None);
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
