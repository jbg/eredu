//! Lazy row bindings over an exact retained compact checkpoint recipe.
use super::*;
use eredu_checkpoint::rows::{PreparedRowSource, RowReadLimits};
use eredu_core::residency::OffloadUnitRange;
use std::sync::Arc;

/// One uniform row namespace sharing a prepared source and local binding name.
/// Catalog storage scales with shard recipes; only demanded rows gain bindings.
#[derive(Debug, Clone)]
pub struct RowResidencyRange {
    range: OffloadUnitRange,
    source: Arc<PreparedRowSource>,
    binding: String,
}
impl RowResidencyRange {
    /// Validates the row extent and encoded bytes against the prepared source.
    pub fn new(
        range: OffloadUnitRange,
        source: Arc<PreparedRowSource>,
        binding: impl Into<String>,
    ) -> Result<Self, ResidencyControllerError> {
        let binding = validate_name(binding.into())?;
        let rows = source.metadata().shape[0] as u64;
        let bytes = source.metadata().byte_len;
        if range.end() - range.start() != rows || bytes % rows != 0 || range.bytes() != bytes / rows
        {
            return Err(ResidencyControllerError::RowRangeGeometry {
                prefix: range.prefix().clone(),
            });
        }
        let result = Self {
            range,
            source,
            binding,
        };
        // Exercise both boundaries without loading any payload.
        result.unit(&result.range.member_id(result.range.start()).unwrap())?;
        result.unit(&result.range.member_id(result.range.end() - 1).unwrap())?;
        Ok(result)
    }
    /// Exact local row binding.
    pub fn binding(&self) -> &str {
        &self.binding
    }
    /// Retained recipe output geometry (encoded width for packed blocks).
    pub fn metadata(&self) -> &eredu_checkpoint::recipe::RecipeMetadata {
        self.source.metadata()
    }
    /// Exact cold residency namespace.
    pub fn range(&self) -> &OffloadUnitRange {
        &self.range
    }
    /// Source retained for every deferred row acquisition, with dynamic recipe
    /// inference caching disabled.
    pub fn source(&self) -> &dyn eredu_checkpoint::store::CheckpointSource {
        self.source.as_ref()
    }
    /// Whether two bindings retain the same exact prepared source and namespace.
    /// A same-shaped replacement source is not interchangeable after selection.
    pub fn same_binding(&self, other: &Self) -> bool {
        self.range == other.range
            && self.binding == other.binding
            && Arc::ptr_eq(&self.source, &other.source)
    }
    /// Constructs only this demanded row's bounded binding recipe.
    pub fn unit(
        &self,
        id: &OffloadUnitId,
    ) -> Result<Option<OffloadUnit>, ResidencyControllerError> {
        let Some(member) = self.range.member(id) else {
            return Ok(None);
        };
        let plan = self.source.plan(
            &[member - self.range.start()],
            RowReadLimits {
                requests: 1,
                rows_per_read: 1,
            },
        )?;
        let recipe = plan.reads()[0].recipe().clone();
        let binding = WeightBinding::from_recipe(&self.binding, recipe, self.range.bytes())?;
        Ok(Some(OffloadUnit::new(id.clone(), [binding])?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_checkpoint::{
        recipe::DerivedWeightRecipe,
        store::{MemoryWeightStore, SharedCheckpointSource},
    };
    use eredu_core::residency::{OffloadConfig, ResidencyPolicy};
    fn fixture() -> (
        SharedCheckpointSource,
        Arc<PreparedRowSource>,
        OffloadUnitRange,
    ) {
        let source: SharedCheckpointSource = Arc::new(
            MemoryWeightStore::from_safetensors([(
                "table".into(),
                safetensors::Dtype::F32,
                vec![8, 2],
                vec![0; 64],
            )])
            .unwrap(),
        );
        let table = PreparedRowSource::new(
            source.clone(),
            DerivedWeightRecipe::source("table", TensorSelection::Full),
        )
        .unwrap();
        let range = OffloadUnitRange::new(
            OffloadUnitId::new("rows").unwrap(),
            10,
            18,
            8,
            ResidencyPolicy::Cacheable,
        )
        .unwrap();
        (source, table, range)
    }
    #[test]
    fn row_catalog_matches_plan_and_retains_unpublished_reservations() {
        let (source, table, range) = fixture();
        let plan = OffloadPlan::with_ranges(
            OffloadConfig::new(Some(16), Some(16), 1).unwrap(),
            [],
            [range.clone()],
        )
        .unwrap();
        assert!(matches!(
            ResidencyController::new_with_row_ranges(|_| source.as_ref(), plan.clone(), [], vec![]),
            Err(ResidencyControllerError::RangeCatalogMismatch)
        ));
        let rows = RowResidencyRange::new(range.clone(), table, "value").unwrap();
        let mut controller =
            ResidencyController::new_with_row_ranges(|_| source.as_ref(), plan, [], vec![rows])
                .unwrap();
        assert_eq!(controller.units().len(), 0);
        let id = range.member_id(13).unwrap();
        let invalid = OffloadUnitId::new("rows.999").unwrap();
        assert!(controller.prepare_units(&[id.clone(), invalid]).is_err());
        assert_eq!(
            controller.units().len(),
            0,
            "failed preparation must be atomic"
        );
        controller.prepare_units(std::slice::from_ref(&id)).unwrap();
        controller.ledger_mut().mark_initialized();
        controller
            .ledger_mut()
            .reserve_copy(&id, MemoryTier::Device, 8, &BTreeSet::new())
            .unwrap();
        assert!(
            !controller.retire_unit_if_unused(&id).unwrap(),
            "reserved storage has no published copy yet"
        );
        controller
            .ledger_mut()
            .publish_reserved(&id, MemoryTier::Device, 8, None)
            .unwrap();
        assert!(!controller.retire_unit_if_unused(&id).unwrap());
        controller
            .ledger_mut()
            .evict(&id, MemoryTier::Device)
            .unwrap();
        assert!(controller.retire_unit_if_unused(&id).unwrap());
        assert_eq!(controller.units().len(), 0);
        assert!(controller.ledger().unit_reports().is_empty());
    }
    #[test]
    fn malformed_row_catalog_does_not_create_bindings() {
        let (_, table, range) = fixture();
        let bad = OffloadUnitRange::new(
            range.prefix().clone(),
            10,
            18,
            16,
            ResidencyPolicy::Cacheable,
        )
        .unwrap();
        assert!(matches!(
            RowResidencyRange::new(bad, table.clone(), "value"),
            Err(ResidencyControllerError::RowRangeGeometry { .. })
        ));
        assert!(RowResidencyRange::new(range, table, " ").is_err());
    }
}
