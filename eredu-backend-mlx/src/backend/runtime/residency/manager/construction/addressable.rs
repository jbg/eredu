//! Load-time source bank without a fabricated target execution layout.
use super::*;
use eredu_checkpoint::store::RetainedCheckpointSource;
use eredu_core::residency::ResidencyPolicy;

impl ResidencyManager {
    /// Retain the ordinary finite disk/read/materialization source for every
    /// independently addressable bank member. This does not acquire a device
    /// window, choose members or grant request operation authority.
    pub(crate) fn prepare_original_addressable(
        primary: RetainedCheckpointSource,
        plan: &OffloadPlan,
        units: &[OffloadUnit],
        source_stream: &Stream,
        execution_stream: &Stream,
        pool: &WorkingMemoryPool,
    ) -> Result<Option<Self>, OriginalManagerError> {
        if units.is_empty() && plan.units().is_empty() {
            return Ok(None);
        }
        // These are the actual source-bank assignments. A source-only entry
        // cannot silently materialize a pinned target or relabel a host bank.
        if plan.units().len() != units.len()
            || plan.units().iter().any(|unit| {
                unit.policy() != ResidencyPolicy::Cacheable
                    || unit.tier() != MemoryTier::Disk
                    || !units.iter().any(|definition| definition.id() == unit.id())
            })
        {
            return Err(OriginalManagerError(PreparationFailure::Policy(
                WorkingMemoryError::IdentityMismatch,
            )));
        }
        // With no selected target IDs, SupplementarySourcePlan retains all
        // actual members through its existing singleton source-acquire graph.
        // The same initializer authenticates CPU source and the selected CPU/Metal execution streams,
        // captures foreground descriptors and catalogs, pays manager storage,
        // installs ordinary materialization resources, and initializes the
        // supplementary source. Only target-specific setup is absent.
        Self::prepare_original_layerwise_impl(
            primary, BTreeMap::new(), plan, units, &[], None,
            source_stream, execution_stream, pool, true,
        )
    }
}
