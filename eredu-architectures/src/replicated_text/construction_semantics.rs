//! Aliases of semantic owners from a completed exact prepared construction.

use super::config_source::EffectiveConfig;

/// Inline slots owned by the existing prepared-source allocation. Neither a
/// configuration payload nor another outer shared allocation is created here.
#[derive(Default)]
#[cfg_attr(test, derive(Clone))]
pub(crate) struct PreparedConstructionSemantics {
    // Success of the existing full validator for this exact immutable target.
    pub(super) store_handoff: std::sync::OnceLock<()>,
    pub(super) config: std::sync::OnceLock<EffectiveConfig>,
    pub(super) composite: std::sync::OnceLock<super::composite_source::Completed>,
    pub(super) capability: std::sync::OnceLock<crate::capability::CapabilityEstimate>,
    pub(crate) direct_partition: std::sync::OnceLock<crate::prepared_execution::PreparedDirectPartitionSource>,
    pub(crate) routed: std::sync::OnceLock<crate::routed_text::CompletedRoutedConstruction>,
}
