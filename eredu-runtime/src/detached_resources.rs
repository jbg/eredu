//! Bounded observations of detached work; observation never grants release authority.
use crate::resource_lifetimes::{
    compose_resource_peaks, ResourceLifetimeEvent, ResourceLifetimePlan,
};
use eredu_core::resources::{
    ResourceCoverage, ResourceDescriptionError, ResourceLifetimeDescription,
};

use eredu_core::resources::detached::DetachedResourceObservation;
pub use eredu_core::resources::detached::{DetachedResourceLimits, DetachedResourceScan};

/// Immutable, validated snapshot of one producer's detached holdings.
/// Multiple records may retain the same backing/owner. Consumers must compose
/// identities and lifetimes rather than summing records. Empty/truncated/unknown
/// observations cannot prove reclaimed storage or refund execution budgets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DetachedResourceInventory {
    source: String,
    limits: DetachedResourceLimits,
    inspected_nodes: usize,
    undescribed_nodes: usize,
    scan: DetachedResourceScan,
    descriptions: Vec<ResourceLifetimeDescription>,
}
impl DetachedResourceInventory {
    /// Validates and composes the observation from one native producer.
    pub fn from_observation(
        limits: DetachedResourceLimits,
        observation: DetachedResourceObservation,
    ) -> Result<Self, ResourceDescriptionError> {
        let DetachedResourceObservation {
            source,
            inspected_nodes,
            undescribed_nodes,
            scan,
            descriptions,
        } = observation;
        Self::new(
            source,
            limits,
            inspected_nodes,
            undescribed_nodes,
            scan,
            descriptions,
        )
    }

    /// Validates reporting bounds, counts, descriptions and shared backing facts.
    /// Each description preserves its coexistence declaration; retaining owners
    /// may also be held elsewhere.
    fn new(
        source: String,
        limits: DetachedResourceLimits,
        inspected_nodes: usize,
        undescribed_nodes: usize,
        scan: DetachedResourceScan,
        descriptions: Vec<ResourceLifetimeDescription>,
    ) -> Result<Self, ResourceDescriptionError> {
        if source.trim().is_empty()
            || inspected_nodes > limits.nodes()
            || undescribed_nodes.checked_add(descriptions.len()) != Some(inspected_nodes)
        {
            return Err(invalid(
                "detached resource source, counts or node limit are invalid",
            ));
        }
        if let DetachedResourceScan::Unavailable { reason } = &scan {
            if reason.trim().is_empty() || inspected_nodes != 0 {
                return Err(invalid(
                    "unavailable detached scan cannot contain inspected records",
                ));
            }
        }
        let allocations = descriptions.iter().try_fold(0usize, |n, d| {
            n.checked_add(d.resources.allocations.len())
                .ok_or_else(|| invalid("detached allocation count overflowed"))
        })?;
        if allocations > limits.allocations() {
            return Err(invalid(
                "detached descriptions exceed the allocation reporting limit",
            ));
        }
        // Reuse the ordinary lifetime validator, including conflicts between
        // aliases. Validation does not release, poll or inspect native work.
        compose_resource_peaks(&ResourceLifetimePlan {
            coverage: ResourceCoverage::Partial {
                reasons: vec![
                    "detached producer snapshot excludes active work and other producers".into(),
                ],
            },
            events: descriptions
                .iter()
                .cloned()
                .map(ResourceLifetimeEvent::Acquire)
                .collect(),
        })?;
        Ok(Self {
            source,
            limits,
            inspected_nodes,
            undescribed_nodes,
            scan,
            descriptions,
        })
    }
    /// Producer/thread scope, not a checkpoint or proof of global coverage.
    pub fn source(&self) -> &str {
        &self.source
    }
    /// Retained reporting limits.
    pub const fn limits(&self) -> DetachedResourceLimits {
        self.limits
    }
    /// Nodes examined without progressing their completion.
    pub const fn inspected_nodes(&self) -> usize {
        self.inspected_nodes
    }
    /// Nodes with missing facts or descriptions exceeding the remaining allowance.
    pub const fn undescribed_nodes(&self) -> usize {
        self.undescribed_nodes
    }
    /// Traversal status, independent of allocation coverage.
    pub fn scan(&self) -> &DetachedResourceScan {
        &self.scan
    }
    /// Original scoped descriptions and retention references, including their gaps.
    pub fn descriptions(&self) -> &[ResourceLifetimeDescription] {
        &self.descriptions
    }
}
fn invalid(message: &str) -> ResourceDescriptionError {
    ResourceDescriptionError::Invalid(message.into())
}

#[cfg(test)]
mod tests;
