//! Exact detached-work observations; never grant release or progress authority.

use super::{ResourceDescriptionError, ResourceLifetimeDescription};

/// Limits on diagnostic traversal and returned allocation records, including aliases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DetachedResourceLimits {
    nodes: usize,
    allocations: usize,
}
impl DetachedResourceLimits {
    /// Both limits must be positive. These bound reporting, not native execution.
    pub fn new(nodes: usize, allocations: usize) -> Result<Self, ResourceDescriptionError> {
        if nodes == 0 || allocations == 0 {
            return Err(ResourceDescriptionError::Invalid(
                "detached resource reporting limits must be positive".into(),
            ));
        }
        Ok(Self { nodes, allocations })
    }
    /// Maximum detached records visited.
    pub const fn nodes(self) -> usize {
        self.nodes
    }
    /// Maximum allocation records returned across all descriptions, before deduplication.
    pub const fn allocations(self) -> usize {
        self.allocations
    }
}

/// Whether the declared producer's detached list was fully inspected.
/// This says nothing about physical resource coverage or completion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DetachedResourceScan {
    /// No unvisited list entries remained at observation time.
    Complete,
    /// More entries remained after the reporting limit.
    Truncated,
    /// The list could not be observed without blocking or reentrant mutation.
    Unavailable {
        /// Why an observation could not safely inspect the list.
        reason: String,
    },
}

/// One bounded observation from a native detached-work producer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DetachedResourceObservation {
    /// Producer and inspection scope.
    pub source: String,
    /// Number of nodes visited, including those without a description.
    pub inspected_nodes: usize,
    /// Number of visited nodes without resource descriptions.
    pub undescribed_nodes: usize,
    /// Traversal status at the observation boundary.
    pub scan: DetachedResourceScan,
    /// Per-node lifetime and backing facts, before portable composition.
    pub descriptions: Vec<ResourceLifetimeDescription>,
}
