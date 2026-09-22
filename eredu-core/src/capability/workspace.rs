//! Execution workspace accompanying the persistent-state estimate.

use crate::{CapabilityError, InferenceGeometry};
use serde::{Deserialize, Serialize};

/// A defensible upper bound for one selected mechanism, or an explicit gap.
/// Allocator observations and arbitrary safety reserves are not bounds.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "coverage", rename_all = "snake_case")]
pub enum WorkspaceBound {
    /// Maximum bytes alive simultaneously, with the derivation's assumptions.
    Bounded {
        /// Enforceable upper bound, including native temporary allocations.
        bytes: u64,
        /// Selected implementation, representation and lifetime assumptions.
        assumptions: String,
    },
    /// Complete per-domain requirements are available, while their aggregate
    /// cannot be represented by one u64 diagnostic. This grants no authority
    /// without the matching attributed report.
    PerDomain {
        /// Selected placement and simultaneous lifetime estimation basis.
        assumptions: String,
    },
    /// The selected implementation has no proved bound for this component.
    Unknown {
        /// Missing mechanism or geometry detail.
        reason: String,
    },
}

impl WorkspaceBound {
    /// Declares a known bound. Zero must mean an inapplicable or shared
    /// allocation accounted for elsewhere, documented by `assumptions`.
    pub fn bounded(bytes: u64, assumptions: impl Into<String>) -> Self {
        Self::Bounded {
            bytes,
            assumptions: assumptions.into(),
        }
    }

    /// Returns a proved bound; unknown must never become zero in strict policy.
    pub const fn bytes(&self) -> Option<u64> {
        match self {
            Self::Bounded { bytes, .. } => Some(*bytes),
            Self::Unknown { .. } | Self::PerDomain { .. } => None,
        }
    }
}

/// Simultaneously live managed execution storage beyond persistent state,
/// including tensor buffers and disjoint operation-owned host workspace.
/// Every component is mandatory: a missing bound prevents strict admission.
/// Unified-memory aliases must appear in exactly one component.
/// Runtime/driver bookkeeping, allocator caches, JIT programs and unrelated
/// process memory are not guaranteed by an Eredu-managed working-memory bound.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExecutionWorkspaceEstimate {
    /// Physical attribution from the same selected allocation-lifetime trace.
    /// Process-local topology identities cannot be restored from diagnostics.
    #[serde(skip)]
    pub physical_domains: Option<DomainExecutionWorkspaceEstimate>,
    /// Exact request and chunk geometry priced by these bounds.
    pub geometry: InferenceGeometry,
    /// Layer inputs, outputs, MLP intermediates and recurrent work arrays.
    pub activations: WorkspaceBound,
    /// Masks, attention intermediates and native attention kernel scratch.
    pub attention: WorkspaceBound,
    /// Selected vocabulary rows, sampling and retained verification scores.
    pub vocabulary: WorkspaceBound,
    /// Cache growth, copies and rollback versions beyond the final state.
    pub state_update: WorkspaceBound,
    /// Materialization, transfers and collective scratch beyond retained weights.
    pub materialization: WorkspaceBound,
    /// Prediction, observation and snapshot resources not counted elsewhere.
    pub retained: WorkspaceBound,
}

/// Simultaneous execution requirements resolved before allocation sums.
/// Missing this report is an attribution gap under every limit mode.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct DomainExecutionWorkspaceEstimate {
    /// Exact request geometry priced by the selected mechanism.
    pub geometry: InferenceGeometry,
    /// Complete activation lifetimes, retaining candidate-placement allowances.
    pub activations: crate::DomainMemoryRequirements,
    /// Attention allocations and scratch beyond activation storage.
    pub attention: crate::DomainMemoryRequirements,
    /// Vocabulary and sampling allocations.
    pub vocabulary: crate::DomainMemoryRequirements,
    /// Additional state growth, copies and rollback versions.
    pub state_update: crate::DomainMemoryRequirements,
    /// Materialization, transfers and communication allocations.
    pub materialization: crate::DomainMemoryRequirements,
    /// Observation, host controls and other retained request storage.
    pub retained: crate::DomainMemoryRequirements,
}
impl DomainExecutionWorkspaceEstimate {
    /// Checked simultaneous sum independently in every physical domain.
    pub fn requirements(
        &self,
    ) -> Result<crate::DomainMemoryRequirements, crate::MemoryDomainError> {
        self.activations
            .checked_add(&self.attention)?
            .checked_add(&self.vocabulary)?
            .checked_add(&self.state_update)?
            .checked_add(&self.materialization)?
            .checked_add(&self.retained)
    }
}

/// Actual persistent and media allocation placement for one selected request.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct DomainRuntimeStateEstimate {
    /// Exact requested state/media geometry.
    pub geometry: InferenceGeometry,
    /// Complete canonical retained state graph, including capacity and distinct
    /// copies. A complete media traversal includes its retained source/cache
    /// roots here; the separate media terms then contain only additional owners.
    pub decoder_state: crate::DomainMemoryRequirements,
    /// Additional media embeddings outside the selected canonical state graph.
    pub media_embeddings: crate::DomainMemoryRequirements,
    /// Additional media execution overlap outside the selected equation trace.
    pub media_workspace: crate::DomainMemoryRequirements,
}
impl DomainRuntimeStateEstimate {
    /// Checked simultaneous sum without an aggregate physical-memory amount.
    pub fn requirements(
        &self,
    ) -> Result<crate::DomainMemoryRequirements, crate::MemoryDomainError> {
        self.decoder_state
            .checked_add(&self.media_embeddings)?
            .checked_add(&self.media_workspace)
    }
}

impl ExecutionWorkspaceEstimate {
    /// Conservative simultaneous peak. Media workspace is already represented
    /// in `RuntimeStateEstimate`; these components must not include it again.
    pub fn peak_bytes(&self) -> Result<Option<u64>, CapabilityError> {
        self.peak_bytes_fixed().map_err(Into::into)
    }

    /// The same mandatory-component arithmetic with allocation-free failures.
    pub fn peak_bytes_fixed(&self) -> Result<Option<u64>, crate::AdmissionPolicyError> {
        super::ExecutionWorkspaceRequirements::from(self).peak_bytes()
    }
}
