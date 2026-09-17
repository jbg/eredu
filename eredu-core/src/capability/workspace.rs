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
            Self::Unknown { .. } => None,
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
