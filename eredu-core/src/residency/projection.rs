//! Backend-supplied projection storage facts. These records carry no native
//! handles or device-selection policy; absent coverage never implies zero cost.
use crate::checkpoint::TensorDtype;
use serde::{Deserialize, Serialize};

/// A row interval over which one selected mechanism avoids full-weight promotion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectionRowStorage {
    /// First covered logical row count, inclusive.
    pub first: u64,
    /// Last covered logical row count, inclusive.
    pub last: u64,
    /// Backend mechanism identifier for diagnostic attribution.
    pub mechanism: String,
    /// Exact logical native partial-result payload per row (zero for unsplit GEMM).
    pub partial_bytes_per_row: u64,
}

/// Facts for the current immutable resident parameter binding. The backend must
/// establish dtype, final weight layout and device eligibility without execution.
/// They apply only to ordinary dense projection, never routed/packed operations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectionStorageFacts {
    /// Exact bound matrix input width.
    pub input: u64,
    /// Exact bound matrix output width.
    pub output: u64,
    /// Activation representation covered by these facts.
    pub activation: TensorDtype,
    /// Current narrow storage representation.
    pub weight: TensorDtype,
    /// Output element bytes; outputs are accounted separately by topology.
    pub output_scalar_bytes: u64,
    /// Layout-dependent activation copy upper payload per row; lower is zero.
    /// Forecasting does not know the eventual activation strides.
    pub activation_copy_bytes_per_row: u64,
    /// Disjoint coverage intervals. All other rows retain the previous fallback.
    pub rows: Vec<ProjectionRowStorage>,
}
impl ProjectionStorageFacts {
    /// Reject malformed coverage rather than silently treating it as a bound.
    pub fn validate(&self) -> Result<(), crate::CapabilityError> {
        let mut previous = 0;
        for range in &self.rows {
            if self.input == 0
                || self.output == 0
                || self.output_scalar_bytes == 0
                || range.first <= previous
                || range.last < range.first
                || range.mechanism.is_empty()
            {
                return Err(crate::CapabilityError::InvalidConfiguration {
                    field: "projection storage facts",
                    detail: "invalid or overlapping projection coverage".into(),
                });
            }
            previous = range.last;
        }
        Ok(())
    }

    /// Exact matching is required; a fact for another shape or dtype gives no credit.
    pub fn for_invocation(
        &self,
        input: u64,
        output: u64,
        activation: &TensorDtype,
        rows: u64,
    ) -> Option<&ProjectionRowStorage> {
        if self.input != input || self.output != output || &self.activation != activation {
            return None;
        }
        self.rows
            .iter()
            .find(|range| range.first <= rows && rows <= range.last)
    }
}
