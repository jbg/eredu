//! Sparse participation coordinates, independent of native sorting or routing.
use super::*;

/// Exact bank and route provider for a sparse scalar intervention target.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoutedUnitInterventionPoint {
    /// Architecture-owned routing invocation path.
    pub routing: String,
    /// Global bank geometry and per-token route cardinality.
    pub geometry: RoutedUnitGeometry,
}
impl RoutedUnitInterventionPoint {
    pub(super) fn validate(&self, point: &InterventionPoint) -> Result<(), CaptureError> {
        require(self.geometry.experts != 0 && self.geometry.units_per_expert != 0
            && self.geometry.routes_per_token != 0, "empty routed-unit geometry")?;
        let count = self.geometry.components()?;
        require(
            !self.routing.is_empty()
                && self.routing.len() <= 1024
                && point.routing.is_none()
                && point.stage == InterventionStage::Activation
                && point.axes.len() == 2
                && point.axes[0].name == "token"
                && point.axes[0].dimension == crate::SymbolicDimension::TokenRows
                && point.axes[1].name == "component"
                && matches!(point.axes[1].dimension, crate::SymbolicDimension::Known(n) if n as u64 == count),
            "invalid sparse intervention declaration",
        )
    }
}

/// Coordinates for one native unit row. Multiple slots may select the same expert.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoutedUnitLocation {
    /// Sending peer for exchanged rows; absent for ordinary or replicated input.
    /// A coordinate alone does not establish exchange ownership or completion.
    pub source_peer: Option<u64>,
    /// Invocation-global token row.
    pub token: u64,
    /// Original top-k slot, before native sorting.
    pub slot: u64,
    /// Checkpoint-global expert ordinal.
    pub expert: u64,
}

/// One native chunk, in native value-row order; no values are exported.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutedUnitLocations {
    /// Contiguous half-open native input range (receive order after exchange).
    pub source_token_range: [u64; 2],
    /// Exactly one row per native token/slot; coordinates retain original sources.
    pub rows: Vec<RoutedUnitLocation>,
}

/// Fixed-size progress retained with an operation's ordinary diagnostic record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoutedUnitInterventionReceipt {
    /// Full invocation token count, including unselected token positions.
    pub source_tokens: u64,
    /// Contiguous token rows verified and processed from zero.
    pub completed_tokens: u64,
    /// Actual participating scalar values addressed. Compact masks count only
    /// removed values; inactive experts contribute no count.
    pub affected_values: u64,
}
