//! Logical expert and routed-unit geometry shared by discovery and execution.

use serde::{Deserialize, Serialize};

/// Global bank geometry, independent of compact native bank indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoutedUnitGeometry {
    /// Checkpoint-global expert count.
    pub experts: u64,
    /// Scalar units per expert.
    pub units_per_expert: u64,
    /// Original selected route slots per token; duplicate experts retain their slots.
    pub routes_per_token: u64,
}
/// A routed bank cannot be represented by a nonzero component extent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum RoutedGeometryError {
    /// At least one logical extent is zero.
    #[error("empty routed-unit geometry")]
    Empty,
    /// The expert and unit product exceeds the component index range.
    #[error("routed-unit geometry overflow")]
    Overflow,
}

impl RoutedUnitGeometry {
    /// Number of virtual components in the global bank.
    pub fn component_extent(self) -> Result<u64, RoutedGeometryError> {
        if self.experts == 0 || self.units_per_expert == 0 || self.routes_per_token == 0 {
            return Err(RoutedGeometryError::Empty);
        }
        self.experts
            .checked_mul(self.units_per_expert)
            .ok_or(RoutedGeometryError::Overflow)
    }
}
