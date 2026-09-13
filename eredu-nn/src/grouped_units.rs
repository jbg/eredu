//! Sparse unit boundaries inside selected grouped projections.
use crate::{Error, Tensor};

/// Units actually evaluated for a selected route chunk, before its down projection.
/// These tensors are borrowed from ordinary execution; observing them grants no
/// permission to retain, evaluate, copy, or communicate them without admission.
///
/// Rows are sorted by local bank group, not by token. For row `r`, `group_indices[r]`
/// identifies the local parameter group and `selection_indices[r]` identifies the
/// original flattened entry in `coefficients`. `token_indices[r] + token_offset`
/// is the token row of the full operator request. Routing providers must map these
/// coordinates through their own compact-bank and exchange maps before claiming
/// global expert or sequence identities.
pub struct GroupedUnitBatch<'a, T> {
    /// Actual activated/gated values `[selected_routes, local_units]`.
    pub values: &'a T,
    /// Local parameter group per sorted row, `[selected_routes]`.
    pub group_indices: &'a T,
    /// Original flattened top-k selection per sorted row, `[selected_routes]`.
    pub selection_indices: &'a T,
    /// Chunk-relative source token per sorted row, `[selected_routes]`.
    pub token_indices: &'a T,
    /// Original route coefficients, `[chunk_tokens, top_k]`, before weighting.
    pub coefficients: &'a T,
    /// Start token row of this chunk in the operator's complete input.
    pub token_offset: usize,
    /// Token rows in the operator's complete input, before chunking.
    pub total_token_count: usize,
    /// Number of groups in the local parameter bank, including unselected groups.
    pub group_count: usize,
}

impl<T> GroupedUnitBatch<'_, T> {
    /// Keeps route coordinates while borrowing another value tensor.
    pub fn with_values<'a>(&'a self, values: &'a T) -> GroupedUnitBatch<'a, T> {
        GroupedUnitBatch {
            values,
            group_indices: self.group_indices,
            selection_indices: self.selection_indices,
            token_indices: self.token_indices,
            coefficients: self.coefficients,
            token_offset: self.token_offset,
            total_token_count: self.total_token_count,
            group_count: self.group_count,
        }
    }
}

/// Observation and modification of the actual selected units consumed downstream.
/// Hooks run in order: original observation, optional intervention, effective
/// observation, then down projection. A failure stops that projection. Inactive
/// routes have no rows and must never be represented as measured zero. Adding or
/// replacing units on an absent route requires a separately admitted routing
/// change; an observer must not silently count it as an applied intervention.
pub trait GroupedUnitObserver<T: Tensor> {
    /// Borrows original values before any intervention.
    fn observe(&mut self, batch: &GroupedUnitBatch<'_, T>) -> Result<(), Error>;
    /// Replaces the entire selected value view while preserving its exact shape
    /// and dtype. An admitted observer may modify only its selected coordinates.
    /// `None` preserves the original tensor without a copy.
    fn intervene(&mut self, _batch: &GroupedUnitBatch<'_, T>) -> Result<Option<T>, Error> {
        Ok(None)
    }
    /// Borrows effective values before the down projection's input transform.
    /// For input-quantizing projections these are not the transformed multiply
    /// operands; this hook must not be advertised as that separate evidence.
    fn observe_effective(&mut self, _batch: &GroupedUnitBatch<'_, T>) -> Result<(), Error> {
        Ok(())
    }
}

/// Exact failures at a selected-unit mechanism boundary.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GroupedUnitError {
    /// This mechanism has no internal selected-unit hook.
    #[error("selected grouped-unit observation is not implemented by this mechanism")]
    Unavailable,
    /// A replacement would change route or unit geometry.
    #[error("grouped-unit replacement shape {actual:?} differs from {expected:?}")]
    ReplacementShape {
        /// Original shape.
        expected: Vec<i32>,
        /// Attempted shape.
        actual: Vec<i32>,
    },
    /// A replacement would change the selected arithmetic dtype.
    #[error("grouped-unit replacement dtype differs from the original")]
    ReplacementDtype,
}
