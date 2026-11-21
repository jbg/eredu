//! Neutral grouped-compute operation capabilities.

/// One required grouped-compute mechanism.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum GroupedOperationRequirement {
    /// Activated selected linear projections with explicitly owned output rows.
    Linear,
    /// Ordinary grouped gated-product output.
    GatedProduct,
    /// Rank-local gated-product partial with an explicit post-reduce term.
    GatedProductTensorParallelPartial,
    /// Ordinary grouped ReLU-squared output.
    Relu2,
    /// Rank-local ReLU-squared partial with an explicit post-reduce term.
    Relu2TensorParallelPartial,
}
