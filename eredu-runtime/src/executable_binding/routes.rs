//! Typed bundles of model binding capabilities.

/// An explicitly absent optional construction capability.
#[derive(Clone, Copy, Debug, Default)]
pub struct Unavailable;
/// Typed construction routes with optional capabilities absent by default.
pub struct PreparedExecutionRoutes<
    R = Unavailable,
    T = Unavailable,
    X = Unavailable,
    PD = Unavailable,
    PT = Unavailable,
    PX = Unavailable,
> {
    /// Ordinary replicated text route.
    pub replicated: R,
    /// Routed text route.
    pub routed: T,
    /// Composite text route.
    pub composite: X,
    /// Dense rank-local route.
    pub partitioned_dense: PD,
    /// Routed rank-local route.
    pub partitioned_routed: PT,
    /// Composite rank-local route.
    pub partitioned_composite: PX,
}

impl<R, T, X, PD, PT, PX> PreparedExecutionRoutes<R, T, X, PD, PT, PX> {
    /// Assembles ordinary and partitioned binding capabilities for a prepared model.
    pub(super) fn from_parts(ordinary: (R, T, X), partitioned: (PD, PT, PX)) -> Self {
        let (replicated, routed, composite) = ordinary;
        let (partitioned_dense, partitioned_routed, partitioned_composite) = partitioned;
        Self {
            replicated,
            routed,
            composite,
            partitioned_dense,
            partitioned_routed,
            partitioned_composite,
        }
    }
}
