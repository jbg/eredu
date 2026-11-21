//! Indexed tensor movement required by grouped operators.

use crate::{GroupedNeuralBackend, Tensor};

/// Generic indexed tensor movement required by bounded grouped execution.
///
/// Implementations expose integer-index discovery and tensor movement without
/// receiving architecture plans, bank meaning, or text lifecycle policy.
pub trait IndexedMovement<B>
where
    B: GroupedNeuralBackend,
{
    /// Indexed movement failure.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Returns deterministic demand counts for integer indices below `upper_bound`.
    fn index_demands(
        &mut self,
        indices: &B::Tensor,
        upper_bound: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Vec<(usize, u64)>, Self::Error>;

    /// Rewrites source indices through one exact source-to-compact mapping.
    fn remap_indices(
        &mut self,
        indices: &B::Tensor,
        mapping: &[(usize, usize)],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>;

    /// Selects a contiguous range along the leading row axis.
    fn select_rows(
        &mut self,
        value: &B::Tensor,
        start: usize,
        end: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>;

    /// Concatenates row partitions in their original order.
    fn concatenate_rows(
        &mut self,
        values: &[B::Tensor],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>;
}

/// Backend-neutral tensor movement needed by an expert-exchange protocol.
///
/// Architecture code supplies already validated row and flattened-route
/// indices. Implementations retain tensor storage and completion ownership;
/// they do not receive expert identities, topology, or model-family policy.
pub trait ExpertRouteTensorMovement<T> {
    /// Tensor movement failure.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Returns the logical tensor shape without materializing its values.
    fn shape(&self, value: &T) -> Vec<usize>;

    /// Duplicates and reorders leading-axis rows in the supplied order.
    fn gather_rows(&mut self, value: &T, rows: &[usize]) -> Result<T, Self::Error>;

    /// Selects flattened route scalars and returns them as `[routes, 1]`.
    fn gather_route_values(
        &mut self,
        value: &T,
        flattened_routes: &[usize],
    ) -> Result<T, Self::Error>;

    /// Additively combines route rows into their architecture source rows.
    /// With sequential reduction, consume each destination's contributions in
    /// supplied row order, rounding to the input dtype after every addition.
    ///
    /// Every input row must be consumed exactly once. Repeated destination
    /// rows are intentional and implement weighted routed-expert summation.
    fn scatter_add_rows(
        &mut self,
        value: T,
        destination_rows: &[usize],
        output_rows: usize,
        reduction: crate::GroupReduction,
    ) -> Result<T, Self::Error>;
}
