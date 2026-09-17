//! Borrowed sparse invocation ownership from the retained architecture table.
use super::*;
use eredu_core::capture::{RoutedUnitCaptureOwnership, RoutedUnitGeometry};
use std::mem::{size_of, size_of_val};

/// Fixed source refusals; inspecting a declaration grants no native authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum RoutedPartitionCaptureSourceError {
    /// The observation is absent from a retained rank.
    #[error("routed capture is absent from its retained rank table")]
    Missing,
    /// Retained ranks disagree about the invocation or bank.
    #[error("routed invocation declarations disagree between ranks")]
    Declaration,
    /// Bank dimensions or a source's expert/unit/peer coordinates are invalid.
    #[error("routed capture has invalid retained bank or source coordinates")]
    Geometry,
    /// No rank owns an invocation that can acknowledge the selection.
    #[error("routed capture has no selected invocation owner")]
    Producers,
}
impl RoutedPartitionCaptureSourceError {
    pub(super) fn legacy(self, path: &str) -> eredu_core::capture::CaptureError {
        match self {
            Self::Missing => eredu_core::capture::CaptureError::MissingPath(path.into()),
            _ => eredu_core::capture::CaptureError::Invalid(self.to_string()),
        }
    }
}

/// One actual source. A replica still executes its invocation; an empty expert
/// or unit map remains a producer of the ordinary empty acknowledgment.
#[derive(Debug, Clone, Copy)]
pub struct RoutedPartitionCaptureRank<'a> {
    /// Actual world rank of this invocation.
    pub rank: usize,
    /// Exact retained expert/unit and source-peer ownership.
    pub ownership: &'a RoutedUnitCaptureOwnership,
    /// Whether this source exports a fragment or empty acknowledgment.
    pub produces: bool,
}

/// Immutable loan of the same rank table used by ordinary sparse capture.
/// Checkpoint ownership, expert routing and mutable native rows stay distinct.
/// Projection, complete coverage, scalar representation and completion require
/// their own existing source and receipt checks.
#[derive(Debug)]
pub struct RoutedPartitionCaptureSource<'a> {
    layouts: &'a ComponentPartitionLayouts,
    path: &'a str,
    first: &'a PartitionedRoutedObservation,
    sources: usize,
    producers: usize,
}
impl<'a> RoutedPartitionCaptureSource<'a> {
    /// Canonical provider invocation, distinct from this observation path.
    pub fn routing(&self) -> &'a str {
        self.first.routing()
    }
    /// Whether this observes effective values before down projection.
    pub fn effective(&self) -> bool {
        self.first.effective()
    }
    /// Exact global expert, route-slot and scalar-unit dimensions.
    pub fn geometry(&self) -> RoutedUnitGeometry {
        self.first.geometry()
    }
    /// Actual provider input width before route expansion.
    pub fn input_width(&self) -> u64 {
        self.first.input_width()
    }
    /// Complete retained topology, including idle pipeline ranks.
    pub fn world_size(&self) -> usize {
        self.layouts.layouts.len()
    }
    /// Number of actual invocations, including replicas and empty owners.
    pub const fn source_count(&self) -> usize {
        self.sources
    }
    /// Number of canonical producers, including empty acknowledgments.
    pub const fn producer_count(&self) -> usize {
        self.producers
    }
    /// Borrow this rank's actual invocation, or None for an idle/absent rank.
    pub fn rank(&self, rank: usize) -> Option<RoutedPartitionCaptureRank<'a>> {
        let ownership = self
            .layouts
            .layouts
            .get(rank)?
            .routed_observation(self.path)?
            .ownership()?;
        Some(RoutedPartitionCaptureRank {
            rank,
            ownership,
            produces: self
                .layouts
                .is_routed_capture_producer(self.path, rank, ownership),
        })
    }
    /// Fixed lookup, comparison and return controls. The consumer separately
    /// funds any owned projection or copied coordinate table before constructing it.
    pub fn control_bytes() -> Option<usize> {
        let parts = [
            size_of::<Self>() * 2,
            size_of::<RoutedPartitionCaptureRank<'a>>() * 2,
            size_of::<Option<RoutedPartitionCaptureRank<'a>>>(),
            size_of::<Result<Self, RoutedPartitionCaptureSourceError>>(),
            size_of::<RoutedPartitionCaptureSourceError>(),
            size_of::<(
                &ComponentPartitionLayouts,
                &str,
                usize,
                &RoutedUnitCaptureOwnership,
            )>(),
            size_of::<(usize, usize, RoutedUnitGeometry, bool, Option<u64>)>(),
            size_of::<Option<&PartitionedRoutedObservation>>() * 2,
            size_of::<Option<&RoutedUnitCaptureOwnership>>() * 2,
            size_of::<std::slice::Iter<'a, ComponentPartitionLayout>>() * 2,
            size_of::<std::iter::Enumerate<std::slice::Iter<'a, ComponentPartitionLayout>>>() * 2,
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
}
impl ComponentPartitionLayouts {
    // Same ordinary rule: lowest-rank nonempty equal coordinate map exports;
    // every empty owner explicitly acknowledges. All source hooks remain live.
    fn is_routed_capture_producer(
        &self,
        path: &str,
        rank: usize,
        ownership: &RoutedUnitCaptureOwnership,
    ) -> bool {
        let coordinates = &ownership.coordinates;
        coordinates.experts().local_count() == 0
            || coordinates.units().local_count() == 0
            || !self.layouts[..rank].iter().any(|layout| {
                layout
                    .routed_observation(path)
                    .and_then(PartitionedRoutedObservation::ownership)
                    .is_some_and(|earlier| earlier.coordinates == *coordinates)
            })
    }

    /// Inspect the retained sparse source and ordinary replica decision without
    /// allocating a second table, selecting native resources or admitting work.
    pub fn routed_capture_source<'a>(
        &'a self,
        path: &'a str,
    ) -> Result<RoutedPartitionCaptureSource<'a>, RoutedPartitionCaptureSourceError> {
        use RoutedPartitionCaptureSourceError as E;
        let first = self
            .layouts
            .first()
            .and_then(|layout| layout.routed_observation(path))
            .ok_or(E::Missing)?;
        let geometry = first.geometry();
        if geometry.experts == 0
            || geometry.units_per_expert == 0
            || geometry.routes_per_token == 0
            || first.input_width() == 0
            || geometry
                .experts
                .checked_mul(geometry.units_per_expert)
                .is_none()
        {
            return Err(E::Geometry);
        }
        let (mut sources, mut producers) = (0, 0);
        for (rank, layout) in self.layouts.iter().enumerate() {
            let observation = layout.routed_observation(path).ok_or(E::Missing)?;
            if observation.routing() != first.routing()
                || observation.effective() != first.effective()
                || observation.geometry() != geometry
                || observation.input_width() != first.input_width()
            {
                return Err(E::Declaration);
            }
            if let Some(ownership) = observation.ownership() {
                if ownership.coordinates.experts().global_count() as u64 != geometry.experts
                    || ownership.coordinates.units().global_count() as u64
                        != geometry.units_per_expert
                    || ownership.source_peers == 0
                    || match ownership.source_peer {
                        Some(peer) => peer >= ownership.source_peers,
                        None => ownership.source_peers != 1,
                    }
                {
                    return Err(E::Geometry);
                }
                sources += 1;
                producers += usize::from(self.is_routed_capture_producer(path, rank, ownership));
            }
        }
        if producers == 0 {
            return Err(E::Producers);
        }
        Ok(RoutedPartitionCaptureSource {
            layouts: self,
            path,
            first,
            sources,
            producers,
        })
    }
}
