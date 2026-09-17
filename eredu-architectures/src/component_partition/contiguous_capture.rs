//! Borrowed scalar ownership from the retained architecture rank table.
use super::*;
use std::{mem::{size_of, size_of_val}, ops::Range};

/// Fixed source refusals for allocation-free inspection of the retained table.
/// Neither successful inspection nor this diagnostic grants native authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum PartitionCaptureSourceError {
    /// The selected observation is absent from a retained rank.
    #[error("capture observation is absent from its retained rank table")]
    Missing,
    /// Retained ranks declare different assembly equations.
    #[error("partition observation equations disagree")]
    Equation,
    /// Additive inputs must each contain the complete ordered axis.
    #[error("additive write has incomplete ordered coordinates")]
    AdditiveCoordinates,
    /// Every TP term must have an actual local invocation.
    #[error("additive write lacks a complete tensor-parallel invocation group")]
    AdditiveGroup,
    /// Sites, axes, or source widths differ between ranks.
    #[error("component capture source axes or hook sites disagree")]
    Geometry,
    /// This exact source has no contiguous representation.
    #[error("capture source coordinates are not contiguous")]
    NonContiguous,
    /// No authoritative producer, or not all additive terms are present.
    #[error("capture source lacks its authoritative producer set")]
    Producers,
}
impl PartitionCaptureSourceError {
    pub(super) fn legacy(self, path: &str) -> eredu_core::capture::CaptureError {
        use eredu_core::capture::CaptureError;
        match self {
            Self::Missing => CaptureError::MissingPath(path.into()),
            Self::Equation => CaptureError::Invalid("partition observation equations disagree".into()),
            Self::AdditiveCoordinates => CaptureError::Invalid("additive write has incomplete ordered coordinates".into()),
            Self::AdditiveGroup => CaptureError::Invalid("additive write lacks a complete tensor-parallel invocation group".into()),
            _ => CaptureError::Invalid(self.to_string()),
        }
    }
}

/// One actual hook's physical range and ordinary authoritative-producer decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContiguousPartitionCaptureRank {
    /// World rank of the actual invocation, including replicas and empty shards.
    pub rank: usize,
    /// Contiguous physical coordinates along the retained semantic axis.
    pub coordinates: Range<usize>,
    /// Whether this rank exports a fragment or an explicit empty acknowledgment.
    pub produces: bool,
}

/// An immutable loan of the same rank table used by ordinary partition capture.
/// Sparse maps are not reinterpreted as spans, and replicas are never promoted
/// into additive terms. Projection, native scalar facts and paid delivery remain
/// separate required sources.
#[derive(Debug)]
pub struct ComponentPartitionCaptureSource<'a> {
    layouts: &'a ComponentPartitionLayouts,
    path: &'a str,
    axis: &'a str,
    width: usize,
    site: ObservationHookSite,
    combination: PartitionCaptureCombination,
    producers: usize,
}
impl<'a> ComponentPartitionCaptureSource<'a> {
    /// Exact retained observation whose exporter decision this loan describes.
    pub const fn path(&self) -> &'a str { self.path }
    /// Actual semantic axis to resolve against the admitted observation axes.
    pub const fn axis(&self) -> &str { self.axis }
    /// Global physical extent of that semantic axis.
    pub const fn width(&self) -> usize { self.width }
    /// Complete retained topology world.
    pub fn world_size(&self) -> usize { self.layouts.layouts.len() }
    /// Required architecture hook site.
    pub const fn site(&self) -> ObservationHookSite { self.site }
    /// Existing global assembly equation.
    pub const fn combination(&self) -> PartitionCaptureCombination { self.combination }
    /// Number of canonical exporting sources, including empty shards.
    pub const fn producer_count(&self) -> usize { self.producers }
    /// A local hook may exist without exporting. None means no such invocation;
    /// it does not mean an empty shard, whose zero-length range remains explicit.
    pub fn rank(&self, rank: usize) -> Option<ComponentPartitionCaptureRank<'a>> {
        let point = self.layouts.layouts.get(rank)?.observation(self.path)?;
        let coordinates = point.coordinates()?;
        Some(ComponentPartitionCaptureRank { rank, coordinates,
            produces: self.layouts.is_capture_producer(self.path, rank, self.combination) })
    }
    /// Exact fixed source lookup/iteration frames. The caller separately pays
    /// any retained output table before filling it through these borrowed rows.
    pub fn control_bytes() -> Option<usize> {
        let parts = [size_of::<Self>() * 2,
            size_of::<(&ComponentPartitionLayouts,&str)>(),
            size_of::<ComponentPartitionCaptureRank<'_>>() * 2,
            size_of::<Result<Self, PartitionCaptureSourceError>>(),
            size_of::<Option<ComponentPartitionCaptureRank<'_>>>(),
            size_of::<PartitionCaptureSourceError>(),
            size_of::<(&ComponentPartitionLayouts, &str, usize, PartitionCaptureCombination)>(),
            size_of::<(Option<&str>, Option<usize>, Option<ObservationHookSite>, usize)>(),
            size_of::<(Option<&PartitionedObservation>, Option<&ComponentCoordinateMap>, Option<Range<usize>>)>() * 2,
            size_of::<std::iter::Enumerate<std::slice::Iter<'_, ComponentPartitionLayout>>>() * 2,
            size_of::<(usize, &ComponentPartitionLayout, &PartitionedObservation, &ComponentCoordinateMap)>() * 2,
            size_of::<(eredu_core::ParallelRankTopology, eredu_core::ParallelRankTopology, bool)>(),
            size_of::<(Option<PartitionCaptureCombination>, Result<PartitionCaptureCombination, PartitionCaptureSourceError>)>(),
        ];
        parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
    }
}
/// One actual ordinary capture source; its immutable map may contain multiple
/// positive runs. Replicas and empty/absent invocations remain distinct.
#[derive(Debug,Clone,Copy)]
pub struct ComponentPartitionCaptureRank<'a> {
    /// Actual world rank, including replicas and empty shards.
    pub rank:usize,
    /// Original immutable physical-to-global component coordinates.
    pub coordinates:&'a ComponentCoordinateMap,
    /// Same canonical exporter decision used by ordinary capture.
    pub produces:bool,
}

/// Existing range-only view, validated through the same borrowed source worker.
#[derive(Debug)]
pub struct ContiguousPartitionCaptureSource<'a> { source:ComponentPartitionCaptureSource<'a> }
impl ContiguousPartitionCaptureSource<'_> {
    /// Actual semantic component axis.
    pub fn axis(&self)->&str {self.source.axis()}
    /// Global component extent.
    pub const fn width(&self)->usize {self.source.width()}
    /// Complete retained topology world.
    pub fn world_size(&self)->usize {self.source.world_size()}
    /// Actual architecture hook site.
    pub const fn site(&self)->ObservationHookSite {self.source.site()}
    /// Existing global assembly equation.
    pub const fn combination(&self)->PartitionCaptureCombination {self.source.combination()}
    /// Canonical exporting source count, including empty shards.
    pub const fn producer_count(&self)->usize {self.source.producer_count()}
    /// Actual contiguous rank source, preserving replicas and absence.
    pub fn rank(&self,rank:usize)->Option<ContiguousPartitionCaptureRank> {
        let source=self.source.rank(rank)?;
        Some(ContiguousPartitionCaptureRank {rank,coordinates:source.coordinates.contiguous_range()?,produces:source.produces})
    }
    /// Exact fixed source inspection and range-validation frames.
    pub fn control_bytes()->Option<usize> {
        let frames=[size_of::<Self>()*2,size_of::<ContiguousPartitionCaptureRank>()*2,
            size_of::<Result<Self,PartitionCaptureSourceError>>(),size_of::<Option<ContiguousPartitionCaptureRank>>(),
            size_of::<(&ComponentPartitionLayouts,&str)>(),size_of::<std::ops::Range<usize>>(),
            ComponentPartitionCaptureSource::control_bytes()?];
        frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
    }
}
impl ComponentPartitionLayouts {
    /// Existing contiguous view; explicit maps keep their own checked source.
    pub fn contiguous_capture_source<'a>(&'a self,path:&'a str)
        ->Result<ContiguousPartitionCaptureSource<'a>,PartitionCaptureSourceError> {
        let source=self.component_capture_source(path)?;
        for rank in 0..source.world_size() {
            if let Some(rank)=source.rank(rank) {
                rank.coordinates.contiguous_range().ok_or(PartitionCaptureSourceError::NonContiguous)?;
            }
        }
        Ok(ContiguousPartitionCaptureSource {source})
    }
    /// Inspect actual component maps without allocating a second coordinate
    /// table. Uses the ordinary equation and canonical replica-selection worker.
    pub fn component_capture_source<'a>(&'a self,path:&'a str)
        ->Result<ComponentPartitionCaptureSource<'a>,PartitionCaptureSourceError> {
        self.capture_component_source(path)
    }
    /// Inspect complete or contiguous rank sources without allocating a second
    /// coordinate table. Uses the ordinary equation and replica-selection worker.
    fn capture_component_source<'a>(&'a self, path: &'a str)
        -> Result<ComponentPartitionCaptureSource<'a>, PartitionCaptureSourceError>
    {
        use PartitionCaptureSourceError as E;
        let combination = self.capture_combination_source(path)?;
        let mut axis = None;
        let mut width = None;
        let mut site = None;
        let mut producers = 0usize;
        for (rank, layout) in self.layouts.iter().enumerate() {
            let point = layout.observation(path).ok_or(E::Missing)?;
            if axis.is_some_and(|prior| prior != point.axis())
                || site.is_some_and(|prior| prior != point.site()) { return Err(E::Geometry); }
            axis = Some(point.axis()); site = Some(point.site());
            let Some(map) = point.coordinates() else {
                if point.exports() { return Err(E::Producers); }
                continue;
            };
            if width.is_some_and(|prior| prior != map.global_count()) { return Err(E::Geometry); }
            width = Some(map.global_count());
            if self.is_capture_producer(path, rank, combination) { producers += 1; }
        }
        if producers == 0 || (combination == PartitionCaptureCombination::SumF64ToF32
            && producers != self.topology.tensor()) { return Err(E::Producers); }
        Ok(ComponentPartitionCaptureSource { layouts: self, path, axis: axis.ok_or(E::Missing)?,
            width: width.ok_or(E::Producers)?, site: site.ok_or(E::Missing)?, combination, producers })
    }

    // Same canonical first exporter as the ordinary prior-coordinate table.
    // Inspecting retained predecessors uses no scratch allocation. Sum keys add
    // the actual TP coordinate; Disjoint keys retain just the coordinate map.
    pub(super) fn is_capture_producer(&self, path: &str, rank: usize,
        combination: PartitionCaptureCombination) -> bool
    {
        let Some(layout) = self.layouts.get(rank) else { return false; };
        let Some(point) = layout.observation(path) else { return false; };
        let Some(map) = point.coordinates() else { return false; };
        point.exports() && !self.layouts[..rank].iter().any(|prior| {
            prior.observation(path).is_some_and(|other| other.exports()
                && other.coordinates() == Some(map)
                && (combination == PartitionCaptureCombination::Disjoint
                    || prior.topology.tensor_parallel_rank() == layout.topology.tensor_parallel_rank()))
        })
    }

    // Shared equation validation. Public ordinary callers retain their original
    // formatted errors; funded inspection receives this fixed diagnostic. Peer
    // membership uses actual non-Tensor topology coordinates, exactly as the
    // existing tensor subgroup constructor, without creating its Vec.
    pub(super) fn capture_combination_source(&self, path: &str)
        -> Result<PartitionCaptureCombination, PartitionCaptureSourceError>
    {
        use PartitionCaptureSourceError as E;
        let mut combination = None;
        for layout in &self.layouts {
            let current = if let Some(point) = layout.observation(path) { point.combination() }
                else if layout.routed_observation(path).is_some() { PartitionCaptureCombination::Disjoint }
                else { return Err(E::Missing); };
            if combination.is_some_and(|prior| prior != current) { return Err(E::Equation); }
            combination = Some(current);
        }
        let combination = combination.ok_or(E::Missing)?;
        if combination == PartitionCaptureCombination::SumF64ToF32 {
            for layout in &self.layouts {
                let point = layout.observation(path).ok_or(E::Missing)?;
                let Some(coordinates) = point.coordinates() else { continue; };
                if coordinates.contiguous_range() != Some(0..coordinates.global_count()) {
                    return Err(E::AdditiveCoordinates);
                }
                for peer in &self.layouts {
                    let a = layout.topology;
                    let b = peer.topology;
                    if a.pipeline_parallel_rank() != b.pipeline_parallel_rank()
                        || a.expert_parallel_rank() != b.expert_parallel_rank()
                        || a.data_parallel_rank() != b.data_parallel_rank() { continue; }
                    let other = peer.observation(path).ok_or(E::Missing)?;
                    if other.coordinates() != Some(coordinates) || other.axis() != point.axis() {
                        return Err(E::AdditiveGroup);
                    }
                }
            }
        }
        Ok(combination)
    }
}

#[cfg(test)]
mod tests;
