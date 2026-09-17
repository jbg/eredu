//! Exact native slices and their destinations in a globally selected tensor.

use super::{elements, CaptureError, ResolvedCaptureSlice};
use crate::component::ComponentCoordinateMap;
mod contiguous;
mod coordinate_source;
pub use coordinate_source::CaptureCoordinateProjectionPlan;
pub use contiguous::{CaptureContiguousProjectionError, CaptureContiguousProjectionPlan};

/// One native slice and its exact destination within the globally selected result.
/// This geometry is not capture authority and cannot be deserialized as a proof.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct CaptureFragmentGeometry {
    local: ResolvedCaptureSlice,
    destination: ResolvedCaptureSlice,
}

impl CaptureFragmentGeometry {
    /// Slice of the actual local source tensor, before export.
    pub const fn local(&self) -> &ResolvedCaptureSlice {
        &self.local
    }
    /// Positions within the global selected tensor, preserving all stride origins.
    pub const fn destination(&self) -> &ResolvedCaptureSlice {
        &self.destination
    }

    /// Tests exact selected-result overlap without allocating or enumerating
    /// individual elements. Arithmetic progressions preserve sparse stride origins;
    /// bounding-box overlap alone does not establish duplicate coverage.
    pub fn overlaps_destination(&self, other: &Self) -> Result<bool, CaptureError> {
        let left = &self.destination;
        let right = &other.destination;
        if left.shape.len() != right.shape.len() {
            return Err(CaptureError::Invalid(
                "capture fragments have different destination ranks".into(),
            ));
        }
        Ok((0..left.shape.len()).all(|axis| {
            progressions_overlap(
                left.starts[axis],
                left.strides[axis],
                left.shape[axis],
                right.starts[axis],
                right.strides[axis],
                right.shape[axis],
            )
        }))
    }
}

/// Bounded projection of a global selection onto one component-axis partition.
/// Ordered noncontiguous coordinates become ordinary strided native fragments;
/// no full local tensor needs to be exported to emulate a smaller global query.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct CaptureSlicePartition {
    global_shape: Vec<u64>,
    local_shape: Vec<u64>,
    global_slice: ResolvedCaptureSlice,
    axis: usize,
    fragments: Vec<CaptureFragmentGeometry>,
}

impl CaptureSlicePartition {
    /// Validates exact geometry and enforces the fragment bound before native work.
    /// A valid selection with no local overlap has zero fragments, not a zero value.
    pub fn new(
        global_shape: &[u64],
        global_slice: &ResolvedCaptureSlice,
        axis: usize,
        coordinates: &ComponentCoordinateMap,
        max_fragments: usize,
    ) -> Result<Self, CaptureError> {
        CaptureCoordinateProjectionPlan::prepare(global_shape,global_slice,axis,coordinates,max_fragments)
            .map(|source|source.construct()).map_err(|cause|cause.legacy())
    }

    #[allow(clippy::too_many_arguments)]
    fn push(
        &mut self,
        local_start: u64,
        local_last: u64,
        local_stride: u64,
        destination_start: u64,
        destination_last: u64,
        destination_stride: u64,
        max_fragments: usize,
    ) -> Result<(), CaptureError> {
        if self.fragments.len() >= max_fragments {
            return Err(CaptureError::Invalid(
                "partition capture fragment count exceeds its bound".into(),
            ));
        }
        let count = (local_last - local_start) / local_stride + 1;
        let mut local = self.global_slice.clone();
        local.starts[self.axis] = local_start;
        local.ends[self.axis] = local_last.checked_add(1).ok_or(CaptureError::Overflow)?;
        local.strides[self.axis] = local_stride;
        local.shape[self.axis] = count;
        let rank = self.global_shape.len();
        let mut destination = ResolvedCaptureSlice {
            starts: vec![0; rank],
            ends: self.global_slice.shape.clone(),
            strides: vec![1; rank],
            shape: self.global_slice.shape.clone(),
        };
        destination.starts[self.axis] = destination_start;
        destination.ends[self.axis] = destination_last
            .checked_add(1)
            .ok_or(CaptureError::Overflow)?;
        destination.strides[self.axis] = destination_stride;
        destination.shape[self.axis] = count;
        self.fragments
            .push(CaptureFragmentGeometry { local, destination });
        Ok(())
    }

    /// Complete source geometry described by the global catalog and request.
    pub fn global_shape(&self) -> &[u64] {
        &self.global_shape
    }
    /// Exact native source geometry expected on this rank.
    pub fn local_shape(&self) -> &[u64] {
        &self.local_shape
    }
    /// Original globally resolved selection, unchanged by partitioning.
    pub const fn global_slice(&self) -> &ResolvedCaptureSlice {
        &self.global_slice
    }
    /// Numeric storage axis carrying component coordinates.
    pub const fn axis(&self) -> usize {
        self.axis
    }
    /// Bounded fragments, in local storage traversal order.
    pub fn fragments(&self) -> &[CaptureFragmentGeometry] {
        &self.fragments
    }
}

// Intersects finite positive arithmetic progressions by their shared congruence.
// Inputs come from validated geometry. u128 products accommodate every u64 stride;
// extended Euclid's signed coefficients fit i128 for u64 moduli.
fn progressions_overlap(
    a: u64,
    step_a: u64,
    count_a: u64,
    b: u64,
    step_b: u64,
    count_b: u64,
) -> bool {
    if count_a == 0 || count_b == 0 {
        return false;
    }
    let low = a.max(b) as u128;
    let high = (a + (count_a - 1) * step_a).min(b + (count_b - 1) * step_b) as u128;
    if low > high {
        return false;
    }
    let (mut gcd, mut remainder) = (step_a, step_b);
    while remainder != 0 {
        (gcd, remainder) = (remainder, gcd % remainder);
    }
    let difference = b as i128 - a as i128;
    if difference % gcd as i128 != 0 {
        return false;
    }
    let modulus = step_b / gcd;
    let first = if modulus == 1 {
        a as u128
    } else {
        let reduced = step_a / gcd;
        let (mut old_r, mut r) = (reduced as i128, modulus as i128);
        let (mut old_s, mut s) = (1i128, 0i128);
        while r != 0 {
            let quotient = old_r / r;
            (old_r, r) = (r, old_r - quotient * r);
            (old_s, s) = (s, old_s - quotient * s);
        }
        let inverse = old_s.rem_euclid(modulus as i128) as u128;
        let difference = (difference / gcd as i128).rem_euclid(modulus as i128) as u128;
        let offset = difference * inverse % modulus as u128;
        a as u128 + step_a as u128 * offset
    };
    let period = step_a as u128 * modulus as u128;
    let residue = first % period;
    let first_in_range = if residue < low {
        residue + (low - residue).div_ceil(period) * period
    } else {
        residue
    };
    first_in_range <= high
}

#[cfg(test)]
mod tests {
    use super::*;

    fn selected(start: u64, end: u64, stride: u64) -> ResolvedCaptureSlice {
        ResolvedCaptureSlice {
            starts: vec![0, 1, start],
            ends: vec![1, 4, end],
            strides: vec![1, 2, stride],
            shape: vec![1, 2, (end - start).div_ceil(stride)],
        }
    }

    #[test]
    fn partition_capture_preserves_global_stride_origin_and_empty_intersections() {
        let selection = selected(1, 17, 3);
        let left = CaptureSlicePartition::new(
            &[1, 4, 20],
            &selection,
            2,
            &ComponentCoordinateMap::range(20, 0..8).unwrap(),
            1,
        )
        .unwrap();
        let right = CaptureSlicePartition::new(
            &[1, 4, 20],
            &selection,
            2,
            &ComponentCoordinateMap::range(20, 8..20).unwrap(),
            1,
        )
        .unwrap();
        assert_eq!(left.fragments()[0].local().starts, [0, 1, 1]);
        assert_eq!(left.fragments()[0].local().shape, [1, 2, 3]);
        assert_eq!(right.fragments()[0].local().starts, [0, 1, 2]);
        assert_eq!(right.fragments()[0].local().strides, [1, 2, 3]);
        assert_eq!(right.fragments()[0].destination().starts, [0, 0, 3]);
        assert_eq!(right.fragments()[0].destination().shape, [1, 2, 3]);
        let none = CaptureSlicePartition::new(
            &[1, 4, 20],
            &selection,
            2,
            &ComponentCoordinateMap::range(20, 8..10).unwrap(),
            0,
        )
        .unwrap();
        assert!(none.fragments().is_empty());
    }

    #[test]
    fn partition_capture_handles_permutations_without_exporting_unselected_columns() {
        let map = ComponentCoordinateMap::indices(20, vec![16, 10, 0, 1, 4, 7, 19]).unwrap();
        let plan =
            CaptureSlicePartition::new(&[1, 4, 20], &selected(1, 17, 3), 2, &map, 3).unwrap();
        assert_eq!(plan.fragments().len(), 3);
        assert_eq!(plan.fragments()[0].destination().starts[2], 5);
        assert_eq!(plan.fragments()[1].destination().starts[2], 3);
        assert_eq!(plan.fragments()[2].local().starts[2], 3);
        assert_eq!(plan.fragments()[2].local().ends[2], 6);
        assert_eq!(plan.fragments()[2].destination().shape[2], 3);
        assert!(CaptureSlicePartition::new(&[1, 4, 20], &selected(1, 17, 3), 2, &map, 2).is_err());
        assert!(CaptureSlicePartition::new(&[1, 4, 19], &selected(1, 17, 3), 2, &map, 3).is_err());
    }

    #[test]
    fn partition_capture_range_arithmetic_stays_bounded_at_usize_max() {
        let count = usize::MAX;
        let map = ComponentCoordinateMap::range(count, count - 3..count).unwrap();
        let slice = ResolvedCaptureSlice {
            starts: vec![0],
            ends: vec![count as u64],
            strides: vec![count as u64 - 1],
            shape: vec![2],
        };
        let plan = CaptureSlicePartition::new(&[count as u64], &slice, 0, &map, 1).unwrap();
        assert_eq!(plan.fragments()[0].local().starts, [2]);
        assert_eq!(plan.fragments()[0].destination().starts, [1]);
    }

    #[test]
    fn fragment_overlap_matches_explicit_sparse_sets_and_extreme_strides() {
        for a in 0..16 {
            for b in 0..16 {
                for step_a in 1..9 {
                    for step_b in 1..9 {
                        for count_a in 0..6 {
                            for count_b in 0..6 {
                                let expected = (0..count_a).any(|i| {
                                    (0..count_b).any(|j| a + i * step_a == b + j * step_b)
                                });
                                assert_eq!(
                                    progressions_overlap(a, step_a, count_a, b, step_b, count_b),
                                    expected,
                                    "{a}/{step_a}/{count_a} versus {b}/{step_b}/{count_b}"
                                );
                            }
                        }
                    }
                }
            }
        }
        let top = u64::MAX;
        assert!(progressions_overlap(0, top - 1, 2, top - 1, top, 1));
        assert!(!progressions_overlap(0, top - 1, 2, 1, top - 3, 2));
        assert!(progressions_overlap(1, top - 2, 2, top - 1, top - 3, 1));
        assert!(!progressions_overlap(top - 4, 2, 2, top - 3, 2, 2));
    }
}
