//! Complete bounded parameter reads with one producer per global source cell.
use super::*;
use eredu_core::parameters::{
    ParameterCoordinateMap, ParameterProjection, ParameterRegion, ParameterRegionFragment,
};
use sha2::{Digest, Sha256};

/// Bounded common request intent, independent of local admission and source
/// preparation. Accepted requests are hashed completely. Oversized invalid
/// inputs include their full lengths and bounded prefixes; they still require
/// local rejection and never authorize a producer.
pub fn parameter_read_intent(
    parameter: &str,
    region: &ParameterRegion,
    projection: Option<&ParameterProjection>,
) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"eredu-parameter-read-request-v1\0");
    digest.update((parameter.len() as u64).to_le_bytes());
    digest.update(&parameter.as_bytes()[..parameter.len().min(4096)]);
    for values in [&region.starts, &region.shape] {
        digest.update((values.len() as u64).to_le_bytes());
        for value in values.iter().take(32) {
            digest.update(value.to_le_bytes());
        }
    }
    match projection {
        None => digest.update([0]),
        Some(projection) => {
            digest.update([1]);
            digest.update((projection.axis as u64).to_le_bytes());
            digest.update(projection.directions.to_le_bytes());
            digest.update((projection.coefficients.len() as u64).to_le_bytes());
            for coefficient in projection.coefficients.iter().take((16 << 20) / 4) {
                digest.update(coefficient.to_bits().to_le_bytes());
            }
        }
    }
    digest.finalize().into()
}

/// One unique producer's source tile and its result destination. Projection
/// destinations may overlap only because disjoint source tiles contribute sums.
#[derive(Debug)]
pub struct PartitionParameterReadFragment {
    rank: usize,
    source: ParameterRegionFragment,
    destination: ParameterRegion,
}
impl PartitionParameterReadFragment {
    /// Actual rank supplying this tile; replicas with later ranks do not export it.
    pub fn rank(&self) -> usize {
        self.rank
    }
    /// Effective local source selection and original selected-source coordinates.
    pub fn source(&self) -> &ParameterRegionFragment {
        &self.source
    }
    /// Exact destination in the global query or projection result.
    pub fn destination(&self) -> &ParameterRegion {
        &self.destination
    }
}

/// Immutable complete-source plan. Admission and native work remain the live
/// operation owner's responsibility; this only proves geometry and bounds.
pub struct PartitionParameterReadPlan {
    identity: [u8; 32],
    output_shape: Vec<u64>,
    counts: Vec<usize>,
    fragments: Vec<PartitionParameterReadFragment>,
    projection: bool,
}
impl PartitionParameterReadPlan {
    /// Chooses the first rank covering each source cell, including partial
    /// replica overlaps and differently fragmented permutations. No full-weight
    /// index buffer is constructed. All metadata is charged before allocation.
    pub fn new(
        global_shape: &[u64],
        region: &ParameterRegion,
        projection: Option<&ParameterProjection>,
        ranks: &[Option<&ParameterCoordinateMap>],
        max_fragments: usize,
        reservation: &mut impl CaptureReservation,
    ) -> Result<Self, ParameterError> {
        region.validate(global_shape)?;
        if ranks.is_empty() || max_fragments == 0 || global_shape.len() > 32 {
            return Err(ParameterError::Invalid(
                "parameter read topology or fragment bound".into(),
            ));
        }
        charge(
            reservation,
            add(
                512,
                add(
                    mul(ranks.len() as u64, 32)?,
                    mul(global_shape.len() as u64, 128)?,
                )?,
            )?,
        )?;
        let output_shape = match projection {
            Some(projection) if projection.region == *region => {
                projection.output_shape(global_shape)?
            }
            Some(_) => {
                return Err(ParameterError::Invalid(
                    "projection changed the global selection".into(),
                ))
            }
            None => region.shape.clone(),
        };
        let mut digest = Sha256::new();
        digest.update(b"eredu-parameter-read-plan-v1\0");
        for shape in [
            global_shape,
            region.starts.as_slice(),
            region.shape.as_slice(),
        ] {
            digest.update((shape.len() as u64).to_le_bytes());
            for size in shape {
                digest.update(size.to_le_bytes());
            }
        }
        match projection {
            Some(projection) => {
                digest.update([1]);
                digest.update((projection.axis as u64).to_le_bytes());
                digest.update(projection.directions.to_le_bytes());
                for coefficient in &projection.coefficients {
                    digest.update(coefficient.to_bits().to_le_bytes());
                }
            }
            None => digest.update([0]),
        }
        digest.update((ranks.len() as u64).to_le_bytes());
        let mut remaining = vec![ParameterRegion {
            starts: vec![0; region.shape.len()],
            shape: region.shape.clone(),
        }];
        let mut counts = vec![0usize; ranks.len()];
        let mut fragments = Vec::new();
        for (rank, coordinates) in ranks.iter().enumerate() {
            let Some(coordinates) = coordinates else {
                digest.update([0]);
                continue;
            };
            if coordinates.global_shape() != global_shape {
                return Err(ParameterError::Invalid(
                    "parameter rank shape differs from global declaration".into(),
                ));
            }
            digest.update([1]);
            digest.update(coordinates.identity());
            if remaining.is_empty() {
                continue;
            }
            let projected = coordinates.project_region(region, max_fragments, reservation)?;
            for source in projected.fragments() {
                let mut next = Vec::new();
                for uncovered in remaining {
                    let intersection = intersect(&uncovered, source.destination(), reservation)?;
                    let Some(intersection) = intersection else {
                        if next.len() >= max_fragments {
                            return Err(ParameterError::Invalid(
                                "parameter uncovered-region bound exceeded".into(),
                            ));
                        }
                        charge(reservation, 64)?;
                        next.push(uncovered);
                        continue;
                    };
                    if fragments.len() >= max_fragments {
                        return Err(ParameterError::Invalid(
                            "parameter read fragment bound exceeded".into(),
                        ));
                    }
                    charge(reservation, add(256, mul(global_shape.len() as u64, 32)?)?)?;
                    let selected = source.restrict_destination(&intersection, reservation)?;
                    let destination = match projection {
                        Some(projection) => {
                            let mut starts = Vec::with_capacity(intersection.starts.len());
                            let mut shape = Vec::with_capacity(intersection.shape.len());
                            for axis in 0..intersection.shape.len() {
                                if axis != projection.axis {
                                    starts.push(intersection.starts[axis]);
                                    shape.push(intersection.shape[axis]);
                                }
                            }
                            starts.push(0);
                            shape.push(projection.directions);
                            ParameterRegion { starts, shape }
                        }
                        None => intersection.clone(),
                    };
                    let count = usize::try_from(destination.validate(&output_shape)?)
                        .map_err(|_| ParameterError::Overflow)?;
                    counts[rank] = counts[rank]
                        .checked_add(count)
                        .ok_or(ParameterError::Overflow)?;
                    fragments.push(PartitionParameterReadFragment {
                        rank,
                        source: selected,
                        destination,
                    });
                    subtract(
                        uncovered,
                        &intersection,
                        &mut next,
                        max_fragments,
                        reservation,
                    )?;
                }
                if next.len() > max_fragments {
                    return Err(ParameterError::Invalid(
                        "parameter uncovered-region bound exceeded".into(),
                    ));
                }
                remaining = next;
            }
        }
        if !remaining.is_empty() {
            return Err(ParameterError::Incomplete(
                "selected source cells have no owner".into(),
            ));
        }
        Ok(Self {
            identity: digest.finalize().into(),
            output_shape,
            counts,
            fragments,
            projection: projection.is_some(),
        })
    }
    /// Coordinate, source-selection and exact-contraction digest for live agreement.
    pub fn identity(&self) -> &[u8; 32] {
        &self.identity
    }
    /// Full selected result shape, retaining singleton axes.
    pub fn output_shape(&self) -> &[u64] {
        &self.output_shape
    }
    /// Canonical rank-ordered source tiles, including only unique producers.
    pub fn fragments(&self) -> &[PartitionParameterReadFragment] {
        &self.fragments
    }
    /// Exact F32 output count expected from each rank in fragment order.
    pub fn rank_counts(&self) -> &[usize] {
        &self.counts
    }
    /// Largest per-rank F32 payload, suitable for prepaid word transport.
    pub fn max_rank_words(&self) -> usize {
        self.counts.iter().copied().max().unwrap_or(0)
    }

    /// Validates every producer's exact count and finiteness, then assembles the
    /// complete result. Disjoint-source contractions accumulate in host F64 and
    /// round to F32 once, in deterministic rank/tile order. No missing value is zero.
    pub fn assemble(
        &self,
        ranks: &[Vec<f32>],
        reservation: &mut impl CaptureReservation,
    ) -> Result<Vec<f32>, ParameterError> {
        if ranks.len() != self.counts.len()
            || ranks
                .iter()
                .zip(&self.counts)
                .any(|(values, count)| values.len() != *count)
        {
            return Err(ParameterError::Incomplete(
                "producer payload count differs from complete read plan".into(),
            ));
        }
        if ranks.iter().flatten().any(|value| !value.is_finite()) {
            return Err(ParameterError::Invalid(
                "non-finite parameter result".into(),
            ));
        }
        let count =
            usize::try_from(elements(&self.output_shape)?).map_err(|_| ParameterError::Overflow)?;
        charge(
            reservation,
            add(
                256,
                add(mul(count as u64, 12)?, mul(ranks.len() as u64, 8)?)?,
            )?,
        )?;
        let mut accumulated = vec![0.0f64; count];
        let mut offsets = vec![0usize; ranks.len()];
        for fragment in &self.fragments {
            let count = elements(&fragment.destination.shape)? as usize;
            let start = offsets[fragment.rank];
            let values = &ranks[fragment.rank][start..start + count];
            for (index, value) in values.iter().enumerate() {
                let target = result_index(index, &fragment.destination, &self.output_shape);
                if self.projection {
                    accumulated[target] += f64::from(*value);
                } else {
                    accumulated[target] = f64::from(*value);
                }
            }
            offsets[fragment.rank] += count;
        }
        if accumulated.iter().any(|value| !(*value as f32).is_finite()) {
            return Err(ParameterError::Invalid(
                "parameter contraction result exceeds F32".into(),
            ));
        }
        Ok(accumulated.into_iter().map(|value| value as f32).collect())
    }
}
fn charge(
    reservation: &mut impl CaptureReservation,
    host_bytes: u64,
) -> Result<(), ParameterError> {
    reservation.reserve_quota(CaptureUsage {
        host_bytes,
        ..Default::default()
    })?;
    Ok(())
}
fn intersect(
    left: &ParameterRegion,
    right: &ParameterRegion,
    reservation: &mut impl CaptureReservation,
) -> Result<Option<ParameterRegion>, ParameterError> {
    if left
        .starts
        .iter()
        .zip(&left.shape)
        .zip(right.starts.iter().zip(&right.shape))
        .any(|((&a, &n), (&b, &m))| a.max(b) >= (a + n).min(b + m))
    {
        return Ok(None);
    }
    charge(reservation, add(64, mul(left.shape.len() as u64, 16)?)?)?;
    let starts = left
        .starts
        .iter()
        .zip(&right.starts)
        .map(|(a, b)| *a.max(b))
        .collect::<Vec<_>>();
    let shape = left
        .starts
        .iter()
        .zip(&left.shape)
        .zip(right.starts.iter().zip(&right.shape))
        .zip(&starts)
        .map(|(((&a, &n), (&b, &m)), start)| (a + n).min(b + m) - start)
        .collect();
    Ok(Some(ParameterRegion { starts, shape }))
}
fn subtract(
    mut source: ParameterRegion,
    cut: &ParameterRegion,
    output: &mut Vec<ParameterRegion>,
    max: usize,
    reservation: &mut impl CaptureReservation,
) -> Result<(), ParameterError> {
    for axis in 0..source.shape.len() {
        for before in [true, false] {
            let (start, end) = if before {
                (source.starts[axis], cut.starts[axis])
            } else {
                (
                    cut.starts[axis] + cut.shape[axis],
                    source.starts[axis] + source.shape[axis],
                )
            };
            if start == end {
                continue;
            }
            if output.len() >= max {
                return Err(ParameterError::Invalid(
                    "parameter uncovered-region bound exceeded".into(),
                ));
            }
            charge(reservation, add(64, mul(source.shape.len() as u64, 16)?)?)?;
            let mut piece = source.clone();
            piece.starts[axis] = start;
            piece.shape[axis] = end - start;
            output.push(piece);
        }
        source.starts[axis] = cut.starts[axis];
        source.shape[axis] = cut.shape[axis];
    }
    Ok(())
}
fn result_index(mut index: usize, region: &ParameterRegion, shape: &[u64]) -> usize {
    let mut output = 0usize;
    let mut stride = 1usize;
    for axis in (0..shape.len()).rev() {
        output += (region.starts[axis] as usize + index % region.shape[axis] as usize) * stride;
        index /= region.shape[axis] as usize;
        stride *= shape[axis] as usize;
    }
    output
}

#[cfg(test)]
mod tests;
