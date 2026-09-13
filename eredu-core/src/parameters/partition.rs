//! Exact logical parameter coordinates shared by reads, contractions and edits.
//!
//! These immutable geometries grant no loaded-session or communication authority.
//! The live parameter owner must bind them to its admitted operation and epoch.
use super::*;
use crate::{
    capture::{add, mul, CaptureReservation},
    component::ComponentCoordinateMap,
};

/// Independent global-to-local coordinate maps in effective parameter axis order.
/// Packed storage and materialization transforms must already have been resolved
/// by the architecture's retained binding declaration. Empty local axes are legal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParameterCoordinateMap {
    global_shape: Vec<u64>,
    local_shape: Vec<u64>,
    axes: Vec<ComponentCoordinateMap>,
}

impl ParameterCoordinateMap {
    /// Checks exact global dimensions and arithmetic without touching parameters.
    pub fn new(
        global_shape: Vec<u64>,
        axes: Vec<ComponentCoordinateMap>,
    ) -> Result<Self, ParameterError> {
        if global_shape.len() > 32
            || global_shape.len() != axes.len()
            || global_shape
                .iter()
                .zip(&axes)
                .any(|(size, axis)| *size == 0 || *size != axis.global_count() as u64)
        {
            return Err(invalid(
                "parameter coordinate axes disagree with global geometry",
            ));
        }
        elements(&global_shape)?;
        let local_shape = axes
            .iter()
            .map(|axis| axis.local_count() as u64)
            .collect::<Vec<_>>();
        elements(&local_shape)?;
        Ok(Self {
            global_shape,
            local_shape,
            axes,
        })
    }
    /// Complete effective shape before partitioning, including all logical axes.
    pub fn global_shape(&self) -> &[u64] {
        &self.global_shape
    }
    /// Actual effective shape used by this local loaded slot.
    pub fn local_shape(&self) -> &[u64] {
        &self.local_shape
    }
    /// Ordered per-axis maps, retaining permutations and noncontiguous selections.
    pub fn axes(&self) -> &[ComponentCoordinateMap] {
        &self.axes
    }

    /// Shape and ordered-coordinate digest for peer comparison. Equal extents
    /// alone cannot establish equal parameter placement. This is not authority.
    pub fn identity(&self) -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update(b"eredu-parameter-coordinates-v1\0");
        digest.update((self.axes.len() as u64).to_le_bytes());
        for axis in &self.axes {
            digest.update((axis.global_count() as u64).to_le_bytes());
            digest.update((axis.local_count() as u64).to_le_bytes());
            // Canonical consecutive runs make full vocabulary axes constant
            // work while preserving equality with equivalent indexed maps.
            if let Some(range) = axis.contiguous_range() {
                if !range.is_empty() {
                    digest.update((range.start as u64).to_le_bytes());
                    digest.update((range.len() as u64).to_le_bytes());
                }
            } else {
                let mut local = 0;
                while local < axis.local_count() {
                    let start = axis.local_to_global(local).expect("validated coordinates");
                    let mut length = 1;
                    while local + length < axis.local_count()
                        && axis.local_to_global(local + length) == start.checked_add(length)
                    {
                        length += 1;
                    }
                    digest.update((start as u64).to_le_bytes());
                    digest.update((length as u64).to_le_bytes());
                    local += length;
                }
            }
        }
        digest.finalize().into()
    }

    /// Projects a global rectangle into bounded contiguous local rectangles.
    /// Both local and destination coordinates preserve singleton axes. Reserve
    /// metadata before constructing it; failure or unused results never refund.
    pub fn project_region(
        &self,
        region: &ParameterRegion,
        max_fragments: usize,
        reservation: &mut impl CaptureReservation,
    ) -> Result<PartitionParameterRegion, ParameterError> {
        region.validate(&self.global_shape)?;
        if max_fragments == 0 {
            return Err(invalid("parameter fragment limit must be positive"));
        }
        charge(reservation, add(256, mul(self.axes.len() as u64, 128)?)?)?;
        let empty = self
            .axes
            .iter()
            .zip(region.starts.iter().zip(&region.shape))
            .any(|(axis, (&start, &length))| {
                let end = start + length; // Validated against the global extent above.
                if let Some(range) = axis.contiguous_range() {
                    start.max(range.start as u64) >= end.min(range.end as u64)
                } else {
                    !(0..axis.local_count()).any(|local| {
                        (start..end).contains(
                            &(axis.local_to_global(local).expect("validated coordinates") as u64),
                        )
                    })
                }
            });
        if empty {
            return Ok(PartitionParameterRegion {
                coordinate_identity: self.identity(),
                global_shape: self.global_shape.clone(),
                local_shape: self.local_shape.clone(),
                region: region.clone(),
                fragments: Vec::new(),
            });
        }
        let mut axes = Vec::with_capacity(self.axes.len());
        let mut count = 1usize;
        for (axis, (&start, &length)) in self
            .axes
            .iter()
            .zip(region.starts.iter().zip(&region.shape))
        {
            let mut runs: Vec<Run> = Vec::new();
            let end = add(start, length)?;
            if let Some(range) = axis.contiguous_range() {
                let first = start.max(range.start as u64);
                let last = end.min(range.end as u64);
                if first < last {
                    charge(reservation, 64)?;
                    runs.push(Run {
                        local: first - range.start as u64,
                        destination: first - start,
                        length: last - first,
                    });
                }
            } else {
                for local in 0..axis.local_count() {
                    let global = axis.local_to_global(local).expect("validated coordinates") as u64;
                    if !(start..end).contains(&global) {
                        continue;
                    }
                    if let Some(previous) = runs.last_mut().filter(|previous| {
                        previous.local + previous.length == local as u64
                            && previous.destination + previous.length == global - start
                    }) {
                        previous.length += 1;
                    } else {
                        if runs.len() >= max_fragments {
                            return Err(invalid("parameter fragment limit exceeded"));
                        }
                        charge(reservation, 64)?;
                        runs.push(Run {
                            local: local as u64,
                            destination: global - start,
                            length: 1,
                        });
                    }
                }
            }
            count = count
                .checked_mul(runs.len())
                .ok_or(ParameterError::Overflow)?;
            if count > max_fragments {
                return Err(invalid("parameter fragment limit exceeded"));
            }
            axes.push(runs);
        }
        let fragment_bytes = add(256, mul(self.axes.len() as u64, 32)?)?;
        charge(reservation, mul(count as u64, fragment_bytes)?)?;
        let mut fragments = Vec::with_capacity(count);
        for index in 0..count {
            let mut cursor = index;
            let mut local = ParameterRegion {
                starts: vec![0; axes.len()],
                shape: vec![0; axes.len()],
            };
            let mut destination = local.clone();
            for axis in (0..axes.len()).rev() {
                let run = &axes[axis][cursor % axes[axis].len()];
                cursor /= axes[axis].len();
                local.starts[axis] = run.local;
                local.shape[axis] = run.length;
                destination.starts[axis] = run.destination;
                destination.shape[axis] = run.length;
            }
            fragments.push(ParameterRegionFragment { local, destination });
        }
        Ok(PartitionParameterRegion {
            coordinate_identity: self.identity(),
            global_shape: self.global_shape.clone(),
            local_shape: self.local_shape.clone(),
            region: region.clone(),
            fragments,
        })
    }
}
struct Run {
    local: u64,
    destination: u64,
    length: u64,
}

/// One exact local rectangle and its corresponding global selected-result region.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParameterRegionFragment {
    local: ParameterRegion,
    destination: ParameterRegion,
}
impl ParameterRegionFragment {
    /// Contiguous coordinates in the effective local tensor.
    pub fn local(&self) -> &ParameterRegion {
        &self.local
    }
    /// Coordinates in the original selected global result, starting at zero.
    pub fn destination(&self) -> &ParameterRegion {
        &self.destination
    }

    /// Restricts this exact correspondence to a contained selected-result tile.
    /// Useful for choosing one producer when replicas have different tilings.
    pub fn restrict_destination(
        &self,
        selected: &ParameterRegion,
        reservation: &mut impl CaptureReservation,
    ) -> Result<Self, ParameterError> {
        if selected.starts.len() != self.destination.starts.len()
            || selected.shape.len() != self.destination.shape.len()
        {
            return Err(invalid("parameter fragment restriction rank"));
        }
        for axis in 0..selected.shape.len() {
            if selected.shape[axis] == 0
                || selected.starts[axis] < self.destination.starts[axis]
                || add(selected.starts[axis], selected.shape[axis])?
                    > add(self.destination.starts[axis], self.destination.shape[axis])?
            {
                return Err(invalid("parameter fragment restriction exceeds source"));
            }
        }
        charge(
            reservation,
            add(256, mul(selected.shape.len() as u64, 64)?)?,
        )?;
        let local = ParameterRegion {
            starts: self
                .local
                .starts
                .iter()
                .zip(&selected.starts)
                .zip(&self.destination.starts)
                .map(|((&local, &selected), &original)| local + (selected - original))
                .collect(),
            shape: selected.shape.clone(),
        };
        Ok(Self {
            local,
            destination: selected.clone(),
        })
    }

    /// Projects directions for this exact (possibly restricted) source tile.
    /// Coordinates are relative to the supplied original global selection;
    /// neither this geometry nor the supplied shapes grant loaded authority.
    pub fn project_contraction(
        &self,
        projection: &ParameterProjection,
        global_shape: &[u64],
        reservation: &mut impl CaptureReservation,
    ) -> Result<ParameterProjectionFragment, ParameterError> {
        charge(
            reservation,
            add(256, mul(projection.region.shape.len() as u64, 64)?)?,
        )?;
        let output_shape = projection.output_shape(global_shape)?;
        self.destination.validate(&projection.region.shape)?;
        let width = projection.region.shape[projection.axis];
        let selected_width = self.destination.shape[projection.axis];
        let start = self.destination.starts[projection.axis];
        let coefficients = select_values(
            &projection.coefficients,
            &[projection.directions, width],
            &ParameterRegion {
                starts: vec![0, start],
                shape: vec![projection.directions, selected_width],
            },
            reservation,
        )?;
        let mut destination = ParameterRegion {
            starts: Vec::new(),
            shape: Vec::new(),
        };
        for axis in 0..projection.region.shape.len() {
            if axis != projection.axis {
                destination.starts.push(self.destination.starts[axis]);
                destination.shape.push(self.destination.shape[axis]);
            }
        }
        destination.starts.push(0);
        destination.shape.push(projection.directions);
        destination.validate(&output_shape)?;
        Ok(ParameterProjectionFragment {
            projection: ParameterProjection {
                region: self.local.clone(),
                axis: projection.axis,
                directions: projection.directions,
                coefficients,
            },
            destination,
            source_destination: self.destination.clone(),
        })
    }
}

/// Bounded projection of one original global parameter selection. An empty
/// fragment set denotes no overlap, never a measured or manufactured zero.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionParameterRegion {
    coordinate_identity: [u8; 32],
    global_shape: Vec<u64>,
    local_shape: Vec<u64>,
    region: ParameterRegion,
    fragments: Vec<ParameterRegionFragment>,
}
impl PartitionParameterRegion {
    /// Exact coordinate identity retained at projection.
    pub fn coordinate_identity(&self) -> &[u8; 32] {
        &self.coordinate_identity
    }
    /// Effective shape of the original complete parameter.
    pub fn global_shape(&self) -> &[u64] {
        &self.global_shape
    }
    /// Effective shape of this local slot.
    pub fn local_shape(&self) -> &[u64] {
        &self.local_shape
    }
    /// Original admitted rectangle; it is never replaced by a local interpretation.
    pub fn region(&self) -> &ParameterRegion {
        &self.region
    }
    /// Disjoint local pieces in deterministic local axis order.
    pub fn fragments(&self) -> &[ParameterRegionFragment] {
        &self.fragments
    }

    /// Selects an edit's global row-major payload for the exact local fragment.
    /// Copies only intersecting values, with no full-weight index or value buffer.
    pub fn project_update(
        &self,
        fragment: usize,
        update: &ParameterUpdate,
        reservation: &mut impl CaptureReservation,
    ) -> Result<ParameterUpdate, ParameterError> {
        let values = update.values();
        if values.len() as u64 != elements(&self.region.shape)?
            || values.iter().any(|value| !value.is_finite())
        {
            return Err(invalid(
                "global parameter update payload differs from its selection",
            ));
        }
        let fragment = self
            .fragments
            .get(fragment)
            .ok_or_else(|| invalid("unknown parameter fragment"))?;
        let selected = select_values(
            values,
            &self.region.shape,
            &fragment.destination,
            reservation,
        )?;
        Ok(match update {
            ParameterUpdate::Replace { .. } => ParameterUpdate::Replace { values: selected },
            ParameterUpdate::Add { .. } => ParameterUpdate::Add { values: selected },
        })
    }

    /// Projects host contraction directions without exporting parameter values.
    /// Contractions over a sharded axis produce additive pieces; other sharded
    /// axes produce disjoint output regions. The original source destination is
    /// retained so the live receiver can prove complete, nonduplicate coverage.
    pub fn project_contraction(
        &self,
        fragment: usize,
        projection: &ParameterProjection,
        reservation: &mut impl CaptureReservation,
    ) -> Result<ParameterProjectionFragment, ParameterError> {
        if projection.region != self.region {
            return Err(invalid(
                "parameter contraction changed its global selection",
            ));
        }
        self.fragments
            .get(fragment)
            .ok_or_else(|| invalid("unknown parameter fragment"))?
            .project_contraction(projection, &self.global_shape, reservation)
    }
}

/// One native contraction and its exact global source/output correspondence.
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterProjectionFragment {
    projection: ParameterProjection,
    destination: ParameterRegion,
    source_destination: ParameterRegion,
}
impl ParameterProjectionFragment {
    /// Ordinary native projection over the effective local slot.
    pub fn projection(&self) -> &ParameterProjection {
        &self.projection
    }
    /// Global result tile; distinct source fragments may add into the same tile.
    pub fn destination(&self) -> &ParameterRegion {
        &self.destination
    }
    /// Original selected source cells that contribute to this tile.
    pub fn source_destination(&self) -> &ParameterRegion {
        &self.source_destination
    }
}

fn select_values(
    values: &[f32],
    shape: &[u64],
    region: &ParameterRegion,
    reservation: &mut impl CaptureReservation,
) -> Result<Vec<f32>, ParameterError> {
    let count = region.validate(shape)?;
    let count = usize::try_from(count).map_err(|_| ParameterError::Overflow)?;
    if values.len() as u64 != elements(shape)? {
        return Err(invalid("parameter payload geometry differs"));
    }
    charge(reservation, add(64, mul(count as u64, 4)?)?)?;
    let mut selected = Vec::with_capacity(count);
    for index in 0..count {
        let mut cursor = index as u64;
        let mut source = 0u64;
        let mut stride = 1u64;
        for axis in (0..shape.len()).rev() {
            let coordinate = region.starts[axis] + cursor % region.shape[axis];
            cursor /= region.shape[axis];
            source = add(source, mul(coordinate, stride)?)?;
            stride = mul(stride, shape[axis])?;
        }
        selected.push(values[usize::try_from(source).map_err(|_| ParameterError::Overflow)?]);
    }
    Ok(selected)
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
fn invalid(message: &str) -> ParameterError {
    ParameterError::Invalid(message.into())
}

#[cfg(test)]
mod tests;
