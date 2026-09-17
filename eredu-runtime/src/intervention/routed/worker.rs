//! One native-row-order lowering worker; storage is supplied by its caller.
use super::*;
use eredu_core::component::RoutedComponentCoordinateMap;

/// Fixed source/offset refusals; this carries no execution authority.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum RoutedInterventionLoweringError {
    /// Actual row identity, selected axes, or action payload differs.
    #[error("invalid sparse intervention coordinates")]
    Coordinates,
    /// An exact extent or scalar offset cannot be represented.
    #[error("sparse intervention arithmetic overflowed")]
    Overflow,
}
impl From<RoutedInterventionLoweringError> for CaptureError {
    fn from(cause: RoutedInterventionLoweringError) -> Self {
        match cause {
            RoutedInterventionLoweringError::Coordinates => {
                Self::Invalid("invalid sparse intervention coordinates".into())
            }
            RoutedInterventionLoweringError::Overflow => Self::Overflow,
        }
    }
}
pub(super) type RowKey = (Option<u64>, u64, u64);
pub(super) trait Allocation {
    type Error: From<RoutedInterventionLoweringError>;
    fn vector<T>(&mut self, capacity: usize) -> Result<Vec<T>, Self::Error>;
}
pub(super) struct Ordinary;
impl Allocation for Ordinary {
    type Error = CaptureError;
    fn vector<T>(&mut self, capacity: usize) -> Result<Vec<T>, Self::Error> {
        Ok(Vec::with_capacity(capacity))
    }
}
use RoutedInterventionLoweringError::{Coordinates, Overflow};
pub(super) fn validate_slice(
    geometry: RoutedUnitGeometry,
    slice: &ResolvedCaptureSlice,
) -> Result<u64, RoutedInterventionLoweringError> {
    if geometry.experts == 0 || geometry.units_per_expert == 0 || geometry.routes_per_token == 0 {
        return Err(Coordinates);
    }
    let components = geometry
        .experts
        .checked_mul(geometry.units_per_expert)
        .ok_or(Overflow)?;
    if [&slice.starts, &slice.ends, &slice.strides, &slice.shape]
        .iter()
        .any(|v| v.len() != 2)
        || slice.ends[1] > components
    {
        return Err(Coordinates);
    }
    for axis in 0..2 {
        if slice.strides[axis] == 0
            || slice.starts[axis] >= slice.ends[axis]
            || (slice.ends[axis] - slice.starts[axis]).div_ceil(slice.strides[axis])
                != slice.shape[axis]
        {
            return Err(Coordinates);
        }
    }
    Ok(components)
}
/// Exact scalar count for a token selection over the complete virtual expert
/// component axis. The caller must separately establish complete serial route
/// coverage for `range`; this arithmetic is not a completed-row or source proof.
/// `None` means selected components depend on the actual expert identities.
pub fn routed_intervention_full_component_count(
    geometry: RoutedUnitGeometry,
    slice: &ResolvedCaptureSlice,
    range: [u64; 2],
) -> Result<Option<usize>, RoutedInterventionLoweringError> {
    let components = validate_slice(geometry, slice)?;
    if range[0] > range[1] {
        return Err(Coordinates);
    }
    if slice.starts[1] != 0 || slice.ends[1] != components || slice.strides[1] != 1 {
        return Ok(None);
    }
    let first = axis_ordinal(slice, 0, range[0]);
    let end = axis_ordinal(slice, 0, range[1]);
    let count = end
        .checked_sub(first)
        .and_then(|n| n.checked_mul(geometry.routes_per_token))
        .and_then(|n| n.checked_mul(geometry.units_per_expert))
        .ok_or(Overflow)?;
    Ok(Some(usize::try_from(count).map_err(|_| Overflow)?))
}
/// Fixed arithmetic frames of the shared count/selection worker. This excludes
/// any row, slice or index backing, which stays with the existing source owner.
pub fn routed_intervention_full_component_count_control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let parts = [
        size_of::<(RoutedUnitGeometry, &ResolvedCaptureSlice, [u64; 2])>(),
        size_of::<[u64; 4]>(),
        size_of::<Result<Option<usize>, RoutedInterventionLoweringError>>(),
        size_of::<(RoutedUnitGeometry, &ResolvedCaptureSlice)>(),
        size_of::<Result<u64, RoutedInterventionLoweringError>>(),
        selection_control_bytes()?,
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
pub(super) fn selection_control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let parts = [
        size_of::<(&ResolvedCaptureSlice, usize, u64)>() * 2,
        size_of::<Option<u64>>(),
        size_of::<[u64; 3]>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
// Shared ordinal arithmetic anchors both the exact count and ordinary native
// row selection at the immutable logical slice origin.
fn axis_ordinal(slice: &ResolvedCaptureSlice, axis: usize, value: u64) -> u64 {
    value
        .saturating_sub(slice.starts[axis])
        .div_ceil(slice.strides[axis])
        .min(slice.shape[axis])
}
fn selected_axis_index(slice: &ResolvedCaptureSlice, axis: usize, value: u64) -> Option<u64> {
    if value < slice.starts[axis] || value >= slice.ends[axis] {
        return None;
    }
    let ordinal = axis_ordinal(slice, axis, value);
    (ordinal < slice.shape[axis] && (value - slice.starts[axis]) / slice.strides[axis] == ordinal)
        .then_some(ordinal)
}

/// Caller validates the immutable action against its selected region. The paid
/// caller uses the original admitted source and its fixed geometry resolver.
pub(super) fn lower<A: Allocation>(
    geometry: RoutedUnitGeometry,
    rows: &[RoutedUnitLocation],
    slice: &ResolvedCaptureSlice,
    action: &InterventionAction,
    coordinates: Option<&RoutedComponentCoordinateMap>,
    allocation: &mut A,
) -> Result<LoweredRoutedIntervention, A::Error> {
    let components = validate_slice(geometry, slice)?;
    let dtype = action.dtype().ok_or(Coordinates)?;
    let compact = match action {
        InterventionAction::MaskComponents {
            indices,
            keep_selected,
            ..
        } => {
            if slice.starts[1] != 0 || slice.ends[1] != components || slice.strides[1] != 1 {
                return Err(Coordinates.into());
            }
            let mut sorted = allocation.vector(indices.len())?;
            sorted.extend_from_slice(indices);
            sorted.sort_unstable();
            sorted.dedup();
            Some((sorted, *keep_selected))
        }
        InterventionAction::MaskLogits { .. } => return Err(Coordinates.into()),
        _ => None,
    };
    // Sorting only identity scratch preserves the source's native value order.
    let mut seen = allocation.vector::<RowKey>(rows.len())?;
    for row in rows {
        if row.slot >= geometry.routes_per_token
            || row.expert >= geometry.experts
            || coordinates.is_some_and(|map| {
                usize::try_from(row.expert)
                    .ok()
                    .and_then(|expert| map.experts().global_to_local(expert))
                    .is_none()
            })
        {
            return Err(Coordinates.into());
        }
        seen.push((row.source_peer, row.token, row.slot));
    }
    seen.sort_unstable();
    if seen.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(Coordinates.into());
    }
    let local_units = coordinates.map_or(geometry.units_per_expert, |map| {
        map.units().local_count() as u64
    });
    let maximum = rows
        .len()
        .checked_mul(usize::try_from(local_units).map_err(|_| Overflow)?)
        .ok_or(Overflow)?;
    let mut indices = allocation.vector(maximum)?;
    let mut payload_indices = allocation.vector(maximum)?;
    for (row_index, row) in rows.iter().enumerate() {
        let Some(token_index) = selected_axis_index(slice, 0, row.token) else { continue; };
        for local_unit in 0..local_units {
            let unit = match coordinates {
                Some(map) => map
                    .units()
                    .local_to_global(usize::try_from(local_unit).map_err(|_| Overflow)?)
                    .ok_or(Coordinates)? as u64,
                None => local_unit,
            };
            let component = row
                .expert
                .checked_mul(geometry.units_per_expert)
                .and_then(|n| n.checked_add(unit))
                .ok_or(Overflow)?;
            let Some(component_index) = selected_axis_index(slice, 1, component) else { continue; };
            let payload_index = token_index
                .checked_mul(slice.shape[1])
                .and_then(|n| n.checked_add(component_index))
                .ok_or(Overflow)?;
            let payload_index = usize::try_from(payload_index).map_err(|_| Overflow)?;
            if let Some((set, keep)) = &compact {
                let selected = u32::try_from(component)
                    .ok()
                    .is_some_and(|n| set.binary_search(&n).is_ok());
                if selected == *keep {
                    continue;
                }
            }
            if let InterventionAction::Mask { keep, .. } = action {
                if *keep.get(payload_index).ok_or(Coordinates)? {
                    continue;
                }
            }
            indices.push(
                (row_index as u64)
                    .checked_mul(local_units)
                    .and_then(|n| n.checked_add(local_unit))
                    .ok_or(Overflow)?,
            );
            payload_indices.push(payload_index);
        }
    }
    let n = u64::try_from(indices.len()).map_err(|_| Overflow)?;
    let action = if n == 0 {
        None
    } else {
        Some(match RoutedInterventionNumericalAction::from_action(action)? {
            RoutedInterventionNumericalAction::Zero(_) => InterventionAction::Zero { dtype },
            RoutedInterventionNumericalAction::Scale(_, factor) => InterventionAction::Scale { dtype, factor },
            RoutedInterventionNumericalAction::Replace(tensor) => InterventionAction::Replace {
                tensor: gather(tensor, n, &payload_indices, allocation)?,
            },
            RoutedInterventionNumericalAction::Add(tensor) => InterventionAction::Add {
                tensor: gather(tensor, n, &payload_indices, allocation)?,
            },
        })
    };
    Ok(LoweredRoutedIntervention { indices, action })
}
fn gather<A: Allocation>(
    tensor: &InterventionTensor,
    n: u64,
    indices: &[usize],
    allocation: &mut A,
) -> Result<InterventionTensor, A::Error> {
    let mut shape = allocation.vector(1)?;
    shape.push(n);
    let values = match &tensor.values {
        InterventionValues::Float32(v) => {
            InterventionValues::Float32(gather_values(v, indices, allocation)?)
        }
        InterventionValues::Float16(v) => {
            InterventionValues::Float16(gather_values(v, indices, allocation)?)
        }
        InterventionValues::Bfloat16(v) => {
            InterventionValues::Bfloat16(gather_values(v, indices, allocation)?)
        }
    };
    Ok(InterventionTensor { shape, values })
}
fn gather_values<T: Copy, A: Allocation>(
    source: &[T],
    indices: &[usize],
    allocation: &mut A,
) -> Result<Vec<T>, A::Error> {
    let mut values = allocation.vector(indices.len())?;
    for &index in indices {
        values.push(*source.get(index).ok_or(Coordinates)?);
    }
    Ok(values)
}
