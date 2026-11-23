//! Checked host coordinates and record assembly for prepaid routed evidence.
use super::*;
use crate::intervention::{RoutedUnitLocation, RoutedUnitLocations};
use eredu_nn::{Tensor, TensorElementType};

/// Native copies of only the coordinates, route coefficients and selected scalars.
pub trait RoutedReadout<T> {
    /// Native readback failure.
    type Error: From<CaptureError>;
    /// Copies IDs from the entire coordinate vector or from selected flat indices.
    fn ids(&self, tensor: &T, indices: Option<&[u64]>) -> Result<Vec<u32>, Self::Error>;
    /// Copies coefficients at exact flat selection indices.
    fn coefficients(&self, tensor: &T, indices: &[u64]) -> Result<Vec<f32>, Self::Error>;
    /// Copies the requested unit columns of one participating native row.
    fn units(
        &self,
        tensor: &T,
        row: usize,
        columns: &[u64],
    ) -> Result<TensorObservation, Self::Error>;
}

struct Layout {
    native_rows: usize,
    width: u64,
    end: u64,
}

fn invalid() -> CaptureError {
    CaptureError::Invalid("invalid native routed-unit coordinates".into())
}

impl Layout {
    fn new<T: Tensor>(
        source: &RoutedUnitCaptureSource<'_, T>,
        local: RoutedUnitGeometry,
        require_complete: bool,
    ) -> Result<Self, CaptureError> {
        let [routes, units] = *source.values.shape() else {
            return Err(invalid());
        };
        let [tokens, width] = *source.coefficients.shape() else {
            return Err(invalid());
        };
        let [groups, source_width] = *source.source_groups.shape() else {
            return Err(invalid());
        };
        let capacity = mul(
            tokens.try_into().map_err(|_| invalid())?,
            local.routes_per_token,
        )?;
        if units < 0
            || units as u64 != local.units_per_expert
            || width <= 0
            || width as u64 != local.routes_per_token
            || width != source_width
            || routes < 0
            || routes as u64 > capacity
            || (require_complete && routes as u64 != capacity)
            || source.token_indices.shape() != [routes]
            || source.selection_indices.shape() != [routes]
            || !matches!(
                source.values.element_type(),
                Some(
                    TensorElementType::F16
                        | TensorElementType::Bf16
                        | TensorElementType::F32
                        | TensorElementType::F64
                )
            )
        {
            return Err(invalid());
        }
        let end = add(source.token_offset, tokens as u64)?;
        if groups < 0 || end > groups as u64 {
            return Err(invalid());
        }
        Ok(Self {
            native_rows: routes as usize,
            width: width as u64,
            end,
        })
    }

    fn resolve<T, R: RoutedReadout<T>>(
        &self,
        native: &R,
        source: &RoutedUnitCaptureSource<'_, T>,
        experts: u64,
        origins: Option<RoutedUnitOrigins<'_>>,
    ) -> Result<(Vec<RoutedUnitLocation>, Vec<u64>), R::Error> {
        if self.native_rows == 0 {
            return Ok((vec![], vec![]));
        }
        let tokens = native.ids(source.token_indices, None)?;
        let indices: Vec<u64> = native
            .ids(source.selection_indices, None)?
            .into_iter()
            .map(u64::from)
            .collect();
        if tokens.len() != self.native_rows || indices.len() != self.native_rows {
            return Err(invalid().into());
        }
        let mut group_indices = Vec::with_capacity(self.native_rows);
        for (&token, &selection) in tokens.iter().zip(&indices) {
            let token = u64::from(token);
            let original = add(source.token_offset, token)?;
            if original >= self.end || selection / self.width != token {
                return Err(invalid().into());
            }
            group_indices.push(add(mul(original, self.width)?, selection % self.width)?);
        }
        let groups = native.ids(source.source_groups, Some(&group_indices))?;
        if groups.len() != self.native_rows {
            return Err(invalid().into());
        }
        let rows = tokens
            .into_iter()
            .zip(&indices)
            .zip(groups)
            .map(|((token, selection), group)| {
                let original = source.token_offset + u64::from(token);
                let expert = match source.global_groups {
                    Some(map) => *map.get(group as usize).ok_or_else(invalid)? as u64,
                    None => u64::from(group),
                };
                if expert >= experts {
                    return Err(invalid());
                }
                let (source_peer, token, slot) = if let Some(origins) = origins {
                    let origin = origins
                        .resolve(usize::try_from(original).map_err(|_| invalid())?)
                        .ok_or_else(invalid)?;
                    (
                        origin.source_peer.map(|peer| peer as u64),
                        origin.token as u64,
                        origin.slot as u64,
                    )
                } else {
                    (None, original, selection % self.width)
                };
                Ok(RoutedUnitLocation {
                    source_peer,
                    token,
                    slot,
                    expert,
                })
            })
            .collect::<Result<_, CaptureError>>()?;
        Ok((rows, indices))
    }
}

/// Copies and maps the exact route coordinates for sparse edits in native row order.
pub fn routed_unit_locations<T: Tensor, R: RoutedReadout<T>>(
    native: &R,
    source: &RoutedUnitCaptureSource<'_, T>,
    local: RoutedUnitGeometry,
    origins: Option<RoutedUnitOrigins<'_>>,
) -> Result<RoutedUnitLocations, R::Error> {
    local.components()?;
    let layout = Layout::new(source, local, true)?;
    let (rows, _) = layout.resolve(native, source, local.experts, origins)?;
    Ok(RoutedUnitLocations {
        source_token_range: [source.token_offset, layout.end],
        rows,
    })
}

/// Assembles bounded sparse evidence, optionally projecting the retained partition.
pub fn capture_routed_units<T: Tensor, R: RoutedReadout<T>>(
    native: &R,
    source: &RoutedUnitCaptureSource<'_, T>,
    geometry: RoutedUnitGeometry,
    slice: &ResolvedCaptureSlice,
    partition: Option<(
        &PartitionRoutedUnitCaptureSource<'_, T>,
        &PartitionRoutedUnitCaptureRequest<'_>,
    )>,
) -> Result<RoutedUnitCapture, R::Error> {
    geometry.validate_slice(slice)?;
    let mut local = geometry;
    if let Some((partition_source, request)) = partition {
        request.validate()?;
        if geometry != request.geometry
            || slice != request.slice
            || partition_source.unit_coordinates != request.ownership.coordinates.units()
        {
            return Err(invalid().into());
        }
        local.units_per_expert = partition_source.unit_coordinates.local_count() as u64;
        if request.ownership.source_peer.is_some() {
            local.routes_per_token = 1;
        }
    }
    let layout = Layout::new(source, local, false)?;
    if let Some((partition_source, request)) = partition {
        let source_rows = source.source_groups.dim(0) as u64;
        if source_rows > request.native_shape()?[0] {
            return Err(invalid().into());
        }
        match (request.ownership.source_peer, partition_source.origins) {
            (Some(_), Some(origins))
                if origins.peer_count() as u64 == request.ownership.source_peers
                    && origins.row_count() as u64 == source_rows
                    && origins.routes_per_token() as u64 == geometry.routes_per_token => {}
            (None, None) if source_rows == request.source_tokens => {}
            _ => return Err(invalid().into()),
        }
    }
    let source_token_ranges = if source.token_offset == layout.end {
        vec![]
    } else {
        vec![[source.token_offset, layout.end]]
    };
    let mut capture = RoutedUnitCapture {
        geometry,
        source_token_ranges,
        rows: vec![],
    };
    if layout.native_rows == 0
        || slice.shape[..2].contains(&0)
        || (partition.is_some() && slice.shape[2] == 0)
    {
        return Ok(capture);
    }
    let (routes, indices) = layout.resolve(
        native,
        source,
        geometry.experts,
        partition.and_then(|(source, _)| source.origins),
    )?;
    let coefficients = native.coefficients(source.coefficients, &indices)?;
    if coefficients.len() != routes.len() {
        return Err(invalid().into());
    }
    let columns = (0..slice.shape[2])
        .map(|index| {
            let global = add(slice.starts[2], mul(index, slice.strides[2])?)?;
            match partition {
                Some((source, _)) => source
                    .unit_coordinates
                    .global_to_local(usize::try_from(global).map_err(|_| invalid())?)
                    .map(|local| local as u64)
                    .ok_or_else(invalid),
                None => Ok(global),
            }
        })
        .collect::<Result<Vec<_>, CaptureError>>()?;
    let in_slice = |axis: usize, n: u64| {
        n >= slice.starts[axis]
            && n < slice.ends[axis]
            && (n - slice.starts[axis]).is_multiple_of(slice.strides[axis])
    };
    for (row, (location, coefficient)) in routes.into_iter().zip(coefficients).enumerate() {
        let RoutedUnitLocation {
            source_peer,
            token,
            slot,
            expert,
        } = location;
        if !coefficient.is_finite() {
            return Err(invalid().into());
        }
        if let Some((_, request)) = partition {
            if token >= request.source_tokens
                || slot >= geometry.routes_per_token
                || request
                    .ownership
                    .coordinates
                    .experts()
                    .global_to_local(expert as usize)
                    .is_none()
            {
                return Err(invalid().into());
            }
            if source_peer != request.ownership.source_peer {
                continue;
            }
        }
        if in_slice(0, token) && in_slice(1, slot) {
            capture.rows.push(RoutedUnitCaptureRow {
                source_peer,
                token,
                slot,
                expert,
                coefficient,
                unit_start: slice.starts[2],
                unit_stride: slice.strides[2],
                values: native.units(source.values, row, &columns)?,
            });
        }
    }
    Ok(capture)
}
