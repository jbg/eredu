//! Native coordinate materialization only; portable runtime owns edit semantics.
use super::*;

pub(super) fn indices(values: &[u64]) -> Result<Array, Error> {
    let values = values
        .iter()
        .map(|n| i32::try_from(*n).map_err(|_| Error::observation(CaptureError::Overflow)))
        .collect::<Result<Vec<_>, _>>()?;
    let len =
        i32::try_from(values.len()).map_err(|_| Error::observation(CaptureError::Overflow))?;
    Ok(Array::try_from_slice(&values, &[len])?)
}

pub(super) fn locations(
    source: &RoutedUnitCaptureSource<'_, MlxTensor>,
    geometry: RoutedUnitGeometry,
    stream: &Stream,
) -> Result<RoutedUnitLocations, Error> {
    let invalid = || {
        Error::observation(CaptureError::Invalid(
            "invalid native sparse edit coordinates".into(),
        ))
    };
    geometry.components().map_err(Error::observation)?;
    let shape = source.values.shape();
    let coefficients = source.coefficients.shape();
    let groups_shape = source.source_groups.shape();
    if shape.len() != 2
        || shape[1] as u64 != geometry.units_per_expert
        || coefficients.len() != 2
        || coefficients[1] as u64 != geometry.routes_per_token
        || groups_shape.len() != 2
        || groups_shape[1] as u64 != geometry.routes_per_token
        || source.token_indices.shape() != [shape[0]]
        || source.selection_indices.shape() != [shape[0]]
        || u64::try_from(shape[0]).ok()
            != (coefficients[0] as u64).checked_mul(geometry.routes_per_token)
    {
        return Err(invalid());
    }
    let end = source
        .token_offset
        .checked_add(coefficients[0] as u64)
        .ok_or_else(invalid)?;
    if end > groups_shape[0] as u64 {
        return Err(invalid());
    }
    let tokens_array = source
        .token_indices
        .as_array()
        .as_dtype(Dtype::Uint32, stream)?;
    let slots_array = source
        .selection_indices
        .as_array()
        .as_dtype(Dtype::Uint32, stream)?;
    let tokens = tokens_array.evaluated()?;
    let slots = slots_array.evaluated()?;
    let mut at = Vec::with_capacity(shape[0] as usize);
    for (token, selection) in tokens
        .try_iter::<u32>()
        .map_err(eredu_nn::Error::backend_source)?
        .zip(
            slots
                .try_iter::<u32>()
                .map_err(eredu_nn::Error::backend_source)?,
        )
    {
        if token as u64 >= coefficients[0] as u64
            || selection as u64 / geometry.routes_per_token != token as u64
        {
            return Err(invalid());
        }
        at.push(
            (source.token_offset + token as u64)
                .checked_mul(geometry.routes_per_token)
                .and_then(|n| n.checked_add(selection as u64 % geometry.routes_per_token))
                .ok_or_else(invalid)?,
        );
    }
    let group_array = source
        .source_groups
        .as_array()
        .reshape(&[-1], stream)?
        .take(&indices(&at)?, stream)?
        .as_dtype(Dtype::Uint32, stream)?;
    let groups = group_array.evaluated()?;
    let mut rows = Vec::with_capacity(shape[0] as usize);
    for ((token, selection), group) in tokens
        .try_iter::<u32>()
        .map_err(eredu_nn::Error::backend_source)?
        .zip(
            slots
                .try_iter::<u32>()
                .map_err(eredu_nn::Error::backend_source)?,
        )
        .zip(
            groups
                .try_iter::<u32>()
                .map_err(eredu_nn::Error::backend_source)?,
        )
    {
        let expert = match source.global_groups {
            Some(map) => *map.get(group as usize).ok_or_else(invalid)? as u64,
            None => group as u64,
        };
        if expert >= geometry.experts {
            return Err(invalid());
        }
        rows.push(RoutedUnitLocation {
            source_peer: None,
            token: source.token_offset + token as u64,
            slot: selection as u64 % geometry.routes_per_token,
            expert,
        });
    }
    Ok(RoutedUnitLocations {
        source_token_range: [source.token_offset, end],
        rows,
    })
}

pub(super) fn usage(
    routes: u64,
    units: u64,
    action: &InterventionAction,
) -> Result<CaptureUsage, CaptureError> {
    let values = mul(routes, units)?;
    let compact = match action {
        InterventionAction::MaskComponents { indices, .. } => mul(indices.len() as u64, 64)?,
        _ => 0,
    };
    Ok(CaptureUsage {
        captures: 0,
        // Full source, gather/selection, native exact upload, replacement,
        // scatter and all lazy index/where temporaries across every chunk.
        retained_bytes: add(4096, add(mul(values, 128)?, mul(routes, 96)?)?)?,
        // Route copies/duplicate validation, flat indices/payload offsets,
        // compact sets and gathered exact F32/F16/BF16 host payloads.
        host_bytes: add(
            4096,
            add(compact, add(mul(values, 64)?, mul(routes, 256)?)?)?,
        )?,
        encoded_bytes: 0,
    })
}

pub(super) fn partition_locations(
    source: &PartitionRoutedUnitCaptureSource<'_, MlxTensor>,
    geometry: RoutedUnitGeometry,
    stream: &Stream,
) -> Result<RoutedUnitLocations, Error> {
    let local = RoutedUnitGeometry {
        units_per_expert: source.unit_coordinates.local_count() as u64,
        routes_per_token: if source.origins.is_some() {
            1
        } else {
            geometry.routes_per_token
        },
        ..geometry
    };
    let mut locations = locations(&source.source, local, stream)?;
    if let Some(origins) = source.origins {
        for row in &mut locations.rows {
            let origin = origins
                .resolve(
                    usize::try_from(row.token)
                        .map_err(|_| Error::observation(CaptureError::Overflow))?,
                )
                .ok_or_else(|| {
                    Error::observation(CaptureError::Invalid("sparse edit origin is absent".into()))
                })?;
            row.source_peer = origin.source_peer.map(|peer| peer as u64);
            row.token = origin.token as u64;
            row.slot = origin.slot as u64;
        }
    }
    Ok(locations)
}
