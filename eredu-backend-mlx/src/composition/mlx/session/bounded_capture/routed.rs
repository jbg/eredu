//! Bounded native route metadata and selected-unit copies. No expert-dense tensor.
use super::*;

#[cfg(test)]
mod tests;

pub(super) fn estimate_partition(
    request: &PartitionRoutedUnitCaptureRequest<'_>,
) -> Result<CaptureUsage, CaptureError> {
    request.validate()?;
    estimate_storage(&request.native_shape()?, request.slice, true)
}

pub(super) fn estimate(
    shape: &[u64],
    slice: &ResolvedCaptureSlice,
) -> Result<CaptureUsage, CaptureError> {
    if slice.strides.iter().any(|n| *n > i32::MAX as u64) {
        return Err(CaptureError::Unsupported(
            "MLX routed-unit capture stride".into(),
        ));
    }
    estimate_storage(shape, slice, false)
}

fn estimate_storage(
    shape: &[u64],
    slice: &ResolvedCaptureSlice,
    partitioned: bool,
) -> Result<CaptureUsage, CaptureError> {
    if shape.len() != 3
        || slice.shape.len() != 3
        || shape.iter().any(|n| *n > i32::MAX as u64)
        || elements(shape)? > i32::MAX as u64
    {
        return Err(CaptureError::Unsupported(
            "MLX routed-unit capture geometry".into(),
        ));
    }
    let all_routes = mul(shape[0], shape[1])?;
    let routes = if partitioned && slice.shape[2] == 0 {
        0
    } else {
        mul(slice.shape[0], slice.shape[1])?
    };
    let values = mul(routes, slice.shape[2])?;
    if routes == 0 {
        return Ok(CaptureUsage {
            captures: 1,
            retained_bytes: 0,
            host_bytes: add(4096, mul(shape[0], 32)?)?,
            encoded_bytes: add(4096, mul(shape[0], 64)?)?,
        });
    }
    // Full invocation source and conversion backing, bounded route-index arrays,
    // every row envelope and duplicate-detection/sorting storage. Charged once
    // before the first chunk, including when the selected rows occur much later.
    Ok(CaptureUsage {
        captures: 1,
        retained_bytes: add(
            4096,
            add(mul(elements(shape)?, 24)?, mul(all_routes, 192)?)?,
        )?,
        host_bytes: add(
            4096,
            add(
                mul(all_routes, 128)?,
                add(mul(routes, 512)?, mul(values, 16)?)?,
            )?,
        )?,
        // Native chunk evidence covers the whole invocation, even when the
        // exported token selection is tiny or occurs only in its last chunk.
        encoded_bytes: add(
            4096,
            add(
                mul(shape[0], 64)?,
                add(mul(routes, 768)?, mul(values, 32)?)?,
            )?,
        )?,
    })
}

pub(super) fn capture(
    source: &RoutedUnitCaptureSource<'_, MlxTensor>,
    geometry: RoutedUnitGeometry,
    slice: &ResolvedCaptureSlice,
    stream: &Stream,
) -> Result<RoutedUnitCapture, Error> {
    capture_inner(source, geometry, slice, None, stream)
}

pub(super) fn capture_partition(
    source: &PartitionRoutedUnitCaptureSource<'_, MlxTensor>,
    request: &PartitionRoutedUnitCaptureRequest<'_>,
    stream: &Stream,
) -> Result<RoutedUnitCapture, Error> {
    estimate_partition(request).map_err(Error::observation)?;
    if source.unit_coordinates != request.ownership.coordinates.units() {
        return Err(Error::observation(CaptureError::Invalid(
            "routed source differs from prepared scalar placement".into(),
        )));
    }
    capture_inner(
        &source.source,
        request.geometry,
        request.slice,
        Some((source, request)),
        stream,
    )
}

fn capture_inner(
    source: &RoutedUnitCaptureSource<'_, MlxTensor>,
    geometry: RoutedUnitGeometry,
    slice: &ResolvedCaptureSlice,
    partition: Option<(
        &PartitionRoutedUnitCaptureSource<'_, MlxTensor>,
        &PartitionRoutedUnitCaptureRequest<'_>,
    )>,
    stream: &Stream,
) -> Result<RoutedUnitCapture, Error> {
    let invalid = || {
        Error::observation(CaptureError::Invalid(
            "invalid native routed-unit coordinates".into(),
        ))
    };
    geometry.validate_slice(slice).map_err(Error::observation)?;
    let shape = source.values.shape();
    let coefficient_shape = source.coefficients.shape();
    let source_shape = source.source_groups.shape();
    let local_units = partition.map_or(geometry.units_per_expert, |(source, _)| {
        source.unit_coordinates.local_count() as u64
    });
    let native_routes = partition.map_or(geometry.routes_per_token, |(_, request)| {
        if request.ownership.source_peer.is_some() {
            1
        } else {
            geometry.routes_per_token
        }
    });
    if shape.len() != 2
        || shape[1] as u64 != local_units
        || coefficient_shape.len() != 2
        || coefficient_shape[1] as u64 != native_routes
        || source_shape.len() != 2
        || source_shape[1] as u64 != native_routes
        || source.token_indices.shape() != [shape[0]]
        || source.selection_indices.shape() != [shape[0]]
        || shape[0] as u64
            > (coefficient_shape[0] as u64)
                .checked_mul(native_routes)
                .ok_or_else(invalid)?
        || !matches!(
            source.values.as_array().dtype(),
            Dtype::Float16 | Dtype::Bfloat16 | Dtype::Float32 | Dtype::Float64
        )
    {
        return Err(invalid());
    }
    let end = source
        .token_offset
        .checked_add(coefficient_shape[0] as u64)
        .ok_or_else(invalid)?;
    if end > source_shape[0] as u64 {
        return Err(invalid());
    }
    if let Some((partition_source, request)) = partition {
        let maximum = request.native_shape().map_err(Error::observation)?[0];
        if source_shape[0] as u64 > maximum {
            return Err(invalid());
        }
        match (request.ownership.source_peer, partition_source.origins) {
            (Some(_), Some(origins))
                if origins.peer_count() as u64 == request.ownership.source_peers
                    && origins.row_count() == source_shape[0] as usize
                    && origins.routes_per_token() as u64 == geometry.routes_per_token =>
            {
                ()
            }
            (None, None) if source_shape[0] as u64 == request.source_tokens => (),
            _ => return Err(invalid()),
        }
    }
    if coefficient_shape[0] == 0
        || slice.shape[0] == 0
        || slice.shape[1] == 0
        || (partition.is_some() && slice.shape[2] == 0)
    {
        return Ok(RoutedUnitCapture {
            geometry,
            source_token_ranges: if source.token_offset == end {
                vec![]
            } else {
                vec![[source.token_offset, end]]
            },
            rows: vec![],
        });
    }
    let token_array = source
        .token_indices
        .as_array()
        .as_dtype(Dtype::Uint32, stream)?;
    let tokens = token_array.evaluated()?;
    let selection_array = source
        .selection_indices
        .as_array()
        .as_dtype(Dtype::Uint32, stream)?;
    let selections = selection_array.evaluated()?;
    let token_ids = tokens.as_slice::<u32>();
    let selection_ids = selections.as_slice::<u32>();
    let mut groups_at = Vec::with_capacity(token_ids.len());
    for (&token, &selection) in token_ids.iter().zip(selection_ids) {
        if u64::from(token) >= coefficient_shape[0] as u64
            || u64::from(selection) / native_routes != u64::from(token)
        {
            return Err(invalid());
        }
        let original = source
            .token_offset
            .checked_add(u64::from(token))
            .ok_or_else(invalid)?;
        if original >= source_shape[0] as u64 {
            return Err(invalid());
        }
        let index = original
            .checked_mul(native_routes)
            .and_then(|n| n.checked_add(u64::from(selection) % native_routes))
            .and_then(|n| i32::try_from(n).ok())
            .ok_or_else(invalid)?;
        groups_at.push(index);
    }
    let group_indices = Array::from_slice(&groups_at, &[groups_at.len() as i32]);
    let group_array = source
        .source_groups
        .as_array()
        .reshape(&[-1], stream)?
        .take(&group_indices, stream)?
        .as_dtype(Dtype::Uint32, stream)?;
    let groups = group_array.evaluated()?;
    let coefficient_array = source
        .coefficients
        .as_array()
        .reshape(&[-1], stream)?
        .take(source.selection_indices.as_array(), stream)?
        .as_dtype(Dtype::Float32, stream)?;
    let coefficients = coefficient_array.evaluated()?;
    #[cfg(test)]
    for _ in 0..4 {
        record_host_read(token_ids.len());
    }
    let unit_indices = if let Some((partition_source, _)) = partition {
        let indices = (0..slice.shape[2])
            .map(|index| {
                let global = slice.starts[2] + index * slice.strides[2];
                partition_source
                    .unit_coordinates
                    .global_to_local(usize::try_from(global).map_err(|_| invalid())?)
                    .and_then(|local| i32::try_from(local).ok())
                    .ok_or_else(invalid)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Some(Array::from_slice(&indices, &[indices.len() as i32]))
    } else {
        None
    };
    let mut rows = Vec::new();
    for (index, ((&token, &selection), (&group, &coefficient))) in token_ids
        .iter()
        .zip(selection_ids)
        .zip(
            groups
                .as_slice::<u32>()
                .iter()
                .zip(coefficients.as_slice::<f32>()),
        )
        .enumerate()
    {
        let native_token = source.token_offset + u64::from(token);
        let (source_peer, token, slot) = match partition.and_then(|(source, _)| source.origins) {
            Some(origins) => {
                let origin = origins
                    .resolve(usize::try_from(native_token).map_err(|_| invalid())?)
                    .ok_or_else(invalid)?;
                (
                    origin.source_peer.map(|peer| peer as u64),
                    origin.token as u64,
                    origin.slot as u64,
                )
            }
            None => (None, native_token, u64::from(selection) % native_routes),
        };
        let expert = match source.global_groups {
            Some(map) => *map.get(group as usize).ok_or_else(invalid)? as u64,
            None => u64::from(group),
        };
        if expert >= geometry.experts || !coefficient.is_finite() {
            return Err(invalid());
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
                return Err(invalid());
            }
            if source_peer != request.ownership.source_peer {
                continue;
            }
        }
        if token < slice.starts[0]
            || token >= slice.ends[0]
            || !(token - slice.starts[0]).is_multiple_of(slice.strides[0])
            || slot < slice.starts[1]
            || slot >= slice.ends[1]
            || !(slot - slice.starts[1]).is_multiple_of(slice.strides[1])
        {
            continue;
        }
        let selected = if let Some(indices) = &unit_indices {
            source
                .values
                .as_array()
                .try_index_device((index as i32, ..), stream)?
                .take(indices, stream)?
        } else {
            source.values.as_array().try_index_device(
                (
                    index as i32,
                    (slice.starts[2] as i32..slice.ends[2] as i32)
                        .stride_by(slice.strides[2] as i32),
                ),
                stream,
            )?
        }
        .as_dtype(Dtype::Float32, stream)?
        .contiguous(false, stream)?;
        let values =
            super::super::observation::observe_tensor(&MlxTensor::from_array(selected), stream)?;
        rows.push(RoutedUnitCaptureRow {
            source_peer,
            token,
            slot,
            expert,
            coefficient,
            unit_start: slice.starts[2],
            unit_stride: slice.strides[2],
            values,
        });
    }
    Ok(RoutedUnitCapture {
        geometry,
        source_token_ranges: vec![[source.token_offset, end]],
        rows,
    })
}
