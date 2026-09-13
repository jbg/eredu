//! Sparse global-coordinate lowering and ordinary run ownership.
use super::*;
use std::collections::BTreeSet;

/// A bounded lowering recipe; it grants no admission or reservation authority.
#[derive(Debug, Clone, PartialEq)]
pub struct LoweredRoutedIntervention {
    /// Distinct indices in the native `[route_row, unit]` value tensor.
    pub indices: Vec<u64>,
    /// One-dimensional action with exact gathered payload, absent for no matches.
    pub action: Option<InterventionAction>,
}

/// Projects global expert/unit and token coordinates onto this forward's actual
/// participating rows. Callers reserve full-invocation work before collecting
/// coordinates or constructing this recipe. Native sorting never changes identity.
pub fn lower_routed_intervention(
    geometry: RoutedUnitGeometry,
    locations: &RoutedUnitLocations,
    slice: &ResolvedCaptureSlice,
    action: &InterventionAction,
) -> Result<LoweredRoutedIntervention, CaptureError> {
    let [start, end] = locations.source_token_range;
    if start >= end
        || locations.rows.len() as u64 != mul(end - start, geometry.routes_per_token)?
        || locations
            .rows
            .iter()
            .any(|row| row.source_peer.is_some() || row.token < start || row.token >= end)
    {
        return Err(CaptureError::Invalid(
            "invalid ordinary sparse chunk coverage".into(),
        ));
    }
    lower_rows(geometry, &locations.rows, slice, action, None)
}

/// Lowers participating rows for an admitted distributed owner using retained
/// global expert/unit placement. Token coordinates belong to each source peer.
/// This validates identity and local indexing, not global route coverage, peer
/// authorization, completion, transport or budgets; the distributed owner must
/// establish those separately before executing or publishing the recipe.
pub fn lower_partition_routed_intervention(
    geometry: RoutedUnitGeometry,
    source_tokens: u64,
    rows: &[RoutedUnitLocation],
    coordinates: &eredu_core::component::RoutedComponentCoordinateMap,
    slice: &ResolvedCaptureSlice,
    action: &InterventionAction,
) -> Result<LoweredRoutedIntervention, CaptureError> {
    if source_tokens == 0
        || slice.ends.first().is_none_or(|end| *end > source_tokens)
        || rows.iter().any(|row| row.token >= source_tokens)
        || coordinates.experts().global_count() as u64 != geometry.experts
        || coordinates.units().global_count() as u64 != geometry.units_per_expert
    {
        return Err(CaptureError::Invalid(
            "sparse partition geometry differs from global admission".into(),
        ));
    }
    lower_rows(geometry, rows, slice, action, Some(coordinates))
}

fn lower_rows(
    geometry: RoutedUnitGeometry,
    rows: &[RoutedUnitLocation],
    slice: &ResolvedCaptureSlice,
    action: &InterventionAction,
    coordinates: Option<&eredu_core::component::RoutedComponentCoordinateMap>,
) -> Result<LoweredRoutedIntervention, CaptureError> {
    let components = geometry.components()?;
    let invalid = || CaptureError::Invalid("invalid sparse intervention coordinates".into());
    if [&slice.starts, &slice.ends, &slice.strides, &slice.shape]
        .iter()
        .any(|v| v.len() != 2)
        || slice.ends[1] > components
    {
        return Err(invalid());
    }
    for axis in 0..2 {
        if slice.strides[axis] == 0
            || slice.starts[axis] >= slice.ends[axis]
            || (slice.ends[axis] - slice.starts[axis]).div_ceil(slice.strides[axis])
                != slice.shape[axis]
        {
            return Err(invalid());
        }
    }
    let dtype = action.dtype().ok_or_else(invalid)?;
    action.validate_activation_region(dtype, &slice.shape)?;
    let compact: Option<(BTreeSet<u32>, bool)> = match action {
        InterventionAction::MaskComponents {
            indices,
            keep_selected,
            ..
        } => {
            if slice.starts[1] != 0 || slice.ends[1] != components || slice.strides[1] != 1 {
                return Err(invalid());
            }
            Some((indices.iter().copied().collect(), *keep_selected))
        }
        InterventionAction::MaskLogits { .. } => return Err(invalid()),
        _ => None,
    };
    let local_units = coordinates.map_or(geometry.units_per_expert, |map| {
        map.units().local_count() as u64
    });
    let mut seen = BTreeSet::new();
    let mut indices = Vec::new();
    let mut payload_indices = Vec::new();
    for (row_index, row) in rows.iter().enumerate() {
        if row.slot >= geometry.routes_per_token
            || row.expert >= geometry.experts
            || coordinates.is_some_and(|map| {
                usize::try_from(row.expert)
                    .ok()
                    .and_then(|expert| map.experts().global_to_local(expert))
                    .is_none()
            })
            || !seen.insert((row.source_peer, row.token, row.slot))
        {
            return Err(invalid());
        }
        if row.token < slice.starts[0]
            || row.token >= slice.ends[0]
            || !(row.token - slice.starts[0]).is_multiple_of(slice.strides[0])
        {
            continue;
        }
        for local_unit in 0..local_units {
            let unit = match coordinates {
                Some(map) => map
                    .units()
                    .local_to_global(local_unit as usize)
                    .ok_or_else(invalid)? as u64,
                None => local_unit,
            };
            let component = add(mul(row.expert, geometry.units_per_expert)?, unit)?;
            if component < slice.starts[1]
                || component >= slice.ends[1]
                || !(component - slice.starts[1]).is_multiple_of(slice.strides[1])
            {
                continue;
            }
            let payload_index = add(
                mul(
                    (row.token - slice.starts[0]) / slice.strides[0],
                    slice.shape[1],
                )?,
                (component - slice.starts[1]) / slice.strides[1],
            )?;
            let payload_index =
                usize::try_from(payload_index).map_err(|_| CaptureError::Overflow)?;
            if let Some((set, keep)) = &compact {
                let selected = u32::try_from(component)
                    .ok()
                    .is_some_and(|n| set.contains(&n));
                if selected == *keep {
                    continue;
                }
            }
            if let InterventionAction::Mask { keep, .. } = action {
                if *keep.get(payload_index).ok_or_else(invalid)? {
                    continue;
                }
            }
            indices.push(add(mul(row_index as u64, local_units)?, local_unit)?);
            payload_indices.push(payload_index);
        }
    }
    let n = indices.len() as u64;
    if n == 0 {
        return Ok(LoweredRoutedIntervention {
            indices,
            action: None,
        });
    }
    let gather = |tensor: &InterventionTensor| -> Result<InterventionTensor, CaptureError> {
        macro_rules! gather {
            ($values:expr, $kind:ident) => {
                InterventionValues::$kind(
                    payload_indices
                        .iter()
                        .map(|i| $values.get(*i).copied().ok_or_else(invalid))
                        .collect::<Result<Vec<_>, _>>()?,
                )
            };
        }
        Ok(InterventionTensor {
            shape: vec![n],
            values: match &tensor.values {
                InterventionValues::Float32(v) => gather!(v, Float32),
                InterventionValues::Float16(v) => gather!(v, Float16),
                InterventionValues::Bfloat16(v) => gather!(v, Bfloat16),
            },
        })
    };
    let action = match action {
        InterventionAction::Zero { .. }
        | InterventionAction::Mask { .. }
        | InterventionAction::MaskComponents { .. } => InterventionAction::Zero { dtype },
        InterventionAction::Scale { factor, .. } => InterventionAction::Scale {
            dtype,
            factor: *factor,
        },
        InterventionAction::Replace { tensor } => InterventionAction::Replace {
            tensor: gather(tensor)?,
        },
        InterventionAction::Add { tensor } => InterventionAction::Add {
            tensor: gather(tensor)?,
        },
        _ => return Err(invalid()),
    };
    Ok(LoweredRoutedIntervention {
        indices,
        action: Some(action),
    })
}

pub(super) fn cost(
    estimator: &dyn InterventionEstimator,
    geometry: RoutedUnitGeometry,
    source: &[u64],
    slice: &ResolvedCaptureSlice,
    action: &InterventionAction,
) -> Result<CaptureUsage, CaptureError> {
    let cost = estimator.routed_unit_usage(geometry, source, slice, action)?;
    if cost.captures != 0 || cost.encoded_bytes != 0 {
        return Err(CaptureError::Invalid(
            "sparse edit estimate includes record encoding".into(),
        ));
    }
    Ok(cost)
}

impl CaptureSession {
    pub(crate) fn wants_routed_interventions(&self, routing: &str) -> bool {
        self.interventions.as_ref().is_some_and(|run| {
            run.plan
                .points()
                .iter()
                .zip(run.records.iter().flatten())
                .any(|(point, record)| {
                    point
                        .routed_units
                        .as_ref()
                        .is_some_and(|p| p.routing == routing)
                        && record.outcome != InterventionOutcome::Inactive
                })
        })
    }

    pub(crate) fn intervene_routed_units<B: InterventionBackend>(
        &mut self,
        backend: &mut B,
        routing: &str,
        source: &RoutedUnitCaptureSource<'_, B::Tensor>,
    ) -> Result<Option<B::Tensor>, CaptureExecutionError<B::Error>> {
        use CaptureExecutionError::Backend;
        self.validate_ordinary_intervention()?;
        let tensor_geometry = self.tensor_geometry()?;
        let Some(run) = &mut self.interventions else {
            return Ok(None);
        };
        let records = run
            .records
            .as_mut()
            .ok_or_else(|| CaptureError::Invalid("intervention step not started".into()))?;
        let mut effective = None;
        for (index, ((operation, point), record)) in run
            .plan
            .plan()
            .operations
            .iter()
            .zip(run.plan.points())
            .zip(records)
            .enumerate()
        {
            let Some(routed) = &point.routed_units else {
                continue;
            };
            if routed.routing != routing || record.outcome == InterventionOutcome::Inactive {
                continue;
            }
            let started = std::time::Instant::now();
            let result = (|| {
                if record.outcome != InterventionOutcome::Missing {
                    return Err(CaptureError::Invalid(
                        "sparse intervention already completed or failed".into(),
                    )
                    .into());
                }
                let input = effective.as_ref().unwrap_or(source.values);
                let actual = backend.shape(input).map_err(Backend)?;
                let dtype = backend.intervention_dtype(input).map_err(Backend)?;
                let geometry = routed.geometry;
                if actual.len() != 2
                    || actual[1] != geometry.units_per_expert
                    || actual[0] == 0
                    || !actual[0].is_multiple_of(geometry.routes_per_token)
                {
                    return Err(
                        CaptureError::Invalid("invalid native sparse unit shape".into()).into(),
                    );
                }
                let shape = tensor_geometry
                    .resolve(&point.observation_geometry())?
                    .ok_or_else(|| CaptureError::Invalid("unknown sparse unit extent".into()))?;
                let slice = run.plan.validate_at(
                    index,
                    self.phase,
                    self.prediction,
                    self.invocation,
                    &shape,
                    Some(dtype),
                )?;
                let end = add(source.token_offset, actual[0] / geometry.routes_per_token)?;
                if source.token_offset
                    != record
                        .routed_units
                        .as_ref()
                        .map_or(0, |r| r.completed_tokens)
                    || end > shape[0]
                {
                    return Err(CaptureError::Invalid(
                        "sparse intervention chunk gap, overlap or excess".into(),
                    )
                    .into());
                }
                if record.routed_units.is_none() {
                    let usage = cost(
                        run.estimator.as_ref(),
                        geometry,
                        &shape,
                        &slice,
                        &operation.action,
                    )?;
                    reserve_envelope(&mut self.ledger, usage)?;
                    record.charged = record.charged.checked_add(usage)?;
                    record.routed_units = Some(RoutedUnitInterventionReceipt {
                        source_tokens: shape[0],
                        completed_tokens: 0,
                        affected_values: 0,
                    });
                }
                let unavailable = || {
                    CaptureError::Unsupported("sparse native editing primitive unavailable".into())
                };
                let locations = backend
                    .routed_unit_locations(source, geometry)
                    .ok_or_else(unavailable)?
                    .map_err(Backend)?;
                if locations.source_token_range != [source.token_offset, end] {
                    return Err(CaptureError::Invalid(
                        "sparse native receipt changed token range".into(),
                    )
                    .into());
                }
                let lowered =
                    lower_routed_intervention(geometry, &locations, &slice, &operation.action)?;
                let output = if let Some(action) = lowered.action {
                    let selected = backend
                        .select_elements(input, &lowered.indices)
                        .ok_or_else(unavailable)?
                        .map_err(Backend)?;
                    let n = lowered.indices.len() as u64;
                    if backend.shape(&selected).map_err(Backend)? != [n]
                        || backend.intervention_dtype(&selected).map_err(Backend)? != dtype
                    {
                        return Err(CaptureError::Invalid(
                            "sparse gather changed shape or dtype".into(),
                        )
                        .into());
                    }
                    let local = ResolvedCaptureSlice {
                        starts: vec![0],
                        ends: vec![n],
                        strides: vec![1],
                        shape: vec![n],
                    };
                    let replacement = apply_activation(backend, &selected, &action, &local)?;
                    let output = backend
                        .update_elements(input, &lowered.indices, &replacement)
                        .ok_or_else(unavailable)?
                        .map_err(Backend)?;
                    if backend.shape(&output).map_err(Backend)? != actual
                        || backend.intervention_dtype(&output).map_err(Backend)? != dtype
                    {
                        return Err(CaptureError::Invalid(
                            "sparse edit changed shape or dtype".into(),
                        )
                        .into());
                    }
                    Some(output)
                } else {
                    None
                };
                let receipt = record.routed_units.as_mut().unwrap();
                receipt.completed_tokens = end;
                receipt.affected_values =
                    add(receipt.affected_values, lowered.indices.len() as u64)?;
                if end == shape[0] {
                    record.outcome = if receipt.affected_values == 0 {
                        InterventionOutcome::Unmatched
                    } else {
                        InterventionOutcome::Applied
                    };
                }
                Ok(output)
            })();
            self.capture_seconds += started.elapsed().as_secs_f64();
            match result {
                Ok(Some(value)) => effective = Some(value),
                Ok(None) => (),
                Err(error) => {
                    record.outcome = InterventionOutcome::Failed {
                        message: bounded_diagnostic(&error),
                    };
                    return Err(error);
                }
            }
        }
        Ok(effective)
    }
}
