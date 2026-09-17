//! Sparse global-coordinate lowering and ordinary run ownership.
use super::*;
mod worker;
pub(crate) mod progress;
mod prepared;
pub use prepared::{PreparedRoutedInterventionRows, PreparedRoutedIntervention, PreparedRoutedInterventionError};
pub use worker::{RoutedInterventionLoweringError, routed_intervention_full_component_count, routed_intervention_full_component_count_control_bytes};

/// Numerical operation after sparse row selection. Mask predicates only select
/// the overwritten elements; they share the same zero worker after lowering.
/// Payload variants still borrow the actual immutable source tensor.
#[derive(Clone, Copy)]
pub enum RoutedInterventionNumericalAction<'a> {
    /// Zero all lowered selected elements.
    Zero(InterventionDtype),
    /// Scale the lowered selected elements by this exact source scalar.
    Scale(InterventionDtype, f32),
    /// Gather the actual replacement payload in selected native-row order.
    Replace(&'a InterventionTensor),
    /// Gather the actual additive payload in selected native-row order.
    Add(&'a InterventionTensor),
}
impl<'a> RoutedInterventionNumericalAction<'a> {
    /// Shared ordinary/paid lowering classification. This is descriptive and
    /// supplies neither selected indices nor a native execution permission.
    pub fn from_action(action: &'a InterventionAction) -> Result<Self, RoutedInterventionLoweringError> {
        use InterventionAction as A;
        Ok(match action {
            A::Zero { dtype } | A::Mask { dtype, .. } | A::MaskComponents { dtype, .. } => Self::Zero(*dtype),
            A::Scale { dtype, factor } => Self::Scale(*dtype, *factor),
            A::Replace { tensor } => Self::Replace(tensor),
            A::Add { tensor } => Self::Add(tensor),
            _ => return Err(RoutedInterventionLoweringError::Coordinates),
        })
    }
}

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
    geometry.components()?;
    worker::validate_slice(geometry, slice)?;
    let dtype = action.dtype().ok_or_else(|| {
        CaptureError::Invalid("invalid sparse intervention coordinates".into())
    })?;
    action.validate_activation_region(dtype, &slice.shape)?;
    worker::lower(geometry, rows, slice, action, coordinates, &mut worker::Ordinary)
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
        let window_geometry = self
            .invocation_window
            .map(|window| window.validate(tensor_geometry))
            .transpose()?;
        let logical_geometry = window_geometry.unwrap_or(tensor_geometry);
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
                let shape = logical_geometry
                    .resolve(&point.observation_geometry())?
                    .ok_or_else(|| CaptureError::Invalid("unknown sparse unit extent".into()))?;
                // Static ordinary admission keeps its original shape owner and
                // None invocation identity. Only an explicit window needs a
                // second physical extent alongside the logical source extent.
                let window_shape = window_geometry
                    .map(|_| {
                        tensor_geometry
                            .resolve(&point.observation_geometry())?
                            .ok_or_else(|| {
                                CaptureError::Invalid("unknown physical sparse unit extent".into())
                            })
                    })
                    .transpose()?;
                let physical_shape = window_shape.as_deref().unwrap_or(&shape);
                let slice = run.plan.validate_at(
                    index,
                    self.phase,
                    self.prediction,
                    window_geometry.or(self.invocation),
                    &shape,
                    Some(dtype),
                )?;
                let end = add(source.token_offset, actual[0] / geometry.routes_per_token)?;
                let progress = progress::prepare(record.routed_units, physical_shape[0], [source.token_offset,end]).map_err(CaptureError::from)?;
                if record.routed_units.is_none() {
                    let usage = if self.invocation_window.is_some() {
                        let usage = run.estimator.window_routed_unit_usage(
                            geometry,
                            &shape,
                            physical_shape,
                            &slice,
                            &operation.action,
                        )?;
                        if usage.captures != 0 || usage.encoded_bytes != 0 {
                            return Err(CaptureError::Invalid(
                                "sparse window estimate includes record encoding".into(),
                            )
                            .into());
                        }
                        usage
                    } else {
                        cost(
                            run.estimator.as_ref(),
                            geometry,
                            &shape,
                            &slice,
                            &operation.action,
                        )?
                    };
                    reserve_envelope(&mut self.ledger, usage)?;
                    record.charged = record.charged.checked_add(usage)?;
                    record.routed_units = Some(progress);
                }
                let unavailable = || {
                    CaptureError::Unsupported("sparse native editing primitive unavailable".into())
                };
                let mut locations = backend
                    .routed_unit_locations(source, geometry)
                    .ok_or_else(unavailable)?
                    .map_err(Backend)?;
                if locations.source_token_range != [source.token_offset, end] {
                    return Err(CaptureError::Invalid(
                        "sparse native receipt changed token range".into(),
                    )
                    .into());
                }
                if let Some(window) = self.invocation_window {
                    // Validate the actual local receipt before mapping it in place;
                    // no duplicate coordinate buffer or global tensor is created.
                    if locations
                        .rows
                        .iter()
                        .any(|row| row.token < source.token_offset || row.token >= end)
                    {
                        return Err(CaptureError::Invalid(
                            "sparse local coordinate exceeds its actual chunk".into(),
                        )
                        .into());
                    }
                    for row in &mut locations.rows {
                        row.token = add(row.token, window.start)?;
                    }
                    locations.source_token_range = [
                        add(source.token_offset, window.start)?,
                        add(end, window.start)?,
                    ];
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
                *receipt = progress::advance(*receipt,[source.token_offset,end],lowered.indices.len() as u64).map_err(CaptureError::from)?;
                if end == physical_shape[0] {
                    record.outcome = progress::outcome(*receipt).map_err(CaptureError::from)?;
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
