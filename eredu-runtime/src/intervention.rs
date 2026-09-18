//! Prospective intervention scheduling under the ordinary capture/run owner.
//! There is no additional sampling loop or native-resource owner here.

mod static_preflight;
pub use static_preflight::{StaticInterventionPreflight,StaticInterventionScratchError,StaticInterventionPreflightError};
mod hook;
mod prefill;
pub use prefill::{InterventionPrefillWindow,InterventionPrefillSourceError,InterventionPrefillProjectionError};
use crate::capture::{
    CaptureExecutionError, CaptureSession, bounded_diagnostic, metadata_reservation,
};
use eredu_core::{capture::*, intervention::*};
pub use hook::{ActivationHook, activation_hook};

mod activation;
mod partition;
pub(crate) mod routed;
mod session;
pub use activation::{
    apply_activation, apply_activation_with_source_shape, localize_component_mask,
};
pub use partition::{
    PartitionActivationLayout, PartitionActivationMember, PartitionActivationProjection,
    PartitionRoutedActivationMember, ReservedPartitionActivation,
    PreparedPartitionInterventionProjection, PartitionInterventionProjectionSourceError, PartitionInterventionUpdate,
    PartitionInterventionProjectionCost, PartitionInterventionColumnError, validate_partition_column_region, intervention_window_metadata,
    PreparedWindowInterventionPayload, PreparedWindowInterventionPayloadError, WindowInterventionPayloadError,
};
pub use routed::{
    RoutedInterventionNumericalAction, routed_intervention_full_component_count, routed_intervention_full_component_count_control_bytes,
    LoweredRoutedIntervention, lower_partition_routed_intervention, lower_routed_intervention,
    PreparedRoutedInterventionRows, PreparedRoutedIntervention, PreparedRoutedInterventionError,
    RoutedInterventionLoweringError,
};
pub(crate) use session::validate_continuation;
pub use session::{CaptureObserver, install_session, validate_session};

pub(crate) struct InterventionRun {
    pub(crate) plan: AdmittedInterventionPlan,
    pub(crate) records: Option<Vec<InterventionRecord>>,
    pub(crate) routing_pending: Option<PendingRouting>,
    pub(crate) estimator: std::sync::Arc<dyn InterventionEstimator>,
}

#[derive(Clone, Copy)]
pub(crate) struct PendingRouting {
    index: usize,
    phase: CapturePhase,
    prediction: u64,
    invocation: Option<CaptureInvocationShape>,
    window: Option<CaptureInvocationWindow>,
    unmodified: bool,
}

impl InterventionRun {
    pub(crate) fn begin_step(
        &mut self,
        ledger: &mut CaptureLedger,
        phase: CapturePhase,
        prediction: u64,
    ) -> Result<(), CaptureError> {
        if self.records.is_some() {
            return Err(CaptureError::Invalid(
                "previous intervention step has not been consumed".into(),
            ));
        }
        let mut records = Vec::new();
        for (operation, point) in self.plan.plan().operations.iter().zip(self.plan.points()) {
            let charged = intervention_metadata(operation, point, self.plan.identity())?;
            reserve_envelope(ledger, charged)?;
            let active = operation.schedule.includes(phase, prediction);
            let mut evidence = Vec::new();
            for (selection, geometry) in evidence_selections(operation, point) {
                let charged = metadata_reservation(&selection, &geometry)?;
                reserve_envelope(ledger, charged)?;
                evidence.push(CaptureRecord {
                    schema_version: CAPTURE_SCHEMA_VERSION,
                    selection_id: selection.id,
                    path: geometry.path,
                    node_id: geometry.node_id,
                    position: geometry.position,
                    source_shape: None,
                    source_dtype: None,
                    selected_shape: None,
                    outcome: if active {
                        CaptureOutcome::Missing
                    } else {
                        CaptureOutcome::Skipped {
                            reason: CaptureSkipReason::Schedule,
                        }
                    },
                    payload: None,
                    charged,
                });
            }
            records.push(initial_record(
                operation,
                point,
                self.plan.identity(),
                phase,
                prediction,
                evidence,
                charged,
                str::to_owned,
            ));
        }
        self.records = Some(records);
        self.routing_pending = None;
        Ok(())
    }

    pub(crate) fn take_records(&mut self) -> Vec<InterventionRecord> {
        self.records.take().unwrap_or_default()
    }
}

// One attribution/schedule constructor for ordinary and prepaid records. The
// caller supplies its actual string destination; evidence is already owned.
pub(crate) fn initial_record(
    operation: &InterventionOperation,
    point: &InterventionPoint,
    identity: &str,
    phase: CapturePhase,
    prediction: u64,
    evidence: Vec<CaptureRecord>,
    charged: CaptureUsage,
    mut text: impl FnMut(&str) -> String,
) -> InterventionRecord {
    InterventionRecord {
        routed_units: None,
        schema_version: INTERVENTION_SCHEMA_VERSION,
        plan_id: text(identity),
        operation_id: text(&operation.id),
        target: text(&operation.target),
        node_id: text(&point.node_id),
        phase,
        prediction_index: prediction,
        outcome: if operation.schedule.includes(phase, prediction) {
            InterventionOutcome::Missing
        } else {
            InterventionOutcome::Inactive
        },
        evidence,
        charged,
    }
}
impl CaptureSession {
    /// Selects an already admitted plan between drained prediction steps. Used
    /// when speculative target/draft rows or isolated branches share one ledger.
    /// No observation budget or completed evidence is reset by a role switch.
    pub fn select_prediction_interventions(
        &mut self,
        plan: Option<AdmittedInterventionPlan>,
        estimator: std::sync::Arc<dyn InterventionEstimator>,
    ) -> Result<(), CaptureError> {
        if !self.checkpoint_ready || self.records.is_some() {
            return Err(CaptureError::Invalid(
                "intervention selection requires a drained successful step".into(),
            ));
        }
        if let Some(plan) = &plan {
            validate_capture_origin(&self.plan, plan)?;
        }
        if plan.as_ref().is_some_and(|plan| {
            plan.request() != self.plan.request()
                || plan.invocation_bounds() != self.plan.invocation_bounds()
        }) {
            return Err(CaptureError::Invalid(
                "capture and intervention request geometry differs".into(),
            ));
        }
        self.interventions = plan.filter(|p| !p.is_empty()).map(|plan| InterventionRun {
            plan,
            records: None,
            routing_pending: None,
            estimator,
        });
        Ok(())
    }
    /// Installs an immutable intervention plan before the first step. Both plans
    /// share one ledger; capture-none budgets must still allow outcome metadata.
    pub fn enable_interventions(
        &mut self,
        plan: AdmittedInterventionPlan,
        estimator: std::sync::Arc<dyn InterventionEstimator>,
    ) -> Result<(), CaptureError> {
        if self.records.is_some()
            || self.interventions.is_some()
            || self.ledger.total() != CaptureUsage::default()
        {
            return Err(CaptureError::Invalid(
                "interventions must be installed once before generation".into(),
            ));
        }
        validate_capture_origin(&self.plan, &plan)?;
        if plan.request() != self.plan.request()
            || plan.invocation_bounds() != self.plan.invocation_bounds()
        {
            return Err(CaptureError::Invalid(
                "capture and intervention request geometry differs".into(),
            ));
        }
        if !plan.is_empty() {
            self.interventions = Some(InterventionRun {
                plan,
                records: None,
                routing_pending: None,
                estimator,
            });
        }
        Ok(())
    }

    /// Applies scheduled activation operations in plan order. Ordinary observation
    /// must precede this call at the exact same hook. Returns no replacement when
    /// no operation is active, avoiding source cloning and native materialization.
    pub fn intervene<B: InterventionBackend>(
        &mut self,
        backend: &mut B,
        path: &str,
        tensor: &B::Tensor,
    ) -> Result<Option<B::Tensor>, CaptureExecutionError<B::Error>> {
        self.validate_ordinary_intervention()?;
        let tensor_geometry = self.tensor_geometry()?;
        let invocation_window = self.invocation_window;
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
            match activation_hook(operation, point, &record.outcome, path) {
                ActivationHook::Unrelated => continue,
                ActivationHook::Active => (),
                ActivationHook::Repeated => {
                    return Err(CaptureError::Invalid(
                        "intervention point executed more than once in one step".into(),
                    )
                    .into());
                }
            }
            if let Some(window) = invocation_window {
                let started = std::time::Instant::now();
                let result = (|| {
                    let input = effective.as_ref().unwrap_or(tensor);
                    let metadata = intervention_window_metadata(operation, point)?;
                    reserve_envelope(&mut self.ledger, metadata)?;
                    record.charged = record.charged.checked_add(metadata)?;
                    let shape = backend
                        .shape(input)
                        .map_err(CaptureExecutionError::Backend)?;
                    let geometry = point.observation_geometry();
                    let (global, axis, coordinates) =
                        window.source_axes(tensor_geometry, &geometry, &shape)?;
                    let projected = partition::PartitionActivationProjection::new_at(
                        &run.plan,
                        index,
                        self.phase,
                        self.prediction,
                        Some(window.validate(tensor_geometry)?),
                        &global,
                        axis,
                        &coordinates,
                        1,
                    )?;
                    let work = projected.reserve(&mut self.ledger, run.estimator.as_ref())?;
                    record.charged = record.charged.checked_add(work.charged())?;
                    work.validate_source(backend, input)?;
                    let evidence = evidence_selections(operation, point);
                    if let Some((selection, geometry)) = evidence.first() {
                        capture_evidence_in_window(
                            Some(window),
                            backend,
                            input,
                            selection,
                            geometry,
                            &mut record.evidence[0],
                            tensor_geometry,
                            &mut self.ledger,
                        )?;
                    }
                    let output = work.apply(backend, input)?;
                    if let Some((selection, geometry)) = evidence.get(1) {
                        capture_evidence_in_window(
                            Some(window),
                            backend,
                            output.as_ref().unwrap_or(input),
                            selection,
                            geometry,
                            &mut record.evidence[1],
                            tensor_geometry,
                            &mut self.ledger,
                        )?;
                    }
                    Ok(output)
                })();
                self.capture_seconds += started.elapsed().as_secs_f64();
                match result {
                    Ok(output) => {
                        if let Some(output) = output {
                            effective = Some(output);
                        }
                        record.outcome = InterventionOutcome::Applied;
                    }
                    Err(error) => {
                        record.outcome = InterventionOutcome::Failed {
                            message: bounded_diagnostic(&error),
                        };
                        return Err(error);
                    }
                }
                continue;
            }
            let started = std::time::Instant::now();
            let result = (|| {
                let input = effective.as_ref().unwrap_or(tensor);
                let shape = backend
                    .shape(input)
                    .map_err(CaptureExecutionError::Backend)?;
                let dtype = backend
                    .intervention_dtype(input)
                    .map_err(CaptureExecutionError::Backend)?;
                let slice = run.plan.validate_at(
                    index,
                    self.phase,
                    self.prediction,
                    self.invocation,
                    &shape,
                    Some(dtype),
                )?;
                run.estimator.validate_geometry(&shape, &slice)?;
                let usage =
                    activation_cost(run.estimator.as_ref(), &shape, &slice, &operation.action)?;
                reserve_envelope(&mut self.ledger, usage)?;
                record.charged = record.charged.checked_add(usage)?;
                let evidence = evidence_selections(operation, point);
                if let Some((selection, geometry)) = evidence.first() {
                    capture_evidence(
                        backend,
                        input,
                        selection,
                        geometry,
                        &mut record.evidence[0],
                        tensor_geometry,
                        &mut self.ledger,
                    )?;
                }
                let output = apply_activation(backend, input, &operation.action, &slice)?;
                // Backend conformance checks: an implementation cannot replace a
                // value with a different shape or silently promote its dtype.
                let output_shape = backend
                    .shape(&output)
                    .map_err(CaptureExecutionError::Backend)?;
                let output_dtype = backend
                    .intervention_dtype(&output)
                    .map_err(CaptureExecutionError::Backend)?;
                run.plan.validate_at(
                    index,
                    self.phase,
                    self.prediction,
                    self.invocation,
                    &output_shape,
                    Some(output_dtype),
                )?;
                if let Some((selection, geometry)) = evidence.get(1) {
                    capture_evidence(
                        backend,
                        &output,
                        selection,
                        geometry,
                        &mut record.evidence[1],
                        tensor_geometry,
                        &mut self.ledger,
                    )?;
                }
                Ok(output)
            })();
            self.capture_seconds += started.elapsed().as_secs_f64();
            match result {
                Ok(output) => {
                    effective = Some(output);
                    record.outcome = InterventionOutcome::Applied;
                }
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

    /// Checks scheduled targets before committing a prediction. Missing targets are
    /// preserved in records and fail the attempt; this does not establish rollback.
    pub fn finish_interventions(&self) -> Result<(), CaptureError> {
        if let Some(run) = &self.interventions {
            if run.routing_pending.is_some() {
                return Err(CaptureError::Invalid(
                    "routing intervention has not resolved before the boundary".into(),
                ));
            }
            let records = run
                .records
                .as_ref()
                .ok_or_else(|| CaptureError::Invalid("intervention step not started".into()))?;
            if let Some((_, record)) = records.iter().enumerate().find(|(index, r)| {
                matches!(
                    r.outcome,
                    InterventionOutcome::Missing | InterventionOutcome::Failed { .. }
                ) && !self.partition_intervention_completed(*index)
            }) {
                return Err(CaptureError::Invalid(format!(
                    "scheduled intervention {} at {} did not complete",
                    record.operation_id, record.target
                )));
            }
        }
        Ok(())
    }
}

/// Reserve the shared logical intervention envelope; a refusal never refunds.
pub fn reserve_envelope(
    ledger: &mut impl CaptureReservation,
    usage: CaptureUsage,
) -> Result<(), CaptureError> {
    match ledger.reserve(usage)? {
        Some(CaptureSkipReason::Limit { budget, cumulative }) => {
            Err(CaptureError::Limit { budget, cumulative })
        }
        _ => Ok(()),
    }
}

/// Same fixed attributed-record cost for cold inspection and execution.
pub fn intervention_metadata(
    operation: &InterventionOperation,
    point: &InterventionPoint,
    identity: &str,
) -> Result<CaptureUsage, CaptureError> {
    let strings = add(
        add(operation.id.len() as u64, point.path.len() as u64)?,
        add(point.node_id.len() as u64, identity.len() as u64)?,
    )?;
    Ok(CaptureUsage {
        captures: 0,
        retained_bytes: 0,
        host_bytes: add(1024, strings)?,
        encoded_bytes: add(4096, mul(strings, 6)?)?,
    })
}

/// Generates only declared evidence fields, with exact shape and attribution.
pub(crate) fn evidence_selections(
    operation: &InterventionOperation,
    point: &InterventionPoint,
) -> Vec<(CaptureSelection, eredu_core::ObservationPoint)> {
    let layout = eredu_core::capture::InterventionEvidenceLayout::new(operation, point);
    let mut entries = Vec::new();
    for index in 0..layout.len() {
        let descriptor = layout.descriptor(index).expect("bounded evidence ordinal");
        let mut geometry = point.observation_geometry();
        geometry.position = descriptor.position();
        geometry.dtype = descriptor.dtype();
        geometry.path = descriptor.path_parts().concat();
        entries.push((
            CaptureSelection {
                id: descriptor.selection_id_parts().concat(),
                path: geometry.path.clone(),
                schedule: operation.schedule.clone(),
                slices: operation.slices.clone(),
                transform: descriptor.transform(),
            },
            geometry,
        ));
    }
    entries
}

pub(crate) fn capture_evidence<B: CaptureBackend>(
    backend: &mut B,
    tensor: &B::Tensor,
    selection: &CaptureSelection,
    point: &eredu_core::ObservationPoint,
    record: &mut CaptureRecord,
    geometry: CaptureInvocationShape,
    ledger: &mut CaptureLedger,
) -> Result<(), CaptureExecutionError<B::Error>> {
    capture_evidence_in_window(
        None, backend, tensor, selection, point, record, geometry, ledger,
    )
}

fn capture_evidence_in_window<B: CaptureBackend>(
    window: Option<CaptureInvocationWindow>,
    backend: &mut B,
    tensor: &B::Tensor,
    selection: &CaptureSelection,
    point: &eredu_core::ObservationPoint,
    record: &mut CaptureRecord,
    geometry: CaptureInvocationShape,
    ledger: &mut CaptureLedger,
) -> Result<(), CaptureExecutionError<B::Error>> {
    let result = crate::capture::capture_window_value(
        window, backend, tensor, selection, point, record, geometry, ledger,
    );
    if let Err(error) = &result {
        record.payload = None;
        record.outcome = CaptureOutcome::Failed {
            reason: match error {
                CaptureExecutionError::Admission(error) => match error.cause() {
                    CaptureError::Limit { budget, cumulative } => CaptureFailureReason::Limit {
                        budget: *budget,
                        cumulative: *cumulative,
                    },
                    _ => CaptureFailureReason::Invalid,
                },
                CaptureExecutionError::Backend(_) => CaptureFailureReason::Native,
            },
            message: bounded_diagnostic(error),
        };
    }
    result
}

/// Joint preflight counts every ordinary capture, intervention diagnostic and
/// before/after transform against the same per-step and cumulative budgets.
pub fn preflight(
    capture: &AdmittedCapturePlan,
    intervention: &AdmittedInterventionPlan,
    estimator: &dyn InterventionEstimator,
) -> Result<(), CaptureError> {
    preflight_continuation(capture, intervention, estimator, 0, CaptureUsage::default())
}

// Coupled sources must retain the same ordinary opening before any estimator
// call or run mutation. Invocation authority remains a separate mode.
fn validate_capture_origin(
    capture: &AdmittedCapturePlan,
    intervention: &AdmittedInterventionPlan,
) -> Result<(), CaptureError> {
    if !intervention.is_empty() && capture.text_origin() != intervention.text_origin() {
        return Err(CaptureError::Invalid("capture/intervention text origins differ".into()));
    }
    Ok(())
}

pub(crate) fn preflight_continuation(
    capture: &AdmittedCapturePlan,
    intervention: &AdmittedInterventionPlan,
    estimator: &dyn InterventionEstimator,
    next_prediction: u64,
    inherited: CaptureUsage,
) -> Result<(), CaptureError> {
    validate_capture_origin(capture, intervention)?;
    if capture.request() != intervention.request()
        || capture.invocation_bounds() != intervention.invocation_bounds()
    {
        return Err(CaptureError::Invalid(
            "capture/intervention geometry mismatch".into(),
        ));
    }
    let mut base = CaptureUsage::default();
    let mut extra = Vec::new();
    let mut scheduled_costs = Vec::new();
    for (index, (operation, point)) in intervention
        .plan()
        .operations
        .iter()
        .zip(intervention.points())
        .enumerate()
    {
        base = base.checked_add(intervention_metadata(
            operation,
            point,
            intervention.identity(),
        )?)?;
        extra.extend(evidence_selections(operation, point));
        let mut costs = [CaptureUsage::default(); 2];
        for (phase_index, phase) in [CapturePhase::Prefill, CapturePhase::Decode]
            .into_iter()
            .enumerate()
        {
            let range = if intervention.invocation_bounds().is_some() {
                operation.schedule.count_coordinates(
                    phase,
                    0,
                    intervention.request().max_predictions,
                )?
            } else {
                operation.schedule.count_and_last_from(
                    phase,
                    next_prediction,
                    intervention.request().max_predictions,
                )?
            };
            let Some((_, last)) = range else {
                continue;
            };
            if let Some(shape) =
                intervention.estimate_shape(&point.observation_geometry(), phase, last)?
            {
                let slice = intervention.validate_at(
                    index,
                    phase,
                    last,
                    intervention
                        .invocation_bounds()
                        .map(|bounds| bounds.maximum())
                        .transpose()?,
                    &shape,
                    operation.action.dtype(),
                )?;
                if let Some(routed) = &point.routed_units {
                    costs[phase_index] = routed::cost(
                        estimator,
                        routed.geometry,
                        &shape,
                        &slice,
                        &operation.action,
                    )?;
                } else if point.routing.is_none() {
                    estimator.validate_geometry(&shape, &slice)?;
                    costs[phase_index] =
                        activation_cost(estimator, &shape, &slice, &operation.action)?;
                }
            }
            if let Some(policy) = &point.routing {
                if operation.evidence != InterventionEvidence::None {
                    let geometry = match intervention.invocation_bounds() {
                        Some(bounds) => bounds.maximum()?,
                        None => intervention.geometry_at(phase, last, None)?,
                    };
                    let rows = mul(geometry.batch, geometry.sequence)?;
                    costs[phase_index] = original_route_cost(estimator, policy, rows)?;
                }
            }
        }
        if point.routing.is_none() || operation.evidence != InterventionEvidence::None {
            scheduled_costs.push((operation.schedule.clone(), costs));
        }
    }
    crate::capture::preflight_continuation(
        capture,
        &extra,
        base,
        &scheduled_costs,
        next_prediction,
        inherited,
        |source, selection, slice| estimator.capture_usage(source, selection, slice),
    )
}

fn activation_cost(
    estimator: &dyn InterventionEstimator,
    source: &[u64],
    slice: &ResolvedCaptureSlice,
    action: &InterventionAction,
) -> Result<CaptureUsage, CaptureError> {
    let cost = estimator.activation_usage(source, slice, action)?;
    if cost.captures != 0 || cost.encoded_bytes != 0 {
        return Err(CaptureError::Invalid(
            "activation estimate must exclude capture and record encoding".into(),
        ));
    }
    Ok(cost)
}

fn original_route_cost(
    estimator: &dyn InterventionEstimator,
    policy: &InterventionRoutingPolicy,
    rows: u64,
) -> Result<CaptureUsage, CaptureError> {
    let cost = estimator.original_route_usage(policy, rows)?;
    if cost.captures != 0 || cost.encoded_bytes != 0 {
        return Err(CaptureError::Invalid(
            "original routing estimate must exclude evidence transforms and diagnostic encoding"
                .into(),
        ));
    }
    Ok(cost)
}

#[cfg(test)]
mod tests;

impl CaptureSession {
    /// Validates a scheduled routing operation against actual flattened token rows
    /// and reserves any original-decision work before the selector runs.
    pub fn routing_control(
        &mut self,
        path: &str,
        token_rows: u64,
    ) -> Result<Option<eredu_nn::routing_intervention::GroupSelectionControl>, CaptureError> {
        use eredu_nn::routing_intervention::{
            GroupScoreStage, GroupSelectionAction, GroupSelectionControl,
        };
        self.validate_ordinary_intervention()?;
        let physical = self
            .invocation_window
            .map(|_| self.tensor_geometry())
            .transpose()?;
        let logical = self
            .invocation_window
            .zip(physical)
            .map(|(window, physical)| window.validate(physical))
            .transpose()?;
        let Some(run) = &mut self.interventions else {
            return Ok(None);
        };
        let records = run
            .records
            .as_mut()
            .ok_or_else(|| CaptureError::Invalid("intervention step not started".into()))?;
        let Some(index) = run
            .plan
            .plan()
            .operations
            .iter()
            .zip(records.iter())
            .position(|(op, record)| {
                op.target == path && record.outcome != InterventionOutcome::Inactive
            })
        else {
            return Ok(None);
        };
        let operation = &run.plan.plan().operations[index];
        let point = &run.plan.points()[index];
        let record = &mut records[index];
        let result = (|| {
            if run.routing_pending.is_some() || record.outcome != InterventionOutcome::Missing {
                return Err(CaptureError::Invalid(
                    "routing target executed more than once or overlaps another pending target"
                        .into(),
                ));
            }
            let policy = point.routing.as_ref().ok_or_else(|| {
                CaptureError::Invalid("routing control reached activation target".into())
            })?;
            let slice = run.plan.validate_at(
                index,
                self.phase,
                self.prediction,
                logical.or(self.invocation),
                &[
                    logical.map_or(token_rows, |value| value.sequence),
                    policy.top_k as u64,
                ],
                None,
            )?;
            run.estimator.validate_geometry(
                &[
                    logical.map_or(token_rows, |value| value.sequence),
                    policy.top_k as u64,
                ],
                &slice,
            )?;
            if physical.is_some_and(|physical| token_rows != physical.sequence) {
                return Err(CaptureError::Invalid(
                    "routing rows differ from actual prefill window".into(),
                ));
            }
            let (first_row, end_row, payload_start, payload_end) =
                if let Some(window) = self.invocation_window {
                    let end = add(window.start, token_rows)?;
                    let ordinal = window
                        .start
                        .saturating_sub(slice.starts[0])
                        .div_ceil(slice.strides[0]);
                    let first = add(slice.starts[0], mul(ordinal, slice.strides[0])?)?;
                    let limit = end.min(slice.ends[0]);
                    if first >= limit {
                        return Ok(None);
                    }
                    let rows = (limit - first).div_ceil(slice.strides[0]);
                    (
                        first - window.start,
                        limit - window.start,
                        mul(ordinal, policy.top_k as u64)?,
                        mul(add(ordinal, rows)?, policy.top_k as u64)?,
                    )
                } else {
                    (
                        slice.starts[0],
                        slice.ends[0],
                        0,
                        mul(slice.shape[0], policy.top_k as u64)?,
                    )
                };
            // The selector owns its native workspace. This separate reservation
            // covers only the actual host control payload cloned/gathered here.
            let elements = match &operation.action {
                InterventionAction::ExcludeExperts { expert_ids }
                | InterventionAction::ZeroExpertContribution { expert_ids } => {
                    expert_ids.len() as u64
                }
                InterventionAction::ForceExperts { .. } => payload_end - payload_start,
                InterventionAction::BiasRoutingScores {
                    expert_ids, biases, ..
                } => add(expert_ids.len() as u64, biases.len() as u64)?,
                _ => 0,
            };
            let controls = CaptureUsage {
                host_bytes: add(
                    std::mem::size_of::<GroupSelectionControl>() as u64,
                    mul(elements, 4)?,
                )?,
                ..Default::default()
            };
            reserve_envelope(&mut self.ledger, controls)?;
            record.charged = record.charged.checked_add(controls)?;
            let action = match &operation.action {
                InterventionAction::ExcludeExperts { expert_ids } => {
                    GroupSelectionAction::Exclude(expert_ids.clone())
                }
                InterventionAction::ZeroExpertContribution { expert_ids } => {
                    GroupSelectionAction::ZeroContribution(expert_ids.clone())
                }
                InterventionAction::ForceExperts { expert_ids, .. } => GroupSelectionAction::Force(
                    expert_ids
                        .get(
                            usize::try_from(payload_start).map_err(|_| CaptureError::Overflow)?
                                ..usize::try_from(payload_end)
                                    .map_err(|_| CaptureError::Overflow)?,
                        )
                        .ok_or_else(|| {
                            CaptureError::Invalid(
                                "projected routing payload differs from admission".into(),
                            )
                        })?
                        .to_vec(),
                ),
                InterventionAction::BiasRoutingScores {
                    stage,
                    expert_ids,
                    biases,
                } => GroupSelectionAction::Bias {
                    stage: match stage {
                        RoutingScoreStage::RawLogits => GroupScoreStage::RawLogits,
                        RoutingScoreStage::TransformedScores => GroupScoreStage::TransformedScores,
                        RoutingScoreStage::RankingScores => GroupScoreStage::RankingScores,
                    },
                    ids: expert_ids.clone(),
                    values: biases.clone(),
                },
                _ => {
                    return Err(CaptureError::Invalid(
                        "activation operation cannot control routing".into(),
                    ));
                }
            };
            let expected = eredu_nn::TopKGroupSelectionSpec::new(
                i32::try_from(policy.expert_count).map_err(|_| CaptureError::Overflow)?,
                i32::try_from(policy.top_k).map_err(|_| CaptureError::Overflow)?,
                match policy.scoring {
                    RoutingScoring::Softmax => eredu_nn::GroupScoring::Softmax,
                    RoutingScoring::SelectedSoftmax => eredu_nn::GroupScoring::SelectedSoftmax,
                    RoutingScoring::Sigmoid => eredu_nn::GroupScoring::Sigmoid,
                    RoutingScoring::SqrtSoftplus => eredu_nn::GroupScoring::SqrtSoftplus,
                },
                policy.normalize_selected,
            )
            .and_then(|spec| spec.with_groups(policy.groups as i32, policy.selected_groups as i32))
            .and_then(|spec| {
                spec.with_weight_policy(policy.normalization_epsilon, policy.coefficient_scale)
            })
            .map_err(|error| CaptureError::Invalid(error.to_string()))?;
            let capture_original = operation.evidence != InterventionEvidence::None;
            if capture_original {
                let cost = original_route_cost(run.estimator.as_ref(), policy, token_rows)?;
                reserve_envelope(&mut self.ledger, cost)?;
                record.charged = record.charged.checked_add(cost)?;
            }
            Ok(Some(GroupSelectionControl {
                expected,
                learned_coefficient_scale: policy.learned_coefficient_scale,
                first_row,
                end_row,
                row_stride: slice.strides[0],
                action,
                capture_original,
            }))
        })();
        match result {
            Ok(control) => {
                run.routing_pending = Some(PendingRouting {
                    index,
                    phase: self.phase,
                    prediction: self.prediction,
                    invocation: self.invocation,
                    window: self.invocation_window,
                    unmodified: control.is_none(),
                });
                Ok(control)
            }
            Err(error) => {
                record.outcome = InterventionOutcome::Failed {
                    message: bounded_diagnostic(&error),
                };
                Err(error)
            }
        }
    }

    /// Captures attributed IDs/coefficients before the expert provider runs.
    pub fn routing_applied<B: CaptureBackend>(
        &mut self,
        backend: &mut B,
        path: &str,
        original: Option<crate::RoutingDecision<'_, B::Tensor>>,
        effective: crate::RoutingDecision<'_, B::Tensor>,
    ) -> Result<(), CaptureExecutionError<B::Error>> {
        self.finish_routing(backend, path, original, effective, false)
    }

    /// Requests metadata only for the exact pending no-overlap operation.
    pub fn routing_unmodified_interest(&self, path: &str) -> crate::RoutingUnmodifiedInterest {
        if self
            .interventions
            .as_ref()
            .and_then(|run| run.routing_pending.map(|pending| (run, pending)))
            .is_some_and(|(run, pending)| {
                pending.unmodified
                    && run.plan.plan().operations[pending.index].target == path
                    && pending.phase == self.phase
                    && pending.prediction == self.prediction
                    && pending.invocation == self.invocation
                    && pending.window == self.invocation_window
            })
        {
            crate::RoutingUnmodifiedInterest::Metadata
        } else {
            crate::RoutingUnmodifiedInterest::None
        }
    }

    /// Receives the actual ordinary decision only for a proved no-overlap window.
    /// Other ordinary notifications do not create intervention work or records.
    pub fn routing_unmodified<B: CaptureBackend>(
        &mut self,
        backend: &mut B,
        path: &str,
        effective: crate::RoutingDecision<'_, B::Tensor>,
    ) -> Result<(), CaptureExecutionError<B::Error>> {
        if !self
            .interventions
            .as_ref()
            .and_then(|run| run.routing_pending.map(|pending| (run, pending)))
            .is_some_and(|(run, pending)| {
                pending.unmodified && run.plan.plan().operations[pending.index].target == path
            })
        {
            return Ok(());
        }
        let original = crate::RoutingDecision {
            ids: effective.ids,
            coefficients: effective.coefficients,
        };
        self.finish_routing(backend, path, Some(original), effective, true)
    }

    fn finish_routing<B: CaptureBackend>(
        &mut self,
        backend: &mut B,
        path: &str,
        original: Option<crate::RoutingDecision<'_, B::Tensor>>,
        effective: crate::RoutingDecision<'_, B::Tensor>,
        unmodified: bool,
    ) -> Result<(), CaptureExecutionError<B::Error>> {
        self.validate_ordinary_intervention()?;
        let tensor_geometry = self.tensor_geometry()?;
        let run = self
            .interventions
            .as_mut()
            .ok_or_else(|| CaptureError::Invalid("unsolicited routing result".into()))?;
        let pending = run
            .routing_pending
            .ok_or_else(|| CaptureError::Invalid("routing result has no pending control".into()))?;
        if pending.unmodified != unmodified
            || pending.phase != self.phase
            || pending.prediction != self.prediction
            || pending.invocation != self.invocation
            || pending.window != self.invocation_window
        {
            return Err(CaptureError::Invalid(
                "routing result invocation differs from pending control".into(),
            )
            .into());
        }
        let index = pending.index;
        run.routing_pending = None;
        let operation = &run.plan.plan().operations[index];
        if operation.target != path {
            return Err(
                CaptureError::Invalid("routing result target differs from control".into()).into(),
            );
        }
        let record = &mut run
            .records
            .as_mut()
            .ok_or_else(|| CaptureError::Invalid("intervention step not started".into()))?[index];
        let started = std::time::Instant::now();
        let result = (|| {
            let shape = backend
                .shape(effective.ids)
                .map_err(CaptureExecutionError::Backend)?;
            tensor_geometry
                .validate_actual(&run.plan.points()[index].observation_geometry(), &shape)?;
            let logical = self
                .invocation_window
                .map(|window| window.validate(tensor_geometry))
                .transpose()?;
            let logical_shape = [logical.map_or(shape[0], |value| value.sequence), shape[1]];
            run.plan.validate_at(
                index,
                self.phase,
                self.prediction,
                logical.or(self.invocation),
                &logical_shape,
                None,
            )?;
            if backend
                .shape(effective.coefficients)
                .map_err(CaptureExecutionError::Backend)?
                != shape
            {
                return Err(CaptureError::Invalid(
                    "effective route IDs and coefficient shapes differ".into(),
                )
                .into());
            }
            let selections = evidence_selections(operation, &run.plan.points()[index]);
            if !selections.is_empty() {
                let original = original.ok_or_else(|| {
                    CaptureError::Invalid("requested original routing decision is missing".into())
                })?;
                for ((tensor, (selection, geometry)), evidence) in [
                    original.ids,
                    original.coefficients,
                    effective.ids,
                    effective.coefficients,
                ]
                .into_iter()
                .zip(&selections)
                .zip(&mut record.evidence)
                {
                    capture_evidence_in_window(
                        self.invocation_window,
                        backend,
                        tensor,
                        selection,
                        geometry,
                        evidence,
                        tensor_geometry,
                        &mut self.ledger,
                    )?;
                }
            }
            Ok(())
        })();
        self.capture_seconds += started.elapsed().as_secs_f64();
        record.outcome = match &result {
            Ok(()) if unmodified => InterventionOutcome::Unmatched,
            Ok(()) => InterventionOutcome::Applied,
            Err(error) => InterventionOutcome::Failed {
                message: bounded_diagnostic(error),
            },
        };
        result
    }

    /// Retains a bounded selector failure without treating it as proof of rollback.
    pub fn routing_failed(&mut self, path: &str, message: &str) {
        if let Some(run) = &mut self.interventions {
            if let Some(pending) = run.routing_pending.take() {
                let index = pending.index;
                if run.plan.plan().operations[index].target == path {
                    if let Some(records) = &mut run.records {
                        records[index].outcome = InterventionOutcome::Failed {
                            message: bounded_diagnostic(&message),
                        };
                    }
                }
            }
        }
    }
}
