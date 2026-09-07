//! Prospective intervention scheduling under the ordinary capture/run owner.
//! There is no additional sampling loop or native-resource owner here.

use crate::capture::{
    bounded_diagnostic, capture_value, metadata_reservation, CaptureExecutionError, CaptureSession,
};
use eredu_core::{capture::*, intervention::*, ObservationPosition};

mod activation;
mod session;
pub use activation::apply_activation;
pub use session::{install_session, validate_session, CaptureObserver};

pub(crate) struct InterventionRun {
    pub(crate) plan: AdmittedInterventionPlan,
    pub(crate) records: Option<Vec<InterventionRecord>>,
    routing_pending: Option<usize>,
    estimator: std::sync::Arc<dyn InterventionEstimator>,
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
            records.push(InterventionRecord {
                schema_version: INTERVENTION_SCHEMA_VERSION,
                plan_id: self.plan.identity().into(),
                operation_id: operation.id.clone(),
                target: operation.target.clone(),
                node_id: point.node_id.clone(),
                phase,
                prediction_index: prediction,
                outcome: if active {
                    InterventionOutcome::Missing
                } else {
                    InterventionOutcome::Inactive
                },
                evidence,
                charged,
            });
        }
        self.records = Some(records);
        self.routing_pending = None;
        Ok(())
    }

    pub(crate) fn take_records(&mut self) -> Vec<InterventionRecord> {
        self.records.take().unwrap_or_default()
    }
}

impl CaptureSession {
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
        if plan.request() != self.plan.request() {
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
            if operation.target != path
                || point.routing.is_some()
                || record.outcome == InterventionOutcome::Inactive
            {
                continue;
            }
            if record.outcome != InterventionOutcome::Missing {
                return Err(CaptureError::Invalid(
                    "intervention point executed more than once in one step".into(),
                )
                .into());
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
                let slice = run.plan.validate_actual(
                    index,
                    self.phase,
                    self.prediction,
                    &shape,
                    Some(dtype),
                )?;
                run.estimator.validate_geometry(&shape, &slice)?;
                let evidence = evidence_selections(operation, point);
                if let Some((selection, geometry)) = evidence.first() {
                    capture_evidence(
                        backend,
                        input,
                        selection,
                        geometry,
                        &mut record.evidence[0],
                        run.plan.request(),
                        self.phase,
                        self.prediction,
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
                run.plan.validate_actual(
                    index,
                    self.phase,
                    self.prediction,
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
                        run.plan.request(),
                        self.phase,
                        self.prediction,
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
            let records = run
                .records
                .as_ref()
                .ok_or_else(|| CaptureError::Invalid("intervention step not started".into()))?;
            if let Some(record) = records.iter().find(|r| {
                matches!(
                    r.outcome,
                    InterventionOutcome::Missing | InterventionOutcome::Failed { .. }
                )
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

fn reserve_envelope(ledger: &mut CaptureLedger, usage: CaptureUsage) -> Result<(), CaptureError> {
    match ledger.reserve(usage)? {
        Some(CaptureSkipReason::Limit { budget, cumulative }) => {
            Err(CaptureError::Limit { budget, cumulative })
        }
        _ => Ok(()),
    }
}

fn intervention_metadata(
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
    let transform = match operation.evidence {
        InterventionEvidence::None => return vec![],
        InterventionEvidence::Preview { max_elements } => {
            CaptureTransform::Preview { max_elements }
        }
        InterventionEvidence::Summary => CaptureTransform::Summary,
    };
    let mut entries = Vec::new();
    for (label, position) in [
        ("before", ObservationPosition::BeforeIntervention),
        ("after", ObservationPosition::AfterIntervention),
    ] {
        let fields: &[Option<eredu_core::RoutingObservationField>] = if point.routing.is_some() {
            &[
                Some(eredu_core::RoutingObservationField::SelectedExperts),
                Some(eredu_core::RoutingObservationField::Coefficients),
            ]
        } else {
            &[None]
        };
        for field in fields {
            let mut geometry = point.observation_geometry();
            geometry.position = position;
            if let Some(field) = field {
                geometry.path = field.path(&point.path);
                if *field == eredu_core::RoutingObservationField::SelectedExperts {
                    geometry.dtype = eredu_core::ObservationDtype::Integer;
                }
            }
            entries.push((
                CaptureSelection {
                    id: format!("{}:{label}:{:?}", operation.id, field),
                    path: geometry.path.clone(),
                    schedule: operation.schedule.clone(),
                    slices: operation.slices.clone(),
                    transform: transform.clone(),
                },
                geometry,
            ));
        }
    }
    entries
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn capture_evidence<B: CaptureBackend>(
    backend: &mut B,
    tensor: &B::Tensor,
    selection: &CaptureSelection,
    point: &eredu_core::ObservationPoint,
    record: &mut CaptureRecord,
    request: CaptureRequestShape,
    phase: CapturePhase,
    prediction: u64,
    ledger: &mut CaptureLedger,
) -> Result<(), CaptureExecutionError<B::Error>> {
    let result = capture_value(
        backend, tensor, selection, point, record, request, phase, prediction, ledger,
    );
    if let Err(error) = &result {
        record.payload = None;
        record.outcome = CaptureOutcome::Failed {
            reason: match error {
                CaptureExecutionError::Admission(CaptureError::Limit { budget, cumulative }) => {
                    CaptureFailureReason::Limit {
                        budget: *budget,
                        cumulative: *cumulative,
                    }
                }
                CaptureExecutionError::Admission(_) => CaptureFailureReason::Invalid,
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
    if capture.request() != intervention.request() {
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
            let Some((_, last)) = operation
                .schedule
                .count_and_last(phase, intervention.request().max_predictions)?
            else {
                continue;
            };
            if let Some(shape) =
                intervention
                    .request()
                    .resolve(&point.observation_geometry(), phase, last)?
            {
                let slice = intervention.validate_actual(
                    index,
                    phase,
                    last,
                    &shape,
                    operation.action.dtype(),
                )?;
                estimator.validate_geometry(&shape, &slice)?;
            }
            if let Some(policy) = &point.routing {
                if operation.evidence != InterventionEvidence::None {
                    let rows = mul(
                        intervention.request().batch,
                        if phase == CapturePhase::Prefill {
                            intervention.request().prompt_tokens
                        } else {
                            1
                        },
                    )?;
                    costs[phase_index] = original_route_cost(estimator, policy, rows)?;
                }
            }
        }
        if point.routing.is_some() && operation.evidence != InterventionEvidence::None {
            scheduled_costs.push((operation.schedule.clone(), costs));
        }
    }
    crate::capture::preflight_with_extra(
        capture,
        &extra,
        base,
        &scheduled_costs,
        |source, selection, slice| estimator.capture_usage(source, selection, slice),
    )
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
            let slice = run.plan.validate_actual(
                index,
                self.phase,
                self.prediction,
                &[token_rows, policy.top_k as u64],
                None,
            )?;
            run.estimator
                .validate_geometry(&[token_rows, policy.top_k as u64], &slice)?;
            let action = match &operation.action {
                InterventionAction::ExcludeExperts { expert_ids } => {
                    GroupSelectionAction::Exclude(expert_ids.clone())
                }
                InterventionAction::ZeroExpertContribution { expert_ids } => {
                    GroupSelectionAction::ZeroContribution(expert_ids.clone())
                }
                InterventionAction::ForceExperts { expert_ids, .. } => {
                    GroupSelectionAction::Force(expert_ids.clone())
                }
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
                    ))
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
            Ok(GroupSelectionControl {
                expected,
                learned_coefficient_scale: policy.learned_coefficient_scale,
                first_row: slice.starts[0],
                end_row: slice.ends[0],
                row_stride: slice.strides[0],
                action,
                capture_original,
            })
        })();
        match result {
            Ok(control) => {
                run.routing_pending = Some(index);
                Ok(Some(control))
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
        let run = self
            .interventions
            .as_mut()
            .ok_or_else(|| CaptureError::Invalid("unsolicited routing result".into()))?;
        let index = run
            .routing_pending
            .take()
            .ok_or_else(|| CaptureError::Invalid("routing result has no pending control".into()))?;
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
            run.plan
                .validate_actual(index, self.phase, self.prediction, &shape, None)?;
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
                    capture_evidence(
                        backend,
                        tensor,
                        selection,
                        geometry,
                        evidence,
                        run.plan.request(),
                        self.phase,
                        self.prediction,
                        &mut self.ledger,
                    )?;
                }
            }
            Ok(())
        })();
        self.capture_seconds += started.elapsed().as_secs_f64();
        record.outcome = match &result {
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
            if let Some(index) = run.routing_pending.take() {
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
