//! Same-run pre-edit tensors are the oracle for physical and logical edit evidence.
use super::*;
use eredu_core::{
    ObservationPosition, SymbolicDimension, TensorObservationData,
    intervention::*,
    speculative::{SpeculativeActivationDiscovery, SpeculativePrefillReductionStatus},
};
const CASE: &str = "managed_plain::embedded::activations::interventions::native_original_qwen_window_intervention_evidence_matches_ordinary_and_controlled";
const RESULT: &str = "PUBLIC_EMBEDDED_INTERNAL_EVIDENCE_RESULT:";
const PREVIEW: usize = 4;
const BASE: [&str; 2] = ["target residual", "prediction fusion"];
const KIND: [&str; 2] = ["preview", "summary"];

pub(super) fn plan(
    discovery: &SpeculativeActivationDiscovery,
    captures: &[CaptureSelection],
) -> InterventionPlan {
    let mut operations = Vec::new();
    for (ordinal, base) in BASE.into_iter().enumerate() {
        let selection = captures
            .iter()
            .find(|selection| selection.id == base)
            .unwrap();
        let observation = discovery
            .captures
            .catalog
            .points
            .iter()
            .find(|point| point.path == selection.path)
            .unwrap();
        assert_eq!(
            observation.position,
            ObservationPosition::BeforeIntervention
        );
        // Both Qwen sites call instrumentation.apply, whose shared driver
        // observes this raw point before invoking the ordered edit list.
        let point = discovery
            .interventions
            .points
            .iter()
            .find(|point| point.path == selection.path && point.node_id == observation.node_id)
            .expect("loaded direct hook declares its static action");
        assert_eq!(point.stage, InterventionStage::Activation);
        assert!(point.operations.contains(&InterventionKind::Scale));
        assert!(point.dtypes.contains(&InterventionDtype::Float32));
        let scope = if ordinal == 0 {
            SpeculativeCaptureScope::Target
        } else {
            SpeculativeCaptureScope::Prediction { depth: 0 }
        };
        assert!(
            discovery
                .bindings
                .iter()
                .any(|binding| binding.node_id == point.node_id && binding.scope == scope)
        );
        assert_eq!(point.axes.len(), 3);
        assert_eq!(point.axes[1].dimension, SymbolicDimension::Sequence);
        assert_eq!(point.axes[2].dimension, SymbolicDimension::Known(8));
        for (kind, factor) in KIND.into_iter().zip([0.5, -1.0]) {
            operations.push(InterventionOperation {
                id: format!("{base} {kind}"),
                target: point.path.clone(),
                schedule: CaptureSchedule {
                    decode: false,
                    ..Default::default()
                },
                slices: vec![
                    CaptureSlice {
                        axis: point.axes[1].name.clone(),
                        start: 1,
                        end: 4,
                        stride: 2,
                    },
                    CaptureSlice {
                        axis: point.axes[2].name.clone(),
                        start: 1,
                        end: 7,
                        stride: 2,
                    },
                ],
                action: InterventionAction::Scale {
                    dtype: InterventionDtype::Float32,
                    factor,
                },
                evidence: if kind == "preview" {
                    InterventionEvidence::Preview {
                        max_elements: PREVIEW as u64,
                    }
                } else {
                    InterventionEvidence::Summary
                },
            });
        }
    }
    InterventionPlan {
        schema_version: INTERVENTION_SCHEMA_VERSION,
        operations,
    }
}
fn source<'a>(envelope: &'a SpeculativeActivationCapture, base: &str) -> &'a CaptureRecord {
    envelope
        .captures
        .as_step()
        .records
        .iter()
        .find(|record| record.selection_id == base)
        .unwrap()
}
fn full(record: &CaptureRecord) -> &[f32] {
    assert_eq!(record.position, ObservationPosition::BeforeIntervention);
    let tensor = record.payload.as_ref().unwrap().as_tensor().unwrap();
    let TensorObservationData::F32(values) = tensor.data() else {
        panic!("floating direct oracle")
    };
    assert!(values.iter().all(|v| v.is_finite()));
    assert!(values.iter().any(|v| v.abs() > 1e-6));
    values
}
fn physical_selection(envelope: &SpeculativeActivationCapture, base: &str) -> (Vec<f32>, u64) {
    let span = envelope.prefill_span.unwrap();
    let sequence = envelope.captures.as_step().invocation.unwrap().sequence;
    assert!(span.validate(envelope.phase, sequence as usize));
    let shifted = envelope.phase == SpeculativeActivationPhase::PredictionPrefill
        && span.token_start != span.hidden_start;
    let logical = span.prompt_tokens - u64::from(shifted);
    assert!(span.hidden_start + sequence <= logical);
    let values = full(source(envelope, base));
    assert_eq!(values.len(), sequence as usize * 8);
    let mut selected = Vec::new();
    for local in 0..sequence as usize {
        let row = span.hidden_start + local as u64;
        if !(1..4).contains(&row) || (row - 1) % 2 != 0 {
            continue;
        }
        for column in [1, 3, 5] {
            selected.push(values[local * 8 + column]);
        }
    }
    (selected, logical)
}
fn payload(
    record: &CaptureRecord,
    kind: &str,
    values: &[f32],
    side: usize,
    selected_rows: u64,
    source_rows: u64,
    physical: bool,
) -> serde_json::Value {
    assert_eq!(
        record.position,
        if side == 0 {
            ObservationPosition::BeforeIntervention
        } else {
            ObservationPosition::AfterIntervention
        }
    );
    assert_eq!(
        record.source_shape.as_deref(),
        Some([1, source_rows, 8].as_slice())
    );
    if values.is_empty() {
        assert_eq!(
            record.outcome,
            CaptureOutcome::Skipped {
                reason: CaptureSkipReason::NotInvoked
            }
        );
        assert!(record.payload.is_none());
        assert!(record.selected_shape.is_none());
    } else {
        assert_eq!(
            record.selected_shape.as_deref(),
            Some([1, selected_rows, 3].as_slice())
        );
        if physical {
            assert!(record.charged.captures > 0);
        }
        // Logical attribution is checked against every physical contributor by
        // the caller; the report's additional charge is checked separately.
        match (kind, record.payload.as_ref().unwrap()) {
            ("preview", value) => {
                let tensor = value.as_tensor().unwrap();
                let count = values.len().min(PREVIEW);
                assert_eq!(tensor.shape(), [count]);
                let TensorObservationData::F32(actual) = tensor.data() else {
                    panic!("preview floating values")
                };
                compare(
                    &serde_json::json!(actual),
                    &serde_json::json!(&values[..count]),
                    "same-run Before/After Preview oracle",
                );
                let expected = if values.len() > PREVIEW {
                    CaptureOutcome::Truncated {
                        available_elements: values.len() as u64,
                        emitted_elements: PREVIEW as u64,
                    }
                } else {
                    CaptureOutcome::Captured
                };
                assert_eq!(record.outcome, expected);
            }
            ("summary", CapturePayload::Summary(actual)) => {
                let count = values.len() as u64;
                let expected = CaptureSummary {
                    elements: count,
                    finite: count,
                    non_finite: 0,
                    nan: 0,
                    positive_infinity: 0,
                    negative_infinity: 0,
                    min: Some(values.iter().copied().fold(f32::INFINITY, f32::min) as f64),
                    max: Some(values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64),
                    mean: Some(values.iter().map(|&v| v as f64).sum::<f64>() / count as f64),
                    rms: Some(
                        (values.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / count as f64)
                            .sqrt(),
                    ),
                };
                assert_eq!(record.outcome, CaptureOutcome::Captured);
                compare(
                    &serde_json::to_value(actual).unwrap(),
                    &serde_json::to_value(expected).unwrap(),
                    "same-run Before/After Summary oracle",
                );
            }
            _ => panic!("typed evidence selection"),
        }
    }
    serde_json::json!({"id":record.selection_id,"path":record.path,"node":record.node_id,"position":record.position,"source_shape":record.source_shape,"source_dtype":record.source_dtype,"selected_shape":record.selected_shape,"outcome":record.outcome,"payload":record.payload})
}
pub(super) fn rows(records: &[SpeculativeActivationCapture], shared: bool) -> serde_json::Value {
    let mut physical = Vec::new();
    let plan_id = &records[0].captures.as_step().interventions[0].plan_id;
    assert!(!plan_id.is_empty());
    for envelope in records {
        let step = envelope.captures.as_step();
        assert_eq!(step.interventions.len(), 4);
        if shared {
            let owner = &envelope.captures;
            let alias = owner.clone();
            assert!(alias.same_storage(owner));
            assert_eq!(
                alias.as_ref().interventions.as_ptr(),
                step.interventions.as_ptr()
            );
            for (a, b) in alias.as_ref().interventions.iter().zip(&step.interventions) {
                assert_eq!(a.evidence.as_ptr(), b.evidence.as_ptr());
            }
        }
        let mut edits = Vec::new();
        for (index, record) in step.interventions.iter().enumerate() {
            let base = BASE[index / 2];
            let kind = KIND[index % 2];
            assert_eq!(record.operation_id, format!("{base} {kind}"));
            assert_eq!(&record.plan_id, plan_id);
            assert!(record.routed_units.is_none());
            let active = matches!(
                (index / 2, envelope.phase),
                (0, SpeculativeActivationPhase::TargetPrefill)
                    | (1, SpeculativeActivationPhase::PredictionPrefill)
            );
            let mut evidence = Vec::new();
            assert_eq!(record.evidence.len(), 2);
            if active {
                assert_eq!(record.outcome, InterventionOutcome::Applied);
                let (selected, _) = physical_selection(envelope, base);
                assert_eq!(record.target, source(envelope, base).path);
                assert_eq!(record.node_id, source(envelope, base).node_id);
                for side in 0..2 {
                    let factor = match (index % 2, side) {
                        (0, 0) => 1.0,
                        (0, 1) | (1, 0) => 0.5,
                        (1, 1) => -0.5,
                        _ => unreachable!(),
                    };
                    let values: Vec<_> = selected.iter().map(|v| v * factor).collect();
                    evidence.push(payload(
                        &record.evidence[side],
                        kind,
                        &values,
                        side,
                        values.len() as u64 / 3,
                        step.invocation.unwrap().sequence,
                        true,
                    ));
                }
            } else {
                assert_eq!(record.outcome, InterventionOutcome::Inactive);
                for value in &record.evidence {
                    assert!(value.payload.is_none());
                    assert!(matches!(value.outcome, CaptureOutcome::Skipped { .. }));
                    evidence.push(serde_json::json!({"id":value.selection_id,"outcome":value.outcome,"position":value.position}));
                }
            }
            edits.push(serde_json::json!({"operation":record.operation_id,"target":record.target,"node":record.node_id,"phase":record.phase,"prediction":record.prediction_index,"outcome":record.outcome,"evidence":evidence}));
        }
        physical.push(serde_json::json!({"invocation":envelope.invocation,"phase":envelope.phase,"span":envelope.prefill_span,"edits":edits}));
    }
    let delivered: Vec<_> = records
        .iter()
        .filter(|record| record.prefill_reductions.is_some())
        .collect();
    assert_eq!(delivered.len(), 1);
    let delivery = delivered[0].prefill_reductions.as_ref().unwrap();
    if shared {
        let owner = delivery;
        let alias = owner.clone();
        assert!(alias.same_storage(owner));
        assert_eq!(
            alias.as_reductions().interventions.as_ptr(),
            owner.as_reductions().interventions.as_ptr()
        );
    }
    let report = delivery.as_reductions();
    assert!(report.records.is_empty());
    assert_eq!(report.interventions.len(), 4);
    assert_eq!(report.logical_invocation, records[0].invocation);
    assert_eq!(report.origin, records[0].origin);
    assert_eq!(
        report
            .interventions
            .iter()
            .map(|entry| entry.operation_index)
            .collect::<Vec<_>>(),
        [0, 1, 2, 3]
    );
    let mut logical = Vec::new();
    for entry in &report.interventions {
        let base = BASE[entry.operation_index / 2];
        let kind = KIND[entry.operation_index % 2];
        let windows: Vec<_> = records
            .iter()
            .filter(|record| record.phase == entry.phase)
            .collect();
        assert_eq!(entry.status, SpeculativePrefillReductionStatus::Complete);
        assert_eq!(&entry.record.plan_id, plan_id);
        assert_eq!(entry.record.operation_id, format!("{base} {kind}"));
        assert_eq!(entry.record.phase, CapturePhase::Prefill);
        assert_eq!(
            entry.record.prediction_index,
            records[0].origin.prediction as u64
        );
        assert_eq!(entry.record.target, source(windows[0], base).path);
        assert_eq!(entry.record.node_id, source(windows[0], base).node_id);
        assert!(entry.record.routed_units.is_none());
        let mut selected = Vec::new();
        let mut logical_sequence = None;
        for envelope in &windows {
            let (values, sequence) = physical_selection(envelope, base);
            assert!(logical_sequence.is_none_or(|value| value == sequence));
            logical_sequence = Some(sequence);
            selected.extend(values);
        }
        assert_eq!(
            selected.len(),
            6,
            "two selected rows cross physical windows"
        );
        assert!(
            selected.iter().any(|value| value.abs() > 1e-6),
            "selected edit values are nonzero"
        );
        assert_eq!(entry.logical_sequence, logical_sequence.unwrap());
        assert_eq!(entry.covered_sequence, entry.logical_sequence);
        assert_eq!(entry.windows, windows.len() as u64);
        assert_eq!(
            entry.first_invocation,
            Some(windows.first().unwrap().invocation)
        );
        assert_eq!(
            entry.last_invocation,
            Some(windows.last().unwrap().invocation)
        );
        assert_eq!(entry.record.outcome, InterventionOutcome::Applied);
        let mut evidence = Vec::new();
        for side in 0..2 {
            let factor = match (entry.operation_index % 2, side) {
                (0, 0) => 1.0,
                (0, 1) | (1, 0) => 0.5,
                (1, 1) => -0.5,
                _ => unreachable!(),
            };
            let values: Vec<_> = selected.iter().map(|v| v * factor).collect();
            let physical_captures: u64 = windows
                .iter()
                .map(|window| {
                    window.captures.as_step().interventions[entry.operation_index].evidence[side]
                        .charged
                        .captures
                })
                .sum();
            assert_eq!(
                entry.record.evidence[side].charged.captures,
                physical_captures
            );
            assert_eq!(
                report.charged.captures, 0,
                "aggregate delivery adds no physical capture spending"
            );
            evidence.push(payload(
                &entry.record.evidence[side],
                kind,
                &values,
                side,
                2,
                entry.logical_sequence,
                false,
            ));
        }
        logical.push(serde_json::json!({"phase":entry.phase,"operation":entry.operation_index,"sequence":entry.logical_sequence,"covered":entry.covered_sequence,"windows":entry.windows,"first":entry.first_invocation,"last":entry.last_invocation,"status":entry.status,"evidence":evidence}));
    }
    serde_json::json!({"physical":physical,"logical":logical})
}
#[test]
#[ignore = "requires Metal and original Embedded static intervention evidence"]
fn native_original_qwen_window_intervention_evidence_matches_ordinary_and_controlled() {
    compare_modes(CASE, RESULT, |mode| run_configured(mode, false, true));
}
