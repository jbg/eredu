use super::super::construction_tests::{geometry, origin, source, summary, values};
use super::*;
use crate::working_memory::{InferenceExecutionIdentity, MemoryLedger};
use eredu_core::{
    intervention::*, ObservationSupportStatus, TensorObservation, TensorObservationData,
};
fn plan(capture: &AdmittedCapturePlan) -> AdmittedInterventionPlan {
    let point = &capture.points()[0];
    let discovery = InterventionDiscovery {
        schema_version: INTERVENTION_SCHEMA_VERSION,
        artifact_identity: "aggregate fixture".into(),
        session_identity: Some("session".into()),
        points: vec![InterventionPoint {
            path: point.path.clone(),
            node_id: point.node_id.clone(),
            stage: InterventionStage::Activation,
            axes: point.axes.clone().unwrap(),
            dtypes: vec![InterventionDtype::Float32],
            operations: vec![InterventionKind::Scale],
            score_stages: vec![],
            prefill: ObservationSupportStatus::Supported,
            decode: ObservationSupportStatus::Supported,
            conditions: vec![],
            routing: None,
            routed_units: None,
        }],
    };
    InterventionPlan {
        schema_version: INTERVENTION_SCHEMA_VERSION,
        operations: [
            InterventionEvidence::Summary,
            InterventionEvidence::Preview { max_elements: 5 },
        ]
        .into_iter()
        .enumerate()
        .map(|(index, evidence)| InterventionOperation {
            id: format!("edit-{index}"),
            target: point.path.clone(),
            schedule: Default::default(),
            slices: vec![],
            action: InterventionAction::Scale {
                dtype: InterventionDtype::Float32,
                factor: -2.0,
            },
            evidence,
        })
        .collect(),
    }
    .admit_invocations(&discovery, capture.invocation_bounds().unwrap(), "session")
    .unwrap()
}
fn report() -> SpeculativePrefillReductions {
    SpeculativePrefillReductions {
        logical_invocation: 7,
        origin: origin(),
        records: vec![],
        interventions: vec![],
        charged: CaptureUsage::default(),
    }
}
fn finish(
    mut lanes: Interventions,
    mut report: SpeculativePrefillReductions,
    source: &OriginalInterventionSource,
) -> SpeculativePrefillReductions {
    for (n, (start, end)) in [(0, 3), (3, 6), (6, 11)].into_iter().enumerate() {
        let phase = SpeculativeActivationPhase::TargetPrefill;
        lanes
            .validate_begin_fixed(&report.interventions, phase, start, end)
            .unwrap();
        let input = values(start, end);
        let records = source
            .plan()
            .admission()
            .plan()
            .operations
            .iter()
            .enumerate()
            .map(|(index, operation)| {
                let companion = source.plan().evidence(index).unwrap();
                let evidence = companion
                    .geometry_source()
                    .plan()
                    .selections
                    .iter()
                    .zip(companion.geometry_source().points())
                    .enumerate()
                    .map(|(side, (selection, point))| {
                        let actual = input
                            .iter()
                            .map(|value| if side == 0 { *value } else { *value * -2.0 })
                            .collect::<Vec<_>>();
                        let payload = match operation.evidence {
                            InterventionEvidence::Summary => {
                                CapturePayload::Summary(summary(&actual))
                            }
                            InterventionEvidence::Preview { max_elements } => {
                                CapturePayload::Tensor(
                                    TensorObservation::new(
                                        vec![actual.len().min(max_elements as usize)],
                                        TensorObservationData::F32(
                                            actual[..actual.len().min(max_elements as usize)]
                                                .to_vec(),
                                        ),
                                    )
                                    .unwrap(),
                                )
                            }
                            _ => unreachable!(),
                        };
                        CaptureRecord {
                            schema_version: CAPTURE_SCHEMA_VERSION,
                            selection_id: selection.id.clone(),
                            path: selection.path.clone(),
                            node_id: point.node_id.clone(),
                            position: point.position,
                            source_shape: Some(vec![end - start, 2]),
                            source_dtype: Some(eredu_core::checkpoint::TensorDtype::F32),
                            selected_shape: Some(vec![end - start, 2]),
                            outcome: crate::capture::completed_capture_outcome(
                                &selection.transform,
                                actual.len() as u64,
                            ),
                            payload: Some(payload),
                            charged: CaptureUsage {
                                captures: 1,
                                retained_bytes: 128,
                                host_bytes: 512,
                                encoded_bytes: 4096,
                            },
                        }
                    })
                    .collect();
                InterventionRecord {
                    schema_version: INTERVENTION_SCHEMA_VERSION,
                    plan_id: source.plan().admission().identity().into(),
                    operation_id: operation.id.clone(),
                    target: operation.target.clone(),
                    node_id: source.plan().admission().points()[index].node_id.clone(),
                    phase: CapturePhase::Prefill,
                    prediction_index: 0,
                    outcome: InterventionOutcome::Applied,
                    evidence,
                    charged: CaptureUsage::default(),
                    routed_units: None,
                }
            })
            .collect::<Vec<_>>();
        lanes
            .prepare_fixed(&mut report.interventions, &records, phase, start, end)
            .unwrap();
        lanes.finish_window(
            &mut report.interventions,
            &records,
            phase,
            end,
            n as u64 + 7,
            true,
        );
    }
    assert!(lanes.sealed());
    lanes.finish(&mut report.interventions, true);
    report
}
#[test]
fn source_owned_evidence_aggregates_preserve_sides_values_and_funded_retirement() {
    let capture = source();
    let admitted = plan(&capture);
    let scopes = [SpeculativeCaptureScope::Target; 2];
    let pool = crate::working_memory::memory_fixture::host_ledger(1 << 26, 0).unwrap();
    let source = pool
        .compile_intervention_source(PreparedInterventionPlanCopy::inspect(&admitted).unwrap())
        .unwrap();
    let baseline = pool.payload_used_bytes().unwrap();
    let mut ordinary_ledger = CaptureLedger::new(&capture);
    let mut ordinary = report();
    let lanes = Interventions::create_shared(
        Source::Ordinary(&admitted),
        &mut ordinary_ledger,
        &scopes,
        geometry(),
        origin(),
        &mut ordinary,
        Metadata::ordinary(),
    )
    .unwrap();
    let expected = finish(lanes, ordinary, &source);
    let funding = pool
        .prepare_workspace_metadata(
            &InferenceExecutionIdentity::default(),
            crate::working_memory::memory_fixture::resolved_host_limits(&pool, 1 << 26),
        )
        .unwrap();
    let mut ledger = CaptureLedger::new(&capture);
    let mut original = report();
    let lanes = Interventions::create_shared(
        Source::Original(&source),
        &mut ledger,
        &scopes,
        geometry(),
        origin(),
        &mut original,
        Metadata::original(&funding),
    )
    .unwrap();
    assert_eq!(ledger.total(), ordinary_ledger.total());
    let actual = finish(lanes, original, &source);
    assert_eq!(actual, expected);
    let full = values(0, 11);
    let edited = full.iter().map(|value| value * -2.0).collect::<Vec<_>>();
    for (side, values) in [&full, &edited].into_iter().enumerate() {
        let CapturePayload::Summary(value) = actual.interventions[0].record.evidence[side]
            .payload
            .as_ref()
            .unwrap()
        else {
            panic!("summary")
        };
        let oracle = summary(values);
        assert_eq!(value.elements, oracle.elements);
        assert_eq!(value.mean, oracle.mean);
        let CapturePayload::Tensor(value) = actual.interventions[1].record.evidence[side]
            .payload
            .as_ref()
            .unwrap()
        else {
            panic!("preview")
        };
        assert!(
            matches!(value.data(),TensorObservationData::F32(data) if data.as_slice()==&values[..5])
        );
    }
    let charged = pool.payload_used_bytes().unwrap();
    assert!(charged > baseline);
    // The enclosing original aggregate owner retains funding behind all records.
    let escaped = (actual, funding);
    drop(source);
    assert!(pool.payload_used_bytes().unwrap() > 0);
    drop(escaped);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
