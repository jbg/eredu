//! Exact count scalar edits on real sparse target-provider rows.
use super::*;
use eredu_core::{
    ObservationPosition, TensorObservationData,
    capture::{RoutedUnitCapture, RoutedUnitCaptureRow},
    intervention::*,
    speculative::SpeculativeActivationDiscovery,
};

const CASE: &str = "managed_plain::embedded::activations::sparse::native_original_sparse_model_interventions_match_ordinary_and_controlled";
const RESULT: &str = "PUBLIC_ORIGINAL_SPARSE_MODEL_INTERVENTION_RESULT:";
const STRIDED_CASE: &str = "managed_plain::embedded::activations::sparse::native_original_strided_sparse_model_interventions_match_ordinary_and_controlled";
const STRIDED_RESULT: &str = "PUBLIC_ORIGINAL_STRIDED_SPARSE_MODEL_INTERVENTION_RESULT:";

fn plan(
    discovery: &SpeculativeActivationDiscovery,
    paths: &[String; 2],
    strided: bool,
) -> SpeculativeActivationPlan {
    let point = discovery
        .interventions
        .points
        .iter()
        .find(|point| {
            point.path == paths[0]
                && discovery.bindings.iter().any(|binding| {
                    binding.node_id == point.node_id
                        && binding.scope == SpeculativeCaptureScope::Target
                })
        })
        .expect("actual target provider hook");
    let bank = point
        .routed_units
        .as_ref()
        .expect("sparse declaration")
        .geometry;
    assert_eq!(
        (bank.experts, bank.units_per_expert, bank.routes_per_token),
        (4, 6, 2)
    );
    assert!(point.operations.contains(&InterventionKind::Scale));
    assert!(point.operations.contains(&InterventionKind::Zero));
    assert!(point.dtypes.contains(&InterventionDtype::Float32));
    let selections = paths
        .iter()
        .enumerate()
        .map(|(index, path)| {
            let capture = discovery
                .captures
                .catalog
                .points
                .iter()
                .find(|capture| {
                    capture.path == *path
                        && discovery.bindings.iter().any(|binding| {
                            binding.node_id == capture.node_id
                                && binding.scope == SpeculativeCaptureScope::Target
                        })
                })
                .expect("original/effective sparse hook");
            assert!(capture.prefill && capture.decode);
            CaptureSelection {
                id: ["original", "effective"][index].into(),
                path: path.clone(),
                // Existing speculative envelopes retain each real physical span,
                // with complete sparse provider coverage and cumulative charges.
                schedule: CaptureSchedule::default(),
                slices: vec![],
                transform: CaptureTransform::RoutedUnits,
            }
        })
        .collect();
    let operations = [true, false]
        .into_iter()
        .map(|prefill| InterventionOperation {
            id: if prefill { "full-scale" } else { "full-zero" }.into(),
            target: point.path.clone(),
            schedule: CaptureSchedule {
                prefill,
                decode: !prefill,
                ..Default::default()
            },
            // The strided prefill keeps every expert component. Its final
            // one-row physical chunk selects no tokens, but still verifies and
            // completes the actual five sources under the same invocation.
            slices: if strided && prefill {
                vec![CaptureSlice {
                    axis: "token".into(),
                    start: 1,
                    end: 5,
                    stride: 2,
                }]
            } else {
                vec![]
            },
            action: if prefill {
                InterventionAction::Scale {
                    dtype: InterventionDtype::Float32,
                    factor: -0.5,
                }
            } else {
                InterventionAction::Zero {
                    dtype: InterventionDtype::Float32,
                }
            },
            evidence: InterventionEvidence::None,
        })
        .collect();
    let usage = CaptureUsage {
        captures: 512,
        retained_bytes: 32 << 20,
        host_bytes: 32 << 20,
        encoded_bytes: 32 << 20,
    };
    SpeculativeActivationPlan {
        schema_version: SPECULATIVE_ACTIVATION_SCHEMA_VERSION,
        bounds: CaptureInvocationBounds {
            batch: 1,
            max_sequence: 5,
            max_context: None,
            max_predictions: 4,
        },
        captures: CapturePlan {
            schema_version: CAPTURE_SCHEMA_VERSION,
            selections,
            limits: CaptureLimits {
                per_step: usage,
                cumulative: usage,
                physical_native_bytes: None,
                on_limit: CaptureLimitPolicy::Fail,
            },
        },
        interventions: InterventionPlan {
            schema_version: INTERVENTION_SCHEMA_VERSION,
            operations,
        },
    }
}
fn payload<'a>(envelope: &'a SpeculativeActivationCapture, id: &str) -> &'a RoutedUnitCapture {
    let record = envelope
        .captures
        .as_step()
        .records
        .iter()
        .find(|record| record.selection_id == id)
        .unwrap();
    assert_eq!(record.outcome, CaptureOutcome::Captured);
    assert_eq!(
        record.position,
        if id == "original" {
            ObservationPosition::BeforeIntervention
        } else {
            ObservationPosition::AfterIntervention
        }
    );
    let Some(CapturePayload::RoutedUnits(value)) = record.payload.as_ref() else {
        panic!("sparse source owner")
    };
    value
}
fn values(row: &RoutedUnitCaptureRow) -> &[f32] {
    let TensorObservationData::F32(values) = row.values.data() else {
        panic!("actual floating units")
    };
    assert_eq!(row.values.shape(), [6]);
    values
}
fn inspect(
    records: &[SpeculativeActivationCapture],
    shared: bool,
    strided: bool,
) -> serde_json::Value {
    let target: Vec<_> = records
        .iter()
        .filter(|record| SpeculativeCaptureScope::Target.applies(record.phase))
        .collect();
    assert!(
        target
            .iter()
            .any(|record| record.phase == SpeculativeActivationPhase::Verification)
    );
    assert_eq!(
        target
            .iter()
            .filter(|record| record.phase == SpeculativeActivationPhase::TargetPrefill)
            .map(|record| record.prefill_span.unwrap().sequence)
            .collect::<Vec<_>>(),
        [2, 2, 1]
    );
    assert_eq!(
        target
            .iter()
            .filter(|record| record.phase == SpeculativeActivationPhase::TargetPrefill)
            .map(|record| {
                let span = record.prefill_span.unwrap();
                [span.input_start, span.input_end, span.position]
            })
            .collect::<Vec<_>>(),
        [[0, 2, 0], [2, 4, 2], [4, 5, 4]]
    );
    let mut encoded = Vec::new();
    let mut previous = CaptureUsage::default();
    for envelope in target {
        assert!(envelope.completed);
        assert!(
            envelope
                .admission_identity
                .as_deref()
                .is_some_and(|id| !id.is_empty())
        );
        let step = envelope.captures.as_step();
        if shared {
            let owner = &envelope.captures;
            let alias = owner.clone();
            assert!(alias.same_storage(owner));
            assert_eq!(alias.as_ref().records.as_ptr(), step.records.as_ptr());
            assert_eq!(
                alias.as_ref().interventions.as_ptr(),
                step.interventions.as_ptr()
            );
        }
        assert!(step.cumulative_usage.captures > previous.captures);
        assert!(step.cumulative_usage.retained_bytes >= previous.retained_bytes);
        assert!(step.cumulative_usage.host_bytes >= previous.host_bytes);
        previous = step.cumulative_usage;
        assert_eq!(step.prediction_index, envelope.origin.prediction as u64);
        assert_eq!(step.interventions.len(), 2);
        let selected_tokens = if strided && step.phase == CapturePhase::Prefill {
            let span = envelope.prefill_span.unwrap();
            (span.hidden_start..span.hidden_start + span.sequence)
                .filter(|token| *token >= 1 && *token < 5 && (*token - 1) % 2 == 0)
                .count() as u64
        } else {
            step.invocation.unwrap().sequence
        };
        let mut edits = Vec::new();
        for (index, edit) in step.interventions.iter().enumerate() {
            let active = (index == 0) == (step.phase == CapturePhase::Prefill);
            assert_eq!(edit.operation_id, ["full-scale", "full-zero"][index]);
            assert_eq!(
                edit.outcome,
                if active && selected_tokens == 0 {
                    InterventionOutcome::Unmatched
                } else if active {
                    InterventionOutcome::Applied
                } else {
                    InterventionOutcome::Inactive
                }
            );
            assert!(edit.evidence.is_empty());
            assert_eq!(edit.prediction_index, step.prediction_index);
            assert_eq!(edit.phase, step.phase);
            if active {
                let receipt = edit.routed_units.expect("completed original route receipt");
                assert_eq!(receipt.source_tokens, step.invocation.unwrap().sequence);
                assert_eq!(receipt.completed_tokens, receipt.source_tokens);
                assert_eq!(receipt.affected_values, selected_tokens * 2 * 6);
                assert!(edit.charged.retained_bytes > 0);
                assert!(edit.charged.host_bytes > 0);
            }
            edits.push(serde_json::json!({"id":edit.operation_id,"target":edit.target,
                "node":edit.node_id,"phase":edit.phase,"prediction":edit.prediction_index,"outcome":edit.outcome}));
        }
        let original = payload(envelope, "original");
        let effective = payload(envelope, "effective");
        assert_eq!(original.geometry, effective.geometry);
        assert_eq!(original.source_token_ranges, effective.source_token_ranges);
        let geometry = step.invocation.unwrap();
        assert_eq!(
            original.rows.len() as u64,
            geometry.batch * geometry.sequence * 2
        );
        assert_eq!(original.rows.len(), effective.rows.len());
        assert!(
            original
                .rows
                .iter()
                .flat_map(values)
                .any(|value| value.abs() > 1e-6)
        );
        let mut end = 0;
        for range in &original.source_token_ranges {
            assert_eq!(range[0], end);
            assert!(range[1] > range[0]);
            end = range[1];
        }
        assert_eq!(end, geometry.batch * geometry.sequence);
        for (raw, edited) in original.rows.iter().zip(&effective.rows) {
            assert_eq!(
                (
                    raw.source_peer,
                    raw.token,
                    raw.slot,
                    raw.expert,
                    raw.unit_start,
                    raw.unit_stride
                ),
                (
                    edited.source_peer,
                    edited.token,
                    edited.slot,
                    edited.expert,
                    edited.unit_start,
                    edited.unit_stride
                )
            );
            assert_eq!(raw.source_peer, None);
            assert!(raw.token < geometry.sequence && raw.slot < 2 && raw.expert < 4);
            assert_eq!((raw.unit_start, raw.unit_stride), (0, 1));
            assert_eq!(raw.coefficient, edited.coefficient);
            assert!(raw.coefficient.is_finite() && raw.coefficient > 0.0);
            let factor = if step.phase == CapturePhase::Prefill {
                let logical = envelope.prefill_span.unwrap().hidden_start + raw.token;
                if strided && !(logical >= 1 && logical < 5 && (logical - 1) % 2 == 0) {
                    1.0
                } else {
                    -0.5
                }
            } else {
                0.0
            };
            let expected: Vec<_> = values(raw).iter().map(|value| value * factor).collect();
            compare(
                &serde_json::json!(values(edited)),
                &serde_json::json!(expected),
                "same-run sparse scalar edit",
            );
        }
        encoded.push(serde_json::json!({"invocation":envelope.invocation,"phase":envelope.phase,
            "span":envelope.prefill_span,"prediction":envelope.origin.prediction,"committed":envelope.origin.committed_tokens,
            "prefix_digest":envelope.origin.prefix_digest,"shape":geometry,"original":original,"effective":effective,"edits":edits}));
    }
    encoded.into()
}
fn run_selected(mode: &str, strided: bool) -> serde_json::Value {
    // Internal activation admission belongs to the selected complete embedded
    // extension. An independent autoregressive draft exposes prediction logits,
    // but does not provide this separate complete internal-hook declaration.
    let target = managed_fixture(
        super::super::super::super::speculative::captured_prefill::source("inkling"),
    );
    let graph = inspect_architecture(&target.0).unwrap();
    let bank = &graph.routed_components[0];
    let paths = [bank.activation.clone(), bank.effective_activation.clone()];
    let execution =
        ExecutionPlan::fully_resident(eredu_core::DevicePlan::new("mlx", "metal:0").unwrap())
            .with_required_session_capabilities(SessionCapabilities::new(true, true, true))
            .with_drafting(DraftingPlan::Embedded {
                max_draft_tokens: 2,
                lookahead: false,
                adaptive_lookahead: false,
            });
    let (records, mut result) = run_plan(mode, target, execution, |discovery| {
        plan(discovery, &paths, strided)
    });
    drop(graph);
    // run_plan already retired the loaded model, original source and driver.
    result["sparse"] = inspect(&records, mode != "ordinary", strided);
    result
}
#[test]
#[ignore = "requires an accessible Metal device and original sparse model intervention sources"]
fn native_original_sparse_model_interventions_match_ordinary_and_controlled() {
    compare_modes(CASE, RESULT, |mode| run_selected(mode, false));
}

#[test]
#[ignore = "requires an accessible Metal device and original sparse model intervention sources"]
fn native_original_strided_sparse_model_interventions_match_ordinary_and_controlled() {
    compare_modes(STRIDED_CASE, STRIDED_RESULT, |mode| {
        run_selected(mode, true)
    });
}
