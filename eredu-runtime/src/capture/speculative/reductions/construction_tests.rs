use super::*;
use crate::working_memory::{InferenceExecutionIdentity, MemoryLedger};
use eredu_core::{
    DescriptionCompleteness, ObservationCatalog, ObservationDtype, ObservationPoint,
    ObservationPosition, ObservationSupport, ObservationSupportReport, ObservationSupportStatus,
    ObservationValueType, SymbolicDimension, TensorAxis, TensorObservation, TensorObservationData,
};

pub(super) fn source() -> AdmittedCapturePlan {
    let point = ObservationPoint {
        path: "block.output".into(),
        node_id: "block".into(),
        meaning: "actual windows".into(),
        value_type: ObservationValueType::Tensor,
        dtype: ObservationDtype::Floating,
        axes: Some(vec![
            TensorAxis {
                name: "sequence".into(),
                dimension: SymbolicDimension::Sequence,
            },
            TensorAxis {
                name: "width".into(),
                dimension: SymbolicDimension::Known(2),
            },
        ]),
        prefill: true,
        decode: true,
        requirements: Vec::new(),
        position: ObservationPosition::ReadOnly,
        retained_bytes: None,
        host_bytes: None,
    };
    let unlimited = CaptureUsage {
        captures: u64::MAX,
        retained_bytes: u64::MAX,
        host_bytes: u64::MAX,
        encoded_bytes: u64::MAX,
    };
    let plan = CapturePlan {
        schema_version: CAPTURE_SCHEMA_VERSION,
        selections: [
            CaptureTransform::Summary,
            CaptureTransform::Histogram {
                edges: vec![-5.0, 0.0, 5.0, 10.0],
            },
            CaptureTransform::Preview { max_elements: 5 },
        ]
        .into_iter()
        .enumerate()
        .map(|(i, transform)| CaptureSelection {
            id: format!("selection-{i}"),
            path: point.path.clone(),
            schedule: Default::default(),
            slices: Vec::new(),
            transform,
        })
        .collect(),
        limits: CaptureLimits {
            per_step: unlimited,
            cumulative: unlimited,
            on_limit: CaptureLimitPolicy::Fail,
        },
    };
    plan.admit_invocations(
        &ObservationCatalog {
            schema_version: 1,
            points: vec![point.clone()],
            completeness: DescriptionCompleteness::Complete,
        },
        &ObservationSupportReport {
            schema_version: 1,
            capture: Default::default(),
            points: vec![ObservationSupport {
                path: point.path.clone(),
                prefill: ObservationSupportStatus::Supported,
                decode: ObservationSupportStatus::Supported,
                floating_to_f32: true,
            }],
        },
        &CaptureCapabilities {
            transformations: vec![
                CaptureTransformKind::Summary,
                CaptureTransformKind::Histogram,
                CaptureTransformKind::Preview,
            ],
            max_histogram_bins: 3,
            conditions: Vec::new(),
        },
        CaptureInvocationBounds {
            batch: 1,
            max_sequence: 11,
            max_context: None,
            max_predictions: 2,
        },
    )
    .unwrap()
}
pub(super) fn geometry() -> SpeculativePrefillReductionGeometry {
    SpeculativePrefillReductionGeometry {
        target_sequence: 11,
        prediction_sequence: 10,
    }
}
pub(super) fn origin() -> SpeculativeActivationOrigin {
    SpeculativeActivationOrigin {
        request: eredu_core::SpeculativeRequestId::new(83),
        committed_tokens: 0,
        prediction: 0,
        prefix_digest: [4; 32],
        optimistic: false,
    }
}
pub(super) fn values(start: u64, end: u64) -> Vec<f32> {
    (start * 2..end * 2)
        .map(|n| n as f32 * 0.375 - 2.5)
        .collect()
}
pub(super) fn summary(values: &[f32]) -> CaptureSummary {
    let count = values.len() as u64;
    CaptureSummary {
        elements: count,
        finite: count,
        non_finite: 0,
        nan: 0,
        positive_infinity: 0,
        negative_infinity: 0,
        min: Some(values.iter().copied().fold(f32::INFINITY, f32::min) as f64),
        max: Some(values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64),
        mean: Some(values.iter().map(|&x| x as f64).sum::<f64>() / count as f64),
        rms: Some((values.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / count as f64).sqrt()),
    }
}
fn histogram(values: &[f32]) -> CaptureHistogram {
    let mut out = CaptureHistogram {
        edges: vec![-5.0, 0.0, 5.0, 10.0],
        counts: vec![0; 3],
        below: 0,
        above: 0,
        non_finite: 0,
    };
    for &value in values {
        if value < -5.0 {
            out.below += 1
        } else if value > 10.0 {
            out.above += 1
        } else {
            let i = if value < 0.0 {
                0
            } else if value < 5.0 {
                1
            } else {
                2
            };
            out.counts[i] += 1;
        }
    }
    out
}
fn finish(
    mut group: WindowReductions,
    source: &AdmittedCapturePlan,
) -> SpeculativePrefillReductions {
    for (n, (start, end)) in [(0, 3), (3, 6), (6, 11)].into_iter().enumerate() {
        let data = values(start, end);
        let count = data.len() as u64;
        let span = eredu_core::speculative::SpeculativePrefillSpan {
            prompt_tokens: 11,
            input_start: start,
            input_end: end,
            position: start,
            hidden_start: start,
            token_start: start,
            sequence: end - start,
            seed_start: 0,
        };
        group
            .begin_capture_window(
                SpeculativeActivationPhase::TargetPrefill,
                span,
                origin(),
                geometry(),
            )
            .unwrap();
        let records: Vec<_> = source
            .plan()
            .selections
            .iter()
            .enumerate()
            .map(|(i, selection)| {
                let payload = match i {
                    0 => CapturePayload::Summary(summary(&data)),
                    1 => CapturePayload::Histogram(histogram(&data)),
                    _ => CapturePayload::Tensor(
                        TensorObservation::new(
                            vec![5.min(data.len())],
                            TensorObservationData::F32(data[..5.min(data.len())].to_vec()),
                        )
                        .unwrap(),
                    ),
                };
                CaptureRecord {
                    schema_version: CAPTURE_SCHEMA_VERSION,
                    selection_id: selection.id.clone(),
                    path: selection.path.clone(),
                    node_id: "block".into(),
                    position: ObservationPosition::ReadOnly,
                    source_shape: Some(vec![end - start, 2]),
                    source_dtype: Some(eredu_core::checkpoint::TensorDtype::Bf16),
                    selected_shape: Some(vec![end - start, 2]),
                    outcome: crate::capture::completed_capture_outcome(&selection.transform, count),
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
        group.prepare_fixed(&records, &[]).unwrap();
        group.finish_window(&records, &[], n as u64 + 7, true);
    }
    group.seal_fixed().unwrap();
    group.finish(true)
}
#[test]
fn paid_aggregate_constructor_matches_nonzero_window_algebra_and_refuses_one_short() {
    let source = source();
    let scopes = [SpeculativeCaptureScope::Target; 3];
    let mut ordinary = CaptureLedger::new(&source);
    let expected = finish(
        WindowReductions::create_capture(
            &source,
            &mut ordinary,
            &scopes,
            geometry(),
            origin(),
            7,
            false,
            Metadata::ordinary(),
        )
        .unwrap()
        .unwrap(),
        &source,
    );
    let pool = crate::working_memory::memory_fixture::host_ledger(1 << 26, 0).unwrap();
    let funding = pool
        .prepare_workspace_metadata(
            &InferenceExecutionIdentity::default(),
            crate::working_memory::memory_fixture::resolved_host_limits(&pool, 1 << 26),
        )
        .unwrap();
    let mut ledger = CaptureLedger::new(&source);
    let group = WindowReductions::create_capture(
        &source,
        &mut ledger,
        &scopes,
        geometry(),
        origin(),
        7,
        false,
        Metadata::original(&funding),
    )
    .unwrap()
    .unwrap();
    let required = pool.payload_used_bytes().unwrap();
    assert_eq!(ordinary.total(), ledger.total());
    let actual = finish(group, &source);
    assert_eq!(actual, expected);
    let full = values(0, 11);
    let CapturePayload::Summary(total) = actual.records[0].record.payload.as_ref().unwrap() else {
        panic!("Summary")
    };
    let oracle = summary(&full);
    assert_eq!(total.elements, 22);
    assert_eq!(total.mean, oracle.mean);
    assert!((total.rms.unwrap() - oracle.rms.unwrap()).abs() < 1e-12);
    assert_eq!(
        actual.records[1].record.payload,
        Some(CapturePayload::Histogram(histogram(&full)))
    );
    let preview = actual.records[2]
        .record
        .payload
        .as_ref()
        .unwrap()
        .as_tensor()
        .unwrap();
    assert_eq!(
        preview.data(),
        &TensorObservationData::F32(full[..5].to_vec())
    );
    assert_eq!(pool.payload_used_bytes().unwrap(), required);
    drop((actual, expected, funding));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    let short = crate::working_memory::memory_fixture::host_ledger(required - 1, 0).unwrap();
    let funding = short
        .prepare_workspace_metadata(
            &InferenceExecutionIdentity::default(),
            crate::working_memory::memory_fixture::resolved_host_limits(&short, required - 1),
        )
        .unwrap();
    let mut ledger = CaptureLedger::new(&source);
    let refused = WindowReductions::create_capture(
        &source,
        &mut ledger,
        &scopes,
        geometry(),
        origin(),
        7,
        false,
        Metadata::original(&funding),
    );
    assert!(matches!(refused, Err(ConstructionError::Metadata(_))));
    assert!(ledger.total().host_bytes > 0);
    assert!(short.payload_used_bytes().unwrap() > 0);
    assert!(short.payload_used_bytes().unwrap() < required);
    drop(refused);
    drop(funding);
    assert_eq!(short.payload_used_bytes().unwrap(), 0);
}
