//! Direct full tensors provide independent host oracles for every reduction.
use super::*;
use eredu_core::{TensorObservationData, speculative::SpeculativePrefillReductionStatus};

const CASE: &str = "managed_plain::embedded::activations::reductions::native_original_qwen_window_reductions_match_ordinary_and_controlled";
const RESULT: &str = "PUBLIC_EMBEDDED_WINDOW_REDUCTIONS_RESULT:";
const PREVIEW: usize = 11;
const BASE: [&str; 2] = ["target residual", "prediction fusion"];
const KINDS: [&str; 3] = ["summary", "histogram", "preview"];

pub(super) fn extend(selections: &mut Vec<CaptureSelection>) {
    assert_eq!(selections.len(), 2);
    for i in 0..2 {
        let base = selections[i].clone();
        for kind in KINDS {
            let mut selection = base.clone();
            selection.id = format!("{} {kind}", base.id);
            selection.transform = match kind {
                "summary" => CaptureTransform::Summary,
                "histogram" => CaptureTransform::Histogram {
                    edges: vec![-1.0, 0.0, 1.0],
                },
                "preview" => CaptureTransform::Preview {
                    max_elements: PREVIEW as u64,
                },
                _ => unreachable!(),
            };
            selections.push(selection);
        }
    }
}
fn record<'a>(records: &'a [CaptureRecord], name: &str) -> &'a CaptureRecord {
    records
        .iter()
        .find(|record| record.selection_id == name)
        .expect("exact admitted selection")
}
fn full(value: &CaptureRecord) -> Option<&[f32]> {
    value.payload.as_ref().map(|payload| {
        let tensor = payload.as_tensor().expect("direct tensor oracle");
        let TensorObservationData::F32(data) = tensor.data() else {
            panic!("F32 oracle")
        };
        assert!(data.iter().all(|value| value.is_finite()));
        data.as_slice()
    })
}
fn validate(value: &CaptureRecord, kind: &str, values: &[f32]) -> serde_json::Value {
    assert!(!values.is_empty());
    let payload = value.payload.as_ref().expect("selected actual transform");
    match (kind, payload) {
        ("summary", CapturePayload::Summary(summary)) => {
            assert_eq!(value.outcome, CaptureOutcome::Captured);
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
                    (values.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / count as f64).sqrt(),
                ),
            };
            compare(
                &serde_json::to_value(summary).unwrap(),
                &serde_json::to_value(expected).unwrap(),
                "independent full-tensor summary",
            );
        }
        ("histogram", CapturePayload::Histogram(histogram)) => {
            assert_eq!(value.outcome, CaptureOutcome::Captured);
            let mut expected = CaptureHistogram {
                edges: vec![-1.0, 0.0, 1.0],
                counts: vec![0, 0],
                below: 0,
                above: 0,
                non_finite: 0,
            };
            for &value in values {
                if value < -1.0 {
                    expected.below += 1;
                } else if value > 1.0 {
                    expected.above += 1;
                } else {
                    expected.counts[usize::from(value >= 0.0)] += 1;
                }
            }
            assert_eq!(*histogram, expected);
        }
        ("preview", _) => {
            let tensor = payload.as_tensor().expect("bounded Preview");
            let count = values.len().min(PREVIEW);
            assert_eq!(tensor.shape(), &[count]);
            assert_eq!(
                tensor.data(),
                &TensorObservationData::F32(values[..count].to_vec())
            );
            if values.len() > PREVIEW {
                assert_eq!(
                    value.outcome,
                    CaptureOutcome::Truncated {
                        available_elements: values.len() as u64,
                        emitted_elements: count as u64
                    }
                );
            } else {
                assert_eq!(value.outcome, CaptureOutcome::Captured);
            }
        }
        _ => panic!("wrong transform payload for {kind}"),
    }
    serde_json::json!({"id":value.selection_id,"path":value.path,"node":value.node_id,
        "position":value.position,"source_shape":value.source_shape,"source_dtype":value.source_dtype,
        "selected_shape":value.selected_shape,"outcome":value.outcome,"payload":payload})
}

pub(super) fn rows(records: &[SpeculativeActivationCapture], shared: bool) -> serde_json::Value {
    let mut physical = Vec::new();
    for envelope in records {
        let step = envelope.captures.as_step();
        assert_eq!(step.records.len(), 8);
        let mut transformed = Vec::new();
        for base in BASE {
            let original = record(&step.records, base);
            for kind in KINDS {
                let value = record(&step.records, &format!("{base} {kind}"));
                if let Some(values) = full(original) {
                    assert_eq!(value.source_shape, original.source_shape);
                    assert_eq!(value.source_dtype, original.source_dtype);
                    assert_eq!(value.selected_shape, original.selected_shape);
                    assert!(
                        value.charged.captures > 0,
                        "physical capture owns its charge"
                    );
                    transformed.push(validate(value, kind, values));
                } else {
                    assert_eq!(value.outcome, original.outcome);
                    assert!(value.payload.is_none());
                    transformed
                        .push(serde_json::json!({"id":value.selection_id,"outcome":value.outcome}));
                }
            }
        }
        physical.push(
            serde_json::json!({"invocation":envelope.invocation,"phase":envelope.phase,
            "span":envelope.prefill_span,"records":transformed}),
        );
    }
    let delivered: Vec<_> = records
        .iter()
        .filter(|envelope| envelope.prefill_reductions.is_some())
        .collect();
    assert_eq!(
        delivered.len(),
        1,
        "one final report for the actual logical prefill"
    );
    let envelope = delivered[0];
    assert_eq!(
        envelope.phase,
        SpeculativeActivationPhase::PredictionPrefill
    );
    let delivery = envelope.prefill_reductions.as_ref().unwrap();
    if shared {
        let report = delivery;
        let alias = report.clone();
        assert!(alias.same_storage(report));
        assert_eq!(
            alias.as_reductions().records.as_ptr(),
            report.as_reductions().records.as_ptr()
        );
    }
    let report = delivery.as_reductions();
    assert!(report.interventions.is_empty());
    assert_eq!(report.records.len(), 6);
    assert_eq!(report.logical_invocation, records[0].invocation);
    assert_eq!(report.origin, records[0].origin);
    let mut logical = Vec::new();
    for (phase, base, sequence, widths) in [
        (
            SpeculativeActivationPhase::TargetPrefill,
            BASE[0],
            5,
            vec![2, 2, 1],
        ),
        (
            SpeculativeActivationPhase::PredictionPrefill,
            BASE[1],
            4,
            vec![1, 2, 1],
        ),
    ] {
        let windows: Vec<_> = records
            .iter()
            .filter(|envelope| envelope.phase == phase)
            .collect();
        assert_eq!(
            windows
                .iter()
                .map(|e| e.prefill_span.unwrap().sequence)
                .collect::<Vec<_>>(),
            widths
        );
        let values: Vec<_> = windows
            .iter()
            .flat_map(|envelope| {
                full(record(&envelope.captures.as_step().records, base))
                    .unwrap()
                    .iter()
                    .copied()
            })
            .collect();
        assert_eq!(values.len(), sequence as usize * 8);
        for kind in KINDS {
            let name = format!("{base} {kind}");
            let entry = report
                .records
                .iter()
                .find(|entry| entry.phase == phase && entry.record.selection_id == name)
                .unwrap();
            assert_eq!(entry.status, SpeculativePrefillReductionStatus::Complete);
            assert_eq!(
                (
                    entry.logical_sequence,
                    entry.covered_sequence,
                    entry.windows
                ),
                (sequence, sequence, 3)
            );
            assert_eq!(entry.first_invocation, Some(windows[0].invocation));
            assert_eq!(entry.last_invocation, Some(windows[2].invocation));
            assert_eq!(
                entry.record.source_shape.as_deref(),
                Some([1, sequence, 8].as_slice())
            );
            assert_eq!(entry.record.selected_shape, entry.record.source_shape);
            let physical_captures: u64 = records
                .iter()
                .filter(|record| record.phase == phase)
                .map(|record| {
                    record.captures.as_step().records[entry.selection_index]
                        .charged
                        .captures
                })
                .sum();
            assert_eq!(
                entry.record.charged.captures, physical_captures,
                "logical record attributes its physical contributors"
            );
            assert_eq!(
                report.charged.captures, 0,
                "aggregate delivery adds no physical capture spending"
            );
            logical.push(serde_json::json!({"phase":phase,"selection":entry.selection_index,
                "windows":entry.windows,"logical_sequence":entry.logical_sequence,"covered_sequence":entry.covered_sequence,
                "first":entry.first_invocation,"last":entry.last_invocation,"status":entry.status,
                "record":validate(&entry.record,kind,&values)}));
        }
    }
    serde_json::json!({"physical":physical,"logical":logical})
}

#[test]
#[ignore = "requires Metal and the original Embedded aggregate collector"]
fn native_original_qwen_window_reductions_match_ordinary_and_controlled() {
    if let Ok(mode) = std::env::var(MODE) {
        println!("\n{RESULT}{}", run_selected(&mode, true));
        return;
    }
    let mut expected = None;
    for mode in ["ordinary", "managed", "controlled"] {
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", CASE, "--ignored", "--nocapture"])
            .env(MODE, mode)
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&result.stdout);
        assert!(
            result.status.success(),
            "{mode}: {stdout}\n{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let actual: serde_json::Value = serde_json::from_str(
            stdout
                .lines()
                .find_map(|line| line.strip_prefix(RESULT))
                .expect("positive aggregate marker"),
        )
        .unwrap();
        if let Some(expected) = &expected {
            compare(&actual, expected, mode);
        } else {
            expected = Some(actual);
        }
    }
}
