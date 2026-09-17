//! Independent host expectations for each boundary, including strided selections.
use super::*;

pub(super) fn compare(
    record: &InterventionRecord,
    effective: &mut [f32],
    preferred: usize,
    evidence: Evidence,
) {
    if evidence == Evidence::None {
        assert!(record.evidence.is_empty());
        return;
    }
    let indices: Vec<usize> = match record.operation_id.as_str() {
        "scale-strided" => (0..64).step_by(2).collect(),
        "zero-selected" => vec![preferred],
        "add-selected" => vec![preferred, preferred + 1],
        _ => (0..64).collect(),
    };
    let before: Vec<f32> = indices.iter().map(|&i| effective[i]).collect();
    match record.operation_id.as_str() {
        "replace" => {
            effective.fill(-4.0);
            effective[preferred] = 8.0;
            effective[preferred + 1] = 2.0;
        }
        "scale-strided" => {
            for i in (0..64).step_by(2) {
                effective[i] *= 2.0;
            }
        }
        "zero-selected" => effective[preferred] = 0.0,
        "add-selected" => effective[preferred] += 24.0,
        "mask" => effective[preferred + 1] = 0.0,
        "mask-logits" => {
            effective[0] = f32::NEG_INFINITY;
            effective[preferred + 1] = f32::NEG_INFINITY;
        }
        other => panic!("unknown host oracle operation {other}"),
    }
    let after: Vec<f32> = indices.iter().map(|&i| effective[i]).collect();
    assert_eq!(record.evidence.len(), 2, "{}", record.operation_id);
    for (actual, (side, position, expected)) in record.evidence.iter().zip([
        (
            "before",
            eredu_core::ObservationPosition::BeforeIntervention,
            before,
        ),
        (
            "after",
            eredu_core::ObservationPosition::AfterIntervention,
            after,
        ),
    ]) {
        assert_eq!(
            actual.selection_id,
            format!("{}:{side}:None", record.operation_id)
        );
        assert_eq!(actual.path, eredu_core::MODEL_LOGITS_OBSERVATION_PATH);
        assert_eq!(actual.position, position);
        assert_eq!(actual.source_shape.as_deref(), Some([1, 1, 64].as_slice()));
        assert_eq!(
            actual.selected_shape.as_deref(),
            Some([1, 1, indices.len() as u64].as_slice())
        );
        let payload = actual
            .payload
            .as_ref()
            .expect("before/after evidence must not be skipped");
        match evidence {
            Evidence::Preview => {
                let tensor = payload.as_tensor().expect("bounded preview payload");
                let eredu_core::TensorObservationData::F32(values) = tensor.data() else {
                    panic!("floating intervention preview")
                };
                assert_eq!(
                    values,
                    &expected[..expected.len().min(13)],
                    "{} {side}",
                    record.operation_id
                );
            }
            Evidence::Summary => {
                let CapturePayload::Summary(actual) = payload else {
                    panic!("finite summary")
                };
                let finite: Vec<f64> = expected
                    .iter()
                    .filter(|v| v.is_finite())
                    .map(|&v| f64::from(v))
                    .collect();
                assert_eq!(actual.elements, expected.len() as u64);
                assert_eq!(actual.finite, finite.len() as u64);
                assert_eq!(actual.non_finite, (expected.len() - finite.len()) as u64);
                assert_eq!(actual.nan, 0);
                assert_eq!(actual.positive_infinity, 0);
                assert_eq!(
                    actual.negative_infinity,
                    expected.iter().filter(|&&v| v == f32::NEG_INFINITY).count() as u64
                );
                let mean = finite.iter().sum::<f64>() / finite.len() as f64;
                let rms = (finite.iter().map(|v| v * v).sum::<f64>() / finite.len() as f64).sqrt();
                for (actual, expected) in [
                    (
                        actual.min,
                        finite.iter().copied().fold(f64::INFINITY, f64::min),
                    ),
                    (
                        actual.max,
                        finite.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                    ),
                    (actual.mean, mean),
                    (actual.rms, rms),
                ] {
                    assert!(
                        (actual.unwrap() - expected).abs() < 2e-5 + 2e-5 * expected.abs(),
                        "{} {side}: {actual:?} != {expected}",
                        record.operation_id
                    );
                }
            }
            Evidence::None => unreachable!(),
        }
    }
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn native_original_intervention_preview_matches_each_host_boundary_and_replay() {
    compare_modes(Evidence::Preview, "managed_plain::speculative::interventions::evidence::native_original_intervention_preview_matches_each_host_boundary_and_replay");
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn native_original_intervention_summary_matches_each_host_boundary_and_replay() {
    compare_modes(Evidence::Summary, "managed_plain::speculative::interventions::evidence::native_original_intervention_summary_matches_each_host_boundary_and_replay");
}
