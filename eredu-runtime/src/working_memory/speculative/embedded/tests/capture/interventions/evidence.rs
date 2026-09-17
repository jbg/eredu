use super::*;

#[test]
fn model_evidence_preserves_window_sides_logical_skips_and_escaped_custody() {
    run_mode(true, true);
}
fn summary(values: &[f32]) -> CaptureSummary {
    CaptureSummary {
        elements: values.len() as u64,
        finite: values.len() as u64,
        non_finite: 0,
        nan: 0,
        positive_infinity: 0,
        negative_infinity: 0,
        min: Some(values.iter().copied().fold(f32::INFINITY, f32::min) as f64),
        max: Some(values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64),
        mean: Some(values.iter().map(|&v| v as f64).sum::<f64>() / values.len() as f64),
        rms: Some(
            (values.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / values.len() as f64).sqrt(),
        ),
    }
}
pub(super) fn usage(
    backend: &EditBackend,
    _value: &[f32; 6],
    claim: &CaptureInterventionEvidenceClaim<'_, '_>,
) -> Result<(eredu_core::checkpoint::TensorDtype, CaptureUsage), FundedCaptureError<Failure>> {
    claim
        .validate_source(&backend.source)
        .map_err(|cause| FundedCaptureError::Backend(Failure::from(cause)))?;
    claim
        .validate_model_custody(&backend.capture.custody)
        .map_err(|cause| FundedCaptureError::Backend(Failure::from(cause)))?;
    assert!(claim.validate_source(&backend.foreign_source).is_err());
    assert!(claim.validate_model_custody(&backend.foreign_role).is_err());
    assert_eq!(claim.invocation().unwrap().sequence, 3);
    assert_eq!(claim.invocation_window(), backend.window);
    let elements = match claim.kind() {
        CaptureInterventionEvidenceKind::Preview(value) => {
            assert_eq!(value.geometry().source_shape(), [3, 2]);
            value.geometry().elements()
        }
        CaptureInterventionEvidenceKind::Summary(value) => {
            assert_eq!(value.geometry().source_shape(), [3, 2]);
            value.geometry().elements()
        }
    };
    Ok((
        eredu_core::checkpoint::TensorDtype::F32,
        CaptureUsage {
            captures: 1,
            retained_bytes: 24,
            host_bytes: 24,
            encoded_bytes: 512 + elements as u64,
        },
    ))
}
pub(super) fn capture<'a, 'c>(
    backend: &mut EditBackend,
    value: &[f32; 6],
    claim: CaptureInterventionEvidenceClaim<'a, 'c>,
) -> Result<ClaimedInterventionEvidence<'a>, FundedCaptureError<Failure>> {
    usage(backend, value, &claim)?;
    assert!(!claim.window_empty());
    let (receipt, kind) = claim.into_parts();
    Ok(match kind {
        CaptureInterventionEvidenceKind::Preview(claim) => {
            let count = claim.geometry().elements();
            let mut output = claim.prepare()?;
            for &v in &value[..count] {
                output.push_f32(v).unwrap();
            }
            receipt.finish_preview(output.finish().unwrap())?
        }
        CaptureInterventionEvidenceKind::Summary(claim) => receipt.finish_summary(
            claim
                .finish_model(&backend.capture.custody, summary(value))
                .unwrap(),
        )?,
    })
}
pub(super) fn check(frame: &eredu_core::capture::SharedCapturedStep, input: &[f32; 6]) {
    let edits = &frame.as_ref().interventions;
    assert_eq!(edits[0].evidence.len(), 2);
    assert_eq!(
        edits[0].evidence[0].outcome,
        CaptureOutcome::Skipped {
            reason: CaptureSkipReason::Limit {
                budget: CaptureBudget::Captures,
                cumulative: true
            }
        }
    );
    assert_eq!(
        edits[0].evidence[1].outcome,
        CaptureOutcome::Skipped {
            reason: CaptureSkipReason::NotInvoked
        }
    );
    assert_eq!(
        edits[0].evidence[1].source_shape.as_deref(),
        Some([3, 2].as_slice())
    );
    assert_eq!(edits[0].evidence[1].selected_shape, None);
    assert!(edits[0].evidence[1].charged.host_bytes > edits[0].evidence[0].charged.host_bytes);
    for (side, values) in [*input, input.map(|v| v * 3.0)].into_iter().enumerate() {
        assert_eq!(
            edits[1].evidence[side].payload,
            Some(CapturePayload::Summary(summary(&values)))
        );
    }
    for (side, values) in [input.map(|v| v * 3.0), input.map(|v| v * -3.0)]
        .into_iter()
        .enumerate()
    {
        let tensor = edits[2].evidence[side]
            .payload
            .as_ref()
            .unwrap()
            .as_tensor()
            .unwrap();
        assert!(
            matches!(tensor.data(),TensorObservationData::F32(actual) if actual.as_slice()==&values[..3])
        );
    }
    assert!(edits[3].evidence.iter().all(|record| record.outcome
        == CaptureOutcome::Skipped {
            reason: CaptureSkipReason::NotInvoked
        }));
}
