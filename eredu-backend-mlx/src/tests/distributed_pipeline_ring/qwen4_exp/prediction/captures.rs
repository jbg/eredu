//! Native ordinary-preparation capture parity, including image/video replacement.
use super::*;
use eredu_core::{capture::*, speculative::*};
use eredu_runtime::speculative::{
    ControlledSpeculativeOptions, ControlledSpeculativeSession, DriveControlledSpeculation,
};

const PATHS: [&str; 5] = [
    "model.layers.0.input",
    "model.layers.1.output",
    "mtp.layers.0.prediction.fusion",
    "mtp.layers.0.prediction.capture",
    "mtp.layers.0.prediction.attention.channels",
];

pub(super) fn admit(runtime: &ModelRuntime<MlxBackend<'_>>) -> AdmittedSpeculativeActivations {
    let discovery =
        <MlxBackend<'_> as SpeculativeGenerationBackend>::speculative_activation_discovery(runtime)
            .unwrap();
    // Match the existing public distributed fixture's finite receipt/parser
    // allowances. These are admission limits, not measured tensor allocations.
    let allowance = CaptureUsage {
        captures: 32,
        retained_bytes: 128 << 20,
        host_bytes: 128 << 20,
        encoded_bytes: 8 << 20,
    };
    SpeculativeActivationPlan {
        schema_version: SPECULATIVE_ACTIVATION_SCHEMA_VERSION,
        bounds: CaptureInvocationBounds {
            batch: 1,
            max_sequence: 64,
            max_context: None,
            // Greedy rejection can require one proposal per emitted token.
            // Replay keeps cumulative capture usage; it does not widen a run's
            // prediction ordinal bounds or refund the admitted usage limit.
            max_predictions: generation_config().max_tokens as u64,
        },
        captures: CapturePlan {
            schema_version: CAPTURE_SCHEMA_VERSION,
            selections: PATHS
                .iter()
                .map(|path| CaptureSelection {
                    id: (*path).into(),
                    path: (*path).into(),
                    schedule: Default::default(),
                    slices: vec![],
                    transform: CaptureTransform::Preview { max_elements: 4096 },
                })
                .collect(),
            limits: CaptureLimits {
                per_step: allowance,
                cumulative: allowance.checked_mul(64).unwrap(),
                physical_native_bytes: None,
                on_limit: CaptureLimitPolicy::Fail,
            },
        },
        interventions: eredu_core::intervention::InterventionPlan {
            schema_version: eredu_core::intervention::INTERVENTION_SCHEMA_VERSION,
            operations: vec![],
        },
    }
    .admit(&discovery)
    .unwrap()
}

pub(super) fn baseline(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    media: bool,
    expected: &eredu_core::SpeculativeGenerationOutput,
) -> Vec<SpeculativeActivationCapture> {
    let admitted = admit(runtime);
    let mut records = Vec::new();
    let mut failure = None;
    let (output, _) = execute_neutral_embedded_mtp_with(
        runtime,
        prompt(media),
        generation_config(),
        DriveControlledSpeculation::new(
            Default::default(),
            ControlledSpeculativeOptions {
                activations: Some(admitted),
                ..Default::default()
            },
            |session: &mut dyn ControlledSpeculativeSession| {
                while let Some(step) = session.step()? {
                    records.extend(step.activations);
                }
                Ok(())
            },
            &mut failure,
        ),
    );
    assert!(failure.is_none(), "single-rank capture: {failure:?}");
    assert_eq!(output.unwrap().token_ids(), expected.token_ids());
    records
}

pub(super) fn compare(
    actual: &[SpeculativeActivationCapture],
    expected: &[SpeculativeActivationCapture],
) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.phase, expected.phase);
        assert_eq!(actual.completed, expected.completed);
        assert_eq!(
            actual.captures.records.len(),
            expected.captures.records.len()
        );
        for (actual, expected) in actual
            .captures
            .records
            .iter()
            .zip(&expected.captures.records)
        {
            assert_eq!(actual.path, expected.path);
            assert_eq!(actual.source_shape, expected.source_shape);
            assert_eq!(actual.selected_shape, expected.selected_shape);
            assert_eq!(actual.source_dtype, expected.source_dtype);
            assert_eq!(actual.outcome, expected.outcome);
            match (&actual.payload, &expected.payload) {
                (Some(CapturePayload::Tensor(actual)), Some(CapturePayload::Tensor(expected))) => {
                    assert_eq!(actual.shape(), expected.shape());
                    let (
                        eredu_core::TensorObservationData::F32(actual),
                        eredu_core::TensorObservationData::F32(expected),
                    ) = (actual.data(), expected.data())
                    else {
                        panic!("floating activation evidence");
                    };
                    assert_eq!(actual.len(), expected.len());
                    for (actual, expected) in actual.iter().zip(expected) {
                        assert!(actual.is_finite() && expected.is_finite());
                        assert!(
                            (actual - expected).abs() <= 2e-4,
                            "distributed capture {actual} differs from resident {expected}"
                        );
                    }
                }
                (None, None) => {}
                _ => panic!("unexpected capture payload"),
            }
        }
    }
    for (index, path) in PATHS.iter().enumerate() {
        let record = actual
            .iter()
            .filter(|event| {
                event.completed
                    && (index < 2
                        || event.phase == SpeculativeActivationPhase::Proposal { depth: 0 })
            })
            .flat_map(|event| &event.captures.records)
            .find(|record| record.path == *path && record.outcome == CaptureOutcome::Captured)
            .unwrap_or_else(|| panic!("no completed capture for {path}"));
        let Some(CapturePayload::Tensor(tensor)) = &record.payload else {
            panic!("missing {path} tensor");
        };
        let eredu_core::TensorObservationData::F32(values) = tensor.data() else {
            panic!("floating capture");
        };
        assert!(
            values.iter().any(|value| value.abs() > 1e-8),
            "zero capture for {path}"
        );
        if index >= 2 {
            let shape: &[u64] = if index < 4 {
                &[1, 1, 2, 32]
            } else {
                &[1, 1, 32]
            };
            assert_eq!(record.source_shape.as_deref(), Some(shape));
            assert_eq!(record.selected_shape.as_deref(), Some(shape));
            assert_eq!(values.len() as u64, shape.iter().product::<u64>());
        }
    }
}

pub(super) fn exact_replay(
    actual: &[SpeculativeActivationCapture],
    expected: &[SpeculativeActivationCapture],
) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.phase, expected.phase);
        assert_eq!(actual.completed, expected.completed);
        assert_eq!(actual.captures.records, expected.captures.records);
        assert_eq!(actual.admission_identity, expected.admission_identity);
    }
}
