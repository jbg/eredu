use super::*;
use eredu_core::capture::*;

#[derive(Default)]
pub(super) struct State {
    pub capture: Option<eredu_runtime::capture::CaptureSession>,
    pub prediction: u64,
}

pub(super) fn discovery() -> CaptureDiscovery {
    let path = eredu_core::MODEL_LOGITS_OBSERVATION_PATH.to_owned();
    let capabilities = CaptureCapabilities {
        transformations: vec![CaptureTransformKind::Summary],
        max_histogram_bins: 0,
        physical_native_limit: false,
        conditions: vec![],
    };
    CaptureDiscovery {
        artifact_identity: "conformance-mock-artifact".into(),
        catalog: eredu_core::ObservationCatalog {
            schema_version: 1,
            completeness: eredu_core::DescriptionCompleteness::Complete,
            points: vec![eredu_core::ObservationPoint {
                path: path.clone(),
                node_id: "output".into(),
                meaning: "mock logits".into(),
                value_type: eredu_core::ObservationValueType::Tensor,
                dtype: eredu_core::ObservationDtype::Floating,
                axes: Some(vec![
                    eredu_core::TensorAxis {
                        name: "batch".into(),
                        dimension: eredu_core::SymbolicDimension::Batch,
                    },
                    eredu_core::TensorAxis {
                        name: "sequence".into(),
                        dimension: eredu_core::SymbolicDimension::Sequence,
                    },
                    eredu_core::TensorAxis {
                        name: "vocabulary".into(),
                        dimension: eredu_core::SymbolicDimension::Known(1),
                    },
                ]),
                prefill: true,
                decode: true,
                requirements: vec![eredu_core::ObservationRequirement::ActivationHooks],
                position: eredu_core::ObservationPosition::BeforeIntervention,
                retained_bytes: None,
                host_bytes: None,
            }],
        },
        support: eredu_core::ObservationSupportReport {
            schema_version: 1,
            capture: capabilities,
            points: vec![eredu_core::ObservationSupport {
                path,
                prefill: eredu_core::ObservationSupportStatus::Supported,
                decode: eredu_core::ObservationSupportStatus::Supported,
                floating_to_f32: true,
            }],
        },
    }
}

fn cost(shape: &[u64]) -> Result<CaptureUsage, CaptureError> {
    Ok(CaptureUsage {
        captures: 1,
        retained_bytes: mul(elements(shape)?, 4)?,
        host_bytes: 128,
        encoded_bytes: 512,
    })
}

pub(super) fn validate(plan: &AdmittedCapturePlan) -> Result<(), CaptureError> {
    eredu_runtime::capture::preflight(plan, |shape, _, _| cost(shape))
}

struct Mechanism;
impl CaptureBackend for Mechanism {
    type Tensor = Vec<u64>;
    type Error = MockError;
    fn shape(&self, tensor: &Self::Tensor) -> Result<Vec<u64>, Self::Error> {
        Ok(tensor.clone())
    }
    fn estimate(
        &self,
        tensor: &Self::Tensor,
        _: &CaptureSelection,
        _: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        cost(tensor)
    }
    fn transform(
        &mut self,
        tensor: &Self::Tensor,
        selection: &CaptureSelection,
        _: &ResolvedCaptureSlice,
    ) -> Result<CapturePayload, Self::Error> {
        if selection.id == "injected-native-failure" {
            return Err(MockError::Capture("injected capture fault".into()));
        }
        let count = elements(tensor).unwrap();
        Ok(CapturePayload::Summary(CaptureSummary {
            elements: count,
            finite: count,
            non_finite: 0,
            nan: 0,
            positive_infinity: 0,
            negative_infinity: 0,
            min: Some(1.0),
            max: Some(1.0),
            mean: Some(1.0),
            rms: Some(1.0),
        }))
    }
}

impl State {
    pub fn observe(&mut self, phase: CapturePhase, sequence: usize) -> Result<(), MockError> {
        if let Some(capture) = &mut self.capture {
            capture
                .begin_step(phase, self.prediction)
                .map_err(|e| MockError::Capture(e.to_string()))?;
            capture
                .observe(
                    &mut Mechanism,
                    eredu_core::MODEL_LOGITS_OBSERVATION_PATH,
                    &vec![1, sequence as u64, 1],
                )
                .map_err(|e| MockError::Capture(e.to_string()))?;
        }
        self.prediction += 1;
        Ok(())
    }
}

pub(super) fn plan() -> CapturePlan {
    let usage = CaptureUsage {
        captures: 100,
        retained_bytes: 1_000_000,
        host_bytes: 1_000_000,
        encoded_bytes: 1_000_000,
    };
    CapturePlan {
        schema_version: 1,
        selections: vec![CaptureSelection {
            id: "summary".into(),
            path: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
            schedule: CaptureSchedule::default(),
            slices: vec![],
            transform: CaptureTransform::Summary,
        }],
        limits: CaptureLimits {
            per_step: usage,
            cumulative: usage,
            physical_native_bytes: None,
            on_limit: CaptureLimitPolicy::Fail,
        },
    }
}
