//! Single-element activation fixture using the shared capture/intervention owners.
use super::*;

pub(super) struct Records;
pub(super) fn limits() -> CaptureLimits {
    let usage = CaptureUsage {
        captures: 10_000,
        retained_bytes: 32_000_000,
        host_bytes: 32_000_000,
        encoded_bytes: 32_000_000,
    };
    CaptureLimits {
        per_step: usage,
        cumulative: usage,
        physical_native_bytes: None,
        on_limit: CaptureLimitPolicy::Fail,
    }
}
pub(super) fn discovery() -> CaptureDiscovery {
    CaptureDiscovery {
        artifact_identity: "host-source".into(),
        catalog: ObservationCatalog {
            schema_version: 1,
            completeness: DescriptionCompleteness::Complete,
            points: vec![ObservationPoint {
                path: "state".into(),
                node_id: "model".into(),
                meaning: "fixture state".into(),
                value_type: ObservationValueType::Tensor,
                dtype: ObservationDtype::Floating,
                axes: Some(vec![TensorAxis {
                    name: "value".into(),
                    dimension: SymbolicDimension::Known(1),
                }]),
                prefill: true,
                decode: true,
                requirements: vec![ObservationRequirement::ActivationHooks],
                position: ObservationPosition::BeforeIntervention,
                retained_bytes: None,
                host_bytes: None,
            }],
        },
        support: ObservationSupportReport {
            schema_version: 1,
            points: vec![ObservationSupport {
                path: "state".into(),
                prefill: ObservationSupportStatus::Supported,
                decode: ObservationSupportStatus::Supported,
                floating_to_f32: true,
            }],
            capture: CaptureCapabilities {
                transformations: vec![CaptureTransformKind::Preview],
                ..Default::default()
            },
        },
    }
}
pub(super) fn interventions() -> InterventionDiscovery {
    InterventionDiscovery {
        schema_version: 1,
        artifact_identity: "host-source".into(),
        session_identity: Some("host-session".into()),
        points: vec![InterventionPoint {
            path: "state".into(),
            node_id: "model".into(),
            stage: InterventionStage::Activation,
            axes: discovery().catalog.points[0].axes.clone().unwrap(),
            dtypes: vec![InterventionDtype::Float32],
            operations: vec![InterventionKind::Scale],
            score_stages: vec![],
            routing: None,
            prefill: ObservationSupportStatus::Supported,
            decode: ObservationSupportStatus::Supported,
            conditions: vec![],
        }],
    }
}
pub(super) fn plans(mode: u8) -> (AdmittedCapturePlan, AdmittedInterventionPlan) {
    let discovery = discovery();
    let request = CaptureRequestShape {
        batch: 1,
        prompt_tokens: 2,
        max_predictions: 20,
    };
    let capture = CapturePlan {
        schema_version: 1,
        limits: limits(),
        selections: if mode & 1 == 0 {
            vec![]
        } else {
            vec![CaptureSelection {
                id: "state-preview".into(),
                path: "state".into(),
                schedule: CaptureSchedule::default(),
                slices: vec![],
                transform: CaptureTransform::Preview { max_elements: 1 },
            }]
        },
    }
    .admit(
        &discovery.catalog,
        &discovery.support,
        &discovery.support.capture,
        request,
    )
    .unwrap();
    let intervention = InterventionPlan {
        schema_version: 1,
        operations: if mode & 2 == 0 {
            vec![]
        } else {
            vec![InterventionOperation {
                id: "future-scale".into(),
                target: "state".into(),
                schedule: CaptureSchedule {
                    first_prediction: 2,
                    every: 2,
                    prefill: false,
                    ..Default::default()
                },
                slices: vec![],
                action: InterventionAction::Scale {
                    dtype: InterventionDtype::Float32,
                    factor: 0.8,
                },
                evidence: InterventionEvidence::Preview { max_elements: 1 },
            }]
        },
    }
    .admit(&interventions(), request, "host-root")
    .unwrap();
    (capture, intervention)
}
pub(super) fn estimate(
    _: &[u64],
    _: &CaptureSelection,
    _: &ResolvedCaptureSlice,
) -> Result<CaptureUsage, CaptureError> {
    Ok(CaptureUsage {
        captures: 1,
        retained_bytes: 128,
        host_bytes: 128,
        encoded_bytes: 1024,
    })
}
impl CaptureBackend for Records {
    type Tensor = Vec<f32>;
    type Error = io::Error;
    fn shape(&self, tensor: &Vec<f32>) -> io::Result<Vec<u64>> {
        Ok(vec![tensor.len() as u64])
    }
    fn estimate(
        &self,
        _: &Vec<f32>,
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        estimate(&slice.shape, selection, slice)
    }
    fn transform(
        &mut self,
        value: &Vec<f32>,
        _: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> io::Result<CapturePayload> {
        let value = self.select_region(value, slice)?;
        Ok(CapturePayload::Tensor(
            TensorObservation::new(vec![value.len()], TensorObservationData::F32(value)).unwrap(),
        ))
    }
}
impl InterventionEstimator for Records {
    fn validate_geometry(
        &self,
        source: &[u64],
        slice: &ResolvedCaptureSlice,
    ) -> Result<(), CaptureError> {
        if source.len() != 1 || slice.shape.len() != 1 {
            return Err(CaptureError::Unsupported(
                "fixture supports rank one".into(),
            ));
        }
        Ok(())
    }
    fn capture_usage(
        &self,
        source: &[u64],
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        estimate(source, selection, slice)
    }
    fn original_route_usage(
        &self,
        _: &InterventionRoutingPolicy,
        _: u64,
    ) -> Result<CaptureUsage, CaptureError> {
        Err(CaptureError::Unsupported("fixture has no routing".into()))
    }
}
impl InterventionBackend for Records {
    fn intervention_dtype(&self, _: &Vec<f32>) -> io::Result<InterventionDtype> {
        Ok(InterventionDtype::Float32)
    }
    fn validate_intervention_geometry(
        &self,
        source: &[u64],
        slice: &ResolvedCaptureSlice,
    ) -> Result<(), CaptureError> {
        self.validate_geometry(source, slice)
    }
    fn select_region(
        &mut self,
        value: &Vec<f32>,
        slice: &ResolvedCaptureSlice,
    ) -> io::Result<Vec<f32>> {
        Ok((slice.starts[0]..slice.ends[0])
            .step_by(slice.strides[0] as usize)
            .map(|i| value[i as usize])
            .collect())
    }
    fn update_region(
        &mut self,
        value: &Vec<f32>,
        slice: &ResolvedCaptureSlice,
        replacement: &Vec<f32>,
    ) -> io::Result<Vec<f32>> {
        let mut output = value.clone();
        for (i, v) in (slice.starts[0]..slice.ends[0])
            .step_by(slice.strides[0] as usize)
            .zip(replacement)
        {
            output[i as usize] = *v;
        }
        Ok(output)
    }
    fn zeros(&mut self, shape: &[u64], _: InterventionDtype) -> io::Result<Vec<f32>> {
        Ok(vec![0.; shape.iter().product::<u64>() as usize])
    }
    fn scale(&mut self, value: &Vec<f32>, factor: f32) -> io::Result<Vec<f32>> {
        Ok(value.iter().map(|v| v * factor).collect())
    }
    fn fill_masked(&mut self, value: &Vec<f32>, keep: &[bool], fill: f32) -> io::Result<Vec<f32>> {
        Ok(value
            .iter()
            .zip(keep)
            .map(|(v, k)| if *k { *v } else { fill })
            .collect())
    }
    fn realize_tensor(&mut self, tensor: &InterventionTensor) -> io::Result<Vec<f32>> {
        match &tensor.values {
            InterventionValues::Float32(values) => Ok(values.clone()),
            _ => Err(io::Error::other("fixture dtype")),
        }
    }
    fn add(&mut self, a: &Vec<f32>, b: &Vec<f32>) -> io::Result<Vec<f32>> {
        Ok(a.iter().zip(b).map(|(a, b)| a + b).collect())
    }
    fn fill_columns(&mut self, value: &Vec<f32>, ids: &[u32], fill: f32) -> io::Result<Vec<f32>> {
        let mut output = value.clone();
        for id in ids {
            output[*id as usize] = fill;
        }
        Ok(output)
    }
}
