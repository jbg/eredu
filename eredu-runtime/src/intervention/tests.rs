use super::*;
use eredu_core::*;

mod session;

#[derive(Clone)]
struct Value {
    shape: Vec<u64>,
    data: Vec<f32>,
}

#[derive(Default)]
struct Backend {
    applications: usize,
    copies: usize,
    fail_application: Option<usize>,
}

fn indices(shape: &[u64], slice: &ResolvedCaptureSlice) -> Vec<usize> {
    (0..elements(shape).unwrap() as usize)
        .filter(|index| {
            let mut index = *index as u64;
            let mut selected = true;
            for axis in (0..shape.len()).rev() {
                let coordinate = index % shape[axis];
                index /= shape[axis];
                selected &= coordinate >= slice.starts[axis]
                    && coordinate < slice.ends[axis]
                    && (coordinate.saturating_sub(slice.starts[axis]))
                        .is_multiple_of(slice.strides[axis]);
            }
            selected
        })
        .collect()
}

fn estimate(
    _: &[u64],
    _: &CaptureSelection,
    slice: &ResolvedCaptureSlice,
) -> Result<CaptureUsage, CaptureError> {
    Ok(CaptureUsage {
        captures: 1,
        retained_bytes: 128,
        host_bytes: 128,
        encoded_bytes: 512 + elements(&slice.shape)? * 32,
    })
}

impl CaptureBackend for Backend {
    type Tensor = Value;
    type Error = std::io::Error;
    fn shape(&self, v: &Value) -> Result<Vec<u64>, Self::Error> {
        Ok(v.shape.clone())
    }
    fn estimate(
        &self,
        v: &Value,
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        estimate(&v.shape, selection, slice)
    }
    fn transform(
        &mut self,
        v: &Value,
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CapturePayload, Self::Error> {
        self.copies += 1;
        let maximum = match selection.transform {
            CaptureTransform::Preview { max_elements } => max_elements as usize,
            _ => usize::MAX,
        };
        let data: Vec<_> = indices(&v.shape, slice)
            .into_iter()
            .take(maximum)
            .map(|i| v.data[i])
            .collect();
        Ok(CapturePayload::Tensor(
            TensorObservation::new(vec![data.len()], TensorObservationData::F32(data)).unwrap(),
        ))
    }
}

impl InterventionBackend for Backend {
    fn intervention_dtype(&self, _: &Value) -> Result<InterventionDtype, Self::Error> {
        Ok(InterventionDtype::Float32)
    }
    fn validate_intervention_geometry(
        &self,
        source: &[u64],
        slice: &ResolvedCaptureSlice,
    ) -> Result<(), CaptureError> {
        Estimates.validate_geometry(source, slice)
    }
    fn select_region(
        &mut self,
        value: &Value,
        slice: &ResolvedCaptureSlice,
    ) -> Result<Value, Self::Error> {
        self.applications += 1;
        if self.fail_application == Some(self.applications) {
            return Err(std::io::Error::other("native application failed"));
        }
        Ok(Value {
            shape: slice.shape.clone(),
            data: indices(&value.shape, slice)
                .iter()
                .map(|i| value.data[*i])
                .collect(),
        })
    }
    fn update_region(
        &mut self,
        value: &Value,
        slice: &ResolvedCaptureSlice,
        replacement: &Value,
    ) -> Result<Value, Self::Error> {
        let mut output = value.clone();
        for (index, v) in indices(&value.shape, slice).iter().zip(&replacement.data) {
            output.data[*index] = *v;
        }
        Ok(output)
    }
    fn zeros(&mut self, shape: &[u64], _: InterventionDtype) -> Result<Value, Self::Error> {
        Ok(Value {
            shape: shape.to_vec(),
            data: vec![0.0; elements(shape).unwrap() as usize],
        })
    }
    fn scale(&mut self, value: &Value, factor: f32) -> Result<Value, Self::Error> {
        Ok(Value {
            shape: value.shape.clone(),
            data: value.data.iter().map(|v| v * factor).collect(),
        })
    }
    fn fill_masked(
        &mut self,
        value: &Value,
        keep: &[bool],
        fill: f32,
    ) -> Result<Value, Self::Error> {
        Ok(Value {
            shape: value.shape.clone(),
            data: value
                .data
                .iter()
                .zip(keep)
                .map(|(v, k)| if *k { *v } else { fill })
                .collect(),
        })
    }
    fn realize_tensor(&mut self, tensor: &InterventionTensor) -> Result<Value, Self::Error> {
        let InterventionValues::Float32(data) = &tensor.values else {
            return Err(std::io::Error::other("unsupported dtype"));
        };
        Ok(Value {
            shape: tensor.shape.clone(),
            data: data.clone(),
        })
    }
    fn add(&mut self, left: &Value, right: &Value) -> Result<Value, Self::Error> {
        Ok(Value {
            shape: left.shape.clone(),
            data: left
                .data
                .iter()
                .zip(&right.data)
                .map(|(a, b)| a + b)
                .collect(),
        })
    }
    fn fill_columns(
        &mut self,
        value: &Value,
        ids: &[u32],
        fill: f32,
    ) -> Result<Value, Self::Error> {
        let width = *value.shape.last().unwrap() as usize;
        Ok(Value {
            shape: value.shape.clone(),
            data: value
                .data
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    if ids.contains(&((i % width) as u32)) {
                        fill
                    } else {
                        *v
                    }
                })
                .collect(),
        })
    }
}

struct Estimates;
impl InterventionEstimator for Estimates {
    fn validate_geometry(&self, _: &[u64], _: &ResolvedCaptureSlice) -> Result<(), CaptureError> {
        Ok(())
    }
    fn capture_usage(
        &self,
        shape: &[u64],
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        estimate(shape, selection, slice)
    }
    fn original_route_usage(
        &self,
        _: &InterventionRoutingPolicy,
        _: u64,
    ) -> Result<CaptureUsage, CaptureError> {
        Err(CaptureError::Unsupported(
            "test backend has no routing estimator".into(),
        ))
    }
}

fn operation(id: &str, action: InterventionAction) -> InterventionOperation {
    InterventionOperation {
        id: id.into(),
        target: "block.output".into(),
        schedule: CaptureSchedule::default(),
        slices: vec![],
        action,
        evidence: InterventionEvidence::Preview { max_elements: 2 },
    }
}

fn plans(
    operations: Vec<InterventionOperation>,
    ordinary: bool,
) -> (AdmittedCapturePlan, AdmittedInterventionPlan) {
    let point = InterventionPoint {
        path: "block.output".into(),
        node_id: "block".into(),
        stage: InterventionStage::Activation,
        axes: vec![
            TensorAxis {
                name: "sequence".into(),
                dimension: SymbolicDimension::Sequence,
            },
            TensorAxis {
                name: "hidden".into(),
                dimension: SymbolicDimension::Known(2),
            },
        ],
        dtypes: vec![InterventionDtype::Float32],
        operations: vec![
            InterventionKind::Zero,
            InterventionKind::Scale,
            InterventionKind::Replace,
            InterventionKind::Mask,
            InterventionKind::Add,
        ],
        score_stages: vec![],
        prefill: ObservationSupportStatus::Supported,
        decode: ObservationSupportStatus::Supported,
        conditions: vec![],
        routing: None,
    };
    let geometry = point.observation_geometry();
    let discovery = InterventionDiscovery {
        schema_version: 1,
        artifact_identity: "source".into(),
        session_identity: Some("backend-session".into()),
        points: vec![point],
    };
    let request = CaptureRequestShape {
        batch: 1,
        prompt_tokens: 2,
        max_predictions: 3,
    };
    let admitted = InterventionPlan {
        schema_version: 1,
        operations,
    }
    .admit(&discovery, request, "session")
    .unwrap();
    let usage = CaptureUsage {
        captures: 64,
        retained_bytes: 1024 * 1024,
        host_bytes: 1024 * 1024,
        encoded_bytes: 1024 * 1024,
    };
    let capabilities = CaptureCapabilities {
        transformations: vec![CaptureTransformKind::Preview],
        ..Default::default()
    };
    let catalog = ObservationCatalog {
        schema_version: 1,
        points: vec![geometry],
        completeness: DescriptionCompleteness::Complete,
    };
    let support = ObservationSupportReport {
        schema_version: 1,
        capture: capabilities.clone(),
        points: vec![ObservationSupport {
            path: "block.output".into(),
            prefill: ObservationSupportStatus::Supported,
            decode: ObservationSupportStatus::Supported,
            floating_to_f32: true,
        }],
    };
    let selections = if ordinary {
        vec![CaptureSelection {
            id: "original".into(),
            path: "block.output".into(),
            schedule: CaptureSchedule::default(),
            slices: vec![],
            transform: CaptureTransform::Preview { max_elements: 2 },
        }]
    } else {
        vec![]
    };
    let capture = CapturePlan {
        schema_version: 1,
        selections,
        limits: CaptureLimits {
            per_step: usage,
            cumulative: usage,
            physical_native_bytes: None,
            on_limit: CaptureLimitPolicy::Fail,
        },
    }
    .admit(&catalog, &support, &capabilities, request)
    .unwrap();
    (capture, admitted)
}

fn values(record: &CaptureRecord) -> &[f32] {
    let Some(CapturePayload::Tensor(tensor)) = &record.payload else {
        panic!("missing tensor evidence")
    };
    let TensorObservationData::F32(values) = tensor.data() else {
        panic!("wrong data")
    };
    values
}

#[test]
fn ordinary_observation_precedes_ordered_interventions_and_downstream_consumes_result() {
    let ops = vec![
        operation(
            "scale",
            InterventionAction::Scale {
                dtype: InterventionDtype::Float32,
                factor: 2.0,
            },
        ),
        operation(
            "zero",
            InterventionAction::Zero {
                dtype: InterventionDtype::Float32,
            },
        ),
    ];
    let (capture, intervention) = plans(ops, true);
    preflight(&capture, &intervention, &Estimates).unwrap();
    let mut session = CaptureSession::new(capture);
    session
        .enable_interventions(intervention, std::sync::Arc::new(Estimates))
        .unwrap();
    let input = Value {
        shape: vec![2, 2],
        data: vec![1.0, 2.0, 3.0, 4.0],
    };
    let mut backend = Backend::default();
    session.begin_step(CapturePhase::Prefill, 0).unwrap();
    session
        .observe(&mut backend, "block.output", &input)
        .unwrap();
    let output = session
        .intervene(&mut backend, "block.output", &input)
        .unwrap()
        .unwrap();
    assert_eq!(output.data.iter().sum::<f32>(), 0.0);
    session.finish_interventions().unwrap();
    let step = session.take_step().unwrap();
    assert_eq!(values(&step.records[0]), [1.0, 2.0]);
    assert_eq!(values(&step.interventions[0].evidence[0]), [1.0, 2.0]);
    assert_eq!(values(&step.interventions[0].evidence[1]), [2.0, 4.0]);
    assert_eq!(values(&step.interventions[1].evidence[0]), [2.0, 4.0]);
    assert_eq!(values(&step.interventions[1].evidence[1]), [0.0, 0.0]);
    assert_eq!(
        step.interventions[0].evidence[1].position,
        ObservationPosition::AfterIntervention
    );
    assert_eq!(step.step_usage.captures, 5);
    assert!(step
        .interventions
        .iter()
        .all(|r| r.outcome == InterventionOutcome::Applied));
}

#[test]
fn prefill_row_patching_leaves_other_rows_unchanged_and_decode_is_inactive() {
    let mut op = operation(
        "patch",
        InterventionAction::Replace {
            tensor: InterventionTensor {
                shape: vec![1, 2],
                values: InterventionValues::Float32(vec![9.0, 8.0]),
            },
        },
    );
    op.schedule.decode = false;
    op.slices = vec![CaptureSlice {
        axis: "sequence".into(),
        start: 1,
        end: 2,
        stride: 1,
    }];
    let (capture, intervention) = plans(vec![op], false);
    let mut session = CaptureSession::new(capture);
    session
        .enable_interventions(intervention, std::sync::Arc::new(Estimates))
        .unwrap();
    let mut backend = Backend::default();
    session.begin_step(CapturePhase::Prefill, 0).unwrap();
    let input = Value {
        shape: vec![2, 2],
        data: vec![1.0, 2.0, 3.0, 4.0],
    };
    assert_eq!(
        session
            .intervene(&mut backend, "block.output", &input)
            .unwrap()
            .unwrap()
            .data,
        [1.0, 2.0, 9.0, 8.0]
    );
    session.take_step().unwrap();
    session.begin_step(CapturePhase::Decode, 1).unwrap();
    assert!(session
        .intervene(
            &mut backend,
            "block.output",
            &Value {
                shape: vec![1, 2],
                data: vec![2.0, 3.0]
            }
        )
        .unwrap()
        .is_none());
    assert_eq!(
        session.take_step().unwrap().interventions[0].outcome,
        InterventionOutcome::Inactive
    );
    assert_eq!(backend.applications, 1);
}

#[test]
fn none_and_inactive_plans_do_not_touch_native_values() {
    let (capture, intervention) = plans(vec![], false);
    let mut session = CaptureSession::new(capture);
    session
        .enable_interventions(intervention, std::sync::Arc::new(Estimates))
        .unwrap();
    let mut backend = Backend::default();
    session.begin_step(CapturePhase::Prefill, 0).unwrap();
    assert!(session
        .intervene(
            &mut backend,
            "block.output",
            &Value {
                shape: vec![],
                data: vec![]
            }
        )
        .unwrap()
        .is_none());
    assert_eq!((backend.applications, backend.copies), (0, 0));
    assert!(session.take_step().unwrap().interventions.is_empty());
}

#[test]
fn missing_duplicate_and_failed_operations_remain_distinct() {
    let ops = vec![
        operation(
            "first",
            InterventionAction::Zero {
                dtype: InterventionDtype::Float32,
            },
        ),
        operation(
            "second",
            InterventionAction::Zero {
                dtype: InterventionDtype::Float32,
            },
        ),
    ];
    let (capture, intervention) = plans(ops, false);
    let mut session = CaptureSession::new(capture);
    session
        .enable_interventions(intervention, std::sync::Arc::new(Estimates))
        .unwrap();
    session.begin_step(CapturePhase::Prefill, 0).unwrap();
    assert!(session.finish_interventions().is_err());
    let mut backend = Backend {
        fail_application: Some(2),
        ..Default::default()
    };
    let input = Value {
        shape: vec![2, 2],
        data: vec![1.0; 4],
    };
    assert!(session
        .intervene(&mut backend, "block.output", &input)
        .is_err());
    assert!(session.finish_interventions().is_err());
    assert!(session
        .intervene(&mut backend, "block.output", &input)
        .is_err());
    let step = session.take_step().unwrap();
    assert_eq!(step.interventions[0].outcome, InterventionOutcome::Applied);
    assert!(matches!(
        step.interventions[1].outcome,
        InterventionOutcome::Failed { .. }
    ));
    assert_eq!(backend.applications, 2);
}

#[test]
fn joint_preflight_and_runtime_charge_evidence_to_capture_limits() {
    let (capture, intervention) = plans(
        vec![operation(
            "zero",
            InterventionAction::Zero {
                dtype: InterventionDtype::Float32,
            },
        )],
        true,
    );
    let mut low = capture.plan().clone();
    low.limits.per_step.captures = 2;
    let catalog = ObservationCatalog {
        schema_version: 1,
        points: capture.points().to_vec(),
        completeness: DescriptionCompleteness::Complete,
    };
    let caps = CaptureCapabilities {
        transformations: vec![CaptureTransformKind::Preview],
        ..Default::default()
    };
    let support = ObservationSupportReport {
        schema_version: 1,
        capture: caps.clone(),
        points: vec![ObservationSupport {
            path: "block.output".into(),
            prefill: ObservationSupportStatus::Supported,
            decode: ObservationSupportStatus::Supported,
            floating_to_f32: true,
        }],
    };
    let limited = low
        .admit(&catalog, &support, &caps, capture.request())
        .unwrap();
    assert!(matches!(
        preflight(&limited, &intervention, &Estimates),
        Err(CaptureError::Limit {
            budget: CaptureBudget::Captures,
            cumulative: false
        })
    ));
    let mut session = CaptureSession::new(limited);
    session
        .enable_interventions(intervention, std::sync::Arc::new(Estimates))
        .unwrap();
    session.begin_step(CapturePhase::Prefill, 0).unwrap();
    let mut backend = Backend::default();
    let input = Value {
        shape: vec![2, 2],
        data: vec![1.0; 4],
    };
    session
        .observe(&mut backend, "block.output", &input)
        .unwrap();
    assert!(session
        .intervene(&mut backend, "block.output", &input)
        .is_err());
    assert_eq!(
        backend.copies, 2,
        "after evidence must fail before an unreserved copy"
    );
    let step = session.take_step().unwrap();
    assert_eq!(step.step_usage.captures, 2);
    assert!(matches!(
        step.interventions[0].evidence[1].outcome,
        CaptureOutcome::Failed { .. }
    ));
}

#[test]
fn immutable_run_rejects_hot_replacement_and_undrained_steps() {
    let (capture, intervention) = plans(
        vec![operation(
            "zero",
            InterventionAction::Zero {
                dtype: InterventionDtype::Float32,
            },
        )],
        false,
    );
    let mut session = CaptureSession::new(capture);
    session
        .enable_interventions(intervention.clone(), std::sync::Arc::new(Estimates))
        .unwrap();
    assert!(session
        .enable_interventions(intervention, std::sync::Arc::new(Estimates))
        .is_err());
    session.begin_step(CapturePhase::Prefill, 0).unwrap();
    assert!(session.begin_step(CapturePhase::Decode, 1).is_err());
}
