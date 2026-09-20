use super::*;
use eredu_core::*;

mod invocation;
mod session;
mod sparse;
mod text_origin;
mod context_prefix;

#[test]
fn global_component_masks_preserve_positions_and_empty_keep_semantics() {
    use eredu_core::component::ComponentCoordinateMap;
    let input = Value {
        shape: vec![1, 3, 8],
        data: (1..=24).map(|value| value as f32).collect(),
    };
    let slice = ResolvedCaptureSlice {
        starts: vec![0, 1, 0],
        ends: vec![1, 3, 8],
        strides: vec![1, 1, 1],
        shape: vec![1, 2, 8],
    };
    for selected in [vec![], vec![6], vec![7, 2, 0]] {
        for keep_selected in [false, true] {
            let action = InterventionAction::MaskComponents {
                dtype: InterventionDtype::Float32,
                indices: selected.clone(),
                keep_selected,
            };
            let expected =
                apply_activation(&mut Backend::default(), &input, &action, &slice).unwrap();
            for map in [
                ComponentCoordinateMap::range(8, 0..4).unwrap(),
                ComponentCoordinateMap::range(8, 4..8).unwrap(),
                ComponentCoordinateMap::indices(8, vec![7, 2, 0]).unwrap(),
            ] {
                let mut local = Value {
                    shape: vec![1, 3, map.local_count() as u64],
                    data: Vec::new(),
                };
                let mut expected_local = Vec::new();
                for row in 0..3 {
                    for index in 0..map.local_count() {
                        let global = row * 8 + map.local_to_global(index).unwrap();
                        local.data.push(input.data[global]);
                        expected_local.push(expected.data[global]);
                    }
                }
                let (action, local_slice) = localize_component_mask(&action, &slice, &map).unwrap();
                assert_eq!(local_slice.starts[1], 1);
                let actual =
                    apply_activation(&mut Backend::default(), &local, &action, &local_slice)
                        .unwrap();
                assert_eq!(actual.data, expected_local);
            }
        }
    }
}

#[test]
fn global_component_mask_rejects_invalid_remote_indices_and_geometry() {
    let map = eredu_core::component::ComponentCoordinateMap::range(8, 0..4).unwrap();
    let slice = ResolvedCaptureSlice {
        starts: vec![0, 1, 0],
        ends: vec![1, 2, 8],
        strides: vec![1, 1, 1],
        shape: vec![1, 1, 8],
    };
    for indices in [vec![8], vec![7, 7]] {
        let action = InterventionAction::MaskComponents {
            dtype: InterventionDtype::Float32,
            indices,
            keep_selected: false,
        };
        assert!(localize_component_mask(&action, &slice, &map).is_err());
    }
    let action = InterventionAction::MaskComponents {
        dtype: InterventionDtype::Float32,
        indices: vec![7],
        keep_selected: true,
    };
    let mut invalid = slice.clone();
    invalid.strides[1] = 0;
    assert!(localize_component_mask(&action, &invalid, &map).is_err());
    invalid = slice.clone();
    invalid.ends[2] = 4;
    invalid.shape[2] = 4;
    assert!(localize_component_mask(&action, &invalid, &map).is_err());
    assert!(localize_component_mask(
        &InterventionAction::Zero {
            dtype: InterventionDtype::Float32
        },
        &slice,
        &map
    )
    .is_err());
}

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
    dtype: Option<InterventionDtype>,
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
    fn source_dtype(&self, _: &Value) -> Option<eredu_core::checkpoint::TensorDtype> {
        Some(eredu_core::checkpoint::TensorDtype::F32)
    }
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
        if selection.transform == CaptureTransform::Summary {
            let finite: Vec<_> = data
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .map(f64::from)
                .collect();
            let count = finite.len() as f64;
            return Ok(CapturePayload::Summary(CaptureSummary {
                elements: data.len() as u64,
                finite: finite.len() as u64,
                non_finite: (data.len() - finite.len()) as u64,
                nan: data.iter().filter(|v| v.is_nan()).count() as u64,
                positive_infinity: data.iter().filter(|v| **v == f32::INFINITY).count() as u64,
                negative_infinity: data.iter().filter(|v| **v == f32::NEG_INFINITY).count() as u64,
                min: finite.iter().copied().reduce(f64::min),
                max: finite.iter().copied().reduce(f64::max),
                mean: (!finite.is_empty()).then(|| finite.iter().sum::<f64>() / count),
                rms: (!finite.is_empty())
                    .then(|| (finite.iter().map(|v| v * v).sum::<f64>() / count).sqrt()),
            }));
        }
        Ok(CapturePayload::Tensor(
            TensorObservation::new(vec![data.len()], TensorObservationData::F32(data)).unwrap(),
        ))
    }
}

impl InterventionBackend for Backend {
    fn routed_unit_locations(
        &mut self,
        source: &RoutedUnitCaptureSource<'_, Value>,
        geometry: RoutedUnitGeometry,
    ) -> Option<Result<RoutedUnitLocations, std::io::Error>> {
        self.copies += 1;
        Some(Ok(RoutedUnitLocations {
            source_token_range: [
                source.token_offset,
                source.token_offset + source.coefficients.shape[0],
            ],
            rows: source
                .token_indices
                .data
                .iter()
                .zip(&source.selection_indices.data)
                .map(|(token, slot)| {
                    let token = *token as u64 + source.token_offset;
                    let slot = *slot as u64 % geometry.routes_per_token;
                    RoutedUnitLocation {
                        source_peer: None,
                        token,
                        slot,
                        expert: source.source_groups.data
                            [(token * geometry.routes_per_token + slot) as usize]
                            as u64,
                    }
                })
                .collect(),
        }))
    }
    fn select_elements(
        &mut self,
        source: &Value,
        indices: &[u64],
    ) -> Option<Result<Value, std::io::Error>> {
        Some(Ok(Value {
            shape: vec![indices.len() as u64],
            data: indices.iter().map(|i| source.data[*i as usize]).collect(),
        }))
    }
    fn update_elements(
        &mut self,
        source: &Value,
        indices: &[u64],
        replacement: &Value,
    ) -> Option<Result<Value, std::io::Error>> {
        let mut result = source.clone();
        for (index, value) in indices.iter().zip(&replacement.data) {
            result.data[*index as usize] = *value;
        }
        Some(Ok(result))
    }
    fn mask_components(
        &mut self,
        value: &Value,
        ids: &[u32],
        keep_selected: bool,
    ) -> std::result::Result<Value, std::io::Error> {
        let width = *value.shape.last().unwrap() as usize;
        let mut result = value.clone();
        for (index, value) in result.data.iter_mut().enumerate() {
            if ids.contains(&((index % width) as u32)) != keep_selected {
                *value = 0.0;
            }
        }
        Ok(result)
    }

    fn intervention_dtype(&self, _: &Value) -> Result<InterventionDtype, Self::Error> {
        Ok(self.dtype.unwrap_or(InterventionDtype::Float32))
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
    fn routed_unit_usage(
        &self,
        _: RoutedUnitGeometry,
        _: &[u64],
        _: &ResolvedCaptureSlice,
        _: &InterventionAction,
    ) -> Result<CaptureUsage, CaptureError> {
        Ok(CaptureUsage {
            retained_bytes: 4096,
            host_bytes: 4096,
            ..Default::default()
        })
    }
    fn activation_usage(
        &self,
        source: &[u64],
        slice: &ResolvedCaptureSlice,
        _: &InterventionAction,
    ) -> Result<CaptureUsage, CaptureError> {
        let elements = |shape: &[u64]| {
            shape
                .iter()
                .try_fold(1u64, |n, d| n.checked_mul(*d).ok_or(CaptureError::Overflow))
        };
        let bytes = elements(source)?
            .checked_add(
                elements(&slice.shape)?
                    .checked_mul(3)
                    .ok_or(CaptureError::Overflow)?,
            )
            .and_then(|n| n.checked_mul(8))
            .ok_or(CaptureError::Overflow)?;
        Ok(CaptureUsage {
            captures: 0,
            retained_bytes: bytes,
            host_bytes: bytes,
            encoded_bytes: 0,
        })
    }

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
        routed_units: None,
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
    let mut session = CaptureSession::new(eredu_core::capture::SharedCapturePlan::new(capture));
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
    let mut session = CaptureSession::new(eredu_core::capture::SharedCapturePlan::new(capture));
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
    let mut session = CaptureSession::new(eredu_core::capture::SharedCapturePlan::new(capture));
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
fn conditional_intervention_requires_actual_application_before_commitment() {
    let (capture, original) = plans(
        vec![operation(
            "scale",
            InterventionAction::Scale {
                dtype: InterventionDtype::Float32,
                factor: 2.0,
            },
        )],
        false,
    );
    let mut discovery = session::discovery(&original);
    discovery.points[0].prefill =
        ObservationSupportStatus::Conditional("requires media input".into());
    for present in [false, true] {
        let intervention = original
            .plan()
            .clone()
            .admit(&discovery, original.request(), original.session_id())
            .unwrap();
        let mut session = CaptureSession::new(eredu_core::capture::SharedCapturePlan::new(capture.clone()));
        session
            .enable_interventions(intervention, std::sync::Arc::new(Estimates))
            .unwrap();
        session.begin_step(CapturePhase::Prefill, 0).unwrap();
        let mut backend = Backend::default();
        if present {
            let input = Value {
                shape: vec![2, 2],
                data: vec![1.0, -2.0, 3.0, 4.0],
            };
            let output = session
                .intervene(&mut backend, "block.output", &input)
                .unwrap()
                .unwrap();
            assert_eq!(output.data, vec![2.0, -4.0, 6.0, 8.0]);
            session.finish_interventions().unwrap();
        } else {
            assert!(matches!(
                session.finish_interventions(),
                Err(CaptureError::Invalid(_))
            ));
        }
        let step = session.take_step().unwrap();
        assert_eq!(backend.applications, usize::from(present));
        assert_eq!(
            step.interventions[0].outcome,
            if present {
                InterventionOutcome::Applied
            } else {
                InterventionOutcome::Missing
            }
        );
    }
    for status in [
        ObservationSupportStatus::Unsupported("no collector".into()),
        ObservationSupportStatus::Unverified("no collector facts".into()),
    ] {
        discovery.points[0].prefill = status;
        assert!(matches!(
            original
                .plan()
                .clone()
                .admit(&discovery, original.request(), original.session_id()),
            Err(CaptureError::Intervention(eredu_core::intervention::InterventionDeclarationError::Unsupported(_)))
        ));
    }
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
    let mut session = CaptureSession::new(eredu_core::capture::SharedCapturePlan::new(capture));
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
    let mut session = CaptureSession::new(eredu_core::capture::SharedCapturePlan::new(limited));
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
    let mut session = CaptureSession::new(eredu_core::capture::SharedCapturePlan::new(capture));
    session
        .enable_interventions(intervention.clone(), std::sync::Arc::new(Estimates))
        .unwrap();
    assert!(session
        .enable_interventions(intervention, std::sync::Arc::new(Estimates))
        .is_err());
    session.begin_step(CapturePhase::Prefill, 0).unwrap();
    assert!(session.begin_step(CapturePhase::Decode, 1).is_err());
}

#[test]
fn compact_component_masks_select_arbitrary_columns_only_in_explicit_rows() {
    let source = Value {
        shape: vec![1, 3, 4],
        data: (1..=12).map(|n| n as f32).collect(),
    };
    let slice = ResolvedCaptureSlice {
        starts: vec![0, 1, 0],
        ends: vec![1, 2, 4],
        strides: vec![1, 1, 1],
        shape: vec![1, 1, 4],
    };
    for keep_selected in [true, false] {
        let mut backend = Backend::default();
        let out = apply_activation(
            &mut backend,
            &source,
            &InterventionAction::MaskComponents {
                dtype: InterventionDtype::Float32,
                indices: vec![0, 2],
                keep_selected,
            },
            &slice,
        )
        .unwrap();
        assert_eq!(&out.data[..4], &source.data[..4]);
        assert_eq!(&out.data[8..], &source.data[8..]);
        assert_eq!(
            &out.data[4..8],
            if keep_selected {
                &[5.0, 0.0, 7.0, 0.0]
            } else {
                &[0.0, 6.0, 0.0, 8.0]
            }
        );
    }
    for indices in [vec![4], vec![1, 1]] {
        let mut backend = Backend::default();
        assert!(apply_activation(
            &mut backend,
            &source,
            &InterventionAction::MaskComponents {
                dtype: InterventionDtype::Float32,
                indices,
                keep_selected: true,
            },
            &slice
        )
        .is_err());
        assert_eq!(backend.applications, 0);
        assert_eq!(backend.copies, 0);
    }
}

#[test]
fn activation_storage_exhaustion_fails_before_native_edit_or_evidence_copy() {
    let mut op = operation(
        "zero",
        InterventionAction::Zero {
            dtype: InterventionDtype::Float32,
        },
    );
    op.evidence = InterventionEvidence::None;
    let (capture, intervention) = plans(vec![op], false);
    let mut plan = capture.plan().clone();
    plan.limits.per_step.retained_bytes = 1;
    let catalog = ObservationCatalog {
        schema_version: 1,
        points: vec![],
        completeness: DescriptionCompleteness::Complete,
    };
    let caps = CaptureCapabilities::default();
    let support = ObservationSupportReport {
        schema_version: 1,
        capture: caps.clone(),
        points: vec![],
    };
    let limited = plan
        .admit(&catalog, &support, &caps, capture.request())
        .unwrap();
    assert!(matches!(
        preflight(&limited, &intervention, &Estimates),
        Err(CaptureError::Limit {
            budget: CaptureBudget::Retention,
            ..
        })
    ));
    let mut session = CaptureSession::new(eredu_core::capture::SharedCapturePlan::new(limited));
    session
        .enable_interventions(intervention, std::sync::Arc::new(Estimates))
        .unwrap();
    session.begin_step(CapturePhase::Prefill, 0).unwrap();
    let mut backend = Backend::default();
    let result = session.intervene(
        &mut backend,
        "block.output",
        &Value {
            shape: vec![2, 2],
            data: vec![1.0; 4],
        },
    );
    assert!(result.is_err());
    assert_eq!(backend.applications, 0);
    assert_eq!(backend.copies, 0);
    let step = session.take_step().unwrap();
    assert!(matches!(
        step.interventions[0].outcome,
        InterventionOutcome::Failed { .. }
    ));
}
mod partition;

mod static_preflight;
