use super::*;
use crate::*;

fn fixture(dimensions: Vec<SymbolicDimension>) -> (CapturePlan, CaptureDiscovery) {
    let point = ObservationPoint {
        path: "attention.scores".into(),
        node_id: "attention".into(),
        meaning: "actual context geometry".into(),
        value_type: ObservationValueType::Tensor,
        dtype: ObservationDtype::Floating,
        axes: Some(
            dimensions
                .into_iter()
                .enumerate()
                .map(|(i, dimension)| TensorAxis {
                    name: format!("axis{i}"),
                    dimension,
                })
                .collect(),
        ),
        prefill: true,
        decode: true,
        requirements: vec![ObservationRequirement::ActivationHooks],
        position: ObservationPosition::BeforeIntervention,
        retained_bytes: None,
        host_bytes: None,
    };
    let caps = CaptureCapabilities {
        transformations: vec![
            CaptureTransformKind::FullTensor,
            CaptureTransformKind::Slice,
        ],
        ..Default::default()
    };
    let discovery = CaptureDiscovery {
        artifact_identity: "origin-fixture".into(),
        catalog: ObservationCatalog {
            schema_version: 1,
            points: vec![point],
            completeness: DescriptionCompleteness::Complete,
        },
        support: ObservationSupportReport {
            schema_version: 1,
            capture: caps,
            points: vec![ObservationSupport {
                path: "attention.scores".into(),
                prefill: ObservationSupportStatus::Supported,
                decode: ObservationSupportStatus::Supported,
                floating_to_f32: true,
            }],
        },
    };
    let usage = CaptureUsage {
        captures: 64,
        retained_bytes: u64::MAX,
        host_bytes: u64::MAX,
        encoded_bytes: u64::MAX,
    };
    let plan = CapturePlan {
        schema_version: 1,
        selections: vec![CaptureSelection {
            id: "scores".into(),
            path: "attention.scores".into(),
            schedule: CaptureSchedule::default(),
            slices: vec![],
            transform: CaptureTransform::FullTensor,
        }],
        limits: CaptureLimits {
            per_step: usage,
            cumulative: usage,
            physical_native_bytes: None,
            on_limit: CaptureLimitPolicy::Fail,
        },
    };
    (plan, discovery)
}
fn request(max_predictions: u64) -> CaptureRequestShape {
    CaptureRequestShape {
        batch: 1,
        prompt_tokens: 3,
        max_predictions,
    }
}
fn admit(
    plan: CapturePlan,
    d: &CaptureDiscovery,
    request: CaptureRequestShape,
    cached: u64,
) -> Result<AdmittedCapturePlan, CaptureError> {
    plan.admit_with_text_origin(
        &d.catalog,
        &d.support,
        &d.support.capture,
        request,
        CaptureTextOrigin {
            cached_positions: cached,
        },
    )
}

#[test]
fn nonzero_prefix_preserves_new_prompt_axes_and_local_prediction_coordinates() {
    let (plan, d) = fixture(vec![
        SymbolicDimension::Batch,
        SymbolicDimension::Sequence,
        SymbolicDimension::TokenRows,
        SymbolicDimension::Context,
        SymbolicDimension::Known(4),
    ]);
    let admitted = admit(
        plan,
        &d,
        CaptureRequestShape {
            batch: 2,
            ..request(4)
        },
        7,
    )
    .unwrap();
    assert_eq!(admitted.request().prompt_tokens, 3);
    assert_eq!(
        admitted.text_origin(),
        Some(CaptureTextOrigin {
            cached_positions: 7
        })
    );
    for (phase, prediction, shape) in [
        (CapturePhase::Prefill, 0, vec![2, 3, 6, 10, 4]),
        (CapturePhase::Decode, 1, vec![2, 1, 2, 11, 4]),
        (CapturePhase::Decode, 3, vec![2, 1, 2, 13, 4]),
    ] {
        let geometry = admitted.geometry_at(phase, prediction, None).unwrap();
        assert_eq!(
            geometry.resolve(&d.catalog.points[0]).unwrap(),
            Some(shape.clone())
        );
        geometry
            .validate_actual(&d.catalog.points[0], &shape)
            .unwrap();
        assert_eq!(
            admitted
                .estimate_shape(&d.catalog.points[0], phase, prediction)
                .unwrap(),
            Some(shape.clone())
        );
        let tensor = CaptureTensorGeometry::prepare(&admitted, 0, phase, prediction, None).unwrap();
        assert_eq!(
            tensor.source_shape(),
            shape.iter().map(|v| *v as usize).collect::<Vec<_>>()
        );
    }
    assert!(admitted
        .geometry_at(CapturePhase::Decode, 1, None)
        .unwrap()
        .validate_actual(&d.catalog.points[0], &[2, 1, 2, 4, 4])
        .is_err());
    // Raw request access intentionally preserves its legacy zero-origin meaning.
    assert_eq!(
        admitted
            .request()
            .invocation_shape(CapturePhase::Decode, 1)
            .unwrap()
            .context,
        Some(4)
    );
    admitted.request().validate_prefill(2, 3).unwrap();
    assert!(admitted.request().validate_prefill(2, 10).is_err());
}

#[test]
fn context_slice_admission_checks_first_and_last_scheduled_decode_with_origin() {
    let (mut plan, d) = fixture(vec![SymbolicDimension::Context]);
    plan.selections[0].transform = CaptureTransform::Slice;
    plan.selections[0].schedule = CaptureSchedule {
        prefill: false,
        decode: true,
        first_prediction: 1,
        ..Default::default()
    };
    plan.selections[0].slices = vec![CaptureSlice {
        axis: "axis0".into(),
        start: 9,
        end: 11,
        stride: 1,
    }];
    let admitted = admit(plan.clone(), &d, request(4), 7).unwrap();
    let geometry =
        CaptureTensorGeometry::prepare(&admitted, 0, CapturePhase::Decode, 1, None).unwrap();
    assert_eq!(geometry.source_shape(), &[11]);
    assert_eq!(geometry.shape(), &[2]);
    assert!(admit(plan.clone(), &d, request(4), 0).is_err());
    plan.selections[0].slices[0].end = 12;
    assert!(admit(plan.clone(), &d, request(4), 7).is_err()); // Last decode fits, first does not.
    plan.selections[0].schedule.first_prediction = 2;
    let later = admit(plan, &d, request(4), 7).unwrap();
    assert_eq!(
        CaptureTensorGeometry::prepare(&later, 0, CapturePhase::Decode, 2, None)
            .unwrap()
            .source_shape(),
        &[12]
    );
}

#[test]
fn one_prediction_has_only_prefill_and_no_phantom_origin_overflow() {
    let (plan, d) = fixture(vec![SymbolicDimension::Context]);
    let admitted = admit(plan, &d, request(1), 7).unwrap();
    let schedule = &admitted.plan().selections[0].schedule;
    assert_eq!(
        schedule.count_and_last(CapturePhase::Prefill, 1).unwrap(),
        Some((1, 0))
    );
    assert_eq!(
        schedule.count_and_last(CapturePhase::Decode, 1).unwrap(),
        None
    );
    assert_eq!(
        CaptureTensorGeometry::prepare(&admitted, 0, CapturePhase::Prefill, 0, None)
            .unwrap()
            .shape(),
        &[10]
    );
    assert!(matches!(
        CaptureTensorGeometry::prepare(&admitted, 0, CapturePhase::Decode, 1, None),
        Err(CaptureTensorGeometryError::Inactive)
    ));
    let empty = admit(CapturePlan::none(), &d, request(1), u64::MAX - 3).unwrap();
    assert_eq!(
        empty
            .geometry_at(CapturePhase::Prefill, 0, None)
            .unwrap()
            .context,
        Some(u64::MAX)
    );
    assert!(matches!(
        empty.geometry_at(CapturePhase::Decode, 1, None),
        Err(CaptureError::Overflow)
    ));
}

#[test]
fn checked_origin_span_rejects_overflow_even_without_selected_points() {
    let (_, d) = fixture(vec![SymbolicDimension::Context]);
    for (cached, max_predictions) in [(u64::MAX, 1), (u64::MAX - 3, 2), (u64::MAX - 4, 3)] {
        assert!(matches!(
            admit(CapturePlan::none(), &d, request(max_predictions), cached),
            Err(CaptureError::Overflow)
        ));
    }
    let origin = CaptureTextOrigin {
        cached_positions: u64::MAX,
    };
    assert!(matches!(
        origin.invocation_shape(request(1), CapturePhase::Prefill, 0),
        Err(CaptureError::Overflow)
    ));
    let zero = CaptureTextOrigin::default();
    assert!(matches!(
        zero.invocation_shape(request(2), CapturePhase::Decode, u64::MAX),
        Err(CaptureError::Overflow)
    ));
}

fn digest(bytes: Vec<u8>) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
#[test]
fn zero_origin_keeps_legacy_identity_and_changed_origin_survives_readmission() {
    let (plan, d) = fixture(vec![
        SymbolicDimension::Sequence,
        SymbolicDimension::Context,
    ]);
    let req = request(4);
    let legacy = plan
        .clone()
        .admit(&d.catalog, &d.support, &d.support.capture, req)
        .unwrap();
    let zero = admit(plan.clone(), &d, req, 0).unwrap();
    let legacy_identity = digest(serde_json::to_vec(&(&plan, legacy.points(), req)).unwrap());
    assert_eq!(legacy.identity(), legacy_identity);
    assert_eq!(legacy.identity(), zero.identity());
    let prefix = admit(plan.clone(), &d, req, 7).unwrap();
    let changed = admit(plan, &d, req, 8).unwrap();
    assert_ne!(prefix.identity(), zero.identity());
    assert_ne!(prefix.identity(), changed.identity());
    let readmitted = prefix.readmit(&d).unwrap();
    assert_eq!(readmitted.identity(), prefix.identity());
    assert_eq!(readmitted.text_origin(), prefix.text_origin());
    assert_eq!(
        readmitted
            .geometry_at(CapturePhase::Decode, 1, None)
            .unwrap()
            .context,
        Some(11)
    );
    let mut changed_discovery = d.clone();
    changed_discovery.catalog.points[0].node_id = "other-node".into();
    assert_ne!(
        prefix.readmit(&changed_discovery).unwrap().identity(),
        prefix.identity()
    );
    let value = CaptureTextOrigin {
        cached_positions: 7,
    };
    assert_eq!(
        serde_json::from_value::<CaptureTextOrigin>(serde_json::to_value(value).unwrap()).unwrap(),
        value
    );
}

#[test]
fn independent_invocation_authority_and_identity_are_unchanged() {
    let (plan, d) = fixture(vec![
        SymbolicDimension::Sequence,
        SymbolicDimension::Context,
    ]);
    let bounds = CaptureInvocationBounds {
        batch: 1,
        max_sequence: 3,
        max_context: Some(20),
        max_predictions: 4,
    };
    let invocation = plan
        .clone()
        .admit_invocations(&d.catalog, &d.support, &d.support.capture, bounds)
        .unwrap();
    assert_eq!(invocation.text_origin(), None);
    assert_eq!(
        invocation.identity(),
        digest(serde_json::to_vec(&("invocation", &plan, invocation.points(), bounds)).unwrap())
    );
    let actual = CaptureInvocationShape {
        batch: 1,
        sequence: 2,
        context: Some(17),
    };
    assert_eq!(
        invocation
            .geometry_at(CapturePhase::Decode, 1, Some(actual))
            .unwrap(),
        actual
    );
    assert!(invocation
        .geometry_at(CapturePhase::Decode, 1, None)
        .is_err());
    assert_eq!(
        invocation.readmit(&d).unwrap().identity(),
        invocation.identity()
    );
    assert_eq!(
        invocation
            .estimate_shape(&d.catalog.points[0], CapturePhase::Decode, 1)
            .unwrap(),
        Some(vec![3, 20])
    );
    let ordinary = admit(plan, &d, request(4), 7).unwrap();
    assert!(ordinary
        .geometry_at(CapturePhase::Decode, 1, Some(actual))
        .is_err());
}
