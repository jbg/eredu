use super::*;
use eredu_core::*;
fn admitted(
    axes: Vec<SymbolicDimension>,
    transform: CaptureTransform,
    slices: Vec<CaptureSlice>,
) -> AdmittedCapturePlan {
    let point = ObservationPoint {
        path: "block.output".into(),
        node_id: "block".into(),
        meaning: "actual activation".into(),
        value_type: ObservationValueType::Tensor,
        dtype: ObservationDtype::Floating,
        axes: Some(
            axes.into_iter()
                .enumerate()
                .map(|(index, dimension)| TensorAxis {
                    name: format!("axis{index}"),
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
    let support = ObservationSupportReport {
        schema_version: 1,
        capture: Default::default(),
        points: vec![ObservationSupport {
            path: point.path.clone(),
            prefill: ObservationSupportStatus::Supported,
            decode: ObservationSupportStatus::Supported,
            floating_to_f32: true,
        }],
    };
    let catalog = ObservationCatalog {
        schema_version: 1,
        points: vec![point],
        completeness: DescriptionCompleteness::Complete,
    };
    let usage = CaptureUsage {
        captures: 10,
        retained_bytes: u64::MAX,
        host_bytes: u64::MAX,
        encoded_bytes: u64::MAX,
    };
    CapturePlan {
        schema_version: 1,
        selections: vec![CaptureSelection {
            id: "tensor".into(),
            path: "block.output".into(),
            schedule: CaptureSchedule {
                prefill: false,
                ..Default::default()
            },
            slices,
            transform,
        }],
        limits: CaptureLimits {
            per_step: usage,
            cumulative: usage,
            on_limit: CaptureLimitPolicy::Fail,
        },
    }
    .admit(
        &catalog,
        &support,
        &CaptureCapabilities {
            transformations: vec![
                CaptureTransformKind::Preview,
                CaptureTransformKind::Slice,
                CaptureTransformKind::FullTensor,
                CaptureTransformKind::Summary,
            ],
            max_histogram_bins: 0,
            conditions: vec![],
        },
        CaptureRequestShape {
            batch: 1,
            prompt_tokens: 3,
            max_predictions: 4,
        },
    )
    .unwrap()
}

#[test]
fn semantic_step_preserves_schedules_wrong_coordinates_and_borrowed_geometry() {
    let source = admitted(
        vec![SymbolicDimension::Context, SymbolicDimension::Known(3)],
        CaptureTransform::Preview { max_elements: 4 },
        vec![],
    );
    let before = &source as *const _;
    let prefill = CaptureObservationStep::new(&source, CapturePhase::Prefill, 0).unwrap();
    assert!(!prefill.requires_sequence_readout());
    assert_eq!(
        prefill.initial_status(0).unwrap(),
        CaptureRecordStatus::Skipped
    );
    let decode = CaptureObservationStep::new(&source, CapturePhase::Decode, 2).unwrap();
    assert!(!decode.requires_sequence_readout());
    let geometry = decode.tensor_geometry(0).unwrap();
    assert_eq!(geometry.admission() as *const _, before);
    assert_eq!(geometry.source_shape(), [5, 3]);
    assert_eq!(geometry.shape(), [4]);
    assert!(decode
        .select(0, CaptureRecordStatus::Missing, "block.output")
        .unwrap());
    assert!(!decode
        .select(0, CaptureRecordStatus::Consumed, "unrelated")
        .unwrap());
    assert!(!decode
        .select(0, CaptureRecordStatus::Skipped, "block.output")
        .unwrap());
    assert!(matches!(
        decode.select(0, CaptureRecordStatus::Consumed, "block.output"),
        Err(CaptureProtocolError::Duplicate)
    ));
    assert!(matches!(
        CaptureObservationStep::new(&source, CapturePhase::Decode, 0),
        Err(CaptureProtocolError::Geometry)
    ));
    assert!(matches!(
        CaptureObservationStep::new(&source, CapturePhase::Prefill, 1),
        Err(CaptureProtocolError::Geometry)
    ));
    assert!(matches!(
        CaptureObservationStep::new(&source, CapturePhase::Decode, 4),
        Err(CaptureProtocolError::Prediction)
    ));
}

#[test]
fn metadata_envelopes_and_delivery_controls_cover_skipped_and_selected_steps() {
    let source = admitted(
        vec![SymbolicDimension::Known(3)],
        CaptureTransform::FullTensor,
        vec![],
    );
    let mut real = CaptureSession::new(eredu_core::capture::SharedCapturePlan::new(source.clone()));
    let mut diagnostic = CaptureLedger::new(&source);
    for p in 0..4 {
        let phase = if p == 0 {
            CapturePhase::Prefill
        } else {
            CapturePhase::Decode
        };
        real.begin_step(phase, p).unwrap();
        diagnostic.begin_step();
        let step = CaptureObservationStep::new(&source, phase, p).unwrap();
        step.reserve_metadata(&mut diagnostic).unwrap();
        assert_eq!(real.ledger.step(), diagnostic.step());
        assert_eq!(real.ledger.total(), diagnostic.total());
        assert!(diagnostic.step().host_bytes > 0);
        let records = real.take_step().unwrap();
        assert_eq!(
            CaptureRecordStatus::from_record(&records.records[0]),
            step.initial_status(0).unwrap()
        );
    }
}
