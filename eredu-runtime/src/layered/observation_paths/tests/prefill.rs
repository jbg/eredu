use super::*;

#[test]
fn row_declarations_share_exact_payload_capacity_and_attachment_lifetime() {
    let mut owner = source();
    let before = owner.capacity_bytes().unwrap();
    let path = spare("readout.embedding", 113);
    let extra = size_of::<PrefillObservationDeclaration>() + path.capacity();
    Arc::get_mut(&mut owner.0).unwrap().prefill =
        vec![PrefillObservationDeclaration::causal_ordinary_text(
            path,
            1,
            PrefillReadoutStage::BeforeReadout,
        )]
        .into_boxed_slice();
    let retired = Arc::new(AtomicBool::new(false));
    let drops = Arc::new(AtomicUsize::new(0));
    Arc::get_mut(&mut owner.0).unwrap().retired = Some(PayloadRetired(retired.clone()));
    assert_eq!(owner.capacity_bytes(), Some(before + extra as u64));
    owner
        .try_attach(&SharedStorageAccountingId::default(), || {
            Ok::<Box<dyn Send + Sync>, Infallible>(Box::new(Charge {
                retired: retired.clone(),
                drops: drops.clone(),
                other: source(),
            }))
        })
        .unwrap();
    let alias = owner.clone();
    let path = owner
        .prefill_observation("readout.embedding")
        .unwrap()
        .path()
        .as_ptr();
    assert_eq!(
        path,
        alias
            .prefill_observation("readout.embedding")
            .unwrap()
            .path()
            .as_ptr()
    );
    drop(owner);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    let declaration = alias.prefill_observation("readout.embedding").unwrap();
    assert_eq!(declaration.sequence_axis(), 1);
    assert_eq!(
        declaration.readout_stage(),
        PrefillReadoutStage::BeforeReadout
    );
    drop(alias);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn duplicate_declarations_are_unknown_instead_of_selecting_first_role() {
    let mut owner = source();
    Arc::get_mut(&mut owner.0).unwrap().prefill = [
        PrefillReadoutStage::BeforeReadout,
        PrefillReadoutStage::VocabularyScores,
    ]
    .into_iter()
    .map(|stage| {
        PrefillObservationDeclaration::causal_ordinary_text("actual.hook".into(), 1, stage)
    })
    .collect::<Vec<_>>()
    .into_boxed_slice();
    assert!(owner.prefill_observation("actual.hook").is_none());
    assert!(owner.prefill_observation("missing").is_none());
}

#[test]
fn sparse_prefill_binding_requires_explicit_flattened_equation_and_preserves_readout() {
    use eredu_core::{capture::*, *};
    let point = ObservationPoint {
        path: "bank.units".into(),
        node_id: "bank".into(),
        meaning: "actual sparse rows".into(),
        value_type: ObservationValueType::RoutedUnits {
            routing: "bank".into(),
            geometry: RoutedUnitGeometry {
                experts: 3,
                units_per_expert: 4,
                routes_per_token: 2,
            },
        },
        dtype: ObservationDtype::Floating,
        axes: Some(vec![
            TensorAxis {
                name: "token".into(),
                dimension: SymbolicDimension::TokenRows,
            },
            TensorAxis {
                name: "route".into(),
                dimension: SymbolicDimension::Known(2),
            },
            TensorAxis {
                name: "unit".into(),
                dimension: SymbolicDimension::Known(4),
            },
        ]),
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
    let mut plan = CapturePlan::none();
    plan.selections = vec![CaptureSelection {
        id: "sparse".into(),
        path: "bank.units".into(),
        schedule: CaptureSchedule::default(),
        slices: vec![],
        transform: CaptureTransform::RoutedUnits,
    }];
    let usage = CaptureUsage {
        captures: u64::MAX,
        retained_bytes: u64::MAX,
        host_bytes: u64::MAX,
        encoded_bytes: u64::MAX,
    };
    plan.limits = CaptureLimits {
        per_step: usage,
        cumulative: usage,
        on_limit: CaptureLimitPolicy::Fail,
    };
    let admitted = SharedCapturePlan::new(
        plan.admit_with_text_origin(
            &catalog,
            &support,
            &CaptureCapabilities {
                transformations: vec![CaptureTransformKind::RoutedUnits],
                ..Default::default()
            },
            CaptureRequestShape {
                batch: 2,
                prompt_tokens: 5,
                max_predictions: 3,
            },
            CaptureTextOrigin {
                cached_positions: 7,
            },
        )
        .unwrap(),
    );
    let mut paths = source();
    Arc::get_mut(&mut paths.0).unwrap().prefill =
        vec![PrefillObservationDeclaration::causal_ordinary_text(
            "bank.units".into(),
            0,
            PrefillReadoutStage::BeforeReadout,
        )]
        .into_boxed_slice();
    assert!(matches!(
        paths.prepare_capture_selection(&admitted),
        Err(PreparedCaptureSelectionError::Axes { index: 0 })
    ));
    Arc::get_mut(&mut paths.0).unwrap().prefill =
        vec![PrefillObservationDeclaration::causal_routed_units(
            "bank.units".into(),
        )]
        .into_boxed_slice();
    let selected = paths.prepare_capture_selection(&admitted).unwrap();
    let geometry = InferenceGeometry {
        batch_size: 2,
        cached_positions: 7,
        input_positions: 5,
        max_output_tokens: 3,
        prefill_chunk_positions: 2,
        output: OutputDemand::LastPosition,
    };
    assert_eq!(
        selected.physical_output(OutputDemand::StateOnly),
        OutputDemand::StateOnly
    );
    assert_eq!(
        selected.bind_geometry(geometry).unwrap().geometry(),
        geometry
    );
    let mapping = CaptureRoutedPrefillPlan::prepare(admitted.admission(), 0, geometry).unwrap();
    let tail = mapping.fragment(2).unwrap();
    assert_eq!(tail.source_tokens(), 2);
    assert_eq!(tail.logical_token(0).unwrap(), 4);
    assert_eq!(tail.logical_token(1).unwrap(), 9);

    // The completed media decoder uses the same sparse causal equation. Its
    // declaration remains distinct from the ordinary text declaration owner.
    drop(selected);
    assert!(matches!(
        paths.prepare_media_capture_selection(&admitted),
        Err(PreparedCaptureSelectionError::Undeclared { index: 0 })
    ));
    Arc::get_mut(&mut paths.0).unwrap().media_prefill =
        vec![PrefillObservationDeclaration::prepared_media_decoder(
            "bank.units".into(),
            0,
            PrefillReadoutStage::BeforeReadout,
        )]
        .into_boxed_slice();
    assert!(matches!(
        paths.prepare_media_capture_selection(&admitted),
        Err(PreparedCaptureSelectionError::Axes { index: 0 })
    ));
    Arc::get_mut(&mut paths.0).unwrap().media_prefill =
        vec![PrefillObservationDeclaration::causal_routed_units(
            "bank.units".into(),
        )]
        .into_boxed_slice();
    let media = paths.prepare_media_capture_selection(&admitted).unwrap();
    assert!(media.is_prepared_media());
    assert_eq!(
        media.physical_output(OutputDemand::StateOnly),
        OutputDemand::StateOnly
    );
    assert_eq!(media.bind_geometry(geometry).unwrap().geometry(), geometry);
    let media_mapping =
        CaptureRoutedPrefillPlan::prepare(media.source().admission(), 0, geometry).unwrap();
    for chunk in 0..3 {
        let ordinary = mapping.fragment(chunk).unwrap();
        let decoded = media_mapping.fragment(chunk).unwrap();
        assert_eq!(ordinary.source_tokens(), decoded.source_tokens());
        for local in 0..ordinary.source_tokens() {
            assert_eq!(
                ordinary.logical_token(local).unwrap(),
                decoded.logical_token(local).unwrap()
            );
        }
    }
}
