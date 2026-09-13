use super::*;

fn geometry() -> RoutedUnitGeometry {
    RoutedUnitGeometry {
        experts: 3,
        units_per_expert: 2,
        routes_per_token: 2,
    }
}
fn full() -> ResolvedCaptureSlice {
    ResolvedCaptureSlice {
        starts: vec![0, 0],
        ends: vec![2, 6],
        strides: vec![1, 1],
        shape: vec![2, 6],
    }
}
fn locations() -> RoutedUnitLocations {
    RoutedUnitLocations {
        source_token_range: [0, 2],
        rows: vec![
            RoutedUnitLocation {
                source_peer: None,
                token: 1,
                slot: 1,
                expert: 2,
            },
            RoutedUnitLocation {
                source_peer: None,
                token: 0,
                slot: 0,
                expert: 1,
            },
            RoutedUnitLocation {
                source_peer: None,
                token: 1,
                slot: 0,
                expert: 0,
            },
            RoutedUnitLocation {
                source_peer: None,
                token: 0,
                slot: 1,
                expert: 1,
            },
        ],
    }
}
fn mask(indices: Vec<u32>, keep_selected: bool) -> InterventionAction {
    InterventionAction::MaskComponents {
        dtype: InterventionDtype::Float32,
        indices,
        keep_selected,
    }
}

#[test]
fn sparse_lowering_uses_global_ids_current_routes_and_distinct_duplicate_slots() {
    let recipe =
        lower_routed_intervention(geometry(), &locations(), &full(), &mask(vec![3, 4], false))
            .unwrap();
    assert_eq!(recipe.indices, [0, 3, 7]);
    assert_eq!(
        recipe.action,
        Some(InterventionAction::Zero {
            dtype: InterventionDtype::Float32
        })
    );
    let mut changed = locations();
    changed.rows[1].expert = 0;
    let recipe =
        lower_routed_intervention(geometry(), &changed, &full(), &mask(vec![3, 4], false)).unwrap();
    assert_eq!(recipe.indices, [0, 7]);
    let recipe =
        lower_routed_intervention(geometry(), &locations(), &full(), &mask(vec![3, 4], true))
            .unwrap();
    assert_eq!(recipe.indices, [1, 2, 4, 5, 6]);
    for action in [mask((0..6).collect(), true), mask(vec![], false)] {
        let recipe = lower_routed_intervention(geometry(), &locations(), &full(), &action).unwrap();
        assert!(recipe.indices.is_empty());
        assert!(recipe.action.is_none());
    }
    assert_eq!(
        lower_routed_intervention(geometry(), &locations(), &full(), &mask(vec![], true))
            .unwrap()
            .indices,
        (0..8).collect::<Vec<_>>()
    );
}

#[test]
fn sparse_payloads_follow_selected_token_component_coordinates_and_preserve_bits() {
    let slice = ResolvedCaptureSlice {
        starts: vec![0, 1],
        ends: vec![2, 6],
        strides: vec![1, 2],
        shape: vec![2, 3],
    };
    for values in [
        InterventionValues::Float32(vec![10., 11., 12., 13., 14., 15.]),
        InterventionValues::Float16(vec![0x8000, 0x3c00, 0x4000, 0x4200, 0x4400, 0x4500]),
        InterventionValues::Bfloat16(vec![0x8000, 0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0]),
    ] {
        for add in [false, true] {
            let tensor = InterventionTensor {
                shape: slice.shape.clone(),
                values: values.clone(),
            };
            let action = if add {
                InterventionAction::Add { tensor }
            } else {
                InterventionAction::Replace { tensor }
            };
            let recipe =
                lower_routed_intervention(geometry(), &locations(), &slice, &action).unwrap();
            assert_eq!(recipe.indices, [1, 3, 5, 7]);
            let tensor = match recipe.action.unwrap() {
                InterventionAction::Add { tensor } | InterventionAction::Replace { tensor } => {
                    tensor
                }
                _ => panic!(),
            };
            assert_eq!(tensor.shape, [4]);
            match (tensor.values, &values) {
                (InterventionValues::Float32(a), InterventionValues::Float32(v)) => {
                    assert_eq!(a, [v[5], v[1], v[3], v[1]])
                }
                (InterventionValues::Float16(a), InterventionValues::Float16(v))
                | (InterventionValues::Bfloat16(a), InterventionValues::Bfloat16(v)) => {
                    assert_eq!(a, [v[5], v[1], v[3], v[1]])
                }
                _ => panic!(),
            }
        }
    }
    let action = InterventionAction::Mask {
        dtype: InterventionDtype::Float32,
        shape: vec![2, 6],
        keep: (0..12).map(|i| i != 3 && i != 10).collect(),
    };
    assert_eq!(
        lower_routed_intervention(geometry(), &locations(), &full(), &action)
            .unwrap()
            .indices,
        [0, 3, 7]
    );
    let slice = ResolvedCaptureSlice {
        starts: vec![1, 0],
        ends: vec![2, 6],
        strides: vec![1, 1],
        shape: vec![1, 6],
    };
    let action = InterventionAction::Scale {
        dtype: InterventionDtype::Float32,
        factor: -2.0,
    };
    let recipe = lower_routed_intervention(geometry(), &locations(), &slice, &action).unwrap();
    assert_eq!(recipe.indices, [0, 1, 4, 5]);
    assert_eq!(recipe.action, Some(action));
}

#[test]
fn sparse_lowering_rejects_incomplete_duplicate_out_of_range_and_invalid_payloads() {
    let action = mask(vec![3], false);
    for change in 0..6 {
        let mut rows = locations();
        match change {
            0 => {
                rows.rows.pop();
            }
            1 => rows.rows[0] = rows.rows[1],
            2 => rows.rows[0].expert = 3,
            3 => rows.rows[0].token = 2,
            4 => rows.rows[0].slot = 2,
            _ => rows.source_token_range = [2, 1],
        }
        assert!(lower_routed_intervention(geometry(), &rows, &full(), &action).is_err());
    }
    for action in [
        mask(vec![6], false),
        mask(vec![1, 1], true),
        InterventionAction::Mask {
            dtype: InterventionDtype::Float32,
            shape: vec![2, 6],
            keep: vec![true],
        },
    ] {
        assert!(lower_routed_intervention(geometry(), &locations(), &full(), &action).is_err());
    }
    let mut huge = geometry();
    huge.experts = u64::MAX;
    assert!(matches!(
        lower_routed_intervention(huge, &locations(), &full(), &action),
        Err(CaptureError::Overflow)
    ));
}

fn plans_for(actions: Vec<InterventionAction>) -> (AdmittedCapturePlan, AdmittedInterventionPlan) {
    let (capture, _) = plans(vec![], false);
    let point = InterventionPoint {
        path: "units".into(),
        node_id: "block".into(),
        stage: InterventionStage::Activation,
        axes: vec![
            TensorAxis {
                name: "token".into(),
                dimension: SymbolicDimension::TokenRows,
            },
            TensorAxis {
                name: "component".into(),
                dimension: SymbolicDimension::Known(6),
            },
        ],
        dtypes: vec![InterventionDtype::Float32],
        operations: vec![InterventionKind::MaskComponents, InterventionKind::Scale],
        score_stages: vec![],
        prefill: ObservationSupportStatus::Supported,
        decode: ObservationSupportStatus::Supported,
        conditions: vec![],
        routing: None,
        routed_units: Some(RoutedUnitInterventionPoint {
            routing: "router".into(),
            geometry: geometry(),
        }),
    };
    let discovery = InterventionDiscovery {
        schema_version: 1,
        artifact_identity: "source".into(),
        session_identity: Some("backend-session".into()),
        points: vec![point],
    };
    let plan = InterventionPlan {
        schema_version: 1,
        operations: actions
            .into_iter()
            .enumerate()
            .map(|(i, action)| InterventionOperation {
                id: format!("sparse{i}"),
                target: "units".into(),
                schedule: Default::default(),
                slices: vec![],
                action,
                evidence: InterventionEvidence::None,
            })
            .collect(),
    }
    .admit(&discovery, capture.request(), "session")
    .unwrap();
    (capture, plan)
}
fn session_for(actions: Vec<InterventionAction>) -> CaptureSession {
    let (capture, plan) = plans_for(actions);
    preflight(&capture, &plan, &Estimates).unwrap();
    let mut session = CaptureSession::new(capture);
    session
        .enable_interventions(plan, std::sync::Arc::new(Estimates))
        .unwrap();
    session.begin_step(CapturePhase::Prefill, 0).unwrap();
    session
}
fn chunk(
    session: &mut CaptureSession,
    backend: &mut Backend,
    offset: u64,
    groups: Vec<f32>,
) -> Result<Option<Value>, CaptureExecutionError<std::io::Error>> {
    let value = Value {
        shape: vec![2, 2],
        data: vec![2., 3., 5., 7.],
    };
    let tokens = Value {
        shape: vec![2],
        data: vec![0., 0.],
    };
    let slots = Value {
        shape: vec![2],
        data: vec![1., 0.],
    };
    let coefficients = Value {
        shape: vec![1, 2],
        data: vec![0.3, 0.7],
    };
    let groups = Value {
        shape: vec![2, 2],
        data: groups,
    };
    session.intervene_routed_units(
        backend,
        "router",
        &RoutedUnitCaptureSource {
            values: &value,
            token_indices: &tokens,
            selection_indices: &slots,
            coefficients: &coefficients,
            source_groups: &groups,
            token_offset: offset,
            global_groups: None,
        },
    )
}

#[test]
fn sparse_session_orders_edits_reserves_once_and_requires_full_chunk_completion() {
    let mut session = session_for(vec![
        InterventionAction::Scale {
            dtype: InterventionDtype::Float32,
            factor: 2.,
        },
        mask(vec![3], false),
    ]);
    let mut backend = Backend::default();
    assert!(session.wants_routed_interventions("router"));
    let out = chunk(&mut session, &mut backend, 0, vec![1., 1., 0., 2.])
        .unwrap()
        .unwrap();
    assert_eq!(out.data, [4., 0., 10., 0.]);
    assert!(session.finish_interventions().is_err());
    let usage = session.ledger.total();
    let out = chunk(&mut session, &mut backend, 1, vec![1., 1., 0., 2.])
        .unwrap()
        .unwrap();
    assert_eq!(out.data, [4., 6., 10., 14.]);
    assert_eq!(session.ledger.total(), usage);
    session.finish_interventions().unwrap();
    let step = session.take_step().unwrap();
    assert_eq!(
        step.interventions[0].routed_units.unwrap().affected_values,
        8
    );
    assert_eq!(
        step.interventions[1].routed_units.unwrap().affected_values,
        2
    );
    for record in &step.interventions {
        assert_eq!(record.outcome, InterventionOutcome::Applied);
        assert_eq!(record.routed_units.unwrap().completed_tokens, 2);
    }
    let json = serde_json::to_vec(&step).unwrap();
    let mut decoded = serde_json::from_slice::<CapturedStep>(&json).unwrap();
    // JSON float parsing may round a wall-clock duration by one bit. Receipts,
    // captured values and accounting remain exact.
    assert!(
        (decoded.capture_seconds - step.capture_seconds).abs()
            <= f64::EPSILON * step.capture_seconds.abs()
    );
    decoded.capture_seconds = step.capture_seconds;
    assert_eq!(decoded, step);
}

#[test]
fn sparse_session_no_match_failure_dtype_and_receipt_errors_preserve_accounting() {
    let mut session = session_for(vec![mask(vec![3], false)]);
    let mut backend = Backend::default();
    for offset in 0..2 {
        assert!(
            chunk(&mut session, &mut backend, offset, vec![0., 2., 0., 2.])
                .unwrap()
                .is_none()
        );
    }
    assert_eq!(backend.applications, 0);
    session.finish_interventions().unwrap();
    let step = session.take_step().unwrap();
    assert_eq!(
        step.interventions[0].outcome,
        InterventionOutcome::Unmatched
    );
    for scenario in 0..4 {
        let mut session = session_for(vec![mask(vec![3], false)]);
        let mut backend = Backend::default();
        if scenario == 0 {
            backend.dtype = Some(InterventionDtype::Float16);
        }
        if scenario == 1 {
            backend.fail_application = Some(1);
        }
        let groups = if scenario == 2 {
            vec![1., 9., 0., 2.]
        } else {
            vec![1., 1., 0., 2.]
        };
        let offset = if scenario == 3 { 1 } else { 0 };
        let before = session.ledger.total();
        assert!(chunk(&mut session, &mut backend, offset, groups).is_err());
        if scenario == 0 || scenario == 3 {
            assert_eq!(backend.copies, 0);
            assert_eq!(session.ledger.total(), before);
        } else {
            assert!(session.ledger.total().host_bytes > before.host_bytes);
        }
        assert!(session.finish_interventions().is_err());
        assert!(matches!(
            session.take_step().unwrap().interventions[0].outcome,
            InterventionOutcome::Failed { .. }
        ));
    }
}

#[test]
fn sparse_admission_binds_geometry_and_refuses_dense_evidence_or_unknown_estimator() {
    let (capture, plan) = plans_for(vec![mask(vec![3], false)]);
    assert!(preflight(&capture, &plan, &session::Facts::new(1)).is_err());
    for scenario in 0..4 {
        let mut discovery = session::discovery(&plan);
        let mut host = plan.plan().clone();
        match scenario {
            0 => host.operations[0].evidence = InterventionEvidence::Summary,
            1 => discovery.points[0].axes[1].dimension = SymbolicDimension::Known(7),
            2 => {
                discovery.points[0]
                    .routed_units
                    .as_mut()
                    .unwrap()
                    .geometry
                    .experts = u64::MAX
            }
            _ => host.operations[0].action = mask(vec![6], false),
        }
        assert!(host
            .admit(&discovery, capture.request(), "session")
            .is_err());
    }
}

#[test]
fn sparse_budget_failure_precedes_route_copies_and_duplicate_chunks_cannot_refund() {
    let mut session = session_for(vec![mask(vec![3], false)]);
    let limit = session.plan.plan().limits.per_step.retained_bytes;
    session
        .ledger
        .reserve(CaptureUsage {
            retained_bytes: limit - 1,
            ..Default::default()
        })
        .unwrap();
    let before = session.ledger.total();
    let mut backend = Backend::default();
    assert!(matches!(
        chunk(&mut session, &mut backend, 0, vec![1., 1., 0., 2.]),
        Err(CaptureExecutionError::Admission(CaptureError::Limit { .. }))
    ));
    assert_eq!(backend.copies, 0);
    assert_eq!(backend.applications, 0);
    assert_eq!(session.ledger.total(), before);

    let mut session = session_for(vec![mask(vec![3], false)]);
    chunk(&mut session, &mut backend, 0, vec![1., 1., 0., 2.]).unwrap();
    let charged = session.ledger.total();
    let copies = backend.copies;
    assert!(chunk(&mut session, &mut backend, 0, vec![1., 1., 0., 2.]).is_err());
    assert_eq!(backend.copies, copies);
    assert_eq!(session.ledger.total(), charged);
    assert!(session.finish_interventions().is_err());
}

#[test]
fn sparse_partition_lowering_matches_global_edits_with_permuted_units_and_peer_origins() {
    use eredu_core::component::{ComponentCoordinateMap as Map, RoutedComponentCoordinateMap};
    let global = RoutedUnitGeometry {
        experts: 4,
        units_per_expert: 5,
        routes_per_token: 2,
    };
    let slice = ResolvedCaptureSlice {
        starts: vec![0, 0],
        ends: vec![3, 20],
        strides: vec![2, 1],
        shape: vec![2, 20],
    };
    let rows = vec![
        RoutedUnitLocation {
            source_peer: Some(3),
            token: 2,
            slot: 1,
            expert: 2,
        },
        RoutedUnitLocation {
            source_peer: Some(0),
            token: 0,
            slot: 0,
            expert: 0,
        },
        RoutedUnitLocation {
            source_peer: Some(3),
            token: 0,
            slot: 0,
            expert: 0,
        },
        RoutedUnitLocation {
            source_peer: Some(0),
            token: 1,
            slot: 1,
            expert: 2,
        },
    ];
    let actions = vec![
        InterventionAction::Zero {
            dtype: InterventionDtype::Float32,
        },
        InterventionAction::Scale {
            dtype: InterventionDtype::Float32,
            factor: -0.5,
        },
        mask(vec![1, 4, 10, 13], false),
        mask(vec![1, 4, 10, 13], true),
        InterventionAction::Mask {
            dtype: InterventionDtype::Float32,
            shape: vec![2, 20],
            keep: (0..40).map(|n| n % 3 != 0).collect(),
        },
        InterventionAction::Replace {
            tensor: InterventionTensor {
                shape: vec![2, 20],
                values: InterventionValues::Float32((0..40).map(|n| n as f32 + 0.25).collect()),
            },
        },
        InterventionAction::Add {
            tensor: InterventionTensor {
                shape: vec![2, 20],
                values: InterventionValues::Float32((0..40).map(|n| n as f32 - 0.5).collect()),
            },
        },
    ];
    for action in actions {
        let input = Value {
            shape: vec![3, 20],
            data: (0..60).map(|n| n as f32 * 0.25 + 1.).collect(),
        };
        let expected = apply_activation(&mut Backend::default(), &input, &action, &slice).unwrap();
        for selected in [vec![4, 0, 2], vec![3, 1], vec![]] {
            let coordinates = RoutedComponentCoordinateMap::new(
                Map::indices(4, vec![2, 0]).unwrap(),
                Map::indices(5, selected.clone()).unwrap(),
            );
            let recipe = lower_partition_routed_intervention(
                global,
                3,
                &rows,
                &coordinates,
                &slice,
                &action,
            )
            .unwrap();
            let mut local = Value {
                shape: vec![rows.len() as u64, selected.len() as u64],
                data: rows
                    .iter()
                    .flat_map(|row| {
                        selected.iter().map(|unit| {
                            input.data[(row.token * 20 + row.expert * 5) as usize + unit]
                        })
                    })
                    .collect(),
            };
            if let Some(action) = recipe.action {
                let mut backend = Backend::default();
                let gathered = backend
                    .select_elements(&local, &recipe.indices)
                    .unwrap()
                    .unwrap();
                let n = gathered.data.len() as u64;
                let region = ResolvedCaptureSlice {
                    starts: vec![0],
                    ends: vec![n],
                    strides: vec![1],
                    shape: vec![n],
                };
                let replacement =
                    apply_activation(&mut backend, &gathered, &action, &region).unwrap();
                local = backend
                    .update_elements(&local, &recipe.indices, &replacement)
                    .unwrap()
                    .unwrap();
            }
            let reference: Vec<_> = rows
                .iter()
                .flat_map(|row| {
                    selected.iter().map(|unit| {
                        expected.data[(row.token * 20 + row.expert * 5) as usize + unit]
                    })
                })
                .collect();
            assert_eq!(local.data, reference);
            let serialized = serde_json::to_vec(&coordinates).unwrap();
            assert_eq!(
                serde_json::from_slice::<RoutedComponentCoordinateMap>(&serialized).unwrap(),
                coordinates
            );
        }
    }
    // A partition recipe is not ordinary completion authority.
    let ordinary = RoutedUnitLocations {
        source_token_range: [0, 2],
        rows: rows.clone(),
    };
    assert!(lower_routed_intervention(global, &ordinary, &slice, &mask(vec![], false)).is_err());
    let coordinates = RoutedComponentCoordinateMap::new(
        Map::indices(4, vec![0, 2]).unwrap(),
        Map::range(5, 2..5).unwrap(),
    );
    for fault in 0..4 {
        let mut rows = rows.clone();
        match fault {
            0 => rows[0].expert = 1,
            1 => rows[0].token = 3,
            2 => rows[0] = rows[1],
            _ => rows[0].slot = 2,
        }
        assert!(lower_partition_routed_intervention(
            global,
            3,
            &rows,
            &coordinates,
            &slice,
            &mask(vec![], true)
        )
        .is_err());
    }
}

#[test]
fn independent_cached_invocations_edit_sparse_units_using_physical_token_rows() {
    let (capture, plan) = plans_for(vec![mask(vec![3], false)]);
    let bounds = CaptureInvocationBounds {
        batch: 1,
        max_sequence: 3,
        max_context: None,
        max_predictions: 3,
    };
    let catalog = ObservationCatalog {
        schema_version: 1,
        completeness: DescriptionCompleteness::Complete,
        points: vec![],
    };
    let support = ObservationSupportReport {
        schema_version: 1,
        capture: CaptureCapabilities::default(),
        points: vec![],
    };
    let capture = capture
        .plan()
        .clone()
        .admit_invocations(&catalog, &support, &support.capture, bounds)
        .unwrap();
    let discovery = session::discovery(&plan);
    let plan = plan
        .plan()
        .clone()
        .admit_invocations(&discovery, bounds, "session")
        .unwrap();
    let mut run = CaptureSession::new(capture);
    run.enable_interventions(plan, std::sync::Arc::new(Estimates))
        .unwrap();
    run.begin_invocation(
        CapturePhase::Decode,
        1,
        CaptureInvocationShape {
            batch: 1,
            sequence: 2,
            context: None,
        },
        crate::capture::CaptureInvocationSelection::default(),
    )
    .unwrap();
    let mut backend = Backend::default();
    let first = chunk(&mut run, &mut backend, 0, vec![1., 1., 0., 2.])
        .unwrap()
        .unwrap();
    assert_eq!(first.data, [2., 0., 5., 0.]);
    assert!(
        run.finish_interventions().is_err(),
        "one cached row cannot stand for this two-row invocation"
    );
    let second = chunk(&mut run, &mut backend, 1, vec![1., 1., 0., 2.]).unwrap();
    assert!(second.is_none(), "other experts preserve their live values");
    run.finish_interventions().unwrap();
    let record = run.take_step().unwrap();
    assert_eq!(
        record.interventions[0].outcome,
        InterventionOutcome::Applied
    );
    assert_eq!(
        record.interventions[0]
            .routed_units
            .unwrap()
            .completed_tokens,
        2
    );
}
