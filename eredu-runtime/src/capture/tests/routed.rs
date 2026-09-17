use super::*;

#[test]
fn draining_an_incomplete_sparse_step_fails_every_pending_selection() {
    let (mut plan, catalog, support, capabilities) = fixture();
    for id in ["second", "third"] {
        let mut selection = plan.selections[0].clone();
        selection.id = id.into();
        plan.selections.push(selection);
    }
    let mut session = CaptureSession::new(admit(plan, &catalog, &support, &capabilities).unwrap());
    let mut backend = Backend {
        calls: 0,
        fail: false,
    };
    session.begin_step(CapturePhase::Prefill, 0).unwrap();
    session
        .observe_routed_units(&mut backend, "experts", false, &source(&0))
        .unwrap();
    let reserved = session.ledger.total();
    let step = session.take_step().unwrap();
    assert_eq!(step.cumulative_usage, reserved);
    assert!(step.records.iter().all(|record| record.payload.is_none()
        && matches!(record.outcome, CaptureOutcome::Failed { .. })));
    assert!(!session.checkpoint_ready);
}

fn fixture() -> (
    CapturePlan,
    ObservationCatalog,
    ObservationSupportReport,
    CaptureCapabilities,
) {
    let (plan, mut catalog, support, mut capabilities) =
        super::fixture(CaptureTransform::RoutedUnits);
    capabilities
        .transformations
        .push(CaptureTransformKind::RoutedUnits);
    catalog.points[0].value_type = ObservationValueType::RoutedUnits {
        routing: "experts".into(),
        geometry: RoutedUnitGeometry {
            experts: 4,
            units_per_expert: 3,
            routes_per_token: 2,
        },
    };
    catalog.points[0].axes = Some(vec![
        TensorAxis {
            name: "token".into(),
            dimension: SymbolicDimension::TokenRows,
        },
        TensorAxis {
            name: "route".into(),
            dimension: SymbolicDimension::Known(2),
        },
        TensorAxis {
            name: "component".into(),
            dimension: SymbolicDimension::Known(3),
        },
    ]);
    (plan, catalog, support, capabilities)
}

struct Backend {
    calls: usize,
    fail: bool,
}
impl CaptureBackend for Backend {
    type Tensor = u64;
    type Error = std::io::Error;
    fn shape(&self, _: &u64) -> Result<Vec<u64>, Self::Error> {
        panic!("sparse source uses declared invocation geometry")
    }
    fn estimate(
        &self,
        _: &u64,
        _: &CaptureSelection,
        _: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        panic!("dense estimator")
    }
    fn transform(
        &mut self,
        _: &u64,
        _: &CaptureSelection,
        _: &ResolvedCaptureSlice,
    ) -> Result<CapturePayload, Self::Error> {
        panic!("dense collector")
    }
    fn source_dtype(&self, _: &u64) -> Option<TensorDtype> {
        Some(TensorDtype::F32)
    }
    fn estimate_routed_units(
        &self,
        _: &[u64],
        _: &CaptureSelection,
        _: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        Ok(CaptureUsage {
            captures: 1,
            retained_bytes: 300,
            host_bytes: 3000,
            encoded_bytes: 30000,
        })
    }
    fn capture_routed_units(
        &mut self,
        source: &RoutedUnitCaptureSource<'_, u64>,
        geometry: RoutedUnitGeometry,
        slice: &ResolvedCaptureSlice,
    ) -> Option<Result<RoutedUnitCapture, Self::Error>> {
        self.calls += 1;
        if self.fail {
            return Some(Err(std::io::Error::other("routed sentinel")));
        }
        let token = *source.values;
        let mut rows = vec![];
        if token >= slice.starts[0]
            && token < slice.ends[0]
            && (token - slice.starts[0]).is_multiple_of(slice.strides[0])
        {
            for slot in (slice.starts[1]..slice.ends[1]).step_by(slice.strides[1] as usize) {
                let values: Vec<_> = (slice.starts[2]..slice.ends[2])
                    .step_by(slice.strides[2] as usize)
                    .map(|unit| (token * 10 + slot * 3 + unit) as f32 + 0.5)
                    .collect();
                rows.push(RoutedUnitCaptureRow {
                    source_peer: None,
                    token,
                    slot,
                    expert: (token + slot) % 4,
                    coefficient: 0.5,
                    unit_start: slice.starts[2],
                    unit_stride: slice.strides[2],
                    values: TensorObservation::new(
                        vec![values.len()],
                        TensorObservationData::F32(values),
                    )
                    .unwrap(),
                });
            }
        }
        Some(Ok(RoutedUnitCapture {
            geometry,
            source_token_ranges: vec![[token, token + 1]],
            rows,
        }))
    }
}
fn source(token: &u64) -> RoutedUnitCaptureSource<'_, u64> {
    RoutedUnitCaptureSource {
        values: token,
        token_indices: token,
        selection_indices: token,
        coefficients: token,
        source_groups: token,
        token_offset: *token,
        global_groups: None,
    }
}

#[test]
fn sparse_capture_reserves_once_validates_receipts_and_waits_for_commit() {
    let (mut plan, catalog, support, capabilities) = fixture();
    plan.selections[0].slices = vec![CaptureSlice {
        axis: "component".into(),
        start: 1,
        end: 3,
        stride: 1,
    }];
    let mut session = CaptureSession::new(admit(plan, &catalog, &support, &capabilities).unwrap());
    let mut backend = Backend {
        calls: 0,
        fail: false,
    };
    let epoch = DistributedCommitEpoch::FIRST;
    session
        .prepare_step_transaction(epoch, crate::ExpertPass::Prefill, 0)
        .unwrap();
    assert!(session.wants_routed_units("experts"));
    assert!(!session.wants_routed_units("other"));
    let before = session.ledger.total();
    for token in 0..3 {
        session
            .observe_routed_units(&mut backend, "experts", false, &source(&token))
            .unwrap();
        assert!(session.take_step().is_none());
        assert_eq!(session.ledger.total().captures, before.captures + 1);
        assert_eq!(
            session.ledger.total().retained_bytes,
            before.retained_bytes + 300
        );
    }
    session.complete_transaction(epoch).unwrap();
    session.finish_transaction(epoch, true);
    let step = session.take_step().unwrap();
    assert_eq!(backend.calls, 3);
    assert_eq!(step.records[0].outcome, CaptureOutcome::Captured);
    let Some(CapturePayload::RoutedUnits(units)) = &step.records[0].payload else {
        panic!("sparse")
    };
    assert_eq!(units.rows.len(), 6);
    assert_eq!(units.rows[4].token, 2);
    assert_eq!(units.rows[4].unit_start, 1);
    assert_eq!(
        units.rows[4].values.data(),
        &TensorObservationData::F32(vec![21.5, 22.5])
    );
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
fn sparse_capture_rejects_incomplete_duplicate_and_failed_work_without_refunds() {
    for failure in ["missing", "duplicate", "backend"] {
        let (plan, catalog, support, capabilities) = fixture();
        let mut session =
            CaptureSession::new(admit(plan, &catalog, &support, &capabilities).unwrap());
        let mut backend = Backend {
            calls: 0,
            fail: false,
        };
        let epoch = DistributedCommitEpoch::FIRST;
        session
            .prepare_step_transaction(epoch, crate::ExpertPass::Prefill, 0)
            .unwrap();
        session
            .observe_routed_units(&mut backend, "experts", false, &source(&0))
            .unwrap();
        let reserved = session.ledger.total();
        match failure {
            "missing" => assert!(session.complete_transaction(epoch).is_err()),
            "duplicate" => assert!(session
                .observe_routed_units(&mut backend, "experts", false, &source(&0))
                .is_err()),
            _ => {
                backend.fail = true;
                let error = session
                    .observe_routed_units(&mut backend, "experts", false, &source(&1))
                    .unwrap_err();
                assert!(matches!(error, CaptureExecutionError::Backend(_)));
                assert!(error.to_string().contains("routed sentinel"));
            }
        }
        session.finish_transaction(epoch, false);
        let step = session.take_step().unwrap();
        assert_eq!(step.cumulative_usage, reserved);
        assert!(matches!(
            step.records[0].outcome,
            CaptureOutcome::Failed { .. }
        ));
        assert!(step.records[0].payload.is_none());
    }
}

#[test]
fn sparse_capture_skip_reserves_no_native_work_and_dense_transform_is_rejected() {
    let (mut plan, catalog, support, capabilities) = fixture();
    let mut wrong = plan.clone();
    wrong.selections[0].transform = CaptureTransform::FullTensor;
    assert!(admit(wrong, &catalog, &support, &capabilities).is_err());
    plan.limits.on_limit = CaptureLimitPolicy::Skip;
    plan.limits.per_step.retained_bytes = 0;
    let mut session = CaptureSession::new(admit(plan, &catalog, &support, &capabilities).unwrap());
    let mut backend = Backend {
        calls: 0,
        fail: false,
    };
    session.begin_step(CapturePhase::Prefill, 0).unwrap();
    session
        .observe_routed_units(&mut backend, "experts", false, &source(&0))
        .unwrap();
    assert_eq!(backend.calls, 0);
    assert!(!session.wants_routed_units("experts"));
    assert!(matches!(
        session.take_step().unwrap().records[0].outcome,
        CaptureOutcome::Skipped { .. }
    ));
}

#[test]
fn sparse_receipts_preserve_duplicate_expert_slots_and_reject_bad_coordinates() {
    let geometry = RoutedUnitGeometry {
        experts: 4,
        units_per_expert: 3,
        routes_per_token: 2,
    };
    let slice = ResolvedCaptureSlice {
        starts: vec![0, 0, 0],
        ends: vec![1, 2, 3],
        strides: vec![1, 1, 1],
        shape: vec![1, 2, 3],
    };
    let mut backend = Backend {
        calls: 0,
        fail: false,
    };
    let mut data = backend
        .capture_routed_units(&source(&0), geometry, &slice)
        .unwrap()
        .unwrap();
    data.rows[1].expert = data.rows[0].expert;
    data.finish_ordinary(&slice, 1).unwrap();
    assert_eq!(data.rows[0].slot, 0);
    assert_eq!(data.rows[1].slot, 1);
    for kind in [
        "expert",
        "slot",
        "token",
        "unit",
        "shape",
        "coefficient",
        "peer",
        "coverage",
    ] {
        let mut invalid = data.clone();
        match kind {
            "expert" => invalid.rows[0].expert = 4,
            "slot" => invalid.rows[1].slot = 0,
            "token" => invalid.rows[0].token = 1,
            "unit" => invalid.rows[0].unit_start = 1,
            "shape" => {
                invalid.rows[0].values =
                    TensorObservation::new(vec![1], TensorObservationData::F32(vec![1.])).unwrap()
            }
            "coefficient" => invalid.rows[0].coefficient = f32::NAN,
            "peer" => invalid.rows[0].source_peer = Some(0),
            _ => invalid.source_token_ranges[0][0] = 1,
        }
        assert!(invalid.finish_ordinary(&slice, 1).is_err(), "{kind}");
    }
    data.rows[0].values = TensorObservation::new(
        vec![3],
        TensorObservationData::F32(vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY]),
    )
    .unwrap();
    let wire = serde_json::to_vec(&CapturePayload::RoutedUnits(data)).unwrap();
    let CapturePayload::RoutedUnits(decoded) = serde_json::from_slice(&wire).unwrap() else {
        panic!("sparse")
    };
    let TensorObservationData::F32(values) = decoded.rows[0].values.data() else {
        panic!("floating")
    };
    assert!(values[0].is_nan());
    assert_eq!(&values[1..], &[f32::INFINITY, f32::NEG_INFINITY]);
    assert!(RoutedUnitGeometry {
        experts: u64::MAX,
        units_per_expert: 2,
        routes_per_token: 1
    }
    .components()
    .is_err());
}

#[test]
fn independent_cached_invocation_geometry_controls_sparse_receipt_completeness() {
    let (plan, catalog, support, capabilities) = fixture();
    let bounds = CaptureInvocationBounds {
        batch: 1,
        max_sequence: 4,
        max_context: None,
        max_predictions: 3,
    };
    let admitted = plan
        .admit_invocations(&catalog, &support, &capabilities, bounds)
        .unwrap();
    let mut session = CaptureSession::new(admitted);
    let mut backend = Backend {
        calls: 0,
        fail: false,
    };
    for sequence in [3, 2] {
        session
            .begin_invocation(
                CapturePhase::Decode,
                1,
                CaptureInvocationShape {
                    batch: 1,
                    sequence,
                    context: None,
                },
                CaptureInvocationSelection::default(),
            )
            .unwrap();
        for token in 0..sequence {
            session
                .observe_routed_units(&mut backend, "experts", false, &source(&token))
                .unwrap();
        }
        let record = session.take_step().unwrap();
        assert_eq!(record.records[0].outcome, CaptureOutcome::Captured);
        assert_eq!(record.records[0].source_shape, Some(vec![sequence, 2, 3]));
        let Some(CapturePayload::RoutedUnits(payload)) = &record.records[0].payload else {
            panic!("sparse receipt missing")
        };
        assert_eq!(payload.rows.len(), sequence as usize * 2);
        assert_eq!(record.invocation.unwrap().sequence, sequence);
    }
    assert_eq!(backend.calls, 5);
    assert_eq!(session.cumulative_usage().captures, 2);
}

#[test]
fn sparse_invocation_windows_preserve_logical_stride_actual_ranges_and_spending() {
    let (mut plan, catalog, support, capabilities) = fixture();
    plan.selections[0].slices = vec![
        CaptureSlice {
            axis: "token".into(),
            start: 1,
            end: 5,
            stride: 2,
        },
        CaptureSlice {
            axis: "component".into(),
            start: 1,
            end: 3,
            stride: 1,
        },
    ];
    let admitted = plan
        .admit_invocations(
            &catalog,
            &support,
            &capabilities,
            CaptureInvocationBounds {
                batch: 1,
                max_sequence: 5,
                max_context: None,
                max_predictions: 1,
            },
        )
        .unwrap();
    let mut session = CaptureSession::new(admitted);
    let mut backend = Backend {
        calls: 0,
        fail: false,
    };
    let mut previous = session.cumulative_usage();
    let mut logical_tokens = Vec::new();
    for (start, sequence, selected) in [(0, 2, 1), (2, 2, 1), (4, 1, 0)] {
        session
            .begin_invocation_window(
                CapturePhase::Prefill,
                0,
                CaptureInvocationShape {
                    batch: 1,
                    sequence,
                    context: None,
                },
                CaptureInvocationSelection::default(),
                CaptureInvocationWindow {
                    logical_sequence: 5,
                    start,
                },
            )
            .unwrap();
        let metadata = session.cumulative_usage();
        let mut reserved = None;
        for token in 0..sequence {
            session
                .observe_routed_units(&mut backend, "experts", false, &source(&token))
                .unwrap();
            let current = session.cumulative_usage();
            if let Some(reserved) = reserved {
                assert_eq!(current, reserved);
            } else {
                reserved = Some(current);
            }
        }
        let step = session.take_step().unwrap();
        assert_eq!(step.cumulative_usage, reserved.unwrap());
        assert_eq!(step.cumulative_usage.captures, previous.captures + 1);
        assert_eq!(
            step.cumulative_usage.retained_bytes,
            previous.retained_bytes + 300
        );
        assert!(step.cumulative_usage.host_bytes > metadata.host_bytes + 3000);
        assert!(step.cumulative_usage.encoded_bytes > metadata.encoded_bytes + 30000);
        assert_eq!(step.invocation.unwrap().sequence, sequence);
        assert_eq!(step.phase, CapturePhase::Prefill);
        assert_eq!(step.prediction_index, 0);
        let record = &step.records[0];
        assert_eq!(record.outcome, CaptureOutcome::Captured);
        assert_eq!(record.source_shape, Some(vec![sequence, 2, 3]));
        assert_eq!(record.selected_shape, Some(vec![selected, 2, 2]));
        let Some(CapturePayload::RoutedUnits(payload)) = &record.payload else {
            panic!("actual sparse window missing")
        };
        assert_eq!(
            payload.source_token_ranges,
            (0..sequence)
                .map(|token| [token, token + 1])
                .collect::<Vec<_>>()
        );
        assert_eq!(payload.rows.len(), selected as usize * 2);
        for row in &payload.rows {
            logical_tokens.push(start + row.token);
            assert_eq!((row.unit_start, row.unit_stride), (1, 1));
            let first = (row.token * 10 + row.slot * 3 + 1) as f32 + 0.5;
            assert_eq!(
                row.values.data(),
                &TensorObservationData::F32(vec![first, first + 1.])
            );
        }
        previous = step.cumulative_usage;
    }
    assert_eq!(logical_tokens, [1, 1, 3, 3]);
    assert_eq!(backend.calls, 5);
    assert_eq!(session.cumulative_usage(), previous);
    // Repeated logical coordinates still spend; invalid native overlap poisons
    // that invocation without refunding its already reserved record and payload.
    session
        .begin_invocation_window(
            CapturePhase::Prefill,
            0,
            CaptureInvocationShape {
                batch: 1,
                sequence: 2,
                context: None,
            },
            CaptureInvocationSelection::default(),
            CaptureInvocationWindow {
                logical_sequence: 5,
                start: 0,
            },
        )
        .unwrap();
    session
        .observe_routed_units(&mut backend, "experts", false, &source(&0))
        .unwrap();
    let reserved = session.cumulative_usage();
    assert!(reserved.captures > previous.captures);
    assert!(session
        .observe_routed_units(&mut backend, "experts", false, &source(&0))
        .is_err());
    let failed = session.take_step().unwrap();
    assert_eq!(failed.cumulative_usage, reserved);
    assert!(matches!(
        failed.records[0].outcome,
        CaptureOutcome::Failed { .. }
    ));
    assert!(failed.records[0].payload.is_none());
}
