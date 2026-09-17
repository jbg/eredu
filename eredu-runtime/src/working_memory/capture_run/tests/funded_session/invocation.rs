use super::*;

#[test]
fn explicit_original_invocation_preserves_phase_shape_scope_and_single_use() {
    let mut raw = raw();
    raw.selections[1].schedule = CaptureSchedule::default();
    let source = admit(raw, point(), 4, true);
    let shape = CaptureInvocationShape {
        batch: 1,
        sequence: 3,
        context: Some(9),
    };
    let selected = [true, false, true];
    assert!(
        crate::capture::CaptureObservationStep::with_invocation(
            source.admission(),
            CapturePhase::Decode,
            2,
            Some(shape)
        )
        .unwrap()
        .requires_sequence_readout()
    );
    let make = || {
        CaptureRunHostPlan::prepare_invocation(&source, CapturePhase::Prefill, 2, shape, &selected)
            .unwrap()
    };
    assert!(matches!(
        CaptureRunHostPlan::prepare(&source),
        Err(CaptureRunHostError::ExplicitInvocation)
    ));
    assert_eq!(make().claim_slots(), 4);
    let h = make().initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(h, 0).unwrap();
    let (short_r, short_run) = fresh(&pool, h - 1);
    let before = ledger(&pool);
    let allocations = CLAIM_ALLOCATIONS.get();
    assert!(short_run.prepare_capture_run(&short_r, make()).is_err());
    assert_eq!(ledger(&pool), before);
    assert_eq!(CLAIM_ALLOCATIONS.get(), allocations);
    drop(short_r);
    drop(short_run);
    let (r, run) = fresh(&pool, h);
    let mut funded = run
        .prepare_capture_run(&r, make())
        .unwrap()
        .into_capture_session()
        .unwrap();
    let mut backend = Backend::default();
    let value = Tensor {
        shape: vec![9, 2],
        values: (0..18).map(|n| (n as f32 - 3.0) * 0.375).collect(),
    };
    let epoch = DistributedCommitEpoch::new(11).unwrap();
    funded
        .with_observer(&mut backend, 2, &|cause| cause, |observer| {
            let guard = crate::inspection::ObservationTransactionGuard::new(observer, epoch);
            guard
                .observer
                .prepare_transaction(epoch, ExpertPass::Prefill)?;
            guard.observer.observe("block.output", &value)?;
            guard.observer.complete_transaction(epoch)?;
            guard.finish(true);
            Ok::<_, FundedCaptureError<MechanismError>>(())
        })
        .unwrap()
        .unwrap();
    let actual = funded.take_shared_step().unwrap().unwrap();
    assert_eq!(funded.spent_steps(), 1);
    assert_eq!(backend.calls, 2);
    assert_eq!(actual.invocation(), Some(shape));
    assert!(matches!(
        actual.records()[1].outcome,
        CaptureOutcome::Skipped {
            reason: CaptureSkipReason::NotInvoked
        }
    ));

    let mut ordinary = CaptureSession::from_shared_plan(source.clone());
    let mut ordinary_backend = Backend::default();
    ordinary
        .begin_invocation(
            CapturePhase::Prefill,
            2,
            shape,
            crate::capture::CaptureInvocationSelection {
                captures: Some(&selected),
                interventions: None,
            },
        )
        .unwrap();
    ordinary
        .prepare_transaction(epoch, ExpertPass::Prefill)
        .unwrap();
    ordinary
        .observe(&mut ordinary_backend, "block.output", &value)
        .unwrap();
    ordinary.complete_transaction(epoch).unwrap();
    ordinary.finish_transaction(epoch, true);
    let expected = ordinary.take_step().unwrap();
    let mut a = serde_json::to_value(&actual).unwrap();
    let mut b = serde_json::to_value(&expected).unwrap();
    a.as_object_mut().unwrap().remove("capture_seconds");
    b.as_object_mut().unwrap().remove("capture_seconds");
    assert_eq!(a, b);
    let usage = funded.usage();
    let repeated = funded.with_observer(&mut backend, 2, &|cause| cause, |observer| {
        observer.prepare_transaction(
            DistributedCommitEpoch::new(12).unwrap(),
            ExpertPass::Prefill,
        )
    });
    assert!(repeated.unwrap().is_err());
    assert_eq!(funded.spent_steps(), 1);
    assert_eq!(funded.usage(), usage);
    assert_eq!(backend.calls, 2);
    let CapturePayload::SharedTensor(payload) = actual.records()[0].payload.as_ref().unwrap()
    else {
        panic!("original shared payload")
    };
    let payload = payload.clone();
    assert_eq!(payload.shape(), &[9, 2]);
    drop(actual);
    drop(funded);
    drop(r);
    drop(run);
    assert_eq!(pool.used_bytes().unwrap(), h);
    let TensorObservationData::F32(values) = payload.data() else {
        panic!("F32 payload")
    };
    assert_eq!(values[0], -1.125);
    assert_eq!(values[17], 5.25);
    drop(payload);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn explicit_funded_observer_readout_uses_the_actual_scope_mask() {
    let mut raw = raw();
    raw.selections[1].schedule = CaptureSchedule::default();
    let source = admit(raw, point(), 4, true);
    let selected = [false, false, false];
    let plan = CaptureRunHostPlan::prepare_invocation(
        &source,
        CapturePhase::Prefill,
        2,
        CaptureInvocationShape {
            batch: 1,
            sequence: 3,
            context: Some(9),
        },
        &selected,
    )
    .unwrap();
    let h = plan.initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(h, 0).unwrap();
    let (reservation, run) = fresh(&pool, h);
    let mut funded = run
        .prepare_capture_run(&reservation, plan)
        .unwrap()
        .into_capture_session()
        .unwrap();
    let mut backend = Backend::default();
    funded
        .with_observer(&mut backend, 2, &|cause| cause, |observer| {
            assert!(!observer.requires_sequence_readout());
        })
        .unwrap();
    assert_eq!(funded.usage(), CaptureUsage::default());
    assert_eq!(funded.spent_steps(), 0);
    assert_eq!(backend.calls, 0);
}

#[test]
fn original_window_claims_match_ordinary_global_slices_and_empty_fragment_delivery() {
    #[derive(Default)]
    struct WindowBackend {
        calls: usize,
    }
    impl ScheduledCaptureBackend for WindowBackend {
        type Tensor = Tensor;
        type Error = MechanismError;
        fn validate_source(
            &self,
            value: &Tensor,
            geometry: &CaptureTensorGeometry<'_>,
        ) -> Result<TensorDtype, Self::Error> {
            if value.shape != geometry.source_shape() {
                return Err(MechanismError::Shape);
            }
            Ok(TensorDtype::F32)
        }
        fn estimate(
            &self,
            _: &Tensor,
            geometry: &CaptureTensorGeometry<'_>,
        ) -> Result<CaptureUsage, CaptureError> {
            Ok(usage(geometry.elements()))
        }
        fn transform(
            &mut self,
            value: &Tensor,
            claim: CaptureTensorClaim<'_, '_>,
        ) -> Result<ClaimedCaptureTensor, Self::Error> {
            self.calls += 1;
            let geometry = claim.geometry();
            let (start, end, stride) = (
                geometry.starts()[0],
                geometry.ends()[0],
                geometry.strides()[0],
            );
            let mut output = claim.prepare()?;
            for row in (start..end).step_by(stride as usize) {
                for col in 0..2 {
                    output
                        .push_f32(value.values[row as usize * 2 + col])
                        .unwrap();
                }
            }
            Ok(output.finish().unwrap())
        }
    }
    impl CaptureBackend for WindowBackend {
        type Tensor = Tensor;
        type Error = MechanismError;
        fn shape(&self, value: &Tensor) -> Result<Vec<u64>, Self::Error> {
            Ok(value.shape.iter().map(|&n| n as u64).collect())
        }
        fn source_dtype(&self, _: &Tensor) -> Option<TensorDtype> {
            Some(TensorDtype::F32)
        }
        fn estimate(
            &self,
            _: &Tensor,
            _: &CaptureSelection,
            slice: &ResolvedCaptureSlice,
        ) -> Result<CaptureUsage, CaptureError> {
            Ok(usage(slice.shape.iter().product::<u64>() as usize))
        }
        fn transform(
            &mut self,
            value: &Tensor,
            _: &CaptureSelection,
            slice: &ResolvedCaptureSlice,
        ) -> Result<CapturePayload, Self::Error> {
            self.calls += 1;
            // Brute-force actual source coordinates, independently of the fixed
            // native/claim iterator and global intersection implementation.
            let values = value
                .values
                .iter()
                .enumerate()
                .filter_map(|(flat, &v)| {
                    let row = (flat / 2) as u64;
                    (row >= slice.starts[0]
                        && row < slice.ends[0]
                        && (row - slice.starts[0]) % slice.strides[0] == 0)
                        .then_some(v)
                })
                .collect();
            Ok(CapturePayload::Tensor(
                TensorObservation::new(
                    slice.shape.iter().map(|&n| n as usize).collect(),
                    TensorObservationData::F32(values),
                )
                .unwrap(),
            ))
        }
    }
    let mut raw = raw();
    raw.selections.truncate(1);
    raw.selections[0].transform = CaptureTransform::Slice;
    raw.selections[0].slices = vec![CaptureSlice {
        axis: "rows".into(),
        start: 1,
        end: 11,
        stride: 3,
    }];
    let mut declaration = point();
    declaration.axes.as_mut().unwrap()[0] = TensorAxis {
        name: "rows".into(),
        dimension: SymbolicDimension::Sequence,
    };
    let catalog = ObservationCatalog {
        schema_version: 1,
        points: vec![declaration],
        completeness: DescriptionCompleteness::Complete,
    };
    let support = ObservationSupportReport {
        schema_version: 1,
        capture: Default::default(),
        points: vec![ObservationSupport {
            path: "block.output".into(),
            prefill: ObservationSupportStatus::Supported,
            decode: ObservationSupportStatus::Supported,
            floating_to_f32: true,
        }],
    };
    let capabilities = CaptureCapabilities {
        transformations: vec![CaptureTransformKind::Slice],
        ..Default::default()
    };
    let source = SharedCapturePlan::new(
        raw.admit_invocations(
            &catalog,
            &support,
            &capabilities,
            CaptureInvocationBounds {
                batch: 1,
                max_sequence: 11,
                max_context: None,
                max_predictions: 1,
            },
        )
        .unwrap(),
    );
    let selected = [true];
    for range in [0..1, 1..2, 2..5, 5..9, 9..11] {
        let shape = CaptureInvocationShape {
            batch: 1,
            sequence: range.end - range.start,
            context: None,
        };
        let window = CaptureInvocationWindow {
            logical_sequence: 11,
            start: range.start,
        };
        let make = || {
            CaptureRunHostPlan::prepare_invocation_window(
                &source,
                CapturePhase::Prefill,
                0,
                shape,
                &selected,
                Some(window),
            )
            .unwrap()
        };
        let h = make().initialization_peak_bytes();
        let pool = WorkingMemoryPool::new(h, 0).unwrap();
        let (short_r, short_run) = fresh(&pool, h - 1);
        let before = ledger(&pool);
        let allocations = CLAIM_ALLOCATIONS.get();
        assert!(short_run.prepare_capture_run(&short_r, make()).is_err());
        assert_eq!(ledger(&pool), before);
        assert_eq!(CLAIM_ALLOCATIONS.get(), allocations);
        drop(short_r);
        drop(short_run);
        let (r, run) = fresh(&pool, h);
        let mut funded = run
            .prepare_capture_run(&r, make())
            .unwrap()
            .into_capture_session()
            .unwrap();
        let value = Tensor {
            shape: vec![shape.sequence as usize, 2],
            values: (range.start * 2..range.end * 2)
                .map(|n| n as f32 * 0.375 - 2.5)
                .collect(),
        };
        let mut backend = WindowBackend::default();
        let epoch = DistributedCommitEpoch::new(1).unwrap();
        funded
            .with_observer(&mut backend, 0, &|e| e, |observer| {
                let guard = crate::inspection::ObservationTransactionGuard::new(observer, epoch);
                guard
                    .observer
                    .prepare_transaction(epoch, ExpertPass::Prefill)?;
                guard.observer.observe("block.output", &value)?;
                guard.observer.complete_transaction(epoch)?;
                guard.finish(true);
                Ok::<_, FundedCaptureError<MechanismError>>(())
            })
            .unwrap()
            .unwrap();
        let actual = funded.take_shared_step().unwrap().unwrap();
        let mut ordinary = CaptureSession::from_shared_plan(source.clone());
        let mut oracle = WindowBackend::default();
        ordinary
            .begin_invocation_window(
                CapturePhase::Prefill,
                0,
                shape,
                crate::capture::CaptureInvocationSelection {
                    captures: Some(&selected),
                    interventions: None,
                },
                window,
            )
            .unwrap();
        ordinary
            .prepare_transaction(epoch, ExpertPass::Prefill)
            .unwrap();
        ordinary
            .observe(&mut oracle, "block.output", &value)
            .unwrap();
        ordinary.complete_transaction(epoch).unwrap();
        ordinary.finish_transaction(epoch, true);
        let expected = ordinary.take_step().unwrap();
        let mut a = serde_json::to_value(&actual).unwrap();
        let mut b = serde_json::to_value(&expected).unwrap();
        a.as_object_mut().unwrap().remove("capture_seconds");
        b.as_object_mut().unwrap().remove("capture_seconds");
        assert_eq!(a, b);
        assert_eq!(backend.calls, oracle.calls);
        let expected_values = (range.start..range.end)
            .filter(|&row| row >= 1 && (row - 1) % 3 == 0)
            .flat_map(|row| [row * 2, row * 2 + 1])
            .map(|n| n as f32 * 0.375 - 2.5)
            .collect::<Vec<_>>();
        let record = &actual.records()[0];
        assert_eq!(record.source_shape, Some(vec![shape.sequence, 2]));
        if expected_values.is_empty() {
            assert_eq!(backend.calls, 0);
            assert!(record.payload.is_none());
            assert!(record.selected_shape.is_none());
            assert!(matches!(
                record.outcome,
                CaptureOutcome::Skipped {
                    reason: CaptureSkipReason::NotInvoked
                }
            ));
        } else {
            let Some(CapturePayload::SharedTensor(payload)) = &record.payload else {
                panic!("shared original payload")
            };
            let TensorObservationData::F32(values) = payload.data() else {
                panic!("F32")
            };
            assert_eq!(values, &expected_values);
        }
        let spent = funded.usage();
        assert!(
            funded
                .with_observer(&mut backend, 0, &|e| e, |observer| observer
                    .prepare_transaction(
                        DistributedCommitEpoch::new(2).unwrap(),
                        ExpertPass::Prefill
                    ))
                .unwrap()
                .is_err()
        );
        assert_eq!(funded.usage(), spent);
        drop(funded);
        drop(r);
        drop(run);
        assert_eq!(pool.used_bytes().unwrap(), h);
        drop(actual);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn original_logical_skip_preserves_limit_reason_without_a_tensor_claim() {
    let source = admit(raw(), point(), 4, true);
    let shape = CaptureInvocationShape {
        batch: 1,
        sequence: 3,
        context: Some(9),
    };
    let selected = [false, false, true];
    let reason = CaptureSkipReason::Limit {
        budget: CaptureBudget::Host,
        cumulative: true,
    };
    let skipped = [Some(reason.clone()), Some(reason.clone()), None];
    let make = || {
        CaptureRunHostPlan::prepare_invocation(&source, CapturePhase::Prefill, 2, shape, &selected)
            .unwrap()
    };
    let tensor_bytes = make().tensor_peak_bytes();
    let plan = make().with_skip_reasons(&skipped).unwrap();
    assert_eq!(plan.tensor_peak_bytes(), tensor_bytes);
    assert!(plan.initialization_peak_bytes() > make().initialization_peak_bytes());
    assert!(
        make()
            .with_skip_reasons(&[None, None, Some(reason.clone())])
            .is_err()
    );
    let h = plan.initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(h, 0).unwrap();
    let (reservation, run) = fresh(&pool, h);
    let mut funded = run
        .prepare_capture_run(&reservation, plan)
        .unwrap()
        .into_capture_session()
        .unwrap();
    let epoch = DistributedCommitEpoch::new(51).unwrap();
    let value = Tensor {
        shape: vec![9, 2],
        values: (0..18).map(|n| (n as f32 - 3.0) * 0.375).collect(),
    };
    let mut backend = Backend::default();
    funded
        .with_observer(&mut backend, 2, &|cause| cause, |observer| {
            let guard = crate::inspection::ObservationTransactionGuard::new(observer, epoch);
            guard
                .observer
                .prepare_transaction(epoch, ExpertPass::Prefill)?;
            guard.observer.observe("block.output", &value)?;
            guard.observer.complete_transaction(epoch)?;
            guard.finish(true);
            Ok::<_, FundedCaptureError<MechanismError>>(())
        })
        .unwrap()
        .unwrap();
    let frame = funded.take_shared_step().unwrap().unwrap();
    assert_eq!(backend.calls, 1);
    assert_eq!(
        frame.records()[0].outcome,
        CaptureOutcome::Skipped { reason }
    );
    assert_eq!(
        frame.records()[1].outcome,
        CaptureOutcome::Skipped {
            reason: CaptureSkipReason::Schedule
        }
    );
    for record in &frame.records()[..2] {
        assert!(record.payload.is_none());
        assert!(record.source_shape.is_none());
        assert!(record.selected_shape.is_none());
    }
    let CapturePayload::SharedTensor(payload) = frame.records()[2].payload.as_ref().unwrap() else {
        panic!("original paid Preview")
    };
    let payload = payload.clone();
    let TensorObservationData::F32(values) = payload.data() else {
        panic!("F32")
    };
    assert_eq!(values, &[-1.125, -0.75, -0.375]);
    drop(frame);
    drop(funded);
    drop(reservation);
    drop(run);
    assert_eq!(pool.used_bytes().unwrap(), h);
    drop(payload);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
