use super::*;
use eredu_core::{checkpoint::TensorDtype, *};
use std::{cell::Cell, convert::Infallible};

mod checkpoints;
mod generated;
mod partition;
mod projection;
mod routed;

fn generated_source(creation_bytes: u64) -> GeneratedCaptureSource {
    GeneratedCaptureSource {
        creation_bytes,
        source_dtype: None,
    }
}

fn fixture(
    transform: CaptureTransform,
) -> (
    CapturePlan,
    ObservationCatalog,
    ObservationSupportReport,
    CaptureCapabilities,
) {
    let point = ObservationPoint {
        path: "block.output".into(),
        node_id: "block".into(),
        meaning: "activation".into(),
        value_type: ObservationValueType::Tensor,
        dtype: ObservationDtype::Floating,
        axes: Some(vec![
            TensorAxis {
                name: "sequence".into(),
                dimension: SymbolicDimension::Sequence,
            },
            TensorAxis {
                name: "hidden".into(),
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
    let usage = CaptureUsage {
        captures: 20,
        retained_bytes: 1_000_000,
        host_bytes: 1_000_000,
        encoded_bytes: 1_000_000,
    };
    let plan = CapturePlan {
        schema_version: 1,
        selections: vec![CaptureSelection {
            id: "preview".into(),
            path: "block.output".into(),
            schedule: CaptureSchedule::default(),
            slices: vec![],
            transform,
        }],
        limits: CaptureLimits {
            per_step: usage,
            cumulative: usage,
            physical_native_bytes: None,
            on_limit: CaptureLimitPolicy::Fail,
        },
    };
    let capabilities = CaptureCapabilities {
        transformations: vec![
            CaptureTransformKind::Preview,
            CaptureTransformKind::Slice,
            CaptureTransformKind::FullTensor,
            CaptureTransformKind::Summary,
            CaptureTransformKind::Histogram,
        ],
        max_histogram_bins: 8,
        physical_native_limit: false,
        conditions: vec![],
    };
    (plan, catalog, support, capabilities)
}

#[test]
fn selected_full_vocabulary_scores_reject_slices_duplicates_and_out_of_range_ids() {
    let (mut plan, mut catalog, mut support, mut capabilities) =
        fixture(CaptureTransform::TokenScores {
            token_ids: vec![0, 3],
        });
    capabilities
        .transformations
        .push(CaptureTransformKind::TokenScores);
    plan.selections[0].path = MODEL_LOGITS_OBSERVATION_PATH.into();
    catalog.points[0].path = MODEL_LOGITS_OBSERVATION_PATH.into();
    catalog.points[0].axes.as_mut().unwrap()[1].name = "vocabulary".into();
    support.points[0].path = MODEL_LOGITS_OBSERVATION_PATH.into();
    admit(plan.clone(), &catalog, &support, &capabilities).unwrap();
    for ids in [vec![], vec![0, 0], vec![4], (0..65).collect()] {
        let mut invalid = plan.clone();
        invalid.selections[0].transform = CaptureTransform::TokenScores { token_ids: ids };
        assert!(admit(invalid, &catalog, &support, &capabilities).is_err());
    }
    plan.selections[0].slices = vec![CaptureSlice {
        axis: "vocabulary".into(),
        start: 0,
        end: 2,
        stride: 1,
    }];
    assert!(
        admit(plan, &catalog, &support, &capabilities).is_err(),
        "truncated distributions cannot masquerade as full probability"
    );
}

fn admit(
    plan: CapturePlan,
    catalog: &ObservationCatalog,
    support: &ObservationSupportReport,
    capabilities: &CaptureCapabilities,
) -> Result<AdmittedCapturePlan, CaptureError> {
    plan.admit(
        catalog,
        support,
        capabilities,
        CaptureRequestShape {
            batch: 1,
            prompt_tokens: 3,
            max_predictions: 10,
        },
    )
}

struct ProbeBackend {
    copies: Cell<usize>,
}
impl CaptureBackend for ProbeBackend {
    type Tensor = Vec<u64>;
    type Error = Infallible;
    fn shape(&self, tensor: &Self::Tensor) -> Result<Vec<u64>, Infallible> {
        Ok(tensor.clone())
    }
    fn estimate(
        &self,
        _: &Self::Tensor,
        _: &CaptureSelection,
        _: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        Ok(CaptureUsage {
            captures: 1,
            retained_bytes: 100,
            host_bytes: 16,
            encoded_bytes: 100,
        })
    }
    fn transform(
        &mut self,
        _: &Self::Tensor,
        _: &CaptureSelection,
        _: &ResolvedCaptureSlice,
    ) -> Result<CapturePayload, Infallible> {
        self.copies.set(self.copies.get() + 1);
        Ok(CapturePayload::Tensor(
            TensorObservation::new(vec![2], TensorObservationData::U64(vec![u64::MAX, 1])).unwrap(),
        ))
    }
}

#[test]
fn session_validation_rechecks_loaded_capture_capabilities_before_estimating() {
    let (plan, catalog, mut support, caps) = fixture(CaptureTransform::Summary);
    support.capture = caps.clone();
    let admitted = admit(plan, &catalog, &support, &caps).unwrap();
    let mut discovery = CaptureDiscovery {
        artifact_identity: "fixture".into(),
        catalog,
        support,
    };
    let calls = Cell::new(0);
    let estimate = |_: &[u64], _: &CaptureSelection, _: &ResolvedCaptureSlice| {
        calls.set(calls.get() + 1);
        Ok(CaptureUsage::default())
    };
    validate_session(&admitted, &discovery, estimate).unwrap();
    assert!(calls.get() > 0);
    calls.set(0);
    discovery.catalog.points[0].node_id = "different-loaded-node".into();
    assert!(validate_session(&admitted, &discovery, estimate).is_err());
    assert_eq!(calls.get(), 0);
}

#[test]
fn none_is_not_legacy_capture_all() {
    let (_, catalog, support, capabilities) = fixture(CaptureTransform::Summary);
    let admitted = admit(CapturePlan::none(), &catalog, &support, &capabilities).unwrap();
    let mut session = CaptureSession::new(admitted);
    let mut backend = ProbeBackend {
        copies: Cell::new(0),
    };
    session.begin_step(CapturePhase::Prefill, 0).unwrap();
    session
        .observe(&mut backend, "block.output", &vec![3, 4])
        .unwrap();
    assert!(session.take_step().unwrap().records.is_empty());
    assert_eq!(backend.copies.get(), 0);
    assert!(ObservationRequest::selected([]).matches("block.output"));
}

#[test]
fn conditional_capture_distinguishes_absence_from_measurement() {
    let (plan, catalog, mut support, caps) = fixture(CaptureTransform::Preview { max_elements: 2 });
    support.points[0].prefill =
        ObservationSupportStatus::Conditional("requires media input".into());
    for present in [false, true] {
        let mut session =
            CaptureSession::new(admit(plan.clone(), &catalog, &support, &caps).unwrap());
        let mut backend = ProbeBackend {
            copies: Cell::new(0),
        };
        session.begin_step(CapturePhase::Prefill, 0).unwrap();
        if present {
            session
                .observe(&mut backend, "block.output", &vec![3, 4])
                .unwrap();
        }
        let step = session.take_step().unwrap();
        assert_eq!(step.records.len(), 1);
        assert_eq!(backend.copies.get(), usize::from(present));
        if present {
            assert!(step.records[0].payload.is_some());
            assert_ne!(step.records[0].outcome, CaptureOutcome::Missing);
        } else {
            assert_eq!(step.records[0].outcome, CaptureOutcome::Missing);
            assert!(step.records[0].payload.is_none());
        }
    }
    for status in [
        ObservationSupportStatus::Unsupported("no collector".into()),
        ObservationSupportStatus::Unverified("no collector facts".into()),
    ] {
        support.points[0].prefill = status;
        assert!(matches!(
            admit(plan.clone(), &catalog, &support, &caps),
            Err(CaptureError::Unsupported(_))
        ));
    }
}

#[test]
fn admission_rejects_unknown_paths_axes_histograms_and_unverified_support() {
    let (plan, catalog, mut support, caps) = fixture(CaptureTransform::Preview { max_elements: 2 });
    let mut invalid = plan.clone();
    invalid.selections[0].path = "missing".into();
    assert!(matches!(
        admit(invalid, &catalog, &support, &caps),
        Err(CaptureError::MissingPath(_))
    ));
    let mut invalid = plan.clone();
    invalid.selections[0].slices.push(CaptureSlice {
        axis: "hidden".into(),
        start: 0,
        end: 5,
        stride: 1,
    });
    assert!(admit(invalid, &catalog, &support, &caps).is_err());
    let mut invalid = plan.clone();
    invalid.selections[0].transform = CaptureTransform::Slice;
    assert!(admit(invalid, &catalog, &support, &caps).is_err());
    for edges in [
        vec![],
        vec![0.0, 0.0],
        vec![0.0, f32::INFINITY],
        vec![2.0, 1.0],
    ] {
        let mut invalid = plan.clone();
        invalid.selections[0].transform = CaptureTransform::Histogram { edges };
        assert!(admit(invalid, &catalog, &support, &caps).is_err());
    }
    support.points[0].decode = ObservationSupportStatus::Unverified("partition ownership".into());
    assert!(matches!(
        admit(plan, &catalog, &support, &caps),
        Err(CaptureError::Unsupported(_))
    ));
}

#[test]
fn runtime_budget_failure_or_skip_never_materializes() {
    for budget in [
        CaptureBudget::Captures,
        CaptureBudget::Retention,
        CaptureBudget::Host,
        CaptureBudget::Encoded,
    ] {
        for policy in [CaptureLimitPolicy::Fail, CaptureLimitPolicy::Skip] {
            let (mut plan, catalog, support, caps) =
                fixture(CaptureTransform::Preview { max_elements: 2 });
            let metadata = metadata_reservation(&plan.selections[0], &catalog.points[0]).unwrap();
            match budget {
                CaptureBudget::Captures => plan.limits.per_step.captures = 0,
                CaptureBudget::Retention => plan.limits.per_step.retained_bytes = 99,
                CaptureBudget::Host => plan.limits.per_step.host_bytes = metadata.host_bytes + 15,
                CaptureBudget::Encoded => {
                    plan.limits.per_step.encoded_bytes = metadata.encoded_bytes + 99
                }
            }
            plan.limits.on_limit = policy;
            let mut session = CaptureSession::new(admit(plan, &catalog, &support, &caps).unwrap());
            let mut backend = ProbeBackend {
                copies: Cell::new(0),
            };
            session.begin_step(CapturePhase::Prefill, 0).unwrap();
            let result = session.observe(&mut backend, "block.output", &vec![3, 4]);
            assert_eq!(backend.copies.get(), 0);
            if policy == CaptureLimitPolicy::Fail {
                assert!(result.is_err());
            } else {
                result.unwrap();
                assert!(matches!(
                    session.take_step().unwrap().records[0].outcome,
                    CaptureOutcome::Skipped {
                        reason: CaptureSkipReason::Limit {
                        budget: actual,
                        cumulative: false
                    }
                    } if actual == budget
                ));
            }
        }
    }
}

#[test]
fn frequency_missing_cumulative_bounds_and_consumer_backpressure() {
    let (mut plan, catalog, support, caps) = fixture(CaptureTransform::Preview { max_elements: 2 });
    plan.selections[0].schedule.every = 2;
    plan.limits.cumulative.retained_bytes = 100;
    plan.limits.on_limit = CaptureLimitPolicy::Skip;
    let mut session = CaptureSession::new(admit(plan, &catalog, &support, &caps).unwrap());
    let mut backend = ProbeBackend {
        copies: Cell::new(0),
    };
    session.begin_step(CapturePhase::Prefill, 0).unwrap();
    session
        .observe(&mut backend, "block.output", &vec![3, 4])
        .unwrap();
    assert!(session.begin_step(CapturePhase::Decode, 1).is_err());
    let first = session.take_step().unwrap();
    assert!(matches!(
        first.records[0].outcome,
        CaptureOutcome::Truncated {
            available_elements: 12,
            emitted_elements: 2
        }
    ));
    assert!(serde_json::to_string(&first)
        .unwrap()
        .contains("18446744073709551615"));
    session.begin_step(CapturePhase::Decode, 1).unwrap();
    session
        .observe(&mut backend, "block.output", &vec![1, 4])
        .unwrap();
    assert!(matches!(
        session.take_step().unwrap().records[0].outcome,
        CaptureOutcome::Skipped {
            reason: CaptureSkipReason::Schedule
        }
    ));
    session.begin_step(CapturePhase::Decode, 2).unwrap();
    session
        .observe(&mut backend, "block.output", &vec![1, 4])
        .unwrap();
    assert!(matches!(
        session.take_step().unwrap().records[0].outcome,
        CaptureOutcome::Skipped {
            reason: CaptureSkipReason::Limit {
                cumulative: true,
                ..
            }
        }
    ));
    session.begin_step(CapturePhase::Decode, 4).unwrap();
    assert_eq!(
        session.take_step().unwrap().records[0].outcome,
        CaptureOutcome::Missing
    );
    assert_eq!(backend.copies.get(), 1);
}

#[test]
fn dynamic_shape_is_checked_and_unknown_is_not_zero() {
    let (mut plan, mut catalog, support, caps) =
        fixture(CaptureTransform::Preview { max_elements: 2 });
    catalog.points[0].axes.as_mut().unwrap()[1].dimension = SymbolicDimension::Unknown;
    plan.selections[0].slices.push(CaptureSlice {
        axis: "hidden".into(),
        start: 0,
        end: 5,
        stride: 2,
    });
    let mut session = CaptureSession::new(admit(plan, &catalog, &support, &caps).unwrap());
    let mut backend = ProbeBackend {
        copies: Cell::new(0),
    };
    session.begin_step(CapturePhase::Prefill, 0).unwrap();
    assert!(session
        .observe(&mut backend, "block.output", &vec![3, 4])
        .is_err());
    assert_eq!(backend.copies.get(), 0);
    assert_eq!(elements(&[u64::MAX, 2]), Err(CaptureError::Overflow));
    assert!(CaptureUsage {
        host_bytes: u64::MAX,
        ..Default::default()
    }
    .checked_add(CaptureUsage {
        host_bytes: 1,
        ..Default::default()
    })
    .is_err());
}

#[test]
fn known_partial_shapes_and_sparse_cumulative_costs_are_checked_before_execution() {
    let (mut plan, mut catalog, support, caps) =
        fixture(CaptureTransform::Preview { max_elements: 2 });
    catalog.points[0].axes.as_mut().unwrap()[0].dimension = SymbolicDimension::Unknown;
    plan.selections[0].slices.push(CaptureSlice {
        axis: "hidden".into(),
        start: 0,
        end: 5,
        stride: 1,
    });
    assert!(admit(plan, &catalog, &support, &caps).is_err());

    let (mut plan, catalog, support, caps) = fixture(CaptureTransform::Preview { max_elements: 2 });
    plan.selections[0].schedule.every = 4;
    plan.limits.cumulative.retained_bytes = 299;
    let admitted = admit(plan.clone(), &catalog, &support, &caps).unwrap();
    let estimate = |_: &[u64], _: &CaptureSelection, _: &ResolvedCaptureSlice| {
        Ok(CaptureUsage {
            captures: 1,
            retained_bytes: 100,
            host_bytes: 16,
            encoded_bytes: 100,
        })
    };
    assert!(matches!(
        preflight(&admitted, estimate),
        Err(CaptureError::Limit {
            budget: CaptureBudget::Retention,
            cumulative: true,
        })
    ));
    plan.limits.cumulative.retained_bytes = 300;
    preflight(&admit(plan, &catalog, &support, &caps).unwrap(), estimate).unwrap();
    assert_eq!(elements(&[0, u64::MAX, 2]), Err(CaptureError::Overflow));
}

#[test]
fn generated_capture_reserves_before_factory_and_reuses_it_for_multiple_selections() {
    let (mut plan, catalog, support, caps) = fixture(CaptureTransform::Preview { max_elements: 2 });
    let mut second = plan.selections[0].clone();
    second.id = "second".into();
    plan.selections.push(second);
    let mut session = CaptureSession::new(admit(plan, &catalog, &support, &caps).unwrap());
    let mut backend = ProbeBackend {
        copies: Cell::new(0),
    };
    let generated = Cell::new(0);
    session.begin_step(CapturePhase::Prefill, 0).unwrap();
    session
        .observe_generated(
            &mut backend,
            "block.output",
            &vec![3, 4],
            &generated_source(64),
            &mut || {
                generated.set(generated.get() + 1);
                Ok(vec![3, 4])
            },
            &|error| error,
        )
        .unwrap();
    let step = session.take_step().unwrap();
    assert_eq!(generated.get(), 1);
    assert_eq!(backend.copies.get(), 2);
    assert!(step.records.iter().all(|record| matches!(
        record.outcome,
        CaptureOutcome::Truncated {
            available_elements: 12,
            emitted_elements: 2
        }
    ) && record.charged.retained_bytes >= 164));
}

#[test]
fn absent_skipped_failed_and_overflowed_generated_capture_never_calls_factory() {
    for case in 0..4 {
        let (mut plan, catalog, support, caps) =
            fixture(CaptureTransform::Preview { max_elements: 2 });
        if case == 1 {
            plan.limits.on_limit = CaptureLimitPolicy::Skip;
        }
        let mut session = CaptureSession::new(admit(plan, &catalog, &support, &caps).unwrap());
        let mut backend = ProbeBackend {
            copies: Cell::new(0),
        };
        let calls = Cell::new(0);
        session.begin_step(CapturePhase::Prefill, 0).unwrap();
        let result = session.observe_generated(
            &mut backend,
            if case == 0 { "absent" } else { "block.output" },
            &vec![3, 4],
            &generated_source(if case == 3 { u64::MAX } else { 1_000_000 }),
            &mut || {
                calls.set(calls.get() + 1);
                Ok(vec![3, 4])
            },
            &|error| error,
        );
        assert_eq!(calls.get(), 0);
        assert_eq!(backend.copies.get(), 0);
        let step = session.take_step().unwrap();
        assert!(step
            .records
            .iter()
            .all(|record| record.source_dtype.is_none()));
        match case {
            0 => {
                assert!(result.is_ok());
                assert_eq!(step.records[0].outcome, CaptureOutcome::Missing);
            }
            1 => {
                assert!(result.is_ok());
                assert!(matches!(
                    step.records[0].outcome,
                    CaptureOutcome::Skipped { .. }
                ));
            }
            2 => assert!(matches!(
                result,
                Err(CaptureExecutionError::Admission(CaptureError::Limit {
                    budget: CaptureBudget::Retention,
                    ..
                }))
            )),
            3 => assert!(matches!(
                result,
                Err(CaptureExecutionError::Admission(CaptureError::Overflow))
            )),
            _ => unreachable!(),
        }
    }
}

#[test]
fn generated_capture_preserves_factory_failure_and_charges_work_without_host_copy() {
    let (plan, catalog, support, caps) = fixture(CaptureTransform::Preview { max_elements: 2 });
    let mut session = CaptureSession::new(admit(plan, &catalog, &support, &caps).unwrap());
    let mut backend = ProbeBackend {
        copies: Cell::new(0),
    };
    session.begin_step(CapturePhase::Prefill, 0).unwrap();
    let failure = session.observe_generated(
        &mut backend,
        "block.output",
        &vec![3, 4],
        &generated_source(64),
        &mut || Err("exact generator failure"),
        &|_| "capture failure",
    );
    assert_eq!(failure, Err("exact generator failure"));
    assert_eq!(backend.copies.get(), 0);
    let step = session.take_step().unwrap();
    assert!(step.records[0].charged.retained_bytes >= 164);
    assert!(matches!(
        step.records[0].outcome,
        CaptureOutcome::Failed {
            reason: CaptureFailureReason::Native,
            ..
        }
    ));
}

#[test]
fn source_precision_survives_export_and_deferred_generation_without_using_prototype_dtype() {
    struct TypedProbe {
        fail: bool,
    }
    impl CaptureBackend for TypedProbe {
        type Tensor = (Vec<u64>, TensorDtype);
        type Error = std::io::Error;
        fn shape(&self, value: &Self::Tensor) -> Result<Vec<u64>, Self::Error> {
            Ok(value.0.clone())
        }
        fn source_dtype(&self, value: &Self::Tensor) -> Option<TensorDtype> {
            Some(value.1.clone())
        }
        fn estimate(
            &self,
            _: &Self::Tensor,
            _: &CaptureSelection,
            _: &ResolvedCaptureSlice,
        ) -> Result<CaptureUsage, CaptureError> {
            Ok(CaptureUsage {
                captures: 1,
                retained_bytes: 100,
                host_bytes: 16,
                encoded_bytes: 256,
            })
        }
        fn transform(
            &mut self,
            _: &Self::Tensor,
            _: &CaptureSelection,
            _: &ResolvedCaptureSlice,
        ) -> Result<CapturePayload, Self::Error> {
            if self.fail {
                return Err(std::io::Error::other("reserved transform failure"));
            }
            Ok(CapturePayload::Tensor(
                TensorObservation::new(vec![2], TensorObservationData::F32(vec![1.25, -0.5]))
                    .unwrap(),
            ))
        }
    }
    let prototype = (vec![3, 4], TensorDtype::Bf16);
    for generated in [false, true] {
        for fail in [false, true] {
            let (plan, catalog, support, caps) =
                fixture(CaptureTransform::Preview { max_elements: 2 });
            let mut session = CaptureSession::new(admit(plan, &catalog, &support, &caps).unwrap());
            let mut backend = TypedProbe { fail };
            session.begin_step(CapturePhase::Prefill, 0).unwrap();
            let result = if generated {
                session.observe_generated(
                    &mut backend,
                    "block.output",
                    &prototype,
                    &generated_source(64),
                    &mut || Ok((vec![3, 4], TensorDtype::F32)),
                    &|e| e,
                )
            } else {
                session.observe(&mut backend, "block.output", &prototype)
            };
            assert_eq!(result.is_err(), fail);
            let record = session.take_step().unwrap().records.remove(0);
            assert_eq!(
                record.source_dtype,
                Some(if generated {
                    TensorDtype::F32
                } else {
                    TensorDtype::Bf16
                })
            );
            assert!(record.charged.retained_bytes >= 100);
            if !fail {
                let Some(CapturePayload::Tensor(value)) = &record.payload else {
                    panic!("tensor payload")
                };
                assert_eq!(value.data(), &TensorObservationData::F32(vec![1.25, -0.5]));
            }
            let wire = serde_json::to_value(&record).unwrap();
            assert_eq!(
                serde_json::from_value::<CaptureRecord>(wire.clone()).unwrap(),
                record
            );
            let mut legacy = wire;
            legacy.as_object_mut().unwrap().remove("source_dtype");
            assert_eq!(
                serde_json::from_value::<CaptureRecord>(legacy)
                    .unwrap()
                    .source_dtype,
                None
            );
        }
    }
    let (mut plan, catalog, support, caps) = fixture(CaptureTransform::Preview { max_elements: 2 });
    plan.limits.on_limit = CaptureLimitPolicy::Skip;
    for generated in [false, true] {
        let mut session =
            CaptureSession::new(admit(plan.clone(), &catalog, &support, &caps).unwrap());
        session.begin_step(CapturePhase::Prefill, 0).unwrap();
        // Exhaust value retention while leaving envelope/output dimensions available.
        session
            .ledger
            .reserve(CaptureUsage {
                retained_bytes: 1_000_000,
                ..Default::default()
            })
            .unwrap();
        let mut backend = TypedProbe { fail: false };
        if generated {
            session
                .observe_generated(
                    &mut backend,
                    "block.output",
                    &prototype,
                    &generated_source(64),
                    &mut || -> Result<_, CaptureExecutionError<std::io::Error>> {
                        panic!("skipped factory")
                    },
                    &|e| e,
                )
                .unwrap();
        } else {
            session
                .observe(&mut backend, "block.output", &prototype)
                .unwrap();
        }
        let record = session.take_step().unwrap().records.remove(0);
        assert!(matches!(record.outcome, CaptureOutcome::Skipped { .. }));
        assert_eq!(
            record.source_dtype,
            if generated {
                None
            } else {
                Some(TensorDtype::Bf16)
            }
        );
    }
}

#[test]
fn transaction_records_wait_for_commit_and_restore_does_not_refund_epochs_or_usage() {
    let (plan, catalog, mut support, caps) = fixture(CaptureTransform::Preview { max_elements: 2 });
    support.capture = caps.clone();
    let admitted = admit(plan, &catalog, &support, &caps).unwrap();
    let discovery = CaptureDiscovery {
        artifact_identity: "transaction-fixture".into(),
        catalog,
        support,
    };
    let mut session = CaptureSession::new(admitted);
    let saved = session.checkpoint(&discovery).unwrap();
    let epoch = eredu_core::DistributedCommitEpoch::FIRST;
    let mut backend = ProbeBackend {
        copies: Cell::new(0),
    };
    session.begin_step(CapturePhase::Prefill, 0).unwrap();
    session
        .prepare_transaction(epoch, crate::ExpertPass::Prefill)
        .unwrap();
    session
        .observe(&mut backend, "block.output", &vec![3, 4])
        .unwrap();
    assert!(session.take_step().is_none());
    assert!(session.checkpoint(&discovery).is_err());
    session.complete_transaction(epoch).unwrap();
    assert!(
        session.take_step().is_none(),
        "completed evidence is still provisional until commit"
    );
    session.finish_transaction(epoch, false);
    let failed_usage = session.take_step().unwrap().cumulative_usage;
    assert!(session.checkpoint(&discovery).is_err());
    session.restore(&saved).unwrap();
    assert_eq!(session.cumulative_usage(), failed_usage);
    session.begin_step(CapturePhase::Prefill, 0).unwrap();
    assert!(session
        .prepare_transaction(epoch, crate::ExpertPass::Prefill)
        .is_err());
    session.finish_transaction(epoch, false);
    session.take_step().unwrap();
    assert!(session.checkpoint(&discovery).is_err());
    session.restore(&saved).unwrap();
    session.begin_step(CapturePhase::Prefill, 0).unwrap();
    let next = epoch.next().unwrap();
    session
        .prepare_transaction(next, crate::ExpertPass::Prefill)
        .unwrap();
    session
        .observe(&mut backend, "block.output", &vec![3, 4])
        .unwrap();
    session.complete_transaction(next).unwrap();
    session.finish_transaction(next, true);
    let step = session.take_step().unwrap();
    assert!(step.cumulative_usage.host_bytes > failed_usage.host_bytes);
    assert_eq!(backend.copies.get(), 2);
    session.checkpoint(&discovery).unwrap();
}
