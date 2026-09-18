//! Actual invocation axes survive distributed admission, delivery and restoration.
use super::*;

fn invocation_plan(transform: CaptureTransform) -> AdmittedCapturePlan {
    let (mut plan, mut catalog, support, capabilities) = fixture(transform);
    catalog.points[0].axes.as_mut().unwrap()[1].dimension = SymbolicDimension::Known(20);
    plan.selections[0].slices = vec![CaptureSlice {
        axis: "hidden".into(),
        start: 1,
        end: 20,
        stride: 3,
    }];
    plan.limits.per_step = CaptureUsage {
        captures: 1000,
        retained_bytes: 32_000_000,
        host_bytes: 32_000_000,
        encoded_bytes: 32_000_000,
    };
    plan.limits.cumulative = plan.limits.per_step.checked_mul(16).unwrap();
    plan.admit_invocations(
        &catalog,
        &support,
        &capabilities,
        CaptureInvocationBounds {
            batch: 1,
            max_sequence: 5,
            max_context: Some(32),
            max_predictions: 8,
        },
    )
    .unwrap()
}

fn shape(rows: u64) -> CaptureInvocationShape {
    CaptureInvocationShape {
        batch: 1,
        sequence: rows,
        context: Some(19),
    }
}

fn invocation_producers(plan: &AdmittedCapturePlan, rows: u64) -> Vec<PartitionCaptureProducer> {
    let global = [rows, 20];
    let slice = resolve_slice(&plan.points()[0], &plan.plan().selections[0], &global).unwrap();
    maps(3)
        .into_iter()
        .map(|(rank, map)| PartitionCaptureProducer {
            rank,
            projection: CaptureSlicePartition::new(&global, &slice, 1, &map, 16).unwrap(),
        })
        .collect()
}

#[test]
fn invocation_receipts_preserve_physical_axes_and_monotone_replay_budgets() {
    for transform in [
        CaptureTransform::Slice,
        CaptureTransform::Summary,
        CaptureTransform::Histogram {
            edges: vec![-4., 0., 5., 20.],
        },
    ] {
        let world = world(3);
        let plan = Arc::new(invocation_plan(transform));
        let results = std::thread::scope(|scope| {
            (0..3)
                .map(|rank| {
                    let world = Arc::clone(&world);
                    let plan = Arc::clone(&plan);
                    scope.spawn(move || {
                        let transport = transport(world, rank, Fault::None);
                        let mut session = configured((*plan).clone(), 3);
                        let saved = session.checkpoint(&discovery(&plan)).unwrap();
                        let mut epoch = DistributedCommitEpoch::FIRST;
                        let mut previous = CaptureUsage::default();
                        let mut outputs = Vec::new();
                        // All decode invocations inspect prediction 4, while their physical
                        // widths and native values differ. One is discarded before publication.
                        for (index, (rows, committed)) in
                            [(5, true), (2, true), (1, false), (5, true)]
                                .into_iter()
                                .enumerate()
                        {
                            if index == 3 {
                                let used = session.cumulative_usage();
                                session.restore(&saved).unwrap();
                                assert_eq!(session.cumulative_usage(), used);
                            }
                            let invocation = shape(rows);
                            session
                                .begin_invocation(
                                    CapturePhase::Decode,
                                    4,
                                    invocation,
                                    CaptureInvocationSelection::default(),
                                )
                                .unwrap();
                            session
                                .prepare_transaction(epoch, crate::ExpertPass::Decode)
                                .unwrap();
                            let mut work = session
                                .prepare_partition_capture(
                                    &transport,
                                    0,
                                    invocation_producers(&plan, rows),
                                    PartitionCaptureReceiptLimits {
                                        max_producers: 3,
                                        max_fragments: 16,
                                        max_record_bytes: 16384,
                                    },
                                    estimate,
                                )
                                .unwrap();
                            let coordination =
                                session.prepare_partition_coordination(&transport).unwrap();
                            session.coordinate_partition_capture(coordination).unwrap();
                            let global = Value {
                                shape: vec![rows, 20],
                                data: TensorObservationData::F32(
                                    (0..rows * 20).map(|i| i as f32 * 0.25 - 3.).collect(),
                                ),
                            };
                            let local = maps(3)
                                .into_iter()
                                .find(|(owner, _)| *owner == rank)
                                .map(|(_, map)| local_value(&global, &map))
                                .unwrap_or(Value {
                                    shape: vec![999],
                                    data: TensorObservationData::F32(vec![]),
                                });
                            let mut backend = Backend::default();
                            session
                                .observe_partition(&mut work, &mut backend, &local)
                                .unwrap();
                            assert_eq!(backend.transforms == 0, rank == 1);
                            assert!(session.take_step().is_none());
                            session.complete_partition_capture(work).unwrap();
                            assert!(session.take_step().is_none());
                            session.complete_transaction(epoch).unwrap();
                            session.finish_transaction(epoch, committed);
                            let output = session.take_step().unwrap();
                            assert_eq!(output.partitions.len(), usize::from(committed));
                            if committed {
                                let wire = serde_json::to_vec(&output.partitions[0]).unwrap();
                                let decoded: PartitionCaptureEvidence =
                                    serde_json::from_slice(&wire).unwrap();
                                assert_eq!(decoded, output.partitions[0]);
                                let context = &decoded.context;
                                assert_eq!(context.invocation, Some(invocation));
                                assert_eq!(context.prediction, 4);
                                assert_eq!(context.forward_epoch, epoch.value());
                                assert_eq!(context.capture_plan_identity, plan.identity());
                                let mut ordinary = CaptureSession::new(eredu_core::capture::SharedCapturePlan::new((*plan).clone()));
                                ordinary
                                    .begin_invocation(
                                        CapturePhase::Decode,
                                        4,
                                        invocation,
                                        CaptureInvocationSelection::default(),
                                    )
                                    .unwrap();
                                ordinary
                                    .observe(&mut Backend::default(), "block.output", &global)
                                    .unwrap();
                                let expected = ordinary.take_step().unwrap();
                                assert_reduced_close(
                                    output.records[0].payload.as_ref().unwrap(),
                                    expected.records[0].payload.as_ref().unwrap(),
                                );
                            } else {
                                assert!(output.records[0].payload.is_none());
                            }
                            assert!(output.cumulative_usage.host_bytes > previous.host_bytes);
                            previous = output.cumulative_usage;
                            assert!(session
                                .prepare_transaction(epoch, crate::ExpertPass::Decode)
                                .is_err());
                            epoch = epoch.next().unwrap();
                            outputs.push(output);
                        }
                        assert_eq!(outputs[0].records[0].payload, outputs[3].records[0].payload);
                        outputs
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|worker| worker.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert_eq!(results[0], results[1]);
        assert_eq!(results[0], results[2]);
    }
}

#[test]
fn mismatched_invocation_axes_reject_all_ranks_before_native_transform() {
    // The context axis is bound even when absent from the selected point shape.
    for context_only in [false, true] {
        let world = world(3);
        let plan = Arc::new(invocation_plan(CaptureTransform::Slice));
        std::thread::scope(|scope| {
            let workers = (0..3)
                .map(|rank| {
                    let world = Arc::clone(&world);
                    let plan = Arc::clone(&plan);
                    scope.spawn(move || {
                        let transport = transport(world, rank, Fault::None);
                        let mut session = configured((*plan).clone(), 3);
                        let rows = if rank == 1 && !context_only { 2 } else { 5 };
                        let mut invocation = shape(rows);
                        if rank == 1 && context_only {
                            invocation.context = Some(20);
                        }
                        session
                            .begin_invocation(
                                CapturePhase::Decode,
                                4,
                                invocation,
                                CaptureInvocationSelection::default(),
                            )
                            .unwrap();
                        let epoch = DistributedCommitEpoch::FIRST;
                        session
                            .prepare_transaction(epoch, crate::ExpertPass::Decode)
                            .unwrap();
                        let _work = session
                            .prepare_partition_capture(
                                &transport,
                                0,
                                invocation_producers(&plan, rows),
                                PartitionCaptureReceiptLimits {
                                    max_producers: 3,
                                    max_fragments: 16,
                                    max_record_bytes: 16384,
                                },
                                estimate,
                            )
                            .unwrap();
                        let coordination =
                            session.prepare_partition_coordination(&transport).unwrap();
                        assert!(session.coordinate_partition_capture(coordination).is_err());
                        session.finish_transaction(epoch, false);
                        let output = session.take_step().unwrap();
                        assert!(output.partitions.is_empty());
                        assert!(output.records[0].payload.is_none());
                        assert!(output.cumulative_usage.host_bytes > 0);
                    })
                })
                .collect::<Vec<_>>();
            for worker in workers {
                worker.join().unwrap();
            }
        });
    }
}

#[test]
fn invalid_invocation_fragment_never_calls_a_deferred_factory() {
    let plan = invocation_plan(CaptureTransform::Slice);
    let global = Value {
        shape: vec![5, 20],
        data: TensorObservationData::F32(vec![2.; 100]),
    };
    let map = ComponentCoordinateMap::range(20, 0..20).unwrap();
    let slice =
        resolve_slice(&plan.points()[0], &plan.plan().selections[0], &global.shape).unwrap();
    let projection = CaptureSlicePartition::new(&global.shape, &slice, 1, &map, 16).unwrap();
    for invocation in [None, Some(shape(2)), Some(shape(6))] {
        let mut calls = 0;
        let mut backend = Backend::default();
        let mut ledger = CaptureLedger::new(&plan);
        let error = capture_generated_fragment(
            &mut backend,
            &global,
            &GeneratedCaptureSource {
                creation_bytes: 4096,
                source_dtype: Some(TensorDtype::F32),
            },
            &mut || {
                calls += 1;
                Ok::<_, CaptureExecutionError<std::io::Error>>(global.clone())
            },
            PartitionCaptureRequest {
                plan: &plan,
                selection_index: 0,
                phase: CapturePhase::Decode,
                prediction: 4,
                invocation,
                projection: &projection,
                fragment_index: 0,
                producer_rank: 0,
            },
            &mut ledger,
            &|error| error,
        );
        assert!(matches!(error, Err(CaptureExecutionError::Admission(_))));
        assert_eq!((calls, backend.transforms), (0, 0));
        assert_eq!(ledger.total(), CaptureUsage::default());
    }
}
