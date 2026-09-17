use super::*;
use eredu_core::DistributedCommitEpoch;
mod host_authority;
mod invocation;
mod observer;
mod routed;
mod shared_source;

fn discovery(plan: &AdmittedCapturePlan) -> CaptureDiscovery {
    let (_, _, mut support, capabilities) = fixture(CaptureTransform::Slice);
    support.capture = capabilities;
    CaptureDiscovery {
        artifact_identity: "artifact-exact".into(),
        catalog: ObservationCatalog {
            schema_version: 1,
            completeness: DescriptionCompleteness::Complete,
            points: plan.points().to_vec(),
        },
        support,
    }
}

fn configured(plan: AdmittedCapturePlan, ranks: usize) -> CaptureSession {
    let mut session = CaptureSession::new(plan);
    session
        .configure_partition_capture(
            PartitionCaptureIdentity::new(
                "artifact-exact".into(),
                "retained-execution".into(),
                "coordinated-run".into(),
                Some("effective-overlay".into()),
                ranks,
            )
            .unwrap(),
        )
        .unwrap();
    session
}

fn maps(ranks: usize) -> Vec<(usize, ComponentCoordinateMap)> {
    if ranks == 1 {
        vec![(0, ComponentCoordinateMap::range(20, 0..20).unwrap())]
    } else {
        vec![
            (
                0,
                ComponentCoordinateMap::indices(20, vec![7, 1, 4, 2, 0]).unwrap(),
            ),
            (2, ComponentCoordinateMap::range(20, 8..20).unwrap()),
        ]
    }
}

fn producers(plan: &AdmittedCapturePlan, ranks: usize) -> Vec<PartitionCaptureProducer> {
    let slice = resolve_slice(&plan.points()[0], &plan.plan().selections[0], &[3, 20]).unwrap();
    maps(ranks)
        .into_iter()
        .map(|(rank, map)| PartitionCaptureProducer {
            rank,
            projection: CaptureSlicePartition::new(&[3, 20], &slice, 1, &map, 16).unwrap(),
        })
        .collect()
}

fn estimate(
    shape: &[u64],
    selection: &CaptureSelection,
    slice: &ResolvedCaptureSlice,
) -> Result<PartitionCaptureNativeEstimate, CaptureError> {
    let prototype = Value {
        shape: shape.to_vec(),
        data: TensorObservationData::F32(vec![]),
    };
    Ok(PartitionCaptureNativeEstimate {
        capture: Backend::default().estimate(&prototype, selection, slice)?,
        generated_creation_bytes: 4096,
    })
}

fn prepare<'a>(
    session: &mut CaptureSession,
    transport: &'a Transport,
) -> Result<SessionPartitionCapture<'a, Transport>, PartitionCaptureExchangeError> {
    let producers = producers(session.plan(), transport.participant_count());
    session.prepare_partition_capture(
        transport,
        0,
        producers,
        PartitionCaptureReceiptLimits {
            max_producers: 3,
            max_fragments: 16,
            max_record_bytes: 16384,
        },
        estimate,
    )
}

fn global() -> Value {
    Value {
        shape: vec![3, 20],
        data: TensorObservationData::F32((0..60).map(|i| i as f32 * 0.25 - 3.).collect()),
    }
}

#[test]
fn failed_live_preparation_cannot_restart_through_direct_begin_step() {
    let plan = plan_for(CaptureTransform::Slice, false);
    let catalog = discovery(&plan);
    let mut session = configured(plan.clone(), 1);
    let mut wrong = catalog.clone();
    wrong.artifact_identity = "another-loaded-artifact".into();
    assert!(session.checkpoint(&wrong).is_err());
    let saved = session.checkpoint(&catalog).unwrap();
    let epoch = DistributedCommitEpoch::FIRST;
    assert!(session
        .prepare_step_transaction(
            epoch,
            crate::ExpertPass::Prefill,
            plan.request().max_predictions
        )
        .is_err());
    let usage = session.cumulative_usage();
    assert!(session.begin_step(CapturePhase::Prefill, 0).is_err());
    assert_eq!(session.cumulative_usage(), usage);
    assert!(session.prepare_restore(&saved).is_err());
    session.finish_transaction(epoch, false);
    assert!(
        session.begin_step(CapturePhase::Prefill, 0).is_err(),
        "terminal transaction must be drained too"
    );
    assert!(session.take_step().is_none());
    session.prepare_restore(&saved).unwrap().commit();
    assert!(session
        .prepare_step_transaction(epoch, crate::ExpertPass::Prefill, 0)
        .is_err());
    session
        .prepare_step_transaction(epoch.next().unwrap(), crate::ExpertPass::Prefill, 0)
        .unwrap();
}

#[test]
fn prepaid_quotas_cannot_refund_or_reset_the_live_parent() {
    let plan = plan_for(CaptureTransform::Slice, false);
    let mut ledger = CaptureLedger::new(&plan);
    let allowance = CaptureUsage {
        captures: 2,
        retained_bytes: 200,
        host_bytes: 300,
        encoded_bytes: 400,
    };
    let mut quota = ledger.reserve_quota(allowance).unwrap();
    let mut child = quota
        .reserve_quota(CaptureUsage {
            captures: 1,
            retained_bytes: 150,
            ..Default::default()
        })
        .unwrap();
    assert_eq!(ledger.total(), allowance);
    let before = child.used();
    assert!(child
        .reserve(CaptureUsage {
            retained_bytes: 151,
            ..Default::default()
        })
        .is_err());
    assert_eq!(child.used(), before);
    child
        .reserve(CaptureUsage {
            retained_bytes: 150,
            ..Default::default()
        })
        .unwrap();
    drop(child);
    assert!(quota
        .reserve(CaptureUsage {
            retained_bytes: 51,
            ..Default::default()
        })
        .is_err());
    drop(quota);
    ledger.begin_step();
    assert_eq!(ledger.total(), allowance);
    assert_eq!(ledger.step(), CaptureUsage::default());
}

#[test]
fn live_global_capture_charges_all_ranks_and_publishes_only_after_commit() {
    for transform in [
        CaptureTransform::Slice,
        CaptureTransform::Summary,
        CaptureTransform::Histogram {
            edges: vec![-4., 0., 5., 20.],
        },
    ] {
        for (empty, generated, committed) in [
            (false, false, true),
            (false, true, true),
            (false, false, false),
            (true, false, true),
            (true, true, true),
        ] {
            let world = world(3);
            let plan = Arc::new(plan_for(transform.clone(), empty));
            let results = std::thread::scope(|scope| {
                (0..3)
                    .map(|rank| {
                        let world = Arc::clone(&world);
                        let plan = Arc::clone(&plan);
                        scope.spawn(move || {
                            let transport = transport(world, rank, Fault::None);
                            let mut session = configured((*plan).clone(), 3);
                            let epoch = DistributedCommitEpoch::FIRST;
                            session
                                .prepare_step_transaction(epoch, crate::ExpertPass::Prefill, 0)
                                .unwrap();
                            let mut work = prepare(&mut session, &transport).unwrap();
                            let global_reserved = work.global_reserved();
                            let local_reserved = work.local_reserved();
                            let prepaid = session.cumulative_usage();
                            assert!(prepare(&mut session, &transport).is_err());
                            assert_eq!(
                                session.cumulative_usage(),
                                prepaid,
                                "duplicate admission cannot consume another quota"
                            );
                            let coordination =
                                session.prepare_partition_coordination(&transport).unwrap();
                            session.coordinate_partition_capture(coordination).unwrap();
                            let prepaid = session.cumulative_usage();
                            assert!(
                                session.complete_transaction(epoch).is_err(),
                                "receipt is not delivered yet"
                            );
                            let mut backend = Backend::default();
                            let mut generated_calls = 0;
                            let local = maps(3)
                                .into_iter()
                                .find(|(producer, _)| *producer == rank)
                                .map(|(_, map)| local_value(&global(), &map))
                                .unwrap_or(Value {
                                    shape: vec![999],
                                    data: TensorObservationData::F32(vec![]),
                                });
                            if generated {
                                let prototype = Value {
                                    shape: local.shape.clone(),
                                    data: TensorObservationData::I64(vec![]),
                                };
                                session
                                    .observe_generated_partition(
                                        &mut work,
                                        &mut backend,
                                        &prototype,
                                        &GeneratedCaptureSource {
                                            creation_bytes: 4096,
                                            source_dtype: Some(TensorDtype::F32),
                                        },
                                        &mut || {
                                            generated_calls += 1;
                                            Ok(local.clone())
                                        },
                                        &|error| error,
                                    )
                                    .unwrap();
                            } else {
                                session
                                    .observe_partition(&mut work, &mut backend, &local)
                                    .unwrap();
                            }
                            let transforms = backend.transforms;
                            assert!(session
                                .observe_partition(&mut work, &mut backend, &local)
                                .is_err());
                            assert_eq!(backend.transforms, transforms);
                            assert_eq!(
                                session.cumulative_usage(),
                                prepaid,
                                "native work spends prepaid credits, never a replacement ledger"
                            );
                            assert!(session.take_step().is_none());
                            session.complete_partition_capture(work).unwrap();
                            assert!(
                                session.take_step().is_none(),
                                "global values are provisional before final commit"
                            );
                            session.complete_transaction(epoch).unwrap();
                            session.finish_transaction(epoch, committed);
                            let step = session.take_step().unwrap();
                            assert_eq!(step.cumulative_usage, prepaid);
                            assert_eq!(step.partitions.len(), usize::from(committed));
                            if committed {
                                assert_eq!(step.records[0].outcome, CaptureOutcome::Captured);
                                let evidence = &step.partitions[0];
                                assert_eq!(evidence.context.forward_epoch, epoch.value());
                                assert_eq!(evidence.context.run_identity, "coordinated-run");
                                assert_eq!(evidence.context.capture_plan_identity, plan.identity());
                                assert_eq!(evidence.producers, [0, 2]);
                                assert_eq!(evidence.contributions.is_empty(), empty);
                                let json = serde_json::to_vec(&step).unwrap();
                                assert_eq!(
                                    serde_json::from_slice::<CapturedStep>(&json).unwrap(),
                                    step
                                );
                            } else {
                                assert_eq!(step.records[0].outcome, CaptureOutcome::Missing);
                                assert!(step.records[0].payload.is_none());
                            }
                            if rank == 1 || empty {
                                assert_eq!((transforms, generated_calls), (0, 0));
                            }
                            if committed {
                                assert_eq!(step.records[0].source_dtype, Some(TensorDtype::F32));
                            }
                            (step, global_reserved, local_reserved)
                        })
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
                    .map(|worker| worker.join().unwrap())
                    .collect::<Vec<_>>()
            });
            for (step, global_reserved, _) in &results {
                assert_eq!(step.cumulative_usage, results[0].0.cumulative_usage);
                assert_eq!(*global_reserved, results[0].1);
            }
            let local_total = results
                .iter()
                .try_fold(CaptureUsage::default(), |total, (_, _, local)| {
                    total.checked_add(*local)
                })
                .unwrap();
            assert!(local_total.exceeded(results[0].1).is_none());
            if committed {
                let mut ordinary = CaptureSession::new((*plan).clone());
                ordinary.begin_step(CapturePhase::Prefill, 0).unwrap();
                ordinary
                    .observe(&mut Backend::default(), "block.output", &global())
                    .unwrap();
                let expected = ordinary.take_step().unwrap();
                for (step, ..) in results {
                    assert_eq!(step.records[0].payload, expected.records[0].payload);
                }
            }
        }
    }
}

#[test]
fn live_partition_authority_rejects_foreign_owners_and_restore_replay_before_work() {
    let plan = plan_for(CaptureTransform::Slice, false);
    let catalog = discovery(&plan);
    let mut session = configured(plan.clone(), 1);
    let saved = session.checkpoint(&catalog).unwrap();
    let transport = transport(world(1), 0, Fault::None);
    let epoch = DistributedCommitEpoch::FIRST;
    session
        .prepare_step_transaction(epoch, crate::ExpertPass::Prefill, 0)
        .unwrap();
    let mut work = prepare(&mut session, &transport).unwrap();
    let consumed = session.cumulative_usage();
    let mut foreign = configured(plan, 1);
    foreign
        .prepare_step_transaction(epoch, crate::ExpertPass::Prefill, 0)
        .unwrap();
    let mut backend = Backend::default();
    assert!(foreign
        .observe_partition(&mut work, &mut backend, &global())
        .is_err());
    assert_eq!(backend.transforms, 0);
    assert!(
        session
            .observe(&mut backend, "block.output", &global())
            .is_err(),
        "ordinary local capture cannot bypass partition authority"
    );
    assert_eq!(backend.transforms, 0);
    session.finish_transaction(epoch, false);
    session.take_step().unwrap();
    session.restore(&saved).unwrap();
    assert_eq!(session.cumulative_usage(), consumed);
    let next = epoch.next().unwrap();
    session
        .prepare_step_transaction(next, crate::ExpertPass::Prefill, 0)
        .unwrap();
    assert!(session
        .observe_partition(&mut work, &mut backend, &global())
        .is_err());
    assert!(session.complete_partition_capture(work).is_err());
    assert_eq!(transport.calls.load(Ordering::SeqCst), 0);
    let mut fresh = prepare(&mut session, &transport).unwrap();
    let coordination = session.prepare_partition_coordination(&transport).unwrap();
    session.coordinate_partition_capture(coordination).unwrap();
    session
        .observe_partition(&mut fresh, &mut backend, &global())
        .unwrap();
    session.complete_partition_capture(fresh).unwrap();
    session.complete_transaction(next).unwrap();
    session.finish_transaction(next, true);
    let step = session.take_step().unwrap();
    assert_eq!(step.partitions[0].context.forward_epoch, next.value());
    assert!(step.cumulative_usage.host_bytes > consumed.host_bytes);
}

#[test]
fn global_admission_rejects_a_budget_that_would_cover_only_one_rank() {
    let plan = plan_for(CaptureTransform::Slice, false);
    let transport = transport(world(3), 0, Fault::None);
    let epoch = DistributedCommitEpoch::FIRST;
    let mut probe = configured(plan.clone(), 3);
    probe
        .prepare_step_transaction(epoch, crate::ExpertPass::Prefill, 0)
        .unwrap();
    let work = prepare(&mut probe, &transport).unwrap();
    let limit = (work.local_reserved().host_bytes + work.global_reserved().host_bytes) / 2;
    assert!(limit > work.local_reserved().host_bytes);
    assert!(limit < work.global_reserved().host_bytes);
    let discovery = discovery(&plan);
    let mut limited = plan.plan().clone();
    limited.limits.per_step.host_bytes = limit;
    limited.limits.cumulative.host_bytes = limit;
    let limited = limited
        .admit(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            plan.request(),
        )
        .unwrap();
    let mut session = configured(limited, 3);
    session
        .prepare_step_transaction(epoch, crate::ExpertPass::Prefill, 0)
        .unwrap();
    assert!(matches!(
        prepare(&mut session, &transport),
        Err(PartitionCaptureExchangeError::Capture(
            CaptureError::Limit {
                budget: CaptureBudget::Host,
                ..
            }
        ))
    ));
    assert_eq!(transport.calls.load(Ordering::SeqCst), 0);
    assert!(
        session.cumulative_usage().host_bytes > 0,
        "cold preparation remains charged"
    );
}

#[test]
fn insufficient_local_factory_allowance_creates_no_tensor_or_global_value() {
    let plan = plan_for(CaptureTransform::Slice, false);
    let transport = transport(world(1), 0, Fault::None);
    let epoch = DistributedCommitEpoch::FIRST;
    let mut session = configured(plan, 1);
    session
        .prepare_step_transaction(epoch, crate::ExpertPass::Prefill, 0)
        .unwrap();
    let producers = producers(session.plan(), 1);
    let mut work = session
        .prepare_partition_capture(
            &transport,
            0,
            producers,
            PartitionCaptureReceiptLimits {
                max_producers: 1,
                max_fragments: 16,
                max_record_bytes: 16384,
            },
            |shape, selection, slice| {
                let mut native = estimate(shape, selection, slice)?;
                native.capture.retained_bytes = 0;
                native.generated_creation_bytes = 0;
                Ok(native)
            },
        )
        .unwrap();
    let coordination = session.prepare_partition_coordination(&transport).unwrap();
    session.coordinate_partition_capture(coordination).unwrap();
    let prepaid = session.cumulative_usage();
    let mut calls = 0;
    let mut backend = Backend::default();
    let result = session.observe_generated_partition(
        &mut work,
        &mut backend,
        &global(),
        &generated_source(1),
        &mut || {
            calls += 1;
            Ok(global())
        },
        &|error| error,
    );
    assert!(matches!(
        result,
        Err(CaptureExecutionError::Admission(CaptureError::Limit {
            budget: CaptureBudget::Retention,
            ..
        }))
    ));
    assert_eq!((calls, backend.transforms), (0, 0));
    assert!(matches!(
        session.complete_partition_capture(work),
        Err(PartitionCaptureExchangeError::LocalRejected {
            stage: PartitionCaptureExchangeStage::Preparation,
            ..
        })
    ));
    assert!(session.complete_transaction(epoch).is_err());
    session.finish_transaction(epoch, false);
    let step = session.take_step().unwrap();
    assert_eq!(step.cumulative_usage, prepaid);
    assert!(step.partitions.is_empty());
    assert!(step.records[0].payload.is_none());
}

#[test]
fn common_coordination_rejects_different_runs_quotas_or_selection_sets_before_sources() {
    for scenario in [0, 1, 2] {
        let world = world(3);
        let plan = Arc::new(plan_for(CaptureTransform::Slice, false));
        let results = std::thread::scope(|scope| {
            (0..3)
                .map(|rank| {
                    let world = Arc::clone(&world);
                    let plan = Arc::clone(&plan);
                    scope.spawn(move || {
                        let transport = transport(world, rank, Fault::None);
                        let mut session = CaptureSession::new((*plan).clone());
                        session
                            .configure_partition_capture(
                                PartitionCaptureIdentity::new(
                                    "artifact-exact".into(),
                                    "retained-execution".into(),
                                    if rank == 0 && scenario == 0 {
                                        "another-run"
                                    } else {
                                        "coordinated-run"
                                    }
                                    .into(),
                                    None,
                                    3,
                                )
                                .unwrap(),
                            )
                            .unwrap();
                        let epoch = DistributedCommitEpoch::FIRST;
                        session
                            .prepare_step_transaction(epoch, crate::ExpertPass::Prefill, 0)
                            .unwrap();
                        let mut work = if rank == 0 && scenario == 2 {
                            None
                        } else {
                            let producers = producers(session.plan(), 3);
                            Some(
                                session
                                    .prepare_partition_capture(
                                        &transport,
                                        0,
                                        producers,
                                        PartitionCaptureReceiptLimits {
                                            max_producers: 3,
                                            max_fragments: 16,
                                            max_record_bytes: 16384,
                                        },
                                        |shape, selection, slice| {
                                            let mut cost = estimate(shape, selection, slice)?;
                                            if rank == 0 && scenario == 1 {
                                                cost.generated_creation_bytes += 512;
                                            }
                                            Ok(cost)
                                        },
                                    )
                                    .unwrap(),
                            )
                        };
                        let coordination =
                            session.prepare_partition_coordination(&transport).unwrap();
                        assert!(
                            prepare(&mut session, &transport).is_err(),
                            "frozen preparation cannot add another authority"
                        );
                        let result = session.coordinate_partition_capture(coordination);
                        assert!(matches!(
                            result,
                            Err(PartitionCaptureExchangeError::PeerRejected {
                                stage: PartitionCaptureExchangeStage::Coordination,
                                ..
                            })
                        ));
                        let mut backend = Backend::default();
                        if let Some(work) = &mut work {
                            assert!(session
                                .observe_partition(work, &mut backend, &global())
                                .is_err());
                        }
                        assert_eq!(backend.transforms, 0);
                        assert_eq!(transport.calls.load(Ordering::SeqCst), 1);
                        assert!(
                            !transport.poison.load(Ordering::SeqCst),
                            "completed common rejection preserves healthy transport"
                        );
                        session.finish_transaction(epoch, false);
                        let step = session.take_step().unwrap();
                        assert!(step.partitions.is_empty());
                        assert!(step.records[0].payload.is_none());
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|worker| worker.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert_eq!(results.len(), 3);
    }
}

#[test]
fn producer_authority_cannot_bypass_common_coordination() {
    let plan = plan_for(CaptureTransform::Slice, false);
    let transport = transport(world(1), 0, Fault::None);
    let mut session = configured(plan, 1);
    let epoch = DistributedCommitEpoch::FIRST;
    session
        .prepare_step_transaction(epoch, crate::ExpertPass::Prefill, 0)
        .unwrap();
    let mut work = prepare(&mut session, &transport).unwrap();
    let mut backend = Backend::default();
    assert!(session
        .observe_partition(&mut work, &mut backend, &global())
        .is_err());
    assert_eq!(backend.transforms, 0);
    assert_eq!(transport.calls.load(Ordering::SeqCst), 0);
    let coordination = session.prepare_partition_coordination(&transport).unwrap();
    assert!(session.prepare_partition_coordination(&transport).is_err());
    session.coordinate_partition_capture(coordination).unwrap();
    session
        .observe_partition(&mut work, &mut backend, &global())
        .unwrap();
    session.complete_partition_capture(work).unwrap();
    session.complete_transaction(epoch).unwrap();
    session.finish_transaction(epoch, true);
    assert_eq!(session.take_step().unwrap().partitions.len(), 1);
    assert_eq!(transport.calls.load(Ordering::SeqCst), 4);
}

#[test]
fn coordination_completion_failures_never_resolve_or_enable_sources() {
    for fault in [
        Fault::Wait(0),
        Fault::Deadline(0),
        Fault::Submit(0),
        Fault::Resolve(0),
    ] {
        let plan = plan_for(CaptureTransform::Slice, false);
        let transport = transport(world(1), 0, fault);
        let mut session = configured(plan, 1);
        let epoch = DistributedCommitEpoch::FIRST;
        session
            .prepare_step_transaction(epoch, crate::ExpertPass::Prefill, 0)
            .unwrap();
        let mut work = prepare(&mut session, &transport).unwrap();
        let coordination = session.prepare_partition_coordination(&transport).unwrap();
        assert!(session.coordinate_partition_capture(coordination).is_err());
        assert!(transport.poison.load(Ordering::SeqCst));
        if !matches!(fault, Fault::Resolve(_)) {
            assert_eq!(transport.resolves.load(Ordering::SeqCst), 0);
        }
        let mut backend = Backend::default();
        assert!(session
            .observe_partition(&mut work, &mut backend, &global())
            .is_err());
        assert_eq!(backend.transforms, 0);
        assert!(session.complete_transaction(epoch).is_err());
    }
}

#[test]
fn empty_deferred_producers_must_agree_declared_precision() {
    let all = world(3);
    std::thread::scope(|scope| {
        let workers = (0..3)
            .map(|rank| {
                let all = Arc::clone(&all);
                scope.spawn(move || {
                    let transport = transport(all, rank, Fault::None);
                    let mut session = configured(plan_for(CaptureTransform::Slice, true), 3);
                    let epoch = DistributedCommitEpoch::FIRST;
                    session
                        .prepare_step_transaction(epoch, crate::ExpertPass::Prefill, 0)
                        .unwrap();
                    let mut work = prepare(&mut session, &transport).unwrap();
                    let coordination = session.prepare_partition_coordination(&transport).unwrap();
                    session.coordinate_partition_capture(coordination).unwrap();
                    let prototype = maps(3)
                        .into_iter()
                        .find(|(producer, _)| *producer == rank)
                        .map(|(_, map)| local_value(&global(), &map))
                        .unwrap_or(global());
                    let source = GeneratedCaptureSource {
                        creation_bytes: 4096,
                        source_dtype: Some(if rank == 2 {
                            TensorDtype::U64
                        } else {
                            TensorDtype::F32
                        }),
                    };
                    session
                        .observe_generated_partition(
                            &mut work,
                            &mut Backend::default(),
                            &prototype,
                            &source,
                            &mut || -> Result<Value, CaptureExecutionError<std::io::Error>> {
                                panic!("empty factory")
                            },
                            &|error| error,
                        )
                        .unwrap();
                    assert!(session.complete_partition_capture(work).is_err());
                    session.finish_transaction(epoch, false);
                    let step = session.take_step().unwrap();
                    assert_eq!(step.records[0].outcome, CaptureOutcome::Missing);
                    assert!(step.partitions.is_empty());
                    assert!(!transport.poison.load(Ordering::SeqCst));
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker.join().unwrap();
        }
    });
}

#[test]
fn empty_generated_shard_acknowledges_precision_without_losing_global_values() {
    let all = world(3);
    let steps = std::thread::scope(|scope| {
        let workers = (0..3)
            .map(|rank| {
                let all = Arc::clone(&all);
                scope.spawn(move || {
                    let transport = transport(all, rank, Fault::None);
                    let plan = plan_for(CaptureTransform::Slice, false);
                    let maps = maps_with_idle_producer();
                    let slice =
                        resolve_slice(&plan.points()[0], &plan.plan().selections[0], &[3, 20])
                            .unwrap();
                    let producers = maps
                        .iter()
                        .enumerate()
                        .map(|(rank, map)| PartitionCaptureProducer {
                            rank,
                            projection: CaptureSlicePartition::new(&[3, 20], &slice, 1, map, 16)
                                .unwrap(),
                        })
                        .collect();
                    let mut session = configured(plan, 3);
                    let epoch = DistributedCommitEpoch::FIRST;
                    session
                        .prepare_step_transaction(epoch, crate::ExpertPass::Prefill, 0)
                        .unwrap();
                    let mut work = session
                        .prepare_partition_capture(
                            &transport,
                            0,
                            producers,
                            PartitionCaptureReceiptLimits {
                                max_producers: 3,
                                max_fragments: 16,
                                max_record_bytes: 16_384,
                            },
                            estimate,
                        )
                        .unwrap();
                    let coordination = session.prepare_partition_coordination(&transport).unwrap();
                    session.coordinate_partition_capture(coordination).unwrap();
                    let local = local_value(&global(), &maps[rank]);
                    let prototype = Value {
                        shape: local.shape.clone(),
                        data: TensorObservationData::I64(vec![]),
                    };
                    let source = GeneratedCaptureSource {
                        creation_bytes: 4096,
                        source_dtype: Some(TensorDtype::F32),
                    };
                    let mut calls = 0;
                    let mut backend = Backend::default();
                    session
                        .observe_generated_partition(
                            &mut work,
                            &mut backend,
                            &prototype,
                            &source,
                            &mut || {
                                calls += 1;
                                Ok(local.clone())
                            },
                            &|error| error,
                        )
                        .unwrap();
                    session.complete_partition_capture(work).unwrap();
                    session.complete_transaction(epoch).unwrap();
                    session.finish_transaction(epoch, true);
                    assert_eq!(calls > 0, rank != 1);
                    assert_eq!(backend.transforms > 0, rank != 1);
                    let step = session.take_step().unwrap();
                    assert_eq!(step.records[0].source_dtype, Some(TensorDtype::F32));
                    assert_eq!(step.records[0].outcome, CaptureOutcome::Captured);
                    assert_eq!(step.partitions[0].producers, [0, 1, 2]);
                    step
                })
            })
            .collect::<Vec<_>>();
        workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>()
    });
    let mut ordinary = CaptureSession::new(plan_for(CaptureTransform::Slice, false));
    ordinary.begin_step(CapturePhase::Prefill, 0).unwrap();
    ordinary
        .observe(&mut Backend::default(), "block.output", &global())
        .unwrap();
    let expected = ordinary.take_step().unwrap();
    assert!(steps.iter().all(
        |step| step.records[0].payload == expected.records[0].payload
            && step.cumulative_usage == steps[0].cumulative_usage
    ));
}
