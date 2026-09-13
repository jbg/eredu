use super::*;

pub(super) fn skip_plan(
    mixed: bool,
    adjust: impl FnOnce(&mut CaptureLimits),
) -> AdmittedCapturePlan {
    let admitted = plan_for(CaptureTransform::Slice, false);
    let discovery = discovery(&admitted);
    let mut plan = admitted.plan().clone();
    plan.selections[0].id = "expensive".into();
    if mixed {
        let mut second = plan.selections[0].clone();
        second.id = "ordinary".into();
        plan.selections.push(second);
    }
    plan.limits.on_limit = CaptureLimitPolicy::Skip;
    plan.limits.per_step = CaptureUsage {
        captures: 10_000,
        retained_bytes: 32 << 20,
        host_bytes: 32 << 20,
        encoded_bytes: 32 << 20,
    };
    plan.limits.cumulative = plan.limits.per_step.checked_mul(8).unwrap();
    adjust(&mut plan.limits);
    plan.admit(
        &discovery.catalog,
        &discovery.support,
        &discovery.support.capture,
        admitted.request(),
    )
    .unwrap()
}

struct RepeatedLayout(Layout);
impl PartitionCaptureLayout for RepeatedLayout {
    fn capture_placement(
        &self,
        plan: &AdmittedCapturePlan,
        _: usize,
        phase: CapturePhase,
        prediction: u64,
        limits: PartitionCaptureReceiptLimits,
    ) -> Result<PartitionCapturePlacement, CaptureError> {
        self.0.capture_placement(plan, 0, phase, prediction, limits)
    }
}

#[test]
fn coordinated_skip_keeps_other_values_and_rejects_rank_disagreement() {
    for (mixed, dissent, committed) in [
        (true, false, true),
        (false, false, true),
        (true, false, false),
        (true, true, false),
    ] {
        let all = world(4);
        let hooks = world(3);
        let steps = std::thread::scope(|scope| {
            (0..4)
                .map(|rank| {
                    let all = Arc::clone(&all);
                    let hooks = Arc::clone(&hooks);
                    scope.spawn(move || {
                        let members = vec![0, 2, 3];
                        let transport = HookTransport {
                            transport: transport(all, rank, Fault::None),
                            hook: members
                                .iter()
                                .position(|member| *member == rank)
                                .map(|local| transport(hooks, local, Fault::None)),
                            members: members.clone(),
                        };
                        let layout = RepeatedLayout(Layout { ranks: 4, members });
                        let mut session = configured(skip_plan(mixed, |_| {}), 4);
                        let local = local_source(4, rank);
                        let epoch = DistributedCommitEpoch::FIRST;
                        let mut factory_calls = 0;
                        let mut observer = PartitionCaptureObserver::for_step(
                            &mut session,
                            Backend::default(),
                            &transport,
                            &layout,
                            0,
                            LIMITS,
                            |shape: &[u64],
                             selection: &CaptureSelection,
                             slice: &ResolvedCaptureSlice| {
                                let mut cost = estimate(shape, selection, slice)?;
                                if selection.id == "expensive" && !(dissent && rank == 2) {
                                    cost.capture.retained_bytes = 1 << 40;
                                }
                                Ok(cost)
                            },
                            |error| error,
                        );
                        observer
                            .prepare_transaction(epoch, crate::ExpertPass::Prefill)
                            .unwrap();
                        assert_eq!(transport.transport.calls.load(Ordering::SeqCst), 0);
                        let agreed = observer.coordinate_transaction(epoch);
                        if dissent {
                            assert!(matches!(
                                agreed,
                                Err(PartitionCaptureObserverError::Exchange(
                                    PartitionCaptureExchangeError::PeerRejected {
                                        stage: PartitionCaptureExchangeStage::Coordination,
                                        ..
                                    }
                                ))
                            ));
                        } else {
                            agreed.unwrap();
                            if rank != 1 {
                                observer
                                    .observe_generated(
                                        "block.output",
                                        &local,
                                        &generated_source(4096),
                                        &mut || {
                                            factory_calls += 1;
                                            Ok(local.clone())
                                        },
                                    )
                                    .unwrap();
                            }
                            observer.complete_transaction(epoch).unwrap();
                        }
                        observer.finish_transaction(epoch, committed);
                        drop(observer);
                        assert!(!transport.transport.poison.load(Ordering::SeqCst));
                        assert_eq!(
                            factory_calls > 0,
                            mixed && !dissent && [0, 2].contains(&rank)
                        );
                        assert_eq!(
                            transport
                                .hook
                                .as_ref()
                                .map_or(0, |hook| hook.calls.load(Ordering::SeqCst)),
                            2 * usize::from(mixed && !dissent && rank != 1)
                        );
                        let step = session.take_step().unwrap();
                        assert_eq!(step.partitions.len(), usize::from(mixed && committed));
                        if committed {
                            assert_eq!(
                                step.records[0].outcome,
                                CaptureOutcome::Skipped {
                                    reason: CaptureSkipReason::Limit {
                                        budget: CaptureBudget::Retention,
                                        cumulative: false
                                    }
                                }
                            );
                            assert!(step.records[0].payload.is_none());
                            assert!(
                                step.records[0].charged.host_bytes > 2048,
                                "preparation charges survive skipping"
                            );
                            if mixed {
                                let mut ordinary =
                                    CaptureSession::new(plan_for(CaptureTransform::Slice, false));
                                ordinary.begin_step(CapturePhase::Prefill, 0).unwrap();
                                ordinary
                                    .observe(&mut Backend::default(), "block.output", &global())
                                    .unwrap();
                                assert_eq!(
                                    step.records[1].payload,
                                    ordinary.take_step().unwrap().records[0].payload
                                );
                            }
                        } else {
                            assert!(
                                step.records
                                    .iter()
                                    .all(|record| record.outcome == CaptureOutcome::Missing
                                        && record.payload.is_none()),
                                "aborted local choices never claim a globally committed outcome"
                            );
                        }
                        step
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|worker| worker.join().unwrap())
                .collect::<Vec<_>>()
        });
        if !dissent {
            assert!(steps
                .iter()
                .all(|step| step.cumulative_usage == steps[0].cumulative_usage));
        }
    }
}

fn singleton() -> HookTransport {
    HookTransport {
        transport: transport(world(1), 0, Fault::None),
        hook: None,
        members: vec![0],
    }
}

#[test]
fn skip_releases_partial_claim_without_refunding_reserved_work() {
    let transport = singleton();
    let generous = skip_plan(false, |_| {});
    let mut priced = configured(generous, 1);
    let epoch = DistributedCommitEpoch::FIRST;
    priced
        .prepare_step_transaction(epoch, crate::ExpertPass::Prefill, 0)
        .unwrap();
    let _work = priced
        .prepare_partition_capture(&transport, 0, producers(priced.plan(), 1), LIMITS, estimate)
        .unwrap();
    let _coordination = priced.prepare_partition_coordination(&transport).unwrap();
    let before_hook = priced.ledger.step();
    for cumulative in [false, true] {
        let plan = skip_plan(false, |limits| {
            if cumulative {
                limits.cumulative.retained_bytes = before_hook.retained_bytes;
            } else {
                limits.per_step.retained_bytes = before_hook.retained_bytes;
            }
        });
        let mut session = configured(plan, 1);
        let layout = Layout {
            ranks: 1,
            members: vec![0],
        };
        let mut observer = PartitionCaptureObserver::for_step(
            &mut session,
            Backend::default(),
            &transport,
            &layout,
            0,
            LIMITS,
            estimate,
            |error| error,
        );
        observer
            .prepare_transaction(epoch, crate::ExpertPass::Prefill)
            .unwrap();
        observer.coordinate_transaction(epoch).unwrap();
        observer
            .observe_generated(
                "block.output",
                &global(),
                &generated_source(4096),
                &mut || -> Result<Value, PartitionCaptureObserverError<std::io::Error>> {
                    panic!("skipped producer cannot run its factory")
                },
            )
            .unwrap();
        observer.complete_transaction(epoch).unwrap();
        observer.finish_transaction(epoch, true);
        drop(observer);
        let step = session.take_step().unwrap();
        assert_eq!(step.step_usage, before_hook);
        assert_eq!(
            step.records[0].outcome,
            CaptureOutcome::Skipped {
                reason: CaptureSkipReason::Limit {
                    budget: CaptureBudget::Retention,
                    cumulative
                }
            }
        );
        assert!(step.partitions.is_empty());
    }
}

#[test]
fn skip_does_not_hide_invalid_sources_or_exhausted_mandatory_coordination() {
    for mandatory in [false, true] {
        let transport = singleton();
        let plan = skip_plan(false, |limits| {
            if mandatory {
                limits.per_step.retained_bytes = 0;
            }
        });
        let mut session = configured(plan, 1);
        let layout = Layout {
            ranks: 1,
            members: vec![0],
        };
        let mut observer = PartitionCaptureObserver::for_step(
            &mut session,
            Backend::default(),
            &transport,
            &layout,
            0,
            LIMITS,
            |_: &[u64], _: &CaptureSelection, _: &ResolvedCaptureSlice| {
                Err(CaptureError::Unsupported(
                    "actual native mechanism is absent".into(),
                ))
            },
            |error| error,
        );
        let epoch = DistributedCommitEpoch::FIRST;
        let result = observer.prepare_transaction(epoch, crate::ExpertPass::Prefill);
        assert!(
            matches!(result, Err(PartitionCaptureObserverError::Exchange(PartitionCaptureExchangeError::Capture(
            CaptureError::Limit { budget: CaptureBudget::Retention, cumulative: false }))) if mandatory)
                || matches!(result, Err(PartitionCaptureObserverError::Exchange(PartitionCaptureExchangeError::Capture(
                CaptureError::Unsupported(_)))) if !mandatory)
        );
        observer.finish_transaction(epoch, false);
        drop(observer);
        assert_eq!(transport.transport.calls.load(Ordering::SeqCst), 0);
        assert_eq!(
            session.take_step().unwrap().records[0].outcome,
            CaptureOutcome::Missing
        );
    }
}

#[test]
fn changing_prepaid_bound_is_an_error_even_with_skip_policy() {
    let transport = HookTransport {
        transport: transport(world(1), 0, Fault::ChangingEstimate(3)),
        hook: None,
        members: vec![0],
    };
    let mut session = configured(skip_plan(false, |_| {}), 1);
    let layout = Layout {
        ranks: 1,
        members: vec![0],
    };
    let mut observer = PartitionCaptureObserver::for_step(
        &mut session,
        Backend::default(),
        &transport,
        &layout,
        0,
        LIMITS,
        estimate,
        |error| error,
    );
    let epoch = DistributedCommitEpoch::FIRST;
    assert!(matches!(
        observer.prepare_transaction(epoch, crate::ExpertPass::Prefill),
        Err(PartitionCaptureObserverError::Exchange(
            PartitionCaptureExchangeError::PrepaidBound {
                source: CaptureError::Limit {
                    budget: CaptureBudget::Retention,
                    cumulative: false
                }
            }
        ))
    ));
    observer.finish_transaction(epoch, false);
    drop(observer);
    assert_eq!(transport.transport.calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        session.take_step().unwrap().records[0].outcome,
        CaptureOutcome::Missing
    );
}
