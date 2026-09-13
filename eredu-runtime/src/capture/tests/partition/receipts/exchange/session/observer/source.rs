use super::*;
use std::sync::Condvar;

struct ReplicaLayout(usize);
impl PartitionCaptureLayout for ReplicaLayout {
    fn capture_placement(
        &self,
        plan: &AdmittedCapturePlan,
        _: usize,
        _: CapturePhase,
        _: u64,
        _: PartitionCaptureReceiptLimits,
    ) -> Result<PartitionCapturePlacement, CaptureError> {
        Ok(PartitionCapturePlacement {
            producers: producers(plan, 1),
            hook_members: (0..self.0).collect(),
            source_shapes: vec![vec![3, 20]; self.0],
        })
    }
}

struct LazyBackend {
    inner: Backend,
    ready: Arc<(Mutex<usize>, Condvar)>,
    participants: usize,
    cost: u64,
    fault: Fault,
    exports: Arc<AtomicUsize>,
}

struct ShardedLayout;
impl PartitionCaptureLayout for ShardedLayout {
    fn capture_placement(
        &self,
        plan: &AdmittedCapturePlan,
        _: usize,
        _: CapturePhase,
        _: u64,
        _: PartitionCaptureReceiptLimits,
    ) -> Result<PartitionCapturePlacement, CaptureError> {
        let slice = resolve_slice(&plan.points()[0], &plan.plan().selections[0], &[3, 20])?;
        Ok(PartitionCapturePlacement {
            producers: [0..8, 8..20]
                .into_iter()
                .enumerate()
                .map(|(rank, range)| PartitionCaptureProducer {
                    rank,
                    projection: CaptureSlicePartition::new(
                        &[3, 20],
                        &slice,
                        1,
                        &ComponentCoordinateMap::range(20, range).unwrap(),
                        16,
                    )
                    .unwrap(),
                })
                .collect(),
            hook_members: vec![0, 1],
            source_shapes: vec![vec![3, 8], vec![3, 12]],
        })
    }
}

#[test]
fn empty_local_shard_executes_dependencies_when_another_shard_exports() {
    let all = world(2);
    let hooks = world(2);
    let ready = Arc::new((Mutex::new(0), Condvar::new()));
    let exports = Arc::new(AtomicUsize::new(0));
    std::thread::scope(|scope| {
        let workers = (0..2)
            .map(|rank| {
                let all = Arc::clone(&all);
                let hooks = Arc::clone(&hooks);
                let ready = Arc::clone(&ready);
                let exports = Arc::clone(&exports);
                scope.spawn(move || {
                    let transport = HookTransport {
                        transport: transport(all, rank, Fault::None),
                        hook: Some(transport(hooks, rank, Fault::None)),
                        members: vec![0, 1],
                    };
                    let original = plan_for(CaptureTransform::Slice, false);
                    let discovery = discovery(&original);
                    let mut plan = original.plan().clone();
                    plan.selections[0].slices[1].end = 7;
                    let plan = plan
                        .admit(
                            &discovery.catalog,
                            &discovery.support,
                            &discovery.support.capture,
                            original.request(),
                        )
                        .unwrap();
                    let mut session = configured(plan, 2);
                    let backend = LazyBackend {
                        inner: Backend::default(),
                        ready,
                        participants: 2,
                        cost: 4096,
                        fault: Fault::None,
                        exports,
                    };
                    let mut observer = PartitionCaptureObserver::for_step(
                        &mut session,
                        backend,
                        &transport,
                        &ShardedLayout,
                        0,
                        LIMITS,
                        estimate,
                        |error| error,
                    );
                    let epoch = DistributedCommitEpoch::FIRST;
                    observer
                        .prepare_transaction(epoch, crate::ExpertPass::Prefill)
                        .unwrap();
                    observer.coordinate_transaction(epoch).unwrap();
                    let value = local_value(
                        &global(),
                        &ComponentCoordinateMap::range(20, if rank == 0 { 0..8 } else { 8..20 })
                            .unwrap(),
                    );
                    let mut source = generated_source(4096);
                    source.source_dtype = Some(TensorDtype::F32);
                    let mut factories = 0;
                    observer
                        .observe_generated("block.output", &value, &source, &mut || {
                            factories += 1;
                            Ok(value.clone())
                        })
                        .unwrap();
                    observer.complete_transaction(epoch).unwrap();
                    observer.finish_transaction(epoch, true);
                    drop(observer);
                    assert_eq!(factories, usize::from(rank == 0));
                    assert!(session.take_step().unwrap().records[0].payload.is_some());
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker.join().unwrap();
        }
    });
    assert_eq!(*ready.0.lock().unwrap(), 2);
    assert_eq!(exports.load(Ordering::SeqCst), 1);
}
impl CaptureBackend for LazyBackend {
    type Tensor = Value;
    type Error = std::io::Error;
    fn shape(&self, value: &Value) -> Result<Vec<u64>, Self::Error> {
        self.inner.shape(value)
    }
    fn source_dtype(&self, value: &Value) -> Option<TensorDtype> {
        self.inner.source_dtype(value)
    }
    fn estimate_partition_source(
        &self,
        shape: &[u64],
        _: BoundedCompletionWait,
    ) -> Result<CaptureUsage, CaptureError> {
        assert_eq!(shape.len(), 2);
        assert_eq!(shape[0], 3);
        Ok(CaptureUsage {
            retained_bytes: self.cost,
            ..Default::default()
        })
    }
    fn prepare_partition_source(
        &mut self,
        _: &Value,
        wait: BoundedCompletionWait,
    ) -> Result<BoundedCompletionOutcome, Self::Error> {
        let (lock, wake) = &*self.ready;
        let mut ready = lock.lock().unwrap();
        *ready += 1;
        wake.notify_all();
        let (ready, timeout) = wake
            .wait_timeout_while(ready, std::time::Duration::from_secs(2), |ready| {
                *ready < self.participants
            })
            .unwrap();
        assert!(
            !timeout.timed_out(),
            "an exporting source needs every replica's dependencies"
        );
        assert_eq!(*ready, self.participants);
        match self.fault {
            Fault::Wait(_) => Err(std::io::Error::other("injected source preparation failure")),
            Fault::Deadline(_) => Ok(BoundedCompletionOutcome::DeadlineExceeded {
                cancellation: wait.cancellation(),
            }),
            _ => Ok(BoundedCompletionOutcome::Completed),
        }
    }
    fn estimate(
        &self,
        value: &Value,
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        self.inner.estimate(value, selection, slice)
    }
    fn transform(
        &mut self,
        value: &Value,
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CapturePayload, Self::Error> {
        assert_eq!(*self.ready.0.lock().unwrap(), self.participants);
        self.exports.fetch_add(1, Ordering::SeqCst);
        self.inner.transform(value, selection, slice)
    }
}

#[test]
fn replica_sources_prepare_together_but_export_once_and_never_run_empty_or_skipped_factories() {
    for (generated, empty, skip, invalid_shape) in [
        (false, false, false, false),
        (true, false, false, false),
        (false, true, false, false),
        (true, true, false, false),
        (true, false, true, false),
        (false, false, false, true),
    ] {
        let all = world(2);
        let hooks = world(2);
        let ready = Arc::new((Mutex::new(0), Condvar::new()));
        let exports = Arc::new(AtomicUsize::new(0));
        let steps = std::thread::scope(|scope| {
            (0..2)
                .map(|rank| {
                    let all = Arc::clone(&all);
                    let hooks = Arc::clone(&hooks);
                    let ready = Arc::clone(&ready);
                    let exports = Arc::clone(&exports);
                    scope.spawn(move || {
                        let transport = HookTransport {
                            transport: transport(all, rank, Fault::None),
                            hook: Some(transport(hooks, rank, Fault::None)),
                            members: vec![0, 1],
                        };
                        let plan = if skip {
                            skip::skip_plan(false, |_| {})
                        } else {
                            plan_for(CaptureTransform::Slice, empty)
                        };
                        let mut session = configured(plan, 2);
                        let backend = LazyBackend {
                            inner: Backend::default(),
                            ready,
                            participants: 2,
                            cost: if skip { 1 << 40 } else { 4096 },
                            fault: Fault::None,
                            exports,
                        };
                        let mut observer = PartitionCaptureObserver::for_step(
                            &mut session,
                            backend,
                            &transport,
                            &ReplicaLayout(2),
                            0,
                            LIMITS,
                            estimate,
                            |error| error,
                        );
                        let epoch = DistributedCommitEpoch::FIRST;
                        observer
                            .prepare_transaction(epoch, crate::ExpertPass::Prefill)
                            .unwrap();
                        observer.coordinate_transaction(epoch).unwrap();
                        let mut value = global();
                        if invalid_shape && rank == 1 {
                            value.shape[1] += 1;
                        }
                        let mut factories = 0;
                        let mut generated_source = generated_source(4096);
                        generated_source.source_dtype = Some(TensorDtype::F32);
                        let result = if generated {
                            observer.observe_generated(
                                "block.output",
                                &value,
                                &generated_source,
                                &mut || {
                                    factories += 1;
                                    Ok(value.clone())
                                },
                            )
                        } else if rank == 1 {
                            observer.observe_replica("block.output", &value)
                        } else {
                            observer.observe("block.output", &value)
                        };
                        if invalid_shape {
                            assert!(result.is_err());
                            if rank == 1 {
                                assert!(result.unwrap_err().to_string().contains("source shape"));
                            }
                        } else {
                            result.unwrap();
                            observer.complete_transaction(epoch).unwrap();
                        }
                        observer.finish_transaction(epoch, !invalid_shape);
                        drop(observer);
                        let step = session.take_step().unwrap();
                        assert_eq!(
                            factories,
                            usize::from(generated && rank == 0 && !empty && !skip)
                        );
                        assert!(!transport.transport.poison.load(Ordering::SeqCst));
                        if skip {
                            assert!(matches!(
                                step.records[0].outcome,
                                CaptureOutcome::Skipped { .. }
                            ));
                        }
                        if invalid_shape {
                            assert!(step.partitions.is_empty());
                        }
                        step
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|worker| worker.join().unwrap())
                .collect::<Vec<_>>()
        });
        let executes = !empty && !skip && !invalid_shape;
        assert_eq!(*ready.0.lock().unwrap(), if executes { 2 } else { 0 });
        assert_eq!(exports.load(Ordering::SeqCst), usize::from(executes));
        assert_eq!(steps[0].cumulative_usage, steps[1].cumulative_usage);
        if !empty && !skip {
            assert!(steps[0].step_usage.retained_bytes >= 8192);
        }
        if executes {
            let mut ordinary = CaptureSession::new(plan_for(CaptureTransform::Slice, false));
            ordinary.begin_step(CapturePhase::Prefill, 0).unwrap();
            ordinary
                .observe(&mut Backend::default(), "block.output", &global())
                .unwrap();
            let expected = ordinary.take_step().unwrap();
            assert_eq!(steps[0].records[0].payload, expected.records[0].payload);
        }
    }
}

#[test]
fn failed_source_keeps_original_cause_fences_transport_and_consumes_its_reservation() {
    for fault in [Fault::Wait(0), Fault::Deadline(0)] {
        let transport = HookTransport {
            transport: transport(world(1), 0, Fault::None),
            hook: None,
            members: vec![0],
        };
        let ready = Arc::new((Mutex::new(0), Condvar::new()));
        let exports = Arc::new(AtomicUsize::new(0));
        let mut session = configured(plan_for(CaptureTransform::Slice, false), 1);
        let backend = LazyBackend {
            inner: Backend::default(),
            ready: Arc::clone(&ready),
            participants: 1,
            cost: 4096,
            fault,
            exports: Arc::clone(&exports),
        };
        let mut observer = PartitionCaptureObserver::for_step(
            &mut session,
            backend,
            &transport,
            &ReplicaLayout(1),
            0,
            LIMITS,
            estimate,
            |error| error,
        );
        let epoch = DistributedCommitEpoch::FIRST;
        observer
            .prepare_transaction(epoch, crate::ExpertPass::Prefill)
            .unwrap();
        observer.coordinate_transaction(epoch).unwrap();
        let error = observer.observe("block.output", &global()).unwrap_err();
        match fault {
            Fault::Wait(_) => assert!(error
                .to_string()
                .contains("injected source preparation failure")),
            Fault::Deadline(_) => assert!(matches!(
                error,
                PartitionCaptureObserverError::Exchange(
                    PartitionCaptureExchangeError::Deadline { .. }
                )
            )),
            _ => unreachable!(),
        }
        observer.finish_transaction(epoch, false);
        drop(observer);
        let step = session.take_step().unwrap();
        assert!(step.step_usage.retained_bytes >= 4096);
        assert!(
            step.capture_seconds > 0.0,
            "failed source work remains attributed to capture"
        );
        assert_eq!(step.step_usage, step.cumulative_usage);
        assert!(step.partitions.is_empty());
        assert!(transport.transport.poison.load(Ordering::SeqCst));
        assert_eq!(*ready.0.lock().unwrap(), 1);
        assert_eq!(exports.load(Ordering::SeqCst), 0);
    }
}
