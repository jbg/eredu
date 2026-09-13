use super::*;
use crate::ActivationObserver;
mod intervention;
mod skip;
mod source;
mod sum;

pub(super) struct HookTransport {
    pub(super) transport: Transport,
    pub(super) hook: Option<Transport>,
    pub(super) members: Vec<usize>,
}
impl ConsensusTransport for HookTransport {
    type Error = std::io::Error;
    fn participant_count(&self) -> usize {
        self.transport.participant_count()
    }
    fn all_gather_words(&self, _: &[u32]) -> Result<Vec<u32>, Self::Error> {
        panic!("unbounded")
    }
}
impl BoundedConsensusTransport for HookTransport {
    type Completion = Done;
    type GatherOutput = usize;
    fn submit_all_gather_words(
        &self,
        words: &[u32],
    ) -> Result<Submission<usize, Done>, Self::Error> {
        self.transport.submit_all_gather_words(words)
    }
    fn resolve_all_gather_words(&self, output: usize) -> Result<Vec<u32>, Self::Error> {
        self.transport.resolve_all_gather_words(output)
    }
}
impl PartitionCaptureTransport for HookTransport {
    fn capture_rank(&self) -> usize {
        self.transport.capture_rank()
    }
    fn capture_wait(&self) -> Result<BoundedCompletionWait, CaptureError> {
        self.transport.capture_wait()
    }
    fn ensure_capture_active(&self) -> Result<(), BackendFailure> {
        self.transport.ensure_capture_active()
    }
    fn estimate_capture_gather(&self, words: usize) -> Result<CaptureUsage, CaptureError> {
        self.transport.estimate_capture_gather(words)
    }
    fn fail_capture_exchange(&self, error: &PartitionCaptureExchangeError) {
        self.transport.fail_capture_exchange(error)
    }
}
impl PartitionCaptureHookTransport for HookTransport {
    type HookOutput = usize;
    fn estimate_capture_hook(&self, members: &[usize]) -> Result<CaptureUsage, CaptureError> {
        assert_eq!(members, self.members);
        Ok(CaptureUsage {
            retained_bytes: 256,
            host_bytes: 128,
            ..Default::default()
        })
    }
    fn submit_capture_hook(
        &self,
        members: &[usize],
        success: bool,
    ) -> Result<Submission<usize, Done>, Self::Error> {
        assert_eq!(members, self.members);
        self.hook
            .as_ref()
            .expect("inactive pipeline ranks must not enter hook transport")
            .submit_all_gather_words(&[u32::from(success)])
    }
    fn resolve_capture_hook(&self, output: usize) -> Result<bool, Self::Error> {
        Ok(self
            .hook
            .as_ref()
            .unwrap()
            .resolve_all_gather_words(output)?
            .iter()
            .all(|word| *word == 1))
    }
}

struct Layout {
    ranks: usize,
    members: Vec<usize>,
}
impl PartitionCaptureLayout for Layout {
    fn capture_placement(
        &self,
        plan: &AdmittedCapturePlan,
        index: usize,
        _: CapturePhase,
        _: u64,
        _: PartitionCaptureReceiptLimits,
    ) -> Result<PartitionCapturePlacement, CaptureError> {
        assert_eq!(index, 0);
        Ok(PartitionCapturePlacement {
            producers: producers(plan, self.ranks),
            hook_members: self.members.clone(),
            source_shapes: self
                .members
                .iter()
                .map(|rank| local_source(self.ranks, *rank).shape)
                .collect(),
        })
    }
}

fn local_source(ranks: usize, rank: usize) -> Value {
    let maps = maps(ranks);
    let map = &maps
        .iter()
        .find(|(producer, _)| *producer == rank)
        .unwrap_or(&maps[0])
        .1;
    local_value(&global(), map)
}

const LIMITS: PartitionCaptureReceiptLimits = PartitionCaptureReceiptLimits {
    max_producers: 4,
    max_fragments: 16,
    max_record_bytes: 16384,
};

#[test]
fn partition_observer_uses_shared_callbacks_and_only_active_stage_votes() {
    // Rank 1 is another pipeline stage; rank 3 executes a replicated invocation
    // without producing duplicate values. Both still receive the global record.
    for (generated, failed, oversized, committed) in [
        (false, false, false, true),
        (true, false, false, true),
        (false, false, false, false),
        (false, true, false, false),
        (true, false, true, false),
    ] {
        let all = world(4);
        let hooks = world(3);
        let boundary = Arc::new(Barrier::new(4));
        let next_model = Arc::new(AtomicUsize::new(0));
        let results = std::thread::scope(|scope| {
            (0..4).map(|rank| {
            let all = Arc::clone(&all); let hooks = Arc::clone(&hooks);
            let boundary = Arc::clone(&boundary); let next_model = Arc::clone(&next_model);
            scope.spawn(move || {
                let members = vec![0, 2, 3];
                let transport = HookTransport {
                    transport: transport(all, rank, Fault::None),
                    hook: members.iter().position(|member| *member == rank).map(|local| transport(hooks, local, Fault::None)),
                    members: members.clone(),
                };
                let layout = Layout { ranks: 4, members };
                let mut session = configured(plan_for(CaptureTransform::Slice, false), 4);
                let epoch = DistributedCommitEpoch::FIRST;
                let local = local_source(4, rank);
                let mut factory_calls = 0;
                let mut observer = PartitionCaptureObserver::for_step(&mut session,
                    Backend { fail: failed && rank == 2, ..Default::default() }, &transport, &layout,
                    0, LIMITS, estimate, |error| error);
                assert!(observer.transactional());
                observer.prepare_transaction(epoch, crate::ExpertPass::Prefill).unwrap();
                assert_eq!(transport.transport.calls.load(Ordering::SeqCst), 0, "local preparation submits nothing");
                assert!(transport.hook.as_ref().is_none_or(|hook| hook.calls.load(Ordering::SeqCst) == 0));
                observer.coordinate_transaction(epoch).unwrap();
                let source = if rank == 1 { Ok(()) } else if generated {
                    observer.observe_generated("unselected", &local, &generated_source(u64::MAX),
                        &mut || -> Result<Value, PartitionCaptureObserverError<std::io::Error>> { panic!("unselected factory") }).unwrap();
                    observer.observe_generated("block.output", &local, &generated_source(if oversized && rank == 2 { 1 << 40 } else { 4096 }),
                        &mut || { factory_calls += 1; Ok(local.clone()) })
                } else { observer.observe("block.output", &local) };
                if rank != 1 && source.is_ok() { next_model.fetch_add(1, Ordering::SeqCst); }
                if (failed || oversized) && rank != 1 { assert!(source.is_err()); }
                if rank == 2 && failed {
                    let error = source.as_ref().unwrap_err();
                    let mut cause: &dyn std::error::Error = error;
                    let mut found = false;
                    while let Some(source) = cause.source() { found |= source.to_string().contains("injected fragment failure"); cause = source; }
                    assert!(found, "native capture cause survives the observer boundary");
                }
                boundary.wait();
                assert_eq!(next_model.load(Ordering::SeqCst), if failed || oversized { 0 } else { 3 }, "no active peer enters the next model collective after local failure");
                if !(failed || oversized) { observer.complete_transaction(epoch).unwrap(); }
                observer.finish_transaction(epoch, committed);
                drop(observer);
                let step = session.take_step().unwrap();
                assert_eq!(step.partitions.len(), usize::from(committed));
                assert_eq!(step.records[0].payload.is_some(), committed);
                assert_eq!(transport.hook.as_ref().map_or(0, |hook| hook.calls.load(Ordering::SeqCst)), 2 * usize::from(rank != 1));
                assert!(!transport.transport.poison.load(Ordering::SeqCst), "agreed capture rejection is recoverable");
                if rank == 3 || rank == 1 || (oversized && rank == 2) { assert_eq!(factory_calls, 0); }
                step
            })
        }).collect::<Vec<_>>().into_iter().map(|worker| worker.join().unwrap()).collect::<Vec<_>>()
        });
        assert!(results
            .iter()
            .all(|step| step.cumulative_usage == results[0].cumulative_usage));
        if committed {
            let mut ordinary = CaptureSession::new(plan_for(CaptureTransform::Slice, false));
            ordinary.begin_step(CapturePhase::Prefill, 0).unwrap();
            ordinary
                .observe(&mut Backend::default(), "block.output", &global())
                .unwrap();
            let expected = ordinary.take_step().unwrap();
            assert!(results
                .iter()
                .all(|step| step.records[0].payload == expected.records[0].payload));
        }
    }
}

#[test]
fn omitted_hook_rejects_at_common_delivery_without_early_return_or_publication() {
    let all = world(3);
    let hooks = world(2);
    std::thread::scope(|scope| {
        let workers = (0..3)
            .map(|rank| {
                let all = Arc::clone(&all);
                let hooks = Arc::clone(&hooks);
                scope.spawn(move || {
                    let members = vec![0, 2];
                    let transport = HookTransport {
                        transport: transport(all, rank, Fault::None),
                        hook: members
                            .iter()
                            .position(|member| *member == rank)
                            .map(|local| transport(hooks, local, Fault::None)),
                        members: members.clone(),
                    };
                    let layout = Layout { ranks: 3, members };
                    let mut session = configured(plan_for(CaptureTransform::Slice, false), 3);
                    let epoch = DistributedCommitEpoch::FIRST;
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
                    assert!(observer.complete_transaction(epoch).is_err());
                    observer.finish_transaction(epoch, false);
                    drop(observer);
                    let step = session.take_step().unwrap();
                    assert!(step.partitions.is_empty() && step.records[0].payload.is_none());
                    assert_eq!(transport.transport.calls.load(Ordering::SeqCst), 2);
                    assert!(!transport.transport.poison.load(Ordering::SeqCst));
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker.join().unwrap();
        }
    });
}

#[test]
fn hook_completion_failure_never_resolves_status_or_accepts_the_source() {
    for fault in [
        Fault::Wait(0),
        Fault::Deadline(0),
        Fault::Submit(0),
        Fault::Resolve(0),
    ] {
        // Failure before result resolution needs no peer; a singleton native
        // transport fixture models the local failing completion of a two-member group.
        let all = world(3);
        let peers = (1..3)
            .map(|rank| {
                let all = Arc::clone(&all);
                std::thread::spawn(move || {
                    let transport = HookTransport {
                        transport: transport(all, rank, Fault::None),
                        hook: None,
                        members: vec![0, 2],
                    };
                    let mut session = configured(plan_for(CaptureTransform::Slice, false), 3);
                    session
                        .prepare_step_transaction(
                            DistributedCommitEpoch::FIRST,
                            crate::ExpertPass::Prefill,
                            0,
                        )
                        .unwrap();
                    let mut work = session
                        .prepare_partition_capture(
                            &transport,
                            0,
                            producers(session.plan(), 3),
                            LIMITS,
                            estimate,
                        )
                        .unwrap();
                    let _hook = session
                        .prepare_partition_hook(&mut work, vec![0, 2])
                        .unwrap();
                    let coordination = session.prepare_partition_coordination(&transport).unwrap();
                    session.coordinate_partition_capture(coordination).unwrap();
                })
            })
            .collect::<Vec<_>>();
        let transport = HookTransport {
            transport: transport(all, 0, Fault::None),
            hook: Some(transport(world(1), 0, fault)),
            members: vec![0, 2],
        };
        let mut session = configured(plan_for(CaptureTransform::Slice, false), 3);
        let epoch = DistributedCommitEpoch::FIRST;
        session
            .prepare_step_transaction(epoch, crate::ExpertPass::Prefill, 0)
            .unwrap();
        let mut work = session
            .prepare_partition_capture(
                &transport,
                0,
                producers(session.plan(), 3),
                LIMITS,
                estimate,
            )
            .unwrap();
        let hook = session
            .prepare_partition_hook(&mut work, vec![0, 2])
            .unwrap();
        let coordination = session.prepare_partition_coordination(&transport).unwrap();
        session.coordinate_partition_capture(coordination).unwrap();
        for peer in peers {
            peer.join().unwrap();
        }
        let local = local_value(&global(), &maps(3)[0].1);
        session
            .observe_partition(&mut work, &mut Backend::default(), &local)
            .unwrap();
        assert!(session.agree_partition_hook(&mut work, hook, true).is_err());
        assert_eq!(
            transport
                .hook
                .as_ref()
                .unwrap()
                .resolves
                .load(Ordering::SeqCst),
            usize::from(fault == Fault::Resolve(0))
        );
        assert!(transport.transport.poison.load(Ordering::SeqCst));
        session.finish_transaction(epoch, false);
        assert!(session.take_step().unwrap().partitions.is_empty());
    }
}
