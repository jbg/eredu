use super::*;

struct SumLayout;
impl PartitionCaptureLayout for SumLayout {
    fn capture_combination(&self, _: &str) -> Result<PartitionCaptureCombination, CaptureError> {
        Ok(PartitionCaptureCombination::SumF64ToF32)
    }
    fn capture_placement(
        &self,
        plan: &AdmittedCapturePlan,
        index: usize,
        _: CapturePhase,
        _: u64,
        _: PartitionCaptureReceiptLimits,
    ) -> Result<PartitionCapturePlacement, CaptureError> {
        let slice = resolve_slice(
            &plan.points()[index],
            &plan.plan().selections[index],
            &[3, 20],
        )?;
        let members = vec![0, 2, 3];
        Ok(PartitionCapturePlacement {
            producers: members
                .iter()
                .map(|rank| {
                    Ok(PartitionCaptureProducer {
                        rank: *rank,
                        projection: CaptureSlicePartition::new(
                            &[3, 20],
                            &slice,
                            1,
                            &ComponentCoordinateMap::range(20, 0..20).unwrap(),
                            1,
                        )?,
                    })
                })
                .collect::<Result<_, CaptureError>>()?,
            source_shapes: vec![vec![3, 20]; members.len()],
            hook_members: members,
        })
    }
}

struct Bootstrap(Transport);
impl ConsensusTransport for Bootstrap {
    type Error = std::io::Error;
    fn participant_count(&self) -> usize {
        self.0.participant_count()
    }
    fn all_gather_words(&self, words: &[u32]) -> Result<Vec<u32>, Self::Error> {
        let submission = self.0.submit_all_gather_words(words)?;
        submission.completion.wait()?;
        self.0.resolve_all_gather_words(submission.output)
    }
}

struct SourceBackend {
    inner: Backend,
    sources: Arc<AtomicUsize>,
    exports: Arc<AtomicUsize>,
}
impl CaptureBackend for SourceBackend {
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
        _: &[u64],
        _: BoundedCompletionWait,
    ) -> Result<CaptureUsage, CaptureError> {
        Ok(CaptureUsage {
            retained_bytes: 512,
            ..Default::default()
        })
    }
    fn prepare_partition_source(
        &mut self,
        _: &Value,
        _: BoundedCompletionWait,
    ) -> Result<BoundedCompletionOutcome, Self::Error> {
        self.sources.fetch_add(1, Ordering::SeqCst);
        Ok(BoundedCompletionOutcome::Completed)
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
        self.exports.fetch_add(1, Ordering::SeqCst);
        self.inner.transform(value, selection, slice)
    }
}

#[test]
fn summed_observer_prices_raw_terms_and_publishes_only_after_shared_completion() {
    for transform in [
        CaptureTransform::Slice,
        CaptureTransform::Preview { max_elements: 5 },
        CaptureTransform::Summary,
        CaptureTransform::Histogram {
            edges: vec![-2.0, 0.0, 2.0],
        },
    ] {
        for (generated, fail, committed) in [
            (false, false, true),
            (true, false, true),
            (false, false, false),
            (false, true, false),
        ] {
            let setup = world(4);
            let all = world(4);
            let hooks = world(3);
            let sources = Arc::new(AtomicUsize::new(0));
            let exports = Arc::new(AtomicUsize::new(0));
            let results = std::thread::scope(|scope| {
                (0..4)
                    .map(|rank| {
                        let setup = Arc::clone(&setup);
                        let all = Arc::clone(&all);
                        let hooks = Arc::clone(&hooks);
                        let sources = Arc::clone(&sources);
                        let exports = Arc::clone(&exports);
                        let transform = transform.clone();
                        scope.spawn(move || {
                            let identity = crate::establish_communication_session(
                                &Bootstrap(transport(setup, rank, Fault::None)),
                                &crate::CommunicationManifest::new(4, rank, vec![], vec![])
                                    .unwrap(),
                                Some([rank as u8 + 1; 32]),
                            )
                            .unwrap()
                            .identity();
                            let identity = PartitionCaptureIdentity::for_session(
                                "artifact-exact".into(),
                                "sum-execution".into(),
                                identity,
                                Some("effective-overlay".into()),
                            )
                            .unwrap();
                            let members = vec![0, 2, 3];
                            let transport = HookTransport {
                                transport: transport(all, rank, Fault::None),
                                hook: members
                                    .iter()
                                    .position(|member| *member == rank)
                                    .map(|local| transport(hooks, local, Fault::None)),
                                members,
                            };
                            let original = plan_for(transform, false);
                            let discovery = discovery(&original);
                            let mut plan = original.plan().clone();
                            plan.limits.per_step.host_bytes = 128 << 20;
                            plan.limits.per_step.encoded_bytes = 8 << 20;
                            plan.limits.cumulative = plan.limits.per_step.checked_mul(2).unwrap();
                            let plan = plan
                                .admit(
                                    &discovery.catalog,
                                    &discovery.support,
                                    &discovery.support.capture,
                                    original.request(),
                                )
                                .unwrap();
                            let mut session = CaptureSession::new(plan);
                            session
                                .configure_partition_capture(identity.clone())
                                .unwrap();
                            let initial = session.checkpoint(&discovery).unwrap();
                            let local = if rank == 2 {
                                global()
                            } else {
                                Value {
                                    shape: vec![3, 20],
                                    data: TensorObservationData::F32(vec![
                                        if rank == 0 {
                                            1e20
                                        } else {
                                            -1e20
                                        };
                                        60
                                    ]),
                                }
                            };
                            let mut calls = 0;
                            let mut observer = PartitionCaptureObserver::for_step(
                                &mut session,
                                SourceBackend {
                                    inner: Backend {
                                        fail: fail && rank == 2,
                                        ..Default::default()
                                    },
                                    sources,
                                    exports,
                                },
                                &transport,
                                &SumLayout,
                                0,
                                PartitionCaptureReceiptLimits {
                                    max_record_bytes: 65_536,
                                    ..LIMITS
                                },
                                estimate,
                                |error| error,
                            )
                            .with_session_identity(identity);
                            let epoch = DistributedCommitEpoch::FIRST;
                            observer
                                .prepare_transaction(epoch, crate::ExpertPass::Prefill)
                                .unwrap();
                            assert_eq!(transport.transport.calls.load(Ordering::SeqCst), 0);
                            observer.coordinate_transaction(epoch).unwrap();
                            if rank != 1 {
                                let result = if generated {
                                    observer.observe_generated(
                                        "block.output",
                                        &local,
                                        &generated_source(4096),
                                        &mut || {
                                            calls += 1;
                                            Ok(local.clone())
                                        },
                                    )
                                } else {
                                    observer.observe("block.output", &local)
                                };
                                assert_eq!(result.is_err(), fail);
                            }
                            if !fail {
                                observer.complete_transaction(epoch).unwrap();
                            }
                            observer.finish_transaction(epoch, committed);
                            drop(observer);
                            assert_eq!(calls, usize::from(generated && rank != 1));
                            let step = session.take_step().unwrap();
                            assert_eq!(step.partitions.len(), usize::from(committed));
                            assert_eq!(step.records[0].payload.is_some(), committed);
                            if committed {
                                assert_eq!(
                                    step.partitions[0].combination,
                                    PartitionCaptureCombination::SumF64ToF32
                                );
                                assert_eq!(step.partitions[0].schema_version, 3);
                                assert_eq!(step.partitions[0].producers, [0, 2, 3]);
                            }
                            let spent = session.ledger.total();
                            session.restore(&initial).unwrap();
                            assert_eq!(
                                session.ledger.total(),
                                spent,
                                "restore cannot refund summed source or wire work"
                            );
                            step
                        })
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
                    .map(|worker| worker.join().unwrap())
                    .collect::<Vec<_>>()
            });
            assert_eq!(sources.load(Ordering::SeqCst), 3);
            assert_eq!(exports.load(Ordering::SeqCst), 3);
            assert!(results
                .iter()
                .all(|step| step.cumulative_usage == results[0].cumulative_usage));
            if committed {
                let mut ordinary = CaptureSession::new(plan_for(transform.clone(), false));
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
}
