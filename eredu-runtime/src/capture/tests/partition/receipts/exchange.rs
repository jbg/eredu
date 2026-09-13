use super::*;
mod routed;
mod session;
use eredu_core::{
    consensus::{BoundedConsensusTransport, ConsensusTransport},
    BackendFailure, BoundedCompletion, BoundedCompletionOutcome, BoundedCompletionWait, Completion,
    CompletionCancellationMode, Submission,
};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc, Barrier, Mutex,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fault {
    None,
    ChangingEstimate(usize),
    Header,
    Payload,
    ShortGather,
    Deadline(usize),
    Wait(usize),
    Submit(usize),
    Resolve(usize),
}

struct World {
    barrier: Barrier,
    frames: Mutex<Vec<Vec<u32>>>,
}

struct Transport {
    rank: usize,
    world: Arc<World>,
    calls: AtomicUsize,
    estimates: AtomicUsize,
    resolves: AtomicUsize,
    poison: AtomicBool,
    fault: Fault,
}

struct Done {
    fault: Fault,
    call: usize,
}
impl Completion for Done {
    type Error = std::io::Error;
    fn is_complete(&self) -> Result<bool, Self::Error> {
        Ok(true)
    }
    fn wait(&self) -> Result<(), Self::Error> {
        Ok(())
    }
}
impl BoundedCompletion for Done {
    fn wait_bounded(
        self,
        wait: BoundedCompletionWait,
    ) -> Result<BoundedCompletionOutcome, Self::Error> {
        match self.fault {
            Fault::Deadline(call) if call == self.call => {
                Ok(BoundedCompletionOutcome::DeadlineExceeded {
                    cancellation: wait.cancellation(),
                })
            }
            Fault::Wait(call) if call == self.call => {
                Err(std::io::Error::other("injected exact completion failure"))
            }
            _ => Ok(BoundedCompletionOutcome::Completed),
        }
    }
}
impl ConsensusTransport for Transport {
    type Error = std::io::Error;
    fn participant_count(&self) -> usize {
        self.world.frames.lock().unwrap().len()
    }
    fn all_gather_words(&self, _: &[u32]) -> Result<Vec<u32>, Self::Error> {
        panic!("unbounded gather must never run")
    }
}
impl BoundedConsensusTransport for Transport {
    type Completion = Done;
    type GatherOutput = usize;
    fn submit_all_gather_words(
        &self,
        words: &[u32],
    ) -> Result<Submission<usize, Done>, Self::Error> {
        assert!(!self.poison.load(Ordering::SeqCst));
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        if self.fault == Fault::Submit(call) {
            return Err(std::io::Error::other("injected submission failure"));
        }
        let mut words = words.to_vec();
        if self.fault == Fault::Header && call == 0 {
            words[5] ^= 1;
        }
        if self.fault == Fault::Payload && call == 1 {
            words[0] = u32::MAX;
        }
        self.world.frames.lock().unwrap()[self.rank] = words;
        Ok(Submission {
            output: call,
            completion: Done {
                fault: self.fault,
                call,
            },
        })
    }
    fn resolve_all_gather_words(&self, call: usize) -> Result<Vec<u32>, Self::Error> {
        self.resolves.fetch_add(1, Ordering::SeqCst);
        if self.fault == Fault::Resolve(call) {
            return Err(std::io::Error::other("injected resolution failure"));
        }
        self.world.barrier.wait();
        let mut words: Vec<_> = self
            .world
            .frames
            .lock()
            .unwrap()
            .iter()
            .flatten()
            .copied()
            .collect();
        self.world.barrier.wait();
        if self.fault == Fault::ShortGather {
            words.pop();
        }
        Ok(words)
    }
}
impl PartitionCaptureTransport for Transport {
    fn capture_rank(&self) -> usize {
        self.rank
    }
    fn capture_wait(&self) -> Result<BoundedCompletionWait, CaptureError> {
        Ok(BoundedCompletionWait::new(
            std::time::Duration::from_secs(1),
            CompletionCancellationMode::QuarantineUntilComplete,
        )
        .unwrap())
    }
    fn ensure_capture_active(&self) -> Result<(), BackendFailure> {
        if self.poison.load(Ordering::SeqCst) {
            Err(BackendFailure::from_error(std::io::Error::other(
                "poisoned",
            )))
        } else {
            Ok(())
        }
    }
    fn estimate_capture_gather(&self, words: usize) -> Result<CaptureUsage, CaptureError> {
        let estimate = self.estimates.fetch_add(1, Ordering::SeqCst);
        Ok(CaptureUsage {
            retained_bytes: if matches!(self.fault, Fault::ChangingEstimate(after) if estimate >= after)
            {
                1 << 40
            } else {
                mul(words as u64, 4 * (1 + self.participant_count()) as u64)?
            },
            host_bytes: mul(words as u64, 4 * self.participant_count() as u64)?,
            ..Default::default()
        })
    }
    fn fail_capture_exchange(&self, _: &PartitionCaptureExchangeError) {
        self.poison.store(true, Ordering::SeqCst);
    }
}

fn world(ranks: usize) -> Arc<World> {
    Arc::new(World {
        barrier: Barrier::new(ranks),
        frames: Mutex::new(vec![Vec::new(); ranks]),
    })
}
fn transport(world: Arc<World>, rank: usize, fault: Fault) -> Transport {
    Transport {
        rank,
        world,
        fault,
        calls: AtomicUsize::new(0),
        estimates: AtomicUsize::new(0),
        resolves: AtomicUsize::new(0),
        poison: AtomicBool::new(false),
    }
}
fn receipt(
    plan: &AdmittedCapturePlan,
    ledger: &mut CaptureLedger,
    ranks: usize,
) -> PartitionCaptureReceiptPlan {
    // Ranks 0 and 2 own the selected components; rank 1 (when present) is an
    // inactive pipeline stage, not an empty-value producer.
    let slice = resolve_slice(&plan.points()[0], &plan.plan().selections[0], &[3, 20]).unwrap();
    let maps = if ranks == 1 {
        vec![(0, ComponentCoordinateMap::range(20, 0..20).unwrap())]
    } else {
        vec![
            (
                0,
                ComponentCoordinateMap::indices(20, vec![7, 1, 4, 2, 0]).unwrap(),
            ),
            (2, ComponentCoordinateMap::range(20, 8..20).unwrap()),
        ]
    };
    PartitionCaptureReceiptPlan::new(
        plan.clone(),
        context(plan),
        maps.into_iter()
            .map(|(rank, map)| PartitionCaptureProducer {
                rank,
                projection: CaptureSlicePartition::new(&[3, 20], &slice, 1, &map, 16).unwrap(),
            })
            .collect(),
        ranks,
        PartitionCaptureReceiptLimits {
            max_producers: ranks,
            max_fragments: 16,
            max_record_bytes: 16384,
        },
        ledger,
    )
    .unwrap()
}
fn local_record(
    plan: &AdmittedCapturePlan,
    authority: &PartitionCaptureReceiptPlan,
    rank: usize,
    ledger: &mut CaptureLedger,
) -> Option<Vec<u8>> {
    let projection = authority.producer(rank)?;
    let map = if authority.world_size() == 1 {
        ComponentCoordinateMap::range(20, 0..20).unwrap()
    } else if rank == 0 {
        ComponentCoordinateMap::indices(20, vec![7, 1, 4, 2, 0]).unwrap()
    } else {
        ComponentCoordinateMap::range(20, 8..20).unwrap()
    };
    let global = Value {
        shape: vec![3, 20],
        data: TensorObservationData::F32((0..60).map(|i| i as f32 * 0.25 - 3.0).collect()),
    };
    let local = local_value(&global, &map);
    let mut backend = Backend::default();
    let fragments = (0..projection.fragments().len())
        .map(|fragment_index| {
            capture_fragment(
                &mut backend,
                &local,
                PartitionCaptureRequest {
                    invocation: None,
                    plan,
                    selection_index: 0,
                    phase: CapturePhase::Prefill,
                    prediction: 0,
                    projection,
                    fragment_index,
                    producer_rank: rank,
                },
                ledger,
            )
            .unwrap()
        })
        .collect();
    Some(
        authority
            .encode_producer(rank, Some(TensorDtype::F32), fragments, ledger)
            .unwrap(),
    )
}

#[derive(Clone, Copy)]
enum Scenario {
    Success,
    LocalFailure,
    DecodeBudget,
    CorruptPayload,
    DifferentIdentity,
}
fn run(
    ranks: usize,
    transform: CaptureTransform,
    scenario: Scenario,
) -> Vec<(
    Result<ReceivedPartitionCapture, PartitionCaptureExchangeError>,
    usize,
    bool,
    CaptureUsage,
)> {
    let world = world(ranks);
    let plan = Arc::new(plan_for(transform, false));
    (0..ranks)
        .map(|rank| {
            let world = world.clone();
            let plan = plan.clone();
            std::thread::spawn(move || {
                let fault = match (scenario, rank) {
                    (Scenario::CorruptPayload, 0) => Fault::Payload,
                    (Scenario::DifferentIdentity, 0) => Fault::Header,
                    _ => Fault::None,
                };
                let transport = transport(world, rank, fault);
                let mut ledger = CaptureLedger::new(&plan);
                let authority = receipt(&plan, &mut ledger, ranks);
                let exchange =
                    PartitionCaptureExchange::admit(&transport, authority, &mut ledger).unwrap();
                assert_eq!(transport.calls.load(Ordering::SeqCst), 0);
                let reserved = exchange.reserved();
                let local = local_record(&plan, exchange.receipt_plan(), rank, &mut ledger);
                let local = if matches!(scenario, Scenario::LocalFailure) && rank == 0 {
                    Err(CaptureError::Invalid("injected failed producer preparation".into()).into())
                } else {
                    Ok(local)
                };
                if matches!(scenario, Scenario::DecodeBudget) && rank == 1 {
                    let spare = plan.plan().limits.per_step.host_bytes - ledger.step().host_bytes;
                    assert_eq!(
                        ledger
                            .reserve(CaptureUsage {
                                host_bytes: spare,
                                ..Default::default()
                            })
                            .unwrap(),
                        None
                    );
                }
                let before = ledger.total();
                let result = exchange.exchange(local, &mut ledger);
                assert!(ledger.total().host_bytes >= before.host_bytes);
                assert!(ledger.total().retained_bytes >= reserved.retained_bytes);
                (
                    result,
                    transport.calls.load(Ordering::SeqCst),
                    transport.poison.load(Ordering::SeqCst),
                    ledger.total(),
                )
            })
        })
        .collect::<Vec<_>>()
        .into_iter()
        .map(|thread| thread.join().unwrap())
        .collect()
}

#[test]
fn concurrent_rank_exchange_delivers_exact_raw_and_reduced_values_after_agreement() {
    for transform in [
        CaptureTransform::Slice,
        CaptureTransform::Summary,
        CaptureTransform::Histogram {
            edges: vec![-2.0, 0.0, 2.0],
        },
    ] {
        let results = run(3, transform.clone(), Scenario::Success);
        let mut reference = None;
        for (result, calls, poison, _) in results {
            let result = result.unwrap();
            assert_eq!(calls, 3);
            assert!(!poison);
            assert_eq!(result.producers(), [0, 2]);
            let payload = result.capture().record().payload.clone();
            if let Some(reference) = &reference {
                assert_eq!(&payload, reference);
            } else {
                reference = Some(payload);
            }
            assert_eq!(result.context().forward_epoch, 19);
        }
        let plan = plan_for(transform, false);
        let mut ordinary = CaptureSession::new(plan);
        ordinary.begin_step(CapturePhase::Prefill, 0).unwrap();
        ordinary
            .observe(
                &mut Backend::default(),
                "block.output",
                &Value {
                    shape: vec![3, 20],
                    data: TensorObservationData::F32(
                        (0..60).map(|i| i as f32 * 0.25 - 3.0).collect(),
                    ),
                },
            )
            .unwrap();
        let payload = ordinary.take_step().unwrap().records.remove(0).payload;
        if let Some(CapturePayload::Tensor(_)) = &payload {
            assert_eq!(reference.unwrap(), payload);
        } else {
            assert_reduced_close(
                reference.as_ref().unwrap().as_ref().unwrap(),
                payload.as_ref().unwrap(),
            );
        }
    }
}

#[test]
fn failed_preparation_reports_failure_without_payload_or_publication_on_any_rank() {
    let results = run(3, CaptureTransform::Slice, Scenario::LocalFailure);
    for (rank, (result, calls, poison, _)) in results.into_iter().enumerate() {
        assert_eq!(calls, 1);
        assert!(!poison);
        if rank == 0 {
            let Err(PartitionCaptureExchangeError::LocalRejected {
                stage: PartitionCaptureExchangeStage::Preparation,
                source,
                ..
            }) = result
            else {
                panic!("expected agreed local preparation rejection")
            };
            assert!(matches!(
                *source,
                PartitionCaptureExchangeError::Capture(CaptureError::Invalid(_))
            ));
        } else {
            assert!(matches!(
                result,
                Err(PartitionCaptureExchangeError::PeerRejected {
                    rank: 0,
                    stage: PartitionCaptureExchangeStage::Preparation
                })
            ));
        }
    }
}

#[test]
fn prepaid_verdict_survives_one_rank_exhausting_its_decoder_budget() {
    let results = run(3, CaptureTransform::Slice, Scenario::DecodeBudget);
    for (rank, (result, calls, poison, _)) in results.into_iter().enumerate() {
        assert_eq!(calls, 3);
        assert!(!poison);
        if rank == 1 {
            let Err(PartitionCaptureExchangeError::LocalRejected {
                stage: PartitionCaptureExchangeStage::Delivery,
                source,
                ..
            }) = result
            else {
                panic!("expected agreed local decoder rejection")
            };
            assert!(matches!(
                *source,
                PartitionCaptureExchangeError::Capture(CaptureError::Limit {
                    budget: CaptureBudget::Host,
                    ..
                })
            ));
        } else {
            assert!(matches!(
                result,
                Err(PartitionCaptureExchangeError::PeerRejected {
                    rank: 1,
                    stage: PartitionCaptureExchangeStage::Delivery
                })
            ));
        }
    }
}

#[test]
fn malformed_receipts_reject_delivery_and_changed_layout_fences_every_rank() {
    for (scenario, expected_calls) in [
        (Scenario::CorruptPayload, 3),
        (Scenario::DifferentIdentity, 1),
    ] {
        for (result, calls, poison, _) in run(3, CaptureTransform::Slice, scenario) {
            assert!(result.is_err());
            assert_eq!(calls, expected_calls);
            assert_eq!(poison, matches!(scenario, Scenario::DifferentIdentity));
        }
    }
}

#[test]
fn exchange_admission_fails_before_submission_and_cannot_refund_credits() {
    let plan = plan_for(CaptureTransform::Slice, false);
    let transport = transport(world(1), 0, Fault::None);
    let mut ledger = CaptureLedger::new(&plan);
    let authority = receipt(&plan, &mut ledger, 1);
    let spare = plan.plan().limits.per_step.host_bytes - ledger.step().host_bytes;
    ledger
        .reserve(CaptureUsage {
            host_bytes: spare,
            ..Default::default()
        })
        .unwrap();
    let before = ledger.total();
    assert!(matches!(
        PartitionCaptureExchange::admit(&transport, authority, &mut ledger),
        Err(PartitionCaptureExchangeError::Capture(
            CaptureError::Limit { .. }
        ))
    ));
    assert_eq!(transport.calls.load(Ordering::SeqCst), 0);
    assert_eq!(ledger.total(), before);
    assert!(!transport.poison.load(Ordering::SeqCst));
    let mut ledger = CaptureLedger::new(&plan);
    let authority = receipt(&plan, &mut ledger, 1);
    let before = ledger.total();
    let exchange = PartitionCaptureExchange::admit(&transport, authority, &mut ledger).unwrap();
    let reserved = exchange.reserved();
    drop(exchange);
    ledger.begin_step();
    assert_eq!(ledger.total(), before.checked_add(reserved).unwrap());
}

#[test]
fn exact_completion_failures_never_resolve_live_outputs_or_release_a_result() {
    for call in 0..3 {
        for fault in [
            Fault::Deadline(call),
            Fault::Wait(call),
            Fault::Submit(call),
            Fault::Resolve(call),
        ] {
            let plan = plan_for(CaptureTransform::Slice, false);
            let transport = transport(world(1), 0, fault);
            let mut ledger = CaptureLedger::new(&plan);
            let authority = receipt(&plan, &mut ledger, 1);
            let exchange =
                PartitionCaptureExchange::admit(&transport, authority, &mut ledger).unwrap();
            let local = local_record(&plan, exchange.receipt_plan(), 0, &mut ledger);
            let result = exchange.exchange(Ok(local), &mut ledger);
            assert!(result.is_err(), "{fault:?}");
            assert_eq!(transport.calls.load(Ordering::SeqCst), call + 1);
            assert_eq!(
                transport.resolves.load(Ordering::SeqCst),
                call + usize::from(matches!(fault, Fault::Resolve(_)))
            );
            assert!(transport.poison.load(Ordering::SeqCst));
            if matches!(fault, Fault::Deadline(_)) {
                assert!(matches!(
                    result,
                    Err(PartitionCaptureExchangeError::Deadline { .. })
                ));
            } else {
                let Err(PartitionCaptureExchangeError::Backend(error)) = result else {
                    panic!("expected retained backend source")
                };
                assert!(std::error::Error::source(&error)
                    .unwrap()
                    .is::<std::io::Error>());
            }
        }
    }
    let plan = plan_for(CaptureTransform::Slice, false);
    let transport = transport(world(1), 0, Fault::ShortGather);
    let mut ledger = CaptureLedger::new(&plan);
    let authority = receipt(&plan, &mut ledger, 1);
    let exchange = PartitionCaptureExchange::admit(&transport, authority, &mut ledger).unwrap();
    let local = local_record(&plan, exchange.receipt_plan(), 0, &mut ledger);
    assert!(matches!(
        exchange.exchange(Ok(local), &mut ledger),
        Err(PartitionCaptureExchangeError::Protocol(_))
    ));
    assert!(transport.poison.load(Ordering::SeqCst));
}

#[test]
fn empty_global_selection_still_exchanges_each_producer_acknowledgment() {
    let world = world(3);
    let plan = Arc::new(plan_for(CaptureTransform::Slice, true));
    let threads: Vec<_> = (0..3)
        .map(|rank| {
            let world = world.clone();
            let plan = plan.clone();
            std::thread::spawn(move || {
                let transport = transport(world, rank, Fault::None);
                let mut ledger = CaptureLedger::new(&plan);
                let authority = receipt(&plan, &mut ledger, 3);
                let exchange =
                    PartitionCaptureExchange::admit(&transport, authority, &mut ledger).unwrap();
                let local = local_record(&plan, exchange.receipt_plan(), rank, &mut ledger);
                let result = exchange.exchange(Ok(local), &mut ledger).unwrap();
                assert_eq!(transport.calls.load(Ordering::SeqCst), 3);
                assert_eq!(result.producers(), [0, 2]);
                assert!(result.capture().contributions().is_empty());
                let Some(CapturePayload::Tensor(value)) = &result.capture().record().payload else {
                    panic!("expected typed empty data")
                };
                assert_eq!(value.shape(), [2, 0]);
                assert_eq!(value.data(), &TensorObservationData::F32(Vec::new()));
            })
        })
        .collect();
    for thread in threads {
        thread.join().unwrap();
    }
}

#[test]
fn invalid_local_receipt_bounds_are_agreed_before_any_payload_submission() {
    for local in [None, Some(Vec::new()), Some(vec![0; 16385])] {
        let plan = plan_for(CaptureTransform::Slice, false);
        let transport = transport(world(1), 0, Fault::None);
        let mut ledger = CaptureLedger::new(&plan);
        let authority = receipt(&plan, &mut ledger, 1);
        let exchange = PartitionCaptureExchange::admit(&transport, authority, &mut ledger).unwrap();
        let before = ledger.total();
        assert!(matches!(
            exchange.exchange(Ok(local), &mut ledger),
            Err(PartitionCaptureExchangeError::LocalRejected {
                stage: PartitionCaptureExchangeStage::Preparation,
                ..
            })
        ));
        assert_eq!(transport.calls.load(Ordering::SeqCst), 1);
        assert_eq!(ledger.total(), before);
        assert!(!transport.poison.load(Ordering::SeqCst));
    }
}

#[test]
fn overflowing_transport_buffers_are_rejected_before_reservation_or_submission() {
    let plan = plan_for(CaptureTransform::Slice, false);
    let transport = transport(world(1), 0, Fault::None);
    for bound in [u64::MAX, isize::MAX as u64 + 1] {
        let mut ledger = CaptureLedger::new(&plan);
        let slice = resolve_slice(&plan.points()[0], &plan.plan().selections[0], &[3, 20]).unwrap();
        let projection = CaptureSlicePartition::new(
            &[3, 20],
            &slice,
            1,
            &ComponentCoordinateMap::range(20, 0..20).unwrap(),
            16,
        )
        .unwrap();
        let authority = PartitionCaptureReceiptPlan::new(
            plan.clone(),
            context(&plan),
            vec![PartitionCaptureProducer {
                rank: 0,
                projection,
            }],
            1,
            PartitionCaptureReceiptLimits {
                max_producers: 1,
                max_fragments: 16,
                max_record_bytes: bound,
            },
            &mut ledger,
        )
        .unwrap();
        let before = ledger.total();
        assert!(matches!(
            PartitionCaptureExchange::admit(&transport, authority, &mut ledger),
            Err(PartitionCaptureExchangeError::Capture(
                CaptureError::Overflow
            ))
        ));
        assert_eq!(transport.calls.load(Ordering::SeqCst), 0);
        assert_eq!(ledger.total(), before);
    }
}
