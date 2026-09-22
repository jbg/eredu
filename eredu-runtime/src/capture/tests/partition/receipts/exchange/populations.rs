//! Native callback populations are compared with the actual neutral exchange.
use super::*;
use crate::capture::partition::{
    PartitionCaptureBuffer, PartitionCaptureFrame, PartitionCaptureTransportDemand as D,
};
use std::cell::RefCell;
struct Recorded {
    inner: Transport,
    calls: RefCell<Vec<D>>,
}
impl ConsensusTransport for Recorded {
    type Error = std::io::Error;
    fn participant_count(&self) -> usize {
        self.inner.participant_count()
    }
    fn all_gather_words(&self, words: &[u32]) -> Result<Vec<u32>, Self::Error> {
        self.inner.all_gather_words(words)
    }
}
impl BoundedConsensusTransport for Recorded {
    type Completion = Done;
    type GatherOutput = usize;
    fn submit_all_gather_words(
        &self,
        words: &[u32],
    ) -> Result<Submission<usize, Done>, Self::Error> {
        self.inner.submit_all_gather_words(words)
    }
    fn resolve_all_gather_words(&self, words: usize) -> Result<Vec<u32>, Self::Error> {
        self.inner.resolve_all_gather_words(words)
    }
}
impl PartitionCaptureTransport for Recorded {
    fn capture_rank(&self) -> usize {
        self.inner.capture_rank()
    }
    fn capture_wait(&self) -> Result<BoundedCompletionWait, CaptureError> {
        self.inner.capture_wait()
    }
    fn ensure_capture_active(&self) -> Result<(), BackendFailure> {
        self.calls.borrow_mut().push(D::Active);
        self.inner.ensure_capture_active()
    }
    fn estimate_capture_gather(&self, n: usize) -> Result<CaptureUsage, CaptureError> {
        self.inner.estimate_capture_gather(n)
    }
    fn fail_capture_exchange(&self, e: &PartitionCaptureExchangeError) {
        self.inner.fail_capture_exchange(e)
    }
    fn capture_word_destination(
        &self,
        n: usize,
    ) -> Result<PartitionCaptureBuffer<u32>, PartitionCaptureExchangeError> {
        self.calls.borrow_mut().push(D::Words {
            maximum_elements: n,
        });
        self.inner.capture_word_destination(n)
    }
    fn capture_byte_destination(
        &self,
        n: usize,
    ) -> Result<PartitionCaptureBuffer<u8>, PartitionCaptureExchangeError> {
        self.calls.borrow_mut().push(D::Bytes {
            maximum_elements: n,
        });
        self.inner.capture_byte_destination(n)
    }
    fn gather_capture_frame(
        &self,
        frame: &PartitionCaptureFrame<'_>,
        wait: BoundedCompletionWait,
    ) -> Result<PartitionCaptureBuffer<u32>, PartitionCaptureExchangeError> {
        self.calls.borrow_mut().push(D::Gather {
            kind: frame.kind(),
            maximum_words: frame.words().len(),
        });
        self.inner.gather_capture_frame(frame, wait)
    }
}
#[test]
fn receipt_transport_population_covers_real_rank_padding_and_completion() {
    for transform in [CaptureTransform::Slice, CaptureTransform::Summary] {
        let world = world(3);
        let plan = Arc::new(plan_for(transform, false));
        let threads = (0..3)
            .map(|rank| {
                let world = world.clone();
                let plan = plan.clone();
                std::thread::spawn(move || {
                    let transport = Recorded {
                        inner: transport(world, rank, Fault::None),
                        calls: RefCell::new(Vec::new()),
                    };
                    let mut ledger = CaptureLedger::new(&plan);
                    let authority = receipt(&plan, &mut ledger, 3);
                    let expected = authority.transport_demands().unwrap().collect::<Vec<_>>();
                    assert!(
                        transport.calls.borrow().is_empty(),
                        "describing a receipt submits no work"
                    );
                    let local = local_record(&plan, &authority, rank, &mut ledger);
                    let exchange =
                        PartitionCaptureExchange::admit(&transport, authority, &mut ledger)
                            .unwrap();
                    let result = exchange.exchange(Ok(local), &mut ledger).unwrap();
                    assert_eq!(result.producers(), [0, 2]);
                    let actual = transport.calls.into_inner();
                    assert_eq!(actual.len(), expected.len());
                    for (actual, bound) in actual.iter().zip(&expected) {
                        match (actual, bound) {
                            (D::Active, D::Active) => (),
                            (
                                D::Gather {
                                    kind: a,
                                    maximum_words: n,
                                },
                                D::Gather {
                                    kind: b,
                                    maximum_words: m,
                                },
                            ) => {
                                assert_eq!(a, b);
                                assert!(n <= m);
                            }
                            (
                                D::Words {
                                    maximum_elements: n,
                                },
                                D::Words {
                                    maximum_elements: m,
                                },
                            )
                            | (
                                D::Bytes {
                                    maximum_elements: n,
                                },
                                D::Bytes {
                                    maximum_elements: m,
                                },
                            ) => assert!(n <= m),
                            _ => panic!(
                                "transport callback differs: actual {actual:?}, source {bound:?}"
                            ),
                        }
                    }
                    // The inactive pipeline rank still validates every remote rank's
                    // padding. Its absence cannot erase decoder scratch allowance.
                    assert_eq!(
                        actual
                            .iter()
                            .filter(|d| matches!(d, D::Bytes { .. }))
                            .count(),
                        3
                    );
                })
            })
            .collect::<Vec<_>>();
        for thread in threads {
            thread.join().unwrap();
        }
    }
}
