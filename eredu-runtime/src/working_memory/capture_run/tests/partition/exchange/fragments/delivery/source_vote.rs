use super::*;
use std::cell::Cell;
struct Sources {
    base: Ranks,
    scalars: [u32; 4],
    failures: Cell<usize>,
}
impl ConsensusTransport for Sources {
    type Error = Infallible;
    fn participant_count(&self) -> usize {
        4
    }
    fn all_gather_words(&self, _: &[u32]) -> Result<Vec<u32>, Infallible> {
        panic!("unbounded source vote")
    }
}
impl BoundedConsensusTransport for Sources {
    type Completion = Done;
    type GatherOutput = ();
    fn submit_all_gather_words(&self, _: &[u32]) -> Result<Submission<(), Done>, Infallible> {
        panic!("untyped source vote")
    }
    fn resolve_all_gather_words(&self, _: ()) -> Result<Vec<u32>, Infallible> {
        panic!("untyped source output")
    }
}
impl PartitionCaptureTransport for Sources {
    fn capture_rank(&self) -> usize {
        self.base.local
    }
    fn capture_wait(&self) -> Result<BoundedCompletionWait, CaptureError> {
        self.base.capture_wait()
    }
    fn ensure_capture_active(&self) -> Result<(), BackendFailure> {
        Ok(())
    }
    fn estimate_capture_gather(&self, n: usize) -> Result<CaptureUsage, CaptureError> {
        self.base.estimate_capture_gather(n)
    }
    fn fail_capture_exchange(&self, _: &PartitionCaptureExchangeError) {
        self.failures.set(self.failures.get() + 1);
    }
    fn capture_word_destination(
        &self,
        n: usize,
    ) -> Result<PartitionCaptureBuffer<u32>, PartitionCaptureExchangeError> {
        self.base.capture_word_destination(n)
    }
    fn capture_byte_destination(
        &self,
        n: usize,
    ) -> Result<PartitionCaptureBuffer<u8>, PartitionCaptureExchangeError> {
        self.base.capture_byte_destination(n)
    }
    fn gather_capture_frame(
        &self,
        frame: &PartitionCaptureFrame<'_>,
        _: BoundedCompletionWait,
    ) -> Result<PartitionCaptureBuffer<u32>, PartitionCaptureExchangeError> {
        assert_eq!(frame.kind(), PartitionCaptureFrameKind::Source);
        self.base.calls.borrow_mut().push(frame.kind());
        let mut out = PartitionCaptureBuffer::funded(frame.gathered_words(), &self.base.funding)?;
        for rank in 0..4 {
            let mut header: [u32; 16] = frame.words().try_into().unwrap();
            header[3] = rank as u32;
            header[13] = self.scalars[rank];
            out.extend_from_slice(&header)?;
        }
        Ok(out)
    }
}
#[test]
fn contiguous_source_vote_requires_every_nonempty_rank_without_inventing_empty_sources() {
    for scalars in [
        [2, 0, 0, 2],
        [2, 2, 0, 2],
        [2, 0, 0, 0],
        [2, 0, 0, 1],
        [2, 3, 0, 2],
    ] {
        let valid = matches!(scalars, [2, 0, 0, 2] | [2, 2, 0, 2]);
        let source = source();
        let (funding, used, _, _) = funding();
        let mut quota = CaptureLedger::new(source.admission());
        quota.begin_step();
        let receipt = receipt(
            &source,
            PartitionCaptureCombination::Disjoint,
            &funding,
            &mut quota,
        );
        assert!(receipt.producer(1).unwrap().fragments().is_empty());
        assert!(receipt.producer(2).is_none());
        let transport = Sources {
            base: Ranks {
                local: 0,
                bytes: RefCell::new(std::array::from_fn(|_| None)),
                funding: funding.clone(),
                reject: false,
                calls: RefCell::new(vec![]),
            },
            scalars,
            failures: Cell::new(0),
        };
        let before = used.load(Ordering::SeqCst);
        let constructor = PreparedPartitionCaptureCoordination::<Sources>::prepare_metadata_bytes(
            receipt.context(),
        )
        .unwrap();
        assert_eq!(used.load(Ordering::SeqCst), before);
        let coordination = PreparedPartitionCaptureCoordination::prepare(
            &transport,
            &source,
            receipt.context(),
            &funding,
            &mut quota,
        )
        .unwrap();
        assert_eq!(used.load(Ordering::SeqCst) - before, constructor);
        let spent = quota.total();
        let result = coordination.coordinate_contiguous_source(
            0,
            &receipt,
            Some(&TensorDtype::F32),
            estimate(0).capture,
            &quota,
        );
        assert_eq!(result.is_ok(), valid);
        if valid {
            assert_eq!(result.unwrap(), TensorDtype::F32);
        }
        assert_eq!(
            *transport.base.calls.borrow(),
            [PartitionCaptureFrameKind::Source]
        );
        assert_eq!(transport.failures.get(), usize::from(!valid));
        assert_eq!(quota.total(), spent);
    }
}

#[test]
fn empty_overlap_receiver_keeps_actual_local_chunk_shape_and_original_source() {
    use super::super::prefill::{inference, projected_source};
    use crate::capture::partition::PartitionPrefillReceiverSource;
    for transform in [
        CaptureTransform::Slice,
        CaptureTransform::Summary,
        CaptureTransform::Histogram {
            edges: vec![0.0, 1.0, 2.0],
        },
    ] {
        for width in [0, 1] {
            let source = projected_source(transform.clone());
            let (funding, _, _, _) = funding();
            let mut quota = CaptureLedger::new(source.admission());
            quota.begin_step();
            let producers = [
                PartitionCaptureContiguousProducer {
                    rank: 1,
                    coordinates: 0..width,
                },
                PartitionCaptureContiguousProducer {
                    rank: 3,
                    coordinates: width..3,
                },
                PartitionCaptureContiguousProducer {
                    rank: 0,
                    coordinates: 3..7,
                },
            ];
            let receipt = PartitionCaptureReceiptPlan::new_contiguous_shared_funded(
                &source,
                &context(&source, 0),
                2,
                &producers,
                PartitionCaptureCombination::Disjoint,
                4,
                PartitionCaptureReceiptLimits {
                    max_producers: 3,
                    max_fragments: 3,
                    max_record_bytes: 64 << 10,
                },
                &funding,
                &mut quota,
            )
            .unwrap();
            assert!(receipt.producer(1).unwrap().fragments().is_empty());
            let spent = quota.total();
            let foreign = projected_source(transform.clone());
            for k in 0..3 {
                let receiver = PartitionPrefillReceiverSource::prepare(
                    &receipt,
                    1,
                    TensorDtype::F16,
                    inference(),
                    k,
                )
                .unwrap();
                assert_eq!(receiver.source_shape(), [2, 1, width as usize]);
                assert_eq!(receiver.dtype(), &TensorDtype::F16);
                assert!(receiver.matches(&source, 0, 1, k, receipt.context().forward_epoch));
                assert!(!receiver.matches(&foreign, 0, 1, k, receipt.context().forward_epoch));
                assert!(!receiver.matches(&source, 0, 1, k + 1, receipt.context().forward_epoch));
            }
            for (rank, k) in [(0, 0), (2, 0), (1, 3)] {
                assert!(PartitionPrefillReceiverSource::prepare(
                    &receipt,
                    rank,
                    TensorDtype::F16,
                    inference(),
                    k
                )
                .is_err());
            }
            let mut wrong = inference();
            wrong.cached_positions += 1;
            assert!(PartitionPrefillReceiverSource::prepare(
                &receipt,
                1,
                TensorDtype::F16,
                wrong,
                0
            )
            .is_err());
            assert_eq!(quota.total(), spent);
            assert!(PartitionPrefillReceiverSource::control_bytes().unwrap() > 0);
        }
    }
}
