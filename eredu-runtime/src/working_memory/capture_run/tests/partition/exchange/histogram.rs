use super::super::histogram::{histogram_record, histogram_source};
use super::*;

#[test]
fn complete_histogram_delivery_retains_partition_evidence_and_spent_final_vote() {
    for reject in [false, true] {
        let source = histogram_source();
        let h = plan(&source).initialization_peak_bytes();
        let pool = WorkingMemoryPool::new(h, 0).unwrap();
        let (reservation, run) = fresh(&pool, h);
        let mut bank = run
            .prepare_capture_run(&reservation, plan(&source))
            .unwrap();
        let mut frame = bank
            .begin_step(CapturePhase::Prefill, 0)
            .unwrap()
            .prepare()
            .unwrap();
        let context = context(&source, 0);
        let geometry = CaptureHistogramGeometry::prepare(
            source.admission(),
            0,
            CapturePhase::Prefill,
            0,
            None,
        )
        .unwrap();
        let record = histogram_record(frame.records()[0].clone(), &geometry);
        let charged = record.charged;
        let (funding, used, _, retired) = funding();
        let mut quota = CaptureLedger::new(source.admission());
        quota.begin_step();
        let receipt = PartitionCaptureReceiptPlan::new_complete_shared_funded(
            &source,
            &context,
            3,
            4,
            PartitionCaptureReceiptLimits {
                max_producers: 1,
                max_fragments: 1,
                max_record_bytes: 64 << 10,
            },
            &funding,
            &mut quota,
        )
        .unwrap();
        let encoded = PartitionCaptureRecordEncoding::new(
            &context,
            receipt.identity(),
            3,
            PartitionCaptureCombination::Disjoint,
            &record,
            64 << 10,
        )
        .encode(&funding)
        .unwrap();
        let transport = Receipts {
            source: encoded.to_vec(),
            funding: funding.clone(),
            reject_delivery: reject,
            calls: RefCell::new(Vec::new()),
        };
        drop(encoded);
        frame.prepare_partition_evidence(&funding).unwrap();
        let evidence =
            PreparedPartitionCaptureEvidence::prepare(&receipt, charged, &funding, &mut quota)
                .unwrap();
        let exchange = PartitionCaptureExchange::admit(&transport, receipt, &mut quota).unwrap();
        let prepared =
            PreparedPartitionTensorDelivery::prepare(exchange, TensorDtype::F16, charged, &funding)
                .unwrap()
                .with_evidence(evidence);
        let before = quota.total();
        let result = prepared.deliver(&mut frame);
        assert_eq!(result.is_err(), reject);
        assert_eq!(quota.total(), before);
        assert_eq!(frame.records()[0].payload, record.payload);
        assert_eq!(frame.records()[0].charged, charged);
        assert!(frame.take_histogram(0).is_err());
        assert_eq!(frame.partition_evidence().len(), 1);
        assert_eq!(
            frame.partition_evidence()[0].contributions[0].charged,
            charged
        );
        assert_eq!(
            frame.partition_evidence()[0].contributions[0].local.starts,
            geometry.starts()
        );
        assert_eq!(
            *transport.calls.borrow(),
            vec![
                PartitionCaptureFrameKind::Preparation,
                PartitionCaptureFrameKind::Payload,
                PartitionCaptureFrameKind::Delivery
            ]
        );
        assert!(used.load(Ordering::SeqCst) > 0);
        let delivered = frame
            .finish(CaptureStepOutcome::Aborted, unlimited(), unlimited(), 0.0)
            .unwrap();
        let escaped = delivered.clone();
        drop(bank);
        drop(run);
        drop(reservation);
        drop(transport);
        drop(funding);
        drop(result);
        assert_eq!(pool.used_bytes().unwrap(), h);
        assert!(!retired.load(Ordering::SeqCst));
        drop(delivered);
        assert!(!retired.load(Ordering::SeqCst));
        drop(escaped);
        assert_eq!(pool.used_bytes().unwrap(), 0);
        assert!(retired.load(Ordering::SeqCst));
    }
}

#[test]
fn complete_histogram_program_lends_the_prepaid_transform_charge_once() {
    let source = histogram_source();
    let h = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(h, 0).unwrap();
    let (reservation, run) = fresh(&pool, h);
    let mut bank = run
        .prepare_capture_run(&reservation, plan(&source))
        .unwrap();
    let mut frame = bank
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare()
        .unwrap();
    let (funding, _, _, retired) = funding();
    let transport = ProgramVotes {
        funding: funding.clone(),
        fail_at: None,
        participants: 1,
        calls: RefCell::new(Vec::new()),
        failed: Cell::new(false),
    };
    let rows = [PartitionCaptureProducerSource {
        producer: 0,
        dtype: Some(TensorDtype::F16),
        estimate: PartitionCaptureNativeEstimate {
            capture: native_usage(),
            generated_creation_bytes: 0,
        },
    }];
    let mut program = PreparedPartitionCaptureProgram::new(
        &transport,
        &source,
        &context(&source, 0),
        &rows,
        PartitionCaptureReceiptLimits {
            max_producers: 1,
            max_fragments: 1,
            max_record_bytes: 64 << 10,
        },
        &funding,
    )
    .unwrap();
    let mut quota = CaptureLedger::new(source.admission());
    quota.begin_step();
    let epoch = DistributedCommitEpoch::FIRST;
    program
        .prepare(
            &source,
            CapturePhase::Prefill,
            0,
            epoch,
            &mut frame,
            &mut quota,
        )
        .unwrap();
    program.coordinate(epoch, &quota).unwrap();
    let prepaid = quota.total();
    assert!(program
        .reservation(0, &TensorDtype::F32, native_usage())
        .is_err());
    let mut foreign = native_usage();
    foreign.host_bytes += 1;
    assert!(program.reservation(0, &TensorDtype::F16, foreign).is_err());
    let policy =
        crate::capture::CaptureObservationStep::new(source.admission(), CapturePhase::Prefill, 0)
            .unwrap();
    assert!(policy
        .reserve_value(
            program
                .reservation(0, &TensorDtype::F16, native_usage())
                .unwrap(),
            native_usage()
        )
        .unwrap()
        .is_none());
    assert!(policy
        .reserve_value(
            program
                .reservation(0, &TensorDtype::F16, native_usage())
                .unwrap(),
            native_usage()
        )
        .is_err());
    assert_eq!(
        quota.total(),
        prepaid,
        "local reservations consume their paid allowance, never the global ledger twice"
    );
    let claim = frame.take_histogram(0).unwrap();
    let count = claim.geometry().elements() as u64;
    let mut bins = claim.prepare().unwrap();
    bins.add_bin(0, count).unwrap();
    let receipt = bins.finish(0, 0, 0).unwrap();
    frame.record_histogram(receipt, TensorDtype::F16, native_usage())
        .unwrap();
    assert!(frame.take_histogram(0).is_err());
    drop(program);
    drop(frame);
    drop(bank);
    drop(run);
    drop(reservation);
    drop(transport);
    drop(funding);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert!(retired.load(Ordering::SeqCst));
}
