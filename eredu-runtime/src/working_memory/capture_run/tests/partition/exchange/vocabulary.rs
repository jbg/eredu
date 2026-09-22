use super::super::vocabulary::{vocabulary_record, vocabulary_source};
use super::*;
#[test]
fn complete_vocabulary_delivery_binds_terminal_rows_and_keeps_prefill_charge_after_rejected_vote() {
    for scores in [false, true] {
        let source = vocabulary_source(scores, 6);
        let h = plan(&source).initialization_peak_bytes();
        let pool = capture_test_ledger(h, 0).unwrap();
        let (reservation, run) = fresh(&pool, h);
        let mut bank = run
            .prepare_capture_run(&reservation, plan(&source))
            .unwrap();
        let geometry = super::super::super::candidates::g(
            6,
            if scores {
                OutputDemand::LastPosition
            } else {
                OutputDemand::Sequence
            },
        );
        let mut frame = bank
            .begin_step(CapturePhase::Prefill, 0)
            .unwrap()
            .prepare_prefill_with_progression(geometry)
            .unwrap();
        let context = context(&source, 0);
        let rows = frame.partition_terminal_rows(0).unwrap().unwrap();
        assert_eq!(rows, if scores { 1 } else { 2 });
        let record = vocabulary_record(frame.records()[0].clone(), scores, rows as u64, true);
        let charged = record.charged;
        let (funding, used, _, retired) = funding();
        let mut quota = CaptureLedger::new(source.admission());
        quota.begin_step();
        let (mut receipt, _) =
            PartitionCaptureReceiptPlan::new_complete_shared_funded_terminal_global(
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
                Some(rows),
            )
            .unwrap();
        assert_eq!(
            receipt.producers().next().unwrap().1.global_shape(),
            &[1, rows as u64, 4]
        );
        let mut transport = Receipts {
            source: Vec::new(),
            funding: funding.clone(),
            reject_delivery: true,
            calls: RefCell::new(Vec::new()),
        };
        let before = used.load(Ordering::SeqCst);
        let allowance_bytes =
            PreparedPartitionCaptureAllowance::execution_metadata_bytes::<Receipts>(
                &receipt,
                transport.capture_rank(),
            )
            .unwrap();
        assert_eq!(used.load(Ordering::SeqCst), before);
        let mut allowance = PreparedPartitionCaptureAllowance::prepare(
            &transport,
            &mut receipt,
            PartitionCaptureNativeEstimate {
                capture: native_usage(),
                generated_creation_bytes: 0,
            },
            &funding,
            &mut quota,
        )
        .unwrap();
        let constructor_bytes = used.load(Ordering::SeqCst) - before;
        // Cost preparation narrows the admitted wire bound and seals a new
        // receipt identity. A producer encodes only that final borrowed plan.
        let encoded = PartitionCaptureRecordEncoding::new(
            &context,
            receipt.identity(),
            3,
            PartitionCaptureCombination::Disjoint,
            &record,
            receipt.max_record_bytes() as usize,
        )
        .encode(&funding)
        .unwrap();
        transport.source = encoded.to_vec();
        drop(encoded);
        frame.prepare_partition_evidence(&funding).unwrap();
        let evidence = PreparedPartitionCaptureEvidence::prepare(
            &receipt,
            charged,
            &funding,
            allowance.quota_mut(),
        )
        .unwrap();
        let exchange =
            PartitionCaptureExchange::admit(&transport, receipt, allowance.quota_mut()).unwrap();
        let delivery =
            PreparedPartitionTensorDelivery::prepare(exchange, TensorDtype::F16, charged, &funding)
                .unwrap()
                .with_evidence(evidence);
        let prepaid = quota.total();
        for k in 0..3 {
            let chunk = crate::prefill::PrefillChunk {
                input: k * 2..(k + 1) * 2,
                position: 2 + k * 2,
                output: geometry.output.for_chunk(k == 2),
            };
            let decision = frame
                .begin_prefill_hook(0, &chunk, MODEL_LOGITS_OBSERVATION_PATH)
                .unwrap();
            if k < 2 {
                assert_eq!(decision, crate::capture::CapturePrefillHookDecision::Ignore);
            } else {
                assert_eq!(decision, crate::capture::CapturePrefillHookDecision::First);
                let before = used.load(Ordering::SeqCst);
                let charge = allowance.take_remote_charge().unwrap();
                assert_eq!(
                    constructor_bytes + used.load(Ordering::SeqCst) - before,
                    allowance_bytes,
                );
                frame
                    .reserve_remote_prefill_hook(0, TensorDtype::F16, charge)
                    .unwrap();
                assert!(allowance.take_remote_charge().is_err());
                if scores {
                    frame.finish_token_scores_prefill_hook(0, &chunk).unwrap();
                } else {
                    frame.finish_candidate_prefill_hook(0, &chunk).unwrap();
                }
            }
            frame.complete_prefill_chunk(k).unwrap();
        }
        frame.finish_local_prefill_targets().unwrap();
        assert!(frame.finish_prefill_targets().is_err());
        let error = delivery.deliver(&mut frame).unwrap_err();
        assert_eq!(
            frame.records()[0].payload,
            record.payload,
            "delivery failed before the intended final vote: {error:?}"
        );
        assert_eq!(frame.records()[0].source_shape, record.source_shape);
        assert_eq!(frame.records()[0].charged, charged);
        assert_eq!(quota.total(), prepaid);
        frame.finish_prefill_targets().unwrap();
        assert_eq!(
            frame.partition_evidence()[0].contributions[0].local.shape,
            vec![1, rows as u64, 4]
        );
        let output = frame
            .finish(CaptureStepOutcome::Aborted, unlimited(), unlimited(), 0.0)
            .unwrap();
        let escaped = output.clone();
        drop(output);
        drop(allowance);
        drop(error);
        drop(bank);
        drop(run);
        drop(reservation);
        drop(transport);
        drop(funding);
        assert_eq!(pool.payload_used_bytes().unwrap(), h);
        assert!(!retired.load(Ordering::SeqCst));
        drop(escaped);
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
        assert!(retired.load(Ordering::SeqCst));
    }
}
