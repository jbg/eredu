use super::*;

#[test]
fn delayed_fragment_host_owner_binds_actual_epoch_and_retains_refused_source_holds() {
    for wrong_projection in [false, true] {
        let source = source();
        let (funding, _, _, _) = funding();
        let transport = Receipts {
            source: vec![],
            funding: funding.clone(),
            reject_delivery: false,
            calls: RefCell::new(vec![]),
        };
        let make = |epoch, swap, ledger: &mut CaptureLedger| {
            let mut context = context(&source, 0);
            context.forward_epoch = epoch;
            context.run_identity = format!("original-run-{epoch}");
            let rows = [
                PartitionCaptureContiguousProducer {
                    rank: 0,
                    coordinates: if swap { 1..2 } else { 0..1 },
                },
                PartitionCaptureContiguousProducer {
                    rank: 3,
                    coordinates: if swap { 0..1 } else { 1..2 },
                },
                PartitionCaptureContiguousProducer {
                    rank: 1,
                    coordinates: 0..0,
                },
            ];
            PartitionCaptureReceiptPlan::new_contiguous_shared_funded(
                &source,
                &context,
                1,
                &rows,
                PartitionCaptureCombination::Disjoint,
                4,
                PartitionCaptureReceiptLimits {
                    max_producers: 3,
                    max_fragments: 3,
                    max_record_bytes: 64 << 10,
                },
                &funding,
                ledger,
            )
            .unwrap()
        };
        let mut quote = CaptureLedger::new(source.admission());
        quote.begin_step();
        let mut prototype = make(17, false, &mut quote);
        let geometry: Vec<_> = prototype
            .producers()
            .flat_map(|(rank, p)| {
                p.fragments().iter().enumerate().map(move |(fragment, g)| {
                    (rank, fragment, p.local_shape().to_vec(), g.local().clone())
                })
            })
            .collect();
        let transform = &source.admission().plan().selections[0].transform;
        let native: Vec<_> = geometry
            .iter()
            .map(
                |(rank, fragment, shape, slice)| PartitionCaptureFragmentSource {
                    producer: *rank,
                    fragment: *fragment,
                    local_shape: shape,
                    local_slice: slice,
                    transform,
                    dtype: TensorDtype::F16,
                    estimate: estimate(*rank),
                },
            )
            .collect();
        let quoted = PreparedPartitionFragmentAllowance::prepare(
            &transport,
            &mut prototype,
            &native,
            &funding,
            &mut quote,
        )
        .unwrap();
        drop(quoted);
        let h = PartitionFragmentHostPlan::prepare(&prototype)
            .unwrap()
            .initialization_peak_bytes();
        let pool = capture_test_ledger(h, 0).unwrap();
        let (reservation, run) = fresh(&pool, h);
        let prepared = run
            .prepare_partition_fragment_host(&reservation, prototype, None)
            .unwrap();
        assert_eq!(prepared.protected_bytes(), h);
        let mut ledger = CaptureLedger::new(source.admission());
        ledger.begin_step();
        let mut actual = make(91, wrong_projection, &mut ledger);
        let geometry: Vec<_> = actual
            .producers()
            .flat_map(|(rank, p)| {
                p.fragments().iter().enumerate().map(move |(fragment, g)| {
                    (rank, fragment, p.local_shape().to_vec(), g.local().clone())
                })
            })
            .collect();
        let native: Vec<_> = geometry
            .iter()
            .map(
                |(rank, fragment, shape, slice)| PartitionCaptureFragmentSource {
                    producer: *rank,
                    fragment: *fragment,
                    local_shape: shape,
                    local_slice: slice,
                    transform,
                    dtype: TensorDtype::F16,
                    estimate: estimate(*rank),
                },
            )
            .collect();
        let allowance = PreparedPartitionFragmentAllowance::prepare(
            &transport,
            &mut actual,
            &native,
            &funding,
            &mut ledger,
        )
        .unwrap();
        let spent = ledger.total();
        if wrong_projection {
            let error = prepared.bind(&actual, allowance).unwrap_err();
            drop((run, reservation));
            assert!(pool.payload_used_bytes().unwrap() > 0);
            drop(error);
            assert_eq!(pool.payload_used_bytes().unwrap(), 0);
        } else {
            let mut bank = prepared.bind(&actual, allowance).unwrap();
            assert_eq!(actual.context().forward_epoch, 91);
            let claim = bank
                .take_local(&actual, 0, &TensorDtype::F16, estimate(0))
                .unwrap();
            drop(claim);
            assert!(
                bank.take_local(&actual, 0, &TensorDtype::F16, estimate(0))
                    .is_err()
            );
            drop((run, reservation));
            assert!(pool.payload_used_bytes().unwrap() > 0);
            drop(bank);
            assert_eq!(pool.payload_used_bytes().unwrap(), 0);
        }
        assert_eq!(
            ledger.total(),
            spent,
            "binding neither recharges nor refunds logical capture"
        );
    }
}

#[test]
fn host_source_keeps_original_record_ceiling_across_actual_epoch_finalization() {
    for changed_ceiling in [false, true] {
        let source = source();
        let (funding, _, _, _) = funding();
        let transport = Receipts {
            source: vec![],
            funding: funding.clone(),
            reject_delivery: false,
            calls: RefCell::new(vec![]),
        };
        let make = |epoch: u64, label: &str, ceiling: u64, ledger: &mut CaptureLedger| {
            let mut context = context(&source, 0);
            context.forward_epoch = epoch;
            context.run_identity = label.into();
            let rows = [
                PartitionCaptureContiguousProducer {
                    rank: 0,
                    coordinates: 0..1,
                },
                PartitionCaptureContiguousProducer {
                    rank: 3,
                    coordinates: 1..2,
                },
            ];
            PartitionCaptureReceiptPlan::new_contiguous_shared_funded(
                &source,
                &context,
                1,
                &rows,
                PartitionCaptureCombination::Disjoint,
                4,
                PartitionCaptureReceiptLimits {
                    max_producers: 2,
                    max_fragments: 2,
                    max_record_bytes: ceiling,
                },
                &funding,
                ledger,
            )
            .unwrap()
        };
        let mut quotation = CaptureLedger::new(source.admission());
        quotation.begin_step();
        // This geometry-only Host source has no native scalar, transport cost,
        // source vote or issued epoch. Its original record ceiling is retained.
        let prototype = make(0, "retained-setup", 64 << 10, &mut quotation);
        let h = PartitionFragmentHostPlan::prepare(&prototype)
            .unwrap()
            .initialization_peak_bytes();
        let pool = capture_test_ledger(h, 0).unwrap();
        let (reservation, run) = fresh(&pool, h);
        let prepared = run
            .prepare_partition_fragment_host(&reservation, prototype, None)
            .unwrap();
        let mut ledger = CaptureLedger::new(source.admission());
        ledger.begin_step();
        let mut actual = make(
            91327,
            "retained-setup/actual-first-forward-91327",
            (64 << 10) + u64::from(changed_ceiling),
            &mut ledger,
        );
        let geometry: Vec<_> = actual
            .producers()
            .flat_map(|(rank, p)| {
                p.fragments().iter().enumerate().map(move |(fragment, g)| {
                    (rank, fragment, p.local_shape().to_vec(), g.local().clone())
                })
            })
            .collect();
        let transform = &source.admission().plan().selections[0].transform;
        let native: Vec<_> = geometry
            .iter()
            .map(
                |(rank, fragment, shape, slice)| PartitionCaptureFragmentSource {
                    producer: *rank,
                    fragment: *fragment,
                    local_shape: shape,
                    local_slice: slice,
                    transform,
                    dtype: TensorDtype::F16,
                    estimate: estimate(*rank),
                },
            )
            .collect();
        let allowance = PreparedPartitionFragmentAllowance::prepare(
            &transport,
            &mut actual,
            &native,
            &funding,
            &mut ledger,
        )
        .unwrap();
        assert!(
            actual.max_record_bytes() < 64 << 10,
            "actual nonempty envelopes tighten transport independently"
        );
        let spent = ledger.total();
        let result = prepared.bind(&actual, allowance);
        assert_eq!(
            result.is_err(),
            changed_ceiling,
            "Host still authenticates the original admitted ceiling"
        );
        assert_eq!(
            ledger.total(),
            spent,
            "binding cannot refund or reserve logical work"
        );
        drop((run, reservation));
        assert!(pool.payload_used_bytes().unwrap() > 0);
        drop(result);
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    }
}
