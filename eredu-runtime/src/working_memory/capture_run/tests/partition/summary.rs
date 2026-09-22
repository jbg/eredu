//! Fixed remote statistics retain the same spent claim and canonical envelope.
use super::*;
use crate::capture::partition::PartitionCaptureRecordEncoding;

#[test]
fn partition_summary_receipt_checks_scalar_counts_and_retains_failed_claim_custody() {
    for case in ["valid", "counts", "duplicate", "reservation"] {
        let mut raw = raw();
        raw.selections.truncate(1);
        raw.selections[0].transform = CaptureTransform::Summary;
        let source = admit(raw, point(), 4, false);
        let h = plan(&source).initialization_peak_bytes();
        let pool = capture_test_ledger(h, 0).unwrap();
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
        let mut record = frame.records()[0].clone();
        let claim = frame.take_summary(0).unwrap();
        let geometry = claim.geometry();
        assert_eq!(geometry.elements(), 10);
        let summary = CaptureSummary {
            elements: 10,
            finite: 4,
            non_finite: 6,
            nan: 2,
            positive_infinity: 2,
            negative_infinity: 2,
            min: Some(-4.0),
            max: Some(2.0),
            mean: Some(-1.0),
            rms: Some(10f64.sqrt()),
        };
        record.source_shape = Some(geometry.source_shape().iter().map(|n| *n as u64).collect());
        record.selected_shape = Some(geometry.shape().iter().map(|n| *n as u64).collect());
        record.source_dtype = Some(TensorDtype::F16);
        record.outcome = CaptureOutcome::Captured;
        record.payload = Some(CapturePayload::Summary(summary.clone()));
        record.charged = record.charged.checked_add(native_usage()).unwrap();
        let (funding, used, refuse, retired) = funding();
        let encoded = PartitionCaptureRecordEncoding::new(
            &context,
            "retained-receipt",
            3,
            PartitionCaptureCombination::Disjoint,
            &record,
            64 << 10,
        )
        .encode(&funding)
        .unwrap();
        let text = std::str::from_utf8(&encoded).unwrap();
        let bytes = match case {
            "counts" => text.replace("\"finite\":4", "\"finite\":5").into_bytes(),
            "duplicate" => text
                .replace("\"finite\":4", "\"finite\":4,\"finite\":4")
                .into_bytes(),
            _ => text.as_bytes().to_vec(),
        };
        drop(encoded);
        if case == "reservation" {
            refuse.store(true, Ordering::SeqCst);
        }
        let decoded =
            claim.decode_partition_receipt(&bytes, expected(&context, record.charged), &funding);
        assert!(frame.take_summary(0).is_err());
        assert!(used.load(Ordering::SeqCst) > 0);
        let error = if case == "valid" {
            let decoded = decoded.unwrap();
            assert_eq!(decoded.observation(), &summary);
            frame
                .record_summary(decoded, TensorDtype::F16, native_usage())
                .unwrap();
            assert_eq!(frame.records()[0].payload.as_ref(), record.payload.as_ref());
            None
        } else {
            Some(decoded.unwrap_err())
        };
        drop(frame);
        drop(bank);
        drop(run);
        drop(reservation);
        drop(funding);
        if let Some(error) = error {
            assert!(ledger(&pool).0 >= h);
            assert!(!retired.load(Ordering::SeqCst));
            drop(error);
        }
        assert_eq!(ledger(&pool).0, 0);
        assert!(retired.load(Ordering::SeqCst));
    }
}
