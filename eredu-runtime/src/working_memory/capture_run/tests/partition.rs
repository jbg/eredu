use super::*;
mod construction;
mod exchange;
mod histogram;
mod summary;
use eredu_nn::workspace::{HostMetadataAccount, HostMetadataFunding, HostMetadataFundingError};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

#[derive(Debug)]
struct Account {
    used: Arc<AtomicUsize>,
    refuse: Arc<AtomicBool>,
    retired: Arc<AtomicBool>,
}
impl HostMetadataAccount for Account {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
        if self.refuse.load(Ordering::SeqCst) {
            return Err(HostMetadataFundingError::Overflow);
        }
        self.used.fetch_add(bytes, Ordering::SeqCst);
        Ok(())
    }
}
impl Drop for Account {
    fn drop(&mut self) {
        self.retired.store(true, Ordering::SeqCst);
    }
}
fn funding() -> (
    HostMetadataFunding,
    Arc<AtomicUsize>,
    Arc<AtomicBool>,
    Arc<AtomicBool>,
) {
    let used = Arc::new(AtomicUsize::new(0));
    let refuse = Arc::new(AtomicBool::new(false));
    let retired = Arc::new(AtomicBool::new(false));
    let funding = HostMetadataFunding::new(Account {
        used: used.clone(),
        refuse: refuse.clone(),
        retired: retired.clone(),
    })
    .unwrap();
    (funding, used, refuse, retired)
}
fn context(source: &SharedCapturePlan, index: usize) -> PartitionCaptureContext {
    PartitionCaptureContext {
        artifact_identity: "artifact-é".into(),
        execution_identity: "parallel-α".into(),
        run_identity: "run-雪".into(),
        overlay_identity: None,
        capture_plan_identity: source.admission().identity().into(),
        selection_index: index,
        phase: CapturePhase::Prefill,
        prediction: 0,
        forward_epoch: 17,
        invocation: None,
        invocation_window: None,
    }
}
fn native_usage() -> CaptureUsage {
    CaptureUsage {
        captures: 1,
        host_bytes: 307,
        retained_bytes: 811,
        encoded_bytes: 8192,
    }
}
fn wire(
    mut record: CaptureRecord,
    geometry: &CaptureTensorGeometry<'_>,
    context: &PartitionCaptureContext,
) -> (Vec<u8>, CaptureUsage, Vec<f32>) {
    let selected: Vec<u64> = geometry
        .starts()
        .iter()
        .zip(geometry.ends())
        .zip(geometry.strides())
        .map(|((a, b), s)| (b - a).div_ceil(*s))
        .collect();
    let available = selected.iter().product::<u64>();
    let special = [17.25, -0.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
    let values: Vec<_> = (0..geometry.elements())
        .map(|i| special[i % special.len()])
        .collect();
    record.source_shape = Some(geometry.source_shape().iter().map(|v| *v as u64).collect());
    record.selected_shape = Some(selected);
    record.source_dtype = Some(TensorDtype::F16);
    record.outcome = if available > values.len() as u64 {
        CaptureOutcome::Truncated {
            available_elements: available,
            emitted_elements: values.len() as u64,
        }
    } else {
        CaptureOutcome::Captured
    };
    record.payload = Some(CapturePayload::Tensor(
        TensorObservation::new(
            geometry.shape().to_vec(),
            TensorObservationData::F32(values.clone()),
        )
        .unwrap(),
    ));
    record.charged = record.charged.checked_add(native_usage()).unwrap();
    let charged = record.charged;
    let receipt = PartitionCaptureProducerRecord {
        schema_version: PARTITION_CAPTURE_SCHEMA_VERSION,
        combination: PartitionCaptureCombination::Disjoint,
        receipt_plan_identity: "retained-receipt".into(),
        context: context.clone(),
        producer_rank: 3,
        source_dtype: Some(TensorDtype::F16),
        fragments: vec![PartitionCaptureFragmentRecord {
            fragment_index: 0,
            record,
        }],
    };
    // Exercise the ordinary parser's escaped-string scratch while retaining the
    // exact decoded Unicode identity. These wire bytes are independent inputs.
    let text = serde_json::to_string(&receipt)
        .unwrap()
        .replace("é", "\\u00e9")
        .replace("α", "\\u03b1");
    (text.into_bytes(), charged, values)
}
fn expected<'a>(
    context: &'a PartitionCaptureContext,
    charged: CaptureUsage,
) -> PartitionCaptureTensorReceipt<'a> {
    PartitionCaptureTensorReceipt {
        context,
        identity: "retained-receipt",
        producer: 3,
        dtype: TensorDtype::F16,
        charged,
    }
}

#[test]
fn partition_receipt_decodes_directly_into_original_full_and_preview_claims() {
    let source = source();
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
    let (funding, used, _, _) = funding();
    let mut outputs = Vec::new();
    for index in [0, 2] {
        let context = context(&source, index);
        let record = frame.records()[index].clone();
        let claim = frame.take_tensor(index).unwrap();
        let (bytes, charged, values) = wire(record, claim.geometry(), &context);
        let decoded = claim
            .decode_partition_receipt(&bytes, expected(&context, charged), &funding)
            .unwrap();
        let data = match decoded.observation().data() {
            TensorObservationData::F32(data) => data,
            _ => panic!("f32"),
        };
        assert_eq!(data.len(), values.len());
        for (actual, expected) in data.iter().zip(&values) {
            assert!(
                actual.to_bits() == expected.to_bits() || (actual.is_nan() && expected.is_nan())
            );
        }
        outputs.push(decoded.observation().clone());
        frame
            .record_tensor(decoded, TensorDtype::F16, native_usage())
            .unwrap();
        assert_eq!(frame.records()[index].charged, charged);
        assert!(frame.take_tensor(index).is_err());
    }
    assert!(used.load(Ordering::SeqCst) > 0);
    drop(frame);
    drop(bank);
    drop(run);
    drop(reservation);
    assert!(ledger(&pool).0 >= h);
    assert_eq!(outputs[1].shape(), &[3]);
    drop(outputs);
    assert_eq!(ledger(&pool).0, 0);
}

#[test]
fn partition_receipt_rejection_preserves_spent_claim_and_original_error_custody() {
    for case in ["identity", "duplicate", "syntax", "reservation"] {
        let source = source();
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
        let record = frame.records()[0].clone();
        let claim = frame.take_tensor(0).unwrap();
        let (mut bytes, charged, _) = wire(record, claim.geometry(), &context);
        let (funding, used, refuse, retired) = funding();
        match case {
            "identity" => {
                bytes = String::from_utf8(bytes)
                    .unwrap()
                    .replace("retained-receipt", "foreign-receipt")
                    .into_bytes();
            }
            "duplicate" => {
                bytes.splice(1..1, b"\"producer_rank\":3,".iter().copied());
            }
            "syntax" => {
                bytes.truncate(bytes.len() - 4);
            }
            "reservation" => refuse.store(true, Ordering::SeqCst),
            _ => unreachable!(),
        }
        let error = claim
            .decode_partition_receipt(&bytes, expected(&context, charged), &funding)
            .unwrap_err();
        assert!(frame.take_tensor(0).is_err());
        if case != "reservation" {
            assert!(used.load(Ordering::SeqCst) > 0);
        }
        drop(frame);
        drop(bank);
        drop(run);
        drop(reservation);
        drop(funding);
        assert!(ledger(&pool).0 >= h);
        assert!(!retired.load(Ordering::SeqCst));
        drop(error);
        assert_eq!(ledger(&pool).0, 0);
        assert!(retired.load(Ordering::SeqCst));
    }
}

mod vocabulary;
