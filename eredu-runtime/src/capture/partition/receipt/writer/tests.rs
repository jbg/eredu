use super::*;
use eredu_core::{SharedTensorObservation, TensorObservation, TensorObservationData, ObservationPosition};
use eredu_nn::workspace::WorkspaceMetadataAccount;
use std::sync::{Arc, atomic::{AtomicBool, AtomicUsize, Ordering}};

#[derive(Debug)]
struct Account { used: Arc<AtomicUsize>, retired: Arc<AtomicBool>, refuse: Arc<AtomicBool> }
impl WorkspaceMetadataAccount for Account {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), WorkspaceMetadataFundingError> {
        if self.refuse.load(Ordering::SeqCst) { return Err(WorkspaceMetadataFundingError::Overflow); }
        self.used.fetch_add(bytes, Ordering::SeqCst);
        Ok(())
    }
}
impl Drop for Account { fn drop(&mut self) { self.retired.store(true, Ordering::SeqCst); } }
fn source() -> (PartitionCaptureContext, CaptureRecord) {
    let context = PartitionCaptureContext {
        artifact_identity: "artifact-λ".into(), execution_identity: "TP/PP".into(),
        run_identity: "run-雪".into(), overlay_identity: None, capture_plan_identity: "capture-source".into(),
        selection_index: 2, phase: CapturePhase::Decode, prediction: 3, forward_epoch: 11, invocation: None,
    };
    let raw = TensorObservation::new(vec![5], TensorObservationData::F32(vec![
        17.25, -0.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY,
    ])).unwrap();
    let record = CaptureRecord {
        schema_version: CAPTURE_SCHEMA_VERSION, selection_id: "final-雪".into(), path: "model.logits".into(),
        node_id: "decoder-λ".into(), position: ObservationPosition::ReadOnly,
        source_shape: Some(vec![1, 1, 13]), source_dtype: Some(TensorDtype::F16), selected_shape: Some(vec![1, 1, 13]),
        outcome: CaptureOutcome::Truncated { available_elements: 13, emitted_elements: 5 },
        payload: Some(CapturePayload::SharedTensor(SharedTensorObservation::retain(raw, ()))),
        charged: CaptureUsage { host_bytes: 211, encoded_bytes: 8192, ..Default::default() },
    };
    (context, record)
}

#[test]
fn borrowed_partition_receipt_keeps_canonical_unicode_nonfinite_values_and_paid_output() {
    let (context, record) = source();
    let tensor = record.payload.as_ref().unwrap().as_tensor().unwrap();
    let before = match tensor.data() { TensorObservationData::F32(values) => values.as_ptr(), _ => unreachable!() };
    let used = Arc::new(AtomicUsize::new(0)); let retired = Arc::new(AtomicBool::new(false));
    let funding = WorkspaceMetadataFunding::new(Account { used: used.clone(), retired: retired.clone(), refuse: Arc::new(AtomicBool::new(false)) }).unwrap();
    let actual = PartitionCaptureRecordEncoding::new(&context, "receipt-λ", 5, PartitionCaptureCombination::Disjoint, &record, 8192).encode(&funding).unwrap();
    // The caller-owned ordinary wire DTO is an independent expected consumer.
    let ordinary = PartitionCaptureProducerRecord {
        schema_version: PARTITION_CAPTURE_SCHEMA_VERSION, combination: PartitionCaptureCombination::Disjoint,
        receipt_plan_identity: "receipt-λ".into(), context: context.clone(), producer_rank: 5,
        source_dtype: Some(TensorDtype::F16), fragments: vec![PartitionCaptureFragmentRecord { fragment_index: 0, record: record.clone() }],
    };
    assert_eq!(actual.as_ref(), serde_json::to_vec(&ordinary).unwrap());
    let json: serde_json::Value = serde_json::from_slice(&actual).unwrap();
    assert_eq!(json["context"]["forward_epoch"], 11);
    assert_eq!(json["context"]["run_identity"], "run-雪");
    assert_eq!(json["fragments"][0]["record"]["outcome"]["emitted_elements"], 5);
    assert_eq!(json["fragments"][0]["record"]["payload"]["value"]["data"]["values"], serde_json::json!([17.25, -0.0, "nan", "+inf", "-inf"]));
    assert_eq!(match record.payload.as_ref().unwrap().as_tensor().unwrap().data() { TensorObservationData::F32(v) => v.as_ptr(), _ => unreachable!() }, before);
    assert!(used.load(Ordering::SeqCst) > actual.len());
    drop(funding); drop(ordinary); drop(record); drop(context);
    assert!(!retired.load(Ordering::SeqCst));
    assert!(std::str::from_utf8(&actual).unwrap().contains("final-雪"));
    drop(actual); assert!(retired.load(Ordering::SeqCst));
}

#[test]
fn borrowed_partition_receipt_refusal_retains_spending_and_custody() {
    let (context, record) = source();
    let used = Arc::new(AtomicUsize::new(0)); let retired = Arc::new(AtomicBool::new(false)); let refuse = Arc::new(AtomicBool::new(false));
    let funding = WorkspaceMetadataFunding::new(Account { used: used.clone(), retired: retired.clone(), refuse: refuse.clone() }).unwrap();
    let failed = PartitionCaptureRecordEncoding::new(&context, "receipt", 0, PartitionCaptureCombination::Disjoint, &record, 7).encode(&funding).unwrap_err();
    assert!(matches!(failed, PartitionCaptureEncodingError::Source { .. }));
    let spent = used.load(Ordering::SeqCst); assert!(spent > 0);
    refuse.store(true, Ordering::SeqCst);
    let refused = PartitionCaptureRecordEncoding::new(&context, "receipt", 0, PartitionCaptureCombination::Disjoint, &record, 8192).encode(&funding).unwrap_err();
    assert!(matches!(refused, PartitionCaptureEncodingError::Funding { .. }));
    assert_eq!(used.load(Ordering::SeqCst), spent);
    drop(funding); assert!(!retired.load(Ordering::SeqCst));
    drop(failed); assert!(!retired.load(Ordering::SeqCst));
    drop(refused); assert!(retired.load(Ordering::SeqCst));
}


#[test]
fn contiguous_empty_wire_keeps_an_empty_sequence_and_actual_optional_precision() {
    let (context, _) = source();
    let funding = WorkspaceMetadataFunding::new(Account { used: Arc::new(AtomicUsize::new(0)),
        retired: Arc::new(AtomicBool::new(false)), refuse: Arc::new(AtomicBool::new(false)) }).unwrap();
    for dtype in [None, Some(TensorDtype::Bf16)] {
        let output = encode_contiguous(&context, "receipt-empty", 1, PartitionCaptureCombination::Disjoint,
            dtype.as_ref(), None, 8192, &funding).unwrap();
        let expected = PartitionCaptureProducerRecord { schema_version: PARTITION_CAPTURE_SCHEMA_VERSION,
            combination: PartitionCaptureCombination::Disjoint, receipt_plan_identity: "receipt-empty".into(),
            context: context.clone(), producer_rank: 1, source_dtype: dtype, fragments: Vec::new() };
        assert_eq!(output.as_ref(), serde_json::to_vec(&expected).unwrap());
        let wire: serde_json::Value = serde_json::from_slice(output.as_ref()).unwrap();
        assert_eq!(wire["fragments"], serde_json::json!([]));
        assert_eq!(wire["source_dtype"].is_null(), expected.source_dtype.is_none());
    }
}
