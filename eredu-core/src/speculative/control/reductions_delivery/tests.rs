use super::*;
use crate::{
    HostPreparationAuthority, ObservationPosition, SharedTensorObservation, SpeculativeBuffer,
    TensorObservation, TensorObservationData,
    capture::{
        CAPTURE_SCHEMA_VERSION, CaptureHistogram, CaptureOutcome, CapturePayload, CapturePhase,
        CaptureRecord, CaptureStepOutcome, CaptureUsage, CapturedStep, SharedCapturedStep,
    },
    speculative::{
        SpeculativeActivationCapture, SpeculativeActivationOrigin, SpeculativeActivationPhase,
        SpeculativePrefillReduction, SpeculativePrefillReductionStatus,
    },
};
use std::sync::{Arc, Mutex};
struct Retires(u8, Arc<Mutex<Vec<u8>>>);
impl Drop for Retires {
    fn drop(&mut self) {
        self.1.lock().unwrap().push(self.0);
    }
}
fn origin() -> SpeculativeActivationOrigin {
    SpeculativeActivationOrigin {
        request: crate::SpeculativeRequestId::new(9),
        committed_tokens: 0,
        prediction: 0,
        prefix_digest: [3; 32],
        optimistic: false,
    }
}
fn report(drops: Arc<Mutex<Vec<u8>>>) -> SpeculativePrefillReductions {
    let preview = SharedTensorObservation::retain(
        TensorObservation::new(vec![3], TensorObservationData::F32(vec![1.25, -2.5, 7.0])).unwrap(),
        Retires(1, drops),
    );
    let records = [
        CapturePayload::SharedTensor(preview),
        CapturePayload::Histogram(CaptureHistogram {
            edges: vec![-3.0, 0.0, 3.0, 8.0],
            counts: vec![1, 2, 3],
            below: 1,
            above: 2,
            non_finite: 1,
        }),
    ]
    .into_iter()
    .enumerate()
    .map(|(index, payload)| SpeculativePrefillReduction {
        phase: SpeculativeActivationPhase::TargetPrefill,
        selection_index: index,
        logical_sequence: 11,
        covered_sequence: 11,
        first_invocation: Some(17),
        last_invocation: Some(23),
        windows: 5,
        status: SpeculativePrefillReductionStatus::Complete,
        record: CaptureRecord {
            schema_version: CAPTURE_SCHEMA_VERSION,
            selection_id: format!("selected-{index}"),
            path: "blocks.2.output".into(),
            node_id: "block.2".into(),
            position: ObservationPosition::AfterIntervention,
            source_shape: Some(vec![1, 11, 2]),
            source_dtype: Some(crate::checkpoint::TensorDtype::Bf16),
            selected_shape: Some(vec![1, 4, 2]),
            outcome: CaptureOutcome::Captured,
            payload: Some(payload),
            charged: CaptureUsage {
                captures: 5,
                retained_bytes: 32,
                host_bytes: 64,
                encoded_bytes: 256,
            },
        },
    })
    .collect();
    SpeculativePrefillReductions {
        logical_invocation: 17,
        origin: origin(),
        records,
        interventions: Vec::new(),
        charged: CaptureUsage {
            captures: 0,
            retained_bytes: 0,
            host_bytes: 4096,
            encoded_bytes: 8192,
        },
    }
}
fn envelope(report: SharedSpeculativePrefillReductions) -> SpeculativeActivationCapture {
    SpeculativeActivationCapture {
        admission_identity: Some("actual-admission".into()),
        invocation: 23,
        origin: origin(),
        phase: SpeculativeActivationPhase::PredictionPrefill,
        prefill_span: None,
        completed: true,
        captures: crate::capture::SharedCapturedStep::retain(
            CapturedStep {
                outcome: CaptureStepOutcome::Committed,
                phase: CapturePhase::Prefill,
                invocation: None,
                prediction_index: 0,
                records: Vec::new(),
                partitions: Vec::new(),
                interventions: Vec::new(),
                step_usage: CaptureUsage::default(),
                cumulative_usage: CaptureUsage {
                    captures: 10,
                    retained_bytes: 64,
                    host_bytes: 8192,
                    encoded_bytes: 16384,
                },
                capture_seconds: 0.0,
            },
            (),
        ),
        prefill_reductions: Some(report),
    }
}
#[test]
fn aggregate_delivery_preserves_wire_payload_aliases_and_escaped_custody() {
    let drops = Arc::new(Mutex::new(Vec::new()));
    let pending = PreparedSpeculativePrefillReductions::retain(Retires(2, drops.clone()));
    let raw = report(drops.clone());
    let rows = raw.records.as_ptr();
    let path = raw.records[0].record.path.as_ptr();
    let wire = serde_json::to_value(envelope(
        SharedSpeculativePrefillReductions::retain(raw.clone(), ()),
    ))
    .unwrap();
    let shared = pending.finish(raw);
    assert_eq!(shared.as_reductions().records.as_ptr(), rows);
    let source = shared.clone();
    let record = envelope(shared);
    assert_eq!(serde_json::to_value(&record).unwrap(), wire);
    let decoded: SpeculativeActivationCapture = serde_json::from_value(wire).unwrap();
    assert!(decoded.prefill_reductions.is_some());
    assert_eq!(decoded, record);
    drop(decoded);
    let alias = record.clone();
    let reductions = alias.prefill_reductions.as_ref().unwrap();
    assert!(source.same_storage(reductions));
    assert_eq!(reductions.records[0].record.path.as_ptr(), path);
    assert_eq!(reductions.charged.host_bytes, 4096);
    let mut records = SpeculativeBuffer::try_new_retained(
        1,
        HostPreparationAuthority::retain(Retires(3, drops.clone())),
    )
    .unwrap();
    records.try_push(record).unwrap();
    let mut drained = records.into_iter();
    let escaped = drained.next().unwrap();
    drop((drained, source));
    assert_eq!(*drops.lock().unwrap(), [3]);
    assert_eq!(
        escaped.prefill_reductions.as_ref().unwrap().records[1]
            .record
            .payload,
        Some(CapturePayload::Histogram(CaptureHistogram {
            edges: vec![-3.0, 0.0, 3.0, 8.0],
            counts: vec![1, 2, 3],
            below: 1,
            above: 2,
            non_finite: 1
        }))
    );
    drop(escaped);
    assert_eq!(*drops.lock().unwrap(), [3]);
    drop(alias);
    assert_eq!(*drops.lock().unwrap(), [3, 1, 2]);
}
#[test]
fn empty_aggregate_destination_retires_once_without_publication() {
    let drops = Arc::new(Mutex::new(Vec::new()));
    assert!(
        PreparedSpeculativePrefillReductions::retained_control_bytes::<Retires>().unwrap()
            > std::mem::size_of::<SpeculativePrefillReductions>() as u64
    );
    drop(PreparedSpeculativePrefillReductions::retain(Retires(
        4,
        drops.clone(),
    )));
    assert_eq!(*drops.lock().unwrap(), [4]);
}
