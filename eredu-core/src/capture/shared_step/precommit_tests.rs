use super::*;
use crate::{
    checkpoint::TensorDtype, SharedTensorObservation, TensorObservation, TensorObservationData,
};
use std::sync::Mutex;

struct Retired(u8, Arc<Mutex<Vec<u8>>>);
impl Drop for Retired {
    fn drop(&mut self) {
        self.1.lock().unwrap().push(self.0);
    }
}
fn step(retirement: Arc<Mutex<Vec<u8>>>) -> CapturedStep {
    let tensor = SharedTensorObservation::retain(
        TensorObservation::new(
            vec![1, 3],
            TensorObservationData::F32(vec![1.25, -0.0, f32::INFINITY]),
        )
        .unwrap(),
        Retired(1, retirement),
    );
    CapturedStep {
        outcome: CaptureStepOutcome::Untracked,
        phase: CapturePhase::Decode,
        invocation: None,
        prediction_index: 3,
        records: vec![CaptureRecord {
            schema_version: CAPTURE_SCHEMA_VERSION,
            selection_id: "selected-é".into(),
            path: "block.output".into(),
            node_id: "block".into(),
            position: crate::ObservationPosition::BeforeIntervention,
            source_shape: Some(vec![1, 9]),
            source_dtype: Some(TensorDtype::Bf16),
            selected_shape: Some(vec![1, 3]),
            outcome: CaptureOutcome::Truncated {
                available_elements: 9,
                emitted_elements: 3,
            },
            payload: Some(CapturePayload::SharedTensor(tensor)),
            charged: CaptureUsage::default(),
        }],
        partitions: vec![],
        interventions: vec![],
        step_usage: CaptureUsage::default(),
        cumulative_usage: CaptureUsage::default(),
        capture_seconds: 0.25,
    }
}
#[test]
fn preallocated_owner_finalizes_both_outcomes_without_replacing_any_payload() {
    for outcome in [CaptureStepOutcome::Committed, CaptureStepOutcome::Aborted] {
        let retired = Arc::new(Mutex::new(Vec::new()));
        let raw = step(retired.clone());
        let records = raw.records.as_ptr();
        let path = raw.records[0].path.as_ptr();
        let shape = raw.records[0].source_shape.as_ref().unwrap().as_ptr();
        let pending = UnpublishedCapturedStep::retain(raw, Retired(2, retired.clone()));
        let allocation = Arc::as_ptr(pending.0 .0.as_ref().unwrap());
        let published = pending.finish(outcome);
        assert_eq!(Arc::as_ptr(published.0 .0.as_ref().unwrap()), allocation);
        assert_eq!(published.records().as_ptr(), records);
        assert_eq!(published.records()[0].path.as_ptr(), path);
        assert_eq!(
            published.records()[0]
                .source_shape
                .as_ref()
                .unwrap()
                .as_ptr(),
            shape
        );
        assert_eq!(published.outcome(), outcome);
        let decoded: CapturedStep =
            serde_json::from_slice(&serde_json::to_vec(&published).unwrap()).unwrap();
        assert_eq!(
            serde_json::to_value(&decoded).unwrap(),
            serde_json::to_value(&published).unwrap()
        );
        assert!(retired.lock().unwrap().is_empty());
        drop(decoded);
        drop(published);
        assert_eq!(*retired.lock().unwrap(), [1, 2]);
    }
}
#[test]
fn unpublished_drop_retires_real_nested_payload_before_final_custody() {
    let retired = Arc::new(Mutex::new(Vec::new()));
    let pending =
        UnpublishedCapturedStep::retain(step(retired.clone()), Retired(2, retired.clone()));
    drop(pending);
    assert_eq!(*retired.lock().unwrap(), [1, 2]);
}
#[test]
fn finalized_aliases_keep_original_owner_and_escaped_tensor_custody() {
    let retired = Arc::new(Mutex::new(Vec::new()));
    let published =
        UnpublishedCapturedStep::retain(step(retired.clone()), Retired(2, retired.clone()))
            .finish(CaptureStepOutcome::Aborted);
    let alias = published.clone();
    let Some(CapturePayload::SharedTensor(tensor)) = &published.records()[0].payload else {
        panic!("shared tensor")
    };
    let tensor = tensor.clone();
    drop(published);
    assert!(retired.lock().unwrap().is_empty());
    assert_eq!(alias.outcome(), CaptureStepOutcome::Aborted);
    drop(alias);
    assert_eq!(*retired.lock().unwrap(), [2]);
    drop(tensor);
    assert_eq!(*retired.lock().unwrap(), [2, 1]);
}

#[test]
fn concurrent_final_strong_owners_retire_one_frame_and_custody() {
    let retired = Arc::new(Mutex::new(Vec::new()));
    let frame = UnpublishedCapturedStep::retain(step(retired.clone()), Retired(2, retired.clone()))
        .finish(CaptureStepOutcome::Committed);
    let barrier = Arc::new(std::sync::Barrier::new(9));
    let threads: Vec<_> = (0..8)
        .map(|_| {
            let alias = frame.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                assert_eq!(alias.outcome(), CaptureStepOutcome::Committed);
                drop(alias);
            })
        })
        .collect();
    drop(frame);
    assert!(retired.lock().unwrap().is_empty());
    barrier.wait();
    for thread in threads {
        thread.join().unwrap();
    }
    assert_eq!(*retired.lock().unwrap(), [1, 2]);
}
