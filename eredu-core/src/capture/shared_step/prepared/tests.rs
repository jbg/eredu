use super::*;
use std::sync::Mutex;
struct Retired(usize, Arc<Mutex<Vec<usize>>>);
impl Drop for Retired {
    fn drop(&mut self) {
        self.1.lock().unwrap().push(self.0)
    }
}
#[test]
fn empty_prepared_frame_control_is_exact_and_retires_without_publication() {
    let drops = Arc::new(Mutex::new(Vec::new()));
    let owner = PreparedCapturedStep::retain(Retired(7, drops.clone()));
    let block = std::alloc::Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
        .extend(std::alloc::Layout::new::<crate::capture::retained_payload::RetainedCapturePayload<CapturedStep>>())
        .unwrap()
        .0
        .pad_to_align();
    assert_eq!(
        PreparedCapturedStep::retained_control_bytes::<Retired>().unwrap(),
        (block.size() + std::mem::size_of::<Retired>()) as u64
    );
    assert_eq!(
        std::mem::size_of::<PreparedCapturedStep>(),
        std::mem::size_of::<UnpublishedCapturedStep>()
    );
    assert!(owner.0.get().records.is_empty());
    assert!(owner.0.get().partitions.is_empty());
    assert!(owner.0.get().interventions.is_empty());
    drop(owner);
    assert_eq!(*drops.lock().unwrap(), [7]);
}
#[test]
fn prepared_frame_fills_same_control_and_payload_addresses_for_both_outcomes() {
    for outcome in [CaptureStepOutcome::Committed, CaptureStepOutcome::Aborted] {
        let drops = Arc::new(Mutex::new(Vec::new()));
        let mut step = CapturedStep {
            outcome,
            phase: CapturePhase::Decode,
            invocation: None,
            prediction_index: 2,
            records: Vec::new(),
            partitions: Vec::new(),
            interventions: Vec::new(),
            step_usage: CaptureUsage::default(),
            cumulative_usage: CaptureUsage::default(),
            capture_seconds: 0.25,
        };
        step.records.push(CaptureRecord {
            schema_version: 1,
            selection_id: "selected".into(),
            path: "block.output".into(),
            node_id: "block".into(),
            position: crate::ObservationPosition::BeforeIntervention,
            source_shape: Some(vec![1, 4]),
            source_dtype: Some(crate::checkpoint::TensorDtype::F32),
            selected_shape: Some(vec![1, 4]),
            outcome: CaptureOutcome::Failed {
                reason: CaptureFailureReason::Native,
                message: "original native diagnostic".into(),
            },
            payload: None,
            charged: CaptureUsage::default(),
        });
        let records = step.records.as_ptr();
        let path = step.records[0].path.as_ptr();
        let pending = PreparedCapturedStep::retain(Retired(1, drops.clone()));
        let allocation = Arc::as_ptr(pending.0 .0.as_ref().unwrap());
        let published = pending.finish(step);
        assert_eq!(Arc::as_ptr(published.0 .0.as_ref().unwrap()), allocation);
        assert_eq!(published.records().as_ptr(), records);
        assert_eq!(published.records()[0].path.as_ptr(), path);
        assert_eq!(published.outcome(), outcome);
        let decoded: CapturedStep =
            serde_json::from_slice(&serde_json::to_vec(&published).unwrap()).unwrap();
        assert_eq!(&decoded, published.as_step());
        let alias = published.clone();
        drop(published);
        assert!(drops.lock().unwrap().is_empty());
        drop(alias);
        assert_eq!(*drops.lock().unwrap(), [1]);
    }
}
#[test]
fn prepared_frame_concurrent_final_aliases_retire_once_after_nested_tensor() {
    let drops = Arc::new(Mutex::new(Vec::new()));
    let tensor = crate::SharedTensorObservation::retain(
        crate::TensorObservation::new(vec![1], crate::TensorObservationData::F32(vec![2.5]))
            .unwrap(),
        Retired(1, drops.clone()),
    );
    let step = CapturedStep {
        outcome: CaptureStepOutcome::Committed,
        phase: CapturePhase::Prefill,
        invocation: None,
        prediction_index: 0,
        records: vec![CaptureRecord {
            schema_version: 1,
            selection_id: "tensor".into(),
            path: "block.output".into(),
            node_id: "block".into(),
            position: crate::ObservationPosition::BeforeIntervention,
            source_shape: None,
            source_dtype: None,
            selected_shape: None,
            outcome: CaptureOutcome::Captured,
            payload: Some(CapturePayload::SharedTensor(tensor)),
            charged: CaptureUsage::default(),
        }],
        partitions: vec![],
        interventions: vec![],
        step_usage: CaptureUsage::default(),
        cumulative_usage: CaptureUsage::default(),
        capture_seconds: 0.0,
    };
    let frame = PreparedCapturedStep::retain(Retired(2, drops.clone())).finish(step);
    let barrier = Arc::new(std::sync::Barrier::new(5));
    let threads: Vec<_> = (0..4)
        .map(|_| {
            let frame = frame.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                drop(frame)
            })
        })
        .collect();
    drop(frame);
    assert!(drops.lock().unwrap().is_empty());
    barrier.wait();
    for thread in threads {
        thread.join().unwrap()
    }
    assert_eq!(*drops.lock().unwrap(), [1, 2]);
}
