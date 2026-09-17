use super::*;
use crate::{checkpoint::TensorDtype, SharedTensorObservation, TensorObservationData};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

struct Retired(Arc<AtomicUsize>);
impl Drop for Retired {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
fn tensor() -> TensorObservation {
    TensorObservation::new(
        vec![1, 3],
        TensorObservationData::F32(vec![1.25, -2.5, 7.0]),
    )
    .unwrap()
}
fn step(payload: CapturePayload) -> CapturedStep {
    let mut records = Vec::with_capacity(3);
    records.push(CaptureRecord {
        schema_version: CAPTURE_SCHEMA_VERSION,
        selection_id: "actual-selection".into(),
        path: "block.2.output".into(),
        node_id: "block.2".into(),
        position: crate::ObservationPosition::BeforeIntervention,
        source_shape: Some(vec![1, 9]),
        source_dtype: Some(TensorDtype::Bf16),
        selected_shape: Some(vec![1, 3]),
        outcome: CaptureOutcome::Truncated {
            available_elements: 9,
            emitted_elements: 3,
        },
        payload: Some(payload),
        charged: CaptureUsage {
            captures: 1,
            retained_bytes: 128,
            host_bytes: 256,
            encoded_bytes: 512,
        },
    });
    CapturedStep {
        outcome: CaptureStepOutcome::Committed,
        phase: CapturePhase::Decode,
        invocation: Some(CaptureInvocationShape {
            batch: 1,
            sequence: 1,
            context: Some(9),
        }),
        prediction_index: 7,
        records,
        partitions: Vec::new(),
        interventions: Vec::new(),
        step_usage: CaptureUsage {
            captures: 1,
            retained_bytes: 128,
            host_bytes: 256,
            encoded_bytes: 512,
        },
        cumulative_usage: CaptureUsage {
            captures: 4,
            retained_bytes: 1024,
            host_bytes: 2048,
            encoded_bytes: 4096,
        },
        capture_seconds: 0.125,
    }
}

#[test]
fn aliases_borrow_the_same_frame_strings_shapes_and_tensor() {
    let retired = Arc::new(AtomicUsize::new(0));
    let raw = step(CapturePayload::SharedTensor(
        SharedTensorObservation::retain(tensor(), ()),
    ));
    let expected = raw.clone(); // Caller-side reference only; the shared owner never deep-clones.
    let shared = SharedCapturedStep::retain(raw, Retired(retired.clone()));
    let alias = shared.clone();
    assert!(shared.same_storage(&alias));
    assert!(std::ptr::eq(shared.as_step(), alias.as_ref()));
    assert_eq!(shared.records().as_ptr(), alias.records().as_ptr());
    assert_eq!(
        shared.records()[0].path.as_ptr(),
        alias.records()[0].path.as_ptr()
    );
    assert_eq!(
        shared.records()[0].source_shape.as_ref().unwrap().as_ptr(),
        alias.records()[0].source_shape.as_ref().unwrap().as_ptr()
    );
    assert_eq!(shared.as_step(), &expected);
    let decoded: CapturedStep =
        serde_json::from_value(serde_json::to_value(&shared).unwrap()).unwrap();
    assert!(matches!(
        decoded.records[0].payload,
        Some(CapturePayload::Tensor(_))
    ));
    assert_eq!(shared.as_step(), &decoded); // Equality grants no shared custody.
    assert_eq!(shared.outcome(), expected.outcome);
    assert_eq!(shared.phase(), expected.phase);
    assert_eq!(shared.invocation(), expected.invocation);
    assert_eq!(shared.prediction_index(), expected.prediction_index);
    assert_eq!(shared.step_usage(), expected.step_usage);
    assert_eq!(shared.cumulative_usage(), expected.cumulative_usage);
    assert_eq!(shared.capture_seconds(), expected.capture_seconds);
    assert_eq!(shared.partitions(), &expected.partitions);
    assert_eq!(shared.interventions(), &expected.interventions);
    let separate = SharedCapturedStep::retain(expected, ());
    assert_eq!(shared, separate);
    assert!(!shared.same_storage(&separate));
    drop(shared);
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    assert_eq!(
        alias.records()[0]
            .payload
            .as_ref()
            .unwrap()
            .as_tensor()
            .unwrap()
            .shape(),
        &[1, 3]
    );
    drop(alias);
    assert_eq!(retired.load(Ordering::SeqCst), 1);
}

#[test]
fn shared_step_wire_preserves_nonfinite_values_and_original_source_precision() {
    for dtype in [TensorDtype::F32, TensorDtype::F16, TensorDtype::Bf16] {
        let values = TensorObservation::new(
            vec![3],
            TensorObservationData::F32(vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY]),
        )
        .unwrap();
        let mut reference = step(CapturePayload::Tensor(values.clone()));
        reference.records[0].source_dtype = Some(dtype.clone());
        let expected = serde_json::to_value(&reference).unwrap();
        reference.records[0].payload = Some(CapturePayload::SharedTensor(
            SharedTensorObservation::retain(values, ()),
        ));
        let shared = SharedCapturedStep::retain(reference, ());
        let wire = serde_json::to_value(&shared).unwrap();
        assert_eq!(wire, expected);
        assert_eq!(wire["records"][0]["payload"]["kind"], "tensor");
        assert_eq!(
            wire["records"][0]["payload"]["value"]["data"]["values"],
            serde_json::json!(["nan", "+inf", "-inf"])
        );
        let raw: CapturedStep = serde_json::from_value(wire).unwrap();
        assert_eq!(raw.records[0].source_dtype, Some(dtype));
        assert!(matches!(
            raw.records[0].payload,
            Some(CapturePayload::Tensor(_))
        ));
        assert_eq!(raw.step_usage, shared.step_usage());
        assert_eq!(raw.cumulative_usage, shared.cumulative_usage());
    }
}

#[test]
fn frame_payload_drops_before_final_custody_on_last_alias() {
    struct FrameCharge {
        nested: Arc<AtomicUsize>,
        frames: Arc<AtomicUsize>,
    }
    impl Drop for FrameCharge {
        fn drop(&mut self) {
            assert_eq!(self.nested.load(Ordering::SeqCst), 1);
            self.frames.fetch_add(1, Ordering::SeqCst);
        }
    }
    let nested = Arc::new(AtomicUsize::new(0));
    let frames = Arc::new(AtomicUsize::new(0));
    let payload = SharedTensorObservation::retain(tensor(), Retired(nested.clone()));
    let shared = SharedCapturedStep::retain(
        step(CapturePayload::SharedTensor(payload)),
        FrameCharge {
            nested: nested.clone(),
            frames: frames.clone(),
        },
    );
    let last = shared.clone();
    drop(shared);
    assert_eq!(nested.load(Ordering::SeqCst), 0);
    assert_eq!(frames.load(Ordering::SeqCst), 0);
    drop(last);
    assert_eq!(nested.load(Ordering::SeqCst), 1);
    assert_eq!(frames.load(Ordering::SeqCst), 1);
}

#[test]
fn escaped_tensor_alias_retains_its_own_custody_after_frame_retirement() {
    let nested = Arc::new(AtomicUsize::new(0));
    let frames = Arc::new(AtomicUsize::new(0));
    let payload = SharedTensorObservation::retain(tensor(), Retired(nested.clone()));
    let shared = SharedCapturedStep::retain(
        step(CapturePayload::SharedTensor(payload)),
        Retired(frames.clone()),
    );
    let Some(CapturePayload::SharedTensor(payload)) = &shared.records()[0].payload else {
        unreachable!()
    };
    let escaped = payload.clone();
    let pointer = escaped.shape().as_ptr();
    drop(shared);
    assert_eq!(frames.load(Ordering::SeqCst), 1);
    assert_eq!(nested.load(Ordering::SeqCst), 0);
    assert_eq!(escaped.shape().as_ptr(), pointer);
    assert_eq!(escaped.as_observation(), &tensor());
    drop(escaped);
    assert_eq!(nested.load(Ordering::SeqCst), 1);
}

#[test]
fn aborted_and_empty_frames_keep_unmanaged_wire_defaults_and_cleanup() {
    let retired = Arc::new(AtomicUsize::new(0));
    let mut raw = step(CapturePayload::Tensor(tensor()));
    raw.outcome = CaptureStepOutcome::Aborted;
    raw.invocation = None;
    raw.records[0].payload = None;
    raw.records[0].source_dtype = None;
    raw.records[0].outcome = CaptureOutcome::Failed {
        reason: CaptureFailureReason::Native,
        message: "native failure: αβγ".into(),
    };
    let expected = serde_json::to_value(&raw).unwrap();
    let shared = SharedCapturedStep::retain(raw, Retired(retired.clone()));
    assert_eq!(serde_json::to_value(&shared).unwrap(), expected);
    assert_eq!(shared.outcome(), CaptureStepOutcome::Aborted);
    assert!(expected.get("invocation").is_none());
    assert!(expected["records"][0].get("source_dtype").is_none());
    let mut legacy = expected;
    legacy.as_object_mut().unwrap().remove("outcome");
    let mut decoded: CapturedStep = serde_json::from_value(legacy).unwrap();
    assert_eq!(decoded.outcome, CaptureStepOutcome::Untracked);
    decoded.records.clear();
    let empty = SharedCapturedStep::retain(decoded, ());
    assert!(empty.records().is_empty());
    assert_eq!(
        serde_json::from_value::<CapturedStep>(serde_json::to_value(empty).unwrap())
            .unwrap()
            .records
            .len(),
        0
    );
    drop(shared);
    assert_eq!(retired.load(Ordering::SeqCst), 1);
}

#[test]
fn unwinding_consumer_releases_shared_frame_and_nested_payload_once() {
    let nested = Arc::new(AtomicUsize::new(0));
    let frames = Arc::new(AtomicUsize::new(0));
    let result = std::panic::catch_unwind({
        let nested = nested.clone();
        let frames = frames.clone();
        move || {
            let payload = SharedTensorObservation::retain(tensor(), Retired(nested));
            let frame = SharedCapturedStep::retain(
                step(CapturePayload::SharedTensor(payload)),
                Retired(frames),
            );
            let alias = frame.clone();
            drop(frame);
            assert_eq!(alias.prediction_index(), 7);
            panic!("consumer unwind");
        }
    });
    assert!(result.is_err());
    assert_eq!(nested.load(Ordering::SeqCst), 1);
    assert_eq!(frames.load(Ordering::SeqCst), 1);
}
