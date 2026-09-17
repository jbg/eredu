use super::*;
use crate::{SharedTensorObservation, TensorObservationData};
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
fn tensor(values: Vec<f32>) -> TensorObservation {
    TensorObservation::new(vec![values.len()], TensorObservationData::F32(values)).unwrap()
}

#[test]
fn shared_tensor_uses_legacy_nonfinite_wire_and_decodes_without_custody() {
    let raw = tensor(vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0, 3.25]);
    let expected = serde_json::to_value(CapturePayload::Tensor(raw.clone())).unwrap();
    let retired = Arc::new(AtomicUsize::new(0));
    let shared = SharedTensorObservation::retain(raw, Retired(retired.clone()));
    let wire_view = serde_json::to_value(CaptureTensorWire::new(&shared)).unwrap();
    let payload = CapturePayload::SharedTensor(shared);
    let wire = serde_json::to_string(&payload).unwrap();
    let value: serde_json::Value = serde_json::from_str(&wire).unwrap();
    assert_eq!(value, expected);
    assert_eq!(value, wire_view);
    assert_eq!(value["kind"], "tensor");
    assert_eq!(value["value"]["data"]["values"][0], "nan");
    assert_eq!(value["value"]["data"]["values"][1], "+inf");
    assert_eq!(value["value"]["data"]["values"][2], "-inf");
    let decoded: CapturePayload = serde_json::from_str(&wire).unwrap();
    let CapturePayload::Tensor(decoded) = decoded else {
        panic!("wire decoding must never reconstruct a shared owner");
    };
    drop(payload);
    assert_eq!(retired.load(Ordering::SeqCst), 1);
    let TensorObservationData::F32(values) = decoded.data() else {
        unreachable!()
    };
    assert!(values[0].is_nan());
    assert_eq!(values[1], f32::INFINITY);
    assert_eq!(values[2], f32::NEG_INFINITY);
    assert_eq!(values[3].to_bits(), (-0.0f32).to_bits());
    assert_eq!(values[4], 3.25);
    assert!(serde_json::from_str::<CapturePayload>(
        r#"{"kind":"shared_tensor","value":{"shape":[0],"data":{"dtype":"f32","values":[]}}}"#
    )
    .is_err());
}

#[test]
fn shared_payload_clones_share_values_and_keep_semantic_equality() {
    let retired = Arc::new(AtomicUsize::new(0));
    let raw = tensor(vec![2.5, -4.0, 8.25]);
    let expected = CapturePayload::Tensor(raw.clone());
    let payload = CapturePayload::SharedTensor(SharedTensorObservation::retain(
        raw,
        Retired(retired.clone()),
    ));
    let alias = payload.clone();
    assert_eq!(payload, expected);
    assert_eq!(expected, alias);
    let (CapturePayload::SharedTensor(left), CapturePayload::SharedTensor(right)) =
        (&payload, &alias)
    else {
        unreachable!()
    };
    assert!(left.same_storage(right));
    assert_eq!(left.shape().as_ptr(), right.shape().as_ptr());
    let (TensorObservationData::F32(left), TensorObservationData::F32(right)) =
        (left.data(), right.data())
    else {
        unreachable!()
    };
    assert_eq!(left.as_ptr(), right.as_ptr());
    drop(payload);
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    assert_eq!(alias.as_tensor(), expected.as_tensor());
    drop(alias);
    assert_eq!(retired.load(Ordering::SeqCst), 1);
    let nan =
        CapturePayload::SharedTensor(SharedTensorObservation::retain(tensor(vec![f32::NAN]), ()));
    assert_ne!(nan, nan.clone()); // Preserve the original raw floating equality.
}

#[test]
fn all_existing_payload_tags_keep_their_wire_and_unknown_tags_reject() {
    use serde_json::json;
    let values = [
        json!({"kind":"tensor","value":{"shape":[2],"data":{"dtype":"u64","values":[u64::MAX,7]}}}),
        json!({"kind":"summary","value":{"elements":3,"finite":2,"non_finite":1,"nan":1,"positive_infinity":0,"negative_infinity":0,"min":-2.0,"max":4.0,"mean":1.0,"rms":3.0}}),
        json!({"kind":"histogram","value":{"edges":[0.0,1.0,2.0],"counts":[2,3],"below":1,"above":2,"non_finite":1}}),
        json!({"kind":"candidates","value":{"stage":"raw_logits_before_sampling","source":"effective","candidates":[{"token_id":9,"score":2.5,"allowed":false}]}}),
        json!({"kind":"token_scores","value":{"stage":"raw_logits_before_sampling","source":"original","vocabulary":11,"log_partition":3.0,"scores":[{"target":{"token_id":4,"score":2.0,"allowed":true},"log_probability":-1.0,"rank":2,"strongest_alternative":null}],"domain":null}}),
        json!({"kind":"routed_units","value":{"geometry":{"experts":4,"units_per_expert":8,"routes_per_token":2},"source_token_ranges":[[0,1]],"rows":[{"source_peer":null,"token":0,"slot":1,"expert":3,"coefficient":0.5,"unit_start":2,"unit_stride":2,"values":{"shape":[2],"data":{"dtype":"f32","values":[2.0,4.0]}}}]}}),
    ];
    for expected in values {
        let payload: CapturePayload = serde_json::from_value(expected.clone()).unwrap();
        assert!(!matches!(payload, CapturePayload::SharedTensor(_)));
        assert_eq!(serde_json::to_value(&payload).unwrap(), expected);
        let decoded: CapturePayload = serde_json::from_value(expected).unwrap();
        assert_eq!(payload, decoded);
    }
    assert!(
        serde_json::from_str::<CapturePayload>(r#"{"kind":"not_a_capture","value":{}}"#).is_err()
    );
    assert!(serde_json::from_str::<CapturePayload>(
        r#"{"kind":"tensor","value":{"shape":[2],"data":{"dtype":"f32","values":[1]}}}"#
    )
    .is_err());
}
