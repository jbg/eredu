//! Borrowed capture encoding of the same protected tensor payload.
use crate::{SharedTensorObservation, TensorObservation};
use serde::{Serialize, Serializer};

/// Borrowed tensor variant with exactly the existing `CapturePayload` wire policy.
///
/// Unlike standalone `SharedTensorObservation` serialization, this preserves the
/// capture protocol's explicit nonfinite strings. It borrows the same protected
/// payload and invokes the existing serializer without another DTO or value Vec.
/// The serializer's own output/storage remains its caller's responsibility.
pub struct CaptureTensorWire<'a>(&'a SharedTensorObservation);
impl<'a> CaptureTensorWire<'a> {
    /// Borrow the protected payload without copying shape or values.
    pub fn new(tensor: &'a SharedTensorObservation) -> Self {
        Self(tensor)
    }
}
impl Serialize for CaptureTensorWire<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        struct Tensor<'a>(&'a TensorObservation);
        impl Serialize for Tensor<'_> {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                super::tensor_wire::serialize(self.0, serializer)
            }
        }
        #[derive(Serialize)]
        #[serde(tag = "kind", content = "value", rename_all = "snake_case")]
        enum Wire<'a> {
            Tensor(Tensor<'a>),
        }
        Wire::Tensor(Tensor(self.0.as_observation())).serialize(serializer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{capture::CapturePayload, TensorObservationData};

    #[test]
    fn capture_and_standalone_wire_keep_their_original_distinct_policies() {
        let raw = TensorObservation::new(
            vec![5],
            TensorObservationData::F32(vec![
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
                -0.0,
                1.25,
            ]),
        )
        .unwrap();
        // This explicit test-side DTO clone is outside the closed owner worker.
        let expected_capture = serde_json::to_value(CapturePayload::Tensor(raw.clone())).unwrap();
        let expected_standalone = serde_json::to_value(&raw).unwrap();
        let shared = SharedTensorObservation::retain(raw, ());
        let alias = shared.clone();
        let pointer = match shared.data() {
            TensorObservationData::F32(v) => v.as_ptr(),
            _ => unreachable!(),
        };
        assert_eq!(
            serde_json::to_value(CaptureTensorWire::new(&shared)).unwrap(),
            expected_capture
        );
        assert_eq!(serde_json::to_value(&shared).unwrap(), expected_standalone);
        assert_eq!(expected_capture["value"]["data"]["values"][0], "nan");
        assert!(expected_standalone["data"]["values"][0].is_null());
        drop(shared);
        match alias.data() {
            TensorObservationData::F32(v) => {
                assert_eq!(v.as_ptr(), pointer);
                assert_eq!(v[3].to_bits(), (-0.0f32).to_bits());
            }
            _ => unreachable!(),
        }
    }
}
