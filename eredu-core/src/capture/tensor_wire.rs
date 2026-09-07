//! Capture-only JSON encoding; the legacy observation encoding is unchanged.

use crate::{TensorObservation, TensorObservationData};
use serde::{ser::SerializeSeq, Deserialize, Deserializer, Serialize, Serializer};

struct Floats<'a>(&'a [f32]);
impl Serialize for Floats<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut seq = serializer.serialize_seq(Some(self.0.len()))?;
        for value in self.0 {
            if value.is_nan() {
                seq.serialize_element("nan")?;
            } else if *value == f32::INFINITY {
                seq.serialize_element("+inf")?;
            } else if *value == f32::NEG_INFINITY {
                seq.serialize_element("-inf")?;
            } else {
                seq.serialize_element(value)?;
            }
        }
        seq.end()
    }
}

#[derive(Serialize)]
#[serde(tag = "dtype", content = "values", rename_all = "snake_case")]
enum Data<'a> {
    F32(Floats<'a>),
    I64(&'a [i64]),
    U64(&'a [u64]),
    Bool(&'a [bool]),
}

pub(super) fn serialize<S: Serializer>(
    tensor: &TensorObservation,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    #[derive(Serialize)]
    struct Wire<'a> {
        shape: &'a [usize],
        data: Data<'a>,
    }
    let data = match tensor.data() {
        TensorObservationData::F32(v) => Data::F32(Floats(v)),
        TensorObservationData::I64(v) => Data::I64(v),
        TensorObservationData::U64(v) => Data::U64(v),
        TensorObservationData::Bool(v) => Data::Bool(v),
    };
    Wire {
        shape: tensor.shape(),
        data,
    }
    .serialize(serializer)
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Float {
    Number(f32),
    Special(String),
}
#[derive(Deserialize)]
#[serde(tag = "dtype", content = "values", rename_all = "snake_case")]
enum OwnedData {
    F32(Vec<Float>),
    I64(Vec<i64>),
    U64(Vec<u64>),
    Bool(Vec<bool>),
}

pub(super) fn deserialize<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<TensorObservation, D::Error> {
    #[derive(Deserialize)]
    struct Wire {
        shape: Vec<usize>,
        data: OwnedData,
    }
    let wire = Wire::deserialize(deserializer)?;
    let data = match wire.data {
        OwnedData::F32(values) => TensorObservationData::F32(
            values
                .into_iter()
                .map(|v| match v {
                    Float::Number(n) => Ok(n),
                    Float::Special(s) => match s.as_str() {
                        "nan" => Ok(f32::NAN),
                        "+inf" => Ok(f32::INFINITY),
                        "-inf" => Ok(f32::NEG_INFINITY),
                        _ => Err(serde::de::Error::custom("unknown non-finite capture value")),
                    },
                })
                .collect::<Result<_, D::Error>>()?,
        ),
        OwnedData::I64(v) => TensorObservationData::I64(v),
        OwnedData::U64(v) => TensorObservationData::U64(v),
        OwnedData::Bool(v) => TensorObservationData::Bool(v),
    };
    TensorObservation::new(wire.shape, data).map_err(serde::de::Error::custom)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::CapturePayload;
    #[test]
    fn nonfinite_preview_roundtrips_without_null_or_zero_substitution() {
        let payload = CapturePayload::Tensor(
            TensorObservation::new(
                vec![5],
                TensorObservationData::F32(vec![
                    f32::NAN,
                    f32::INFINITY,
                    f32::NEG_INFINITY,
                    -0.0,
                    1.0,
                ]),
            )
            .unwrap(),
        );
        let json = serde_json::to_string(&payload).unwrap();
        assert!(!json.contains("null"));
        let CapturePayload::Tensor(decoded) = serde_json::from_str(&json).unwrap() else {
            panic!()
        };
        let TensorObservationData::F32(values) = decoded.data() else {
            panic!()
        };
        assert!(values[0].is_nan());
        assert_eq!(values[1], f32::INFINITY);
        assert_eq!(values[2], f32::NEG_INFINITY);
        assert_eq!(values[3].to_bits(), (-0.0f32).to_bits());
    }
}
