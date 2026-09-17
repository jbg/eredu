//! One legacy wire representation for raw and shared tensor ownership.
use super::*;
use serde::{Deserializer, Serializer};

impl CapturePayload {
    /// Borrows raw tensor values from either ownership representation.
    ///
    /// A caller's deep clone of this DTO is a separate allocation and does not
    /// transfer the shared owner's custody or any construction authority.
    pub fn as_tensor(&self) -> Option<&TensorObservation> {
        match self {
            Self::Tensor(tensor) => Some(tensor),
            Self::SharedTensor(tensor) => Some(tensor.as_observation()),
            _ => None,
        }
    }
}

impl PartialEq for CapturePayload {
    fn eq(&self, other: &Self) -> bool {
        // Ownership is not a semantic difference. Preserve the raw DTO's
        // existing floating equality, including NaN not comparing equal.
        if let (Some(left), Some(right)) = (self.as_tensor(), other.as_tensor()) {
            return left == right;
        }
        match (self, other) {
            (Self::Summary(left), Self::Summary(right)) => left == right,
            (Self::Histogram(left), Self::Histogram(right)) => left == right,
            (Self::Candidates(left), Self::Candidates(right)) => left == right,
            (Self::TokenScores(left), Self::TokenScores(right)) => left == right,
            (Self::RoutedUnits(left), Self::RoutedUnits(right)) => left == right,
            _ => false,
        }
    }
}

struct Tensor<'a>(&'a TensorObservation);
impl Serialize for Tensor<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        super::tensor_wire::serialize(self.0, serializer)
    }
}

#[derive(Serialize)]
#[serde(
    rename = "CapturePayload",
    tag = "kind",
    content = "value",
    rename_all = "snake_case"
)]
enum Borrowed<'a> {
    Tensor(Tensor<'a>),
    Summary(&'a CaptureSummary),
    Histogram(&'a CaptureHistogram),
    Candidates(&'a CaptureCandidates),
    TokenScores(&'a CaptureTokenScores),
    RoutedUnits(&'a RoutedUnitCapture),
}

/// Read-only canonical capture payload view. This grants neither funding nor
/// source authority; its references keep the caller's actual payload owner live.
#[derive(Clone, Copy, Debug)]
pub enum CapturePayloadWire<'a> {
    /// Exact raw/shared tensor values, using capture's nonfinite wire policy.
    Tensor(&'a TensorObservation),
    /// Completed finite statistics.
    Summary(&'a CaptureSummary),
    /// Completed fixed-edge counts.
    Histogram(&'a CaptureHistogram),
    /// Completed ordered candidates.
    Candidates(&'a CaptureCandidates),
    /// Completed full-distribution selected scores.
    TokenScores(&'a CaptureTokenScores),
    /// Completed routed rows.
    RoutedUnits(&'a RoutedUnitCapture),
}
impl<'a> From<&'a CapturePayload> for CapturePayloadWire<'a> {
    fn from(value: &'a CapturePayload) -> Self {
        match value {
            CapturePayload::Tensor(v) => Self::Tensor(v),
            CapturePayload::SharedTensor(v) => Self::Tensor(v.as_observation()),
            CapturePayload::Summary(v) => Self::Summary(v),
            CapturePayload::Histogram(v) => Self::Histogram(v),
            CapturePayload::Candidates(v) => Self::Candidates(v),
            CapturePayload::TokenScores(v) => Self::TokenScores(v),
            CapturePayload::RoutedUnits(v) => Self::RoutedUnits(v),
        }
    }
}
impl CapturePayloadWire<'_> {
    /// Actual explicit adapter frames; the serializer/destination are separate.
    pub const fn control_bytes() -> usize {
        std::mem::size_of::<Self>() + std::mem::size_of::<Borrowed<'_>>() + std::mem::size_of::<Tensor<'_>>()
    }
}
impl Serialize for CapturePayloadWire<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let borrowed = match self {
            Self::Tensor(v) => Borrowed::Tensor(Tensor(v)),
            Self::Summary(v) => Borrowed::Summary(v),
            Self::Histogram(v) => Borrowed::Histogram(v),
            Self::Candidates(v) => Borrowed::Candidates(v),
            Self::TokenScores(v) => Borrowed::TokenScores(v),
            Self::RoutedUnits(v) => Borrowed::RoutedUnits(v),
        };
        borrowed.serialize(serializer)
    }
}
impl Serialize for CapturePayload {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        CapturePayloadWire::from(self).serialize(serializer)
    }
}

// There is deliberately only one tensor wire tag. A shared in-memory owner
// cannot be reconstructed from serialized values, an identity or a byte count.
#[derive(Deserialize)]
#[serde(
    rename = "CapturePayload",
    tag = "kind",
    content = "value",
    rename_all = "snake_case"
)]
enum Owned {
    Tensor(#[serde(with = "super::tensor_wire")] TensorObservation),
    Summary(CaptureSummary),
    Histogram(CaptureHistogram),
    Candidates(CaptureCandidates),
    TokenScores(CaptureTokenScores),
    RoutedUnits(RoutedUnitCapture),
}

impl<'de> Deserialize<'de> for CapturePayload {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(match Owned::deserialize(deserializer)? {
            Owned::Tensor(value) => Self::Tensor(value),
            Owned::Summary(value) => Self::Summary(value),
            Owned::Histogram(value) => Self::Histogram(value),
            Owned::Candidates(value) => Self::Candidates(value),
            Owned::TokenScores(value) => Self::TokenScores(value),
            Owned::RoutedUnits(value) => Self::RoutedUnits(value),
        })
    }
}

#[cfg(test)]
mod tests;
