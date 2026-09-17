//! Borrowed fields of the existing capture record wire schema.
use super::*;

/// Canonical record fields borrowed from a retained source and completed payload.
/// It carries no admission, completion or funding authority. Encoding storage and
/// validation remain the caller's responsibility, just as for `CaptureRecord`.
#[derive(Debug, Serialize)]
#[serde(rename = "CaptureRecord")]
pub struct CaptureRecordWire<'a> {
    /// Existing schema version.
    pub schema_version: u32,
    /// Selected caller identity.
    pub selection_id: &'a str,
    /// Exact catalog path.
    pub path: &'a str,
    /// Exact catalog node.
    pub node_id: &'a str,
    /// Original/effective observation position.
    pub position: crate::ObservationPosition,
    /// Actual invocation-local extents.
    pub source_shape: Option<&'a [u64]>,
    /// Actual source precision, absent only when unavailable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_dtype: Option<&'a crate::checkpoint::TensorDtype>,
    /// Actual selected extents before a possible preview prefix.
    pub selected_shape: Option<&'a [u64]>,
    /// Existing terminal outcome.
    pub outcome: &'a CaptureOutcome,
    /// Borrowed completed values, preserving their original owner.
    pub payload: Option<CapturePayloadWire<'a>>,
    /// Original conservative logical charge.
    pub charged: CaptureUsage,
}
impl<'a> From<&'a CaptureRecord> for CaptureRecordWire<'a> {
    fn from(record: &'a CaptureRecord) -> Self {
        Self {
            schema_version: record.schema_version, selection_id: &record.selection_id,
            path: &record.path, node_id: &record.node_id, position: record.position,
            source_shape: record.source_shape.as_deref(), source_dtype: record.source_dtype.as_ref(),
            selected_shape: record.selected_shape.as_deref(), outcome: &record.outcome,
            payload: record.payload.as_ref().map(CapturePayloadWire::from), charged: record.charged,
        }
    }
}
impl CaptureRecordWire<'_> {
    /// Explicit borrowed record/payload frames; output and serializer stay separate.
    pub const fn control_bytes() -> usize {
        std::mem::size_of::<Self>() + CapturePayloadWire::control_bytes()
    }
}
impl Serialize for CaptureRecord {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        CaptureRecordWire::from(self).serialize(serializer)
    }
}
