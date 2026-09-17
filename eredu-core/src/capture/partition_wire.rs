//! Versioned host records for bounded partition capture delivery.
//! These records describe evidence; deserialization never grants execution authority.

use super::{CaptureError, CapturePhase, CaptureRecord};
use serde::{Deserialize, Serialize};

/// Version of partition capture receipt records.
pub const PARTITION_CAPTURE_SCHEMA_VERSION: u32 = 3;

/// How measured native regions form the globally selected activation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PartitionCaptureCombination {
    /// Each selected coordinate has one authoritative producer.
    #[default]
    Disjoint,
    /// Every producer supplies the complete selected floating summand. Terms are
    /// converted by the ordinary raw collector to F32, accumulated in canonical
    /// rank order with compensated F64 arithmetic, and converted to F32 before
    /// the requested transform. This is a host assembly,
    /// not evidence that a separate complete native tensor existed.
    SumF64ToF32,
}

/// Host evidence for a numeric region. Deserialization is not slice authority;
/// execution uses the separately validated producer projection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartitionCaptureRegion {
    /// Inclusive starts in each numeric axis.
    pub starts: Vec<u64>,
    /// Exclusive ends in each numeric axis.
    pub ends: Vec<u64>,
    /// Positive strides in each numeric axis.
    pub strides: Vec<u64>,
    /// Selected shape after applying the strides.
    pub shape: Vec<u64>,
}

/// One producer's measured contribution to a committed global capture.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartitionCaptureContributionRecord {
    /// World rank that supplied this fragment.
    pub producer_rank: usize,
    /// Region of the producer's actual local tensor.
    pub local: PartitionCaptureRegion,
    /// Region within the globally selected result.
    pub destination: PartitionCaptureRegion,
    /// Credits reserved before the native transformation.
    pub charged: super::CaptureUsage,
    /// Sparse expert/source placement. Rectangles alone do not describe dynamic
    /// expert participation; this remains evidence, never receipt authority.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routed: Option<super::RoutedUnitCaptureProvenance>,
}

/// Provenance published alongside a complete global capture record. Empty
/// producers remain explicit even though they supply no fragment contributions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartitionCaptureEvidence {
    /// Host-evidence schema version.
    pub schema_version: u32,
    /// Explicit assembly equation; original source precision describes the terms.
    #[serde(default)]
    pub combination: PartitionCaptureCombination,
    /// Exact retained identities and actual forward epoch.
    pub context: PartitionCaptureContext,
    /// Digest of the complete expected ownership and receipt bounds.
    pub receipt_plan_identity: String,
    /// Every producer whose acknowledgment was verified.
    pub producers: Vec<usize>,
    /// Exact local/global placement of all measured fragments.
    pub contributions: Vec<PartitionCaptureContributionRecord>,
}

/// Exact shared execution context attached to a partition capture receipt.
/// The enclosing session establishes these identities from retained admission;
/// accepting caller-provided strings alone does not authenticate a session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartitionCaptureContext {
    /// Exact prepared artifact identity.
    pub artifact_identity: String,
    /// Common selected execution/communication identity, including parameter version.
    pub execution_identity: String,
    /// Coordinated run or branch identity.
    pub run_identity: String,
    /// Active immutable parameter edit, absent for baseline parameters.
    pub overlay_identity: Option<String>,
    /// Original global capture admission digest.
    pub capture_plan_identity: String,
    /// Original global selection ordinal.
    pub selection_index: usize,
    /// Forward phase producing this selection.
    pub phase: CapturePhase,
    /// Run-relative prediction; distinct from within-tensor positions.
    pub prediction: u64,
    /// Monotone forward submission epoch; restore must not reuse it.
    pub forward_epoch: u64,
    /// Independently admitted physical axes, distinct from prediction coordinates.
    /// Absent only for ordinary request-shaped capture authority.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub invocation: Option<super::CaptureInvocationShape>,
}

impl PartitionCaptureContext {
    /// Bounds identity storage and rejects absent required identities.
    pub fn validate(&self) -> Result<(), CaptureError> {
        if let Some(shape) = self.invocation {
            shape.validate()?;
        }
        for identity in [
            &self.artifact_identity,
            &self.execution_identity,
            &self.run_identity,
            &self.capture_plan_identity,
        ]
        .into_iter()
        .chain(self.overlay_identity.iter())
        {
            if identity.is_empty() || identity.len() > 256 {
                return Err(CaptureError::Invalid(
                    "partition capture identity must contain 1..=256 bytes".into(),
                ));
            }
        }
        Ok(())
    }
}

/// One locally transformed fragment. Exact placement comes from retained
/// producer admission, not from untrusted coordinates in a received message.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartitionCaptureFragmentRecord {
    /// Ordinal in this producer's original validated projection.
    pub fragment_index: usize,
    /// Ordinary native capture evidence, with local source/selected geometry.
    pub record: CaptureRecord,
}

/// Explicit receipt from one expected producer, including an empty selection.
/// Native failures/completion are established separately by the forward owner;
/// receipt arrival alone cannot prove a native forward has completed safely.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartitionCaptureProducerRecord {
    /// Exact supported host-record version.
    pub schema_version: u32,
    /// Must agree with the retained receipt authority, never inferred from overlap.
    #[serde(default)]
    pub combination: PartitionCaptureCombination,
    /// Digest of the retained producer geometry, context and delivery bounds.
    pub receipt_plan_identity: String,
    /// Retained execution and submission context.
    pub context: PartitionCaptureContext,
    /// World rank; must agree with the transport's independently known sender.
    pub producer_rank: usize,
    /// Actual local source precision, including when the selected slice is empty.
    pub source_dtype: Option<crate::checkpoint::TensorDtype>,
    /// Every expected local fragment, each appearing exactly once.
    pub fragments: Vec<PartitionCaptureFragmentRecord>,
}

/// Borrowed view of the same canonical fragment record. Construction supplies
/// no provenance, native completion, or allocation authority.
#[derive(Debug, Serialize)]
pub struct BorrowedPartitionCaptureFragmentRecord<'a, R: ?Sized = CaptureRecord> {
    /// Ordinal in the original validated producer projection.
    pub fragment_index: usize,
    /// Existing completed record; its payload is never cloned by this view. The
    /// shared cold counter may instead lend a fixed null sentinel to measure the
    /// same envelope independently of its separately bounded record body.
    pub record: &'a R,
}

/// Borrowed fields for the canonical producer receipt serializer. `fragments`
/// is the exact ordered fragment sequence supplied by the shared runtime; this
/// view does not validate or grant capture/source authority.
#[derive(Debug, Serialize)]
pub struct BorrowedPartitionCaptureProducerRecord<'a, F> {
    /// Exact existing host-record version.
    pub schema_version: u32,
    /// Retained assembly equation.
    pub combination: PartitionCaptureCombination,
    /// Original retained receipt identity.
    pub receipt_plan_identity: &'a str,
    /// Original retained execution context.
    pub context: &'a PartitionCaptureContext,
    /// Independently known sender rank.
    pub producer_rank: usize,
    /// Actual source precision, including an empty selection.
    pub source_dtype: Option<&'a crate::checkpoint::TensorDtype>,
    /// Existing fragment records in their validated order.
    pub fragments: F,
}

impl Serialize for PartitionCaptureFragmentRecord {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        BorrowedPartitionCaptureFragmentRecord {
            fragment_index: self.fragment_index,
            record: &self.record,
        }.serialize(serializer)
    }
}
impl Serialize for PartitionCaptureProducerRecord {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        BorrowedPartitionCaptureProducerRecord {
            schema_version: self.schema_version,
            combination: self.combination,
            receipt_plan_identity: &self.receipt_plan_identity,
            context: &self.context,
            producer_rank: self.producer_rank,
            source_dtype: self.source_dtype.as_ref(),
            fragments: &self.fragments,
        }.serialize(serializer)
    }
}
