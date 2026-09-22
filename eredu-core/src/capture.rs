//! Admitted, bounded observation contracts. Empty selections mean capture none.
//!
//! Budgets measure capture-owned logical storage and transfers, not an accelerator's
//! allocator, inference state, or a consumer's retained history. A physical native
//! allocation ceiling is a separate capability and must never be inferred from these
//! logical bounds. Backends must reserve before retaining or materializing a value.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[cfg(test)]
mod candidate_tests;
mod delivery;
pub use delivery::CaptureDeliveryPending;
mod admission;
mod identity;
mod invocation;
pub use admission::CaptureAdmissionStorageError;
mod source_construction;
pub use source_construction::CaptureSourceConstruction;
mod text_origin;
pub use text_origin::CaptureTextOrigin;
#[cfg(test)]
mod prepared_geometry_tests;
pub use invocation::{
    CaptureAxisError, CaptureInvocationBounds, CaptureInvocationShape, CaptureInvocationWindow,
    CaptureWindowSourceError,
};
mod payload_wire;
pub use payload_wire::CapturePayloadWire;
mod record_wire;
pub use record_wire::CaptureRecordWire;
mod window_geometry;
pub use window_geometry::{CaptureWindowError, CaptureWindowGeometry};
mod prefill_transform;
pub use prefill_transform::{
    CapturePrefillPartitionError, CapturePrefillTransformFragment, CapturePrefillTransformPlan,
};
mod prefill_geometry;
pub use prefill_geometry::{
    CapturePrefillAxisSlice, CapturePrefillElement, CapturePrefillFragment,
    CapturePrefillGeometryError, CapturePrefillRowAssembly,
};
mod retained_error;
mod routed;
pub use retained_error::RetainedCaptureError;
mod intervention_evidence;
pub(crate) mod plan_copy;
pub use intervention_evidence::{
    InterventionEvidenceCompanion, InterventionEvidenceDescriptor, InterventionEvidenceLayout,
    InterventionEvidenceSide,
};
mod shared_plan;
pub use plan_copy::{CapturePlanCopyError, PreparedCapturePlanCopy};
pub(crate) mod retained_payload;
mod shared_step;
mod tensor_wire;
pub use routed::*;
pub use shared_plan::SharedCapturePlan;
pub use shared_step::{PreparedCapturedStep, SharedCapturedStep, UnpublishedCapturedStep};

use crate::{
    ObservationCatalog, ObservationPoint, ObservationSupportReport, ObservationSupportStatus,
    SymbolicDimension, TensorObservation,
};

/// Wire version for capture plans and records.
pub const CAPTURE_SCHEMA_VERSION: u32 = 1;

/// Cold facts about a deferred observation source. Its borrowed prototype gives
/// geometry only; it need not have the factory's actual element type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedCaptureSource {
    /// Conservative logical capture quota for creating the source and temporaries.
    /// Physical native allocation bounds require the selected creation mechanism.
    pub creation_bytes: u64,
    /// Actual output type promised by the source mechanism, checked when generated.
    /// Unknown sources cannot supply empty raw captures without creating a value.
    pub source_dtype: Option<crate::checkpoint::TensorDtype>,
}

/// One ordinary forward operation. Prediction zero is prefill; decode starts at one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapturePhase {
    /// Prompt execution producing the first prediction.
    Prefill,
    /// Cached execution producing a later prediction.
    Decode,
}

/// Half-open selection of prediction indices (zero is the prefill prediction).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptureSchedule {
    /// Whether prompt-prefill observations are selected.
    pub prefill: bool,
    /// Whether cached-decode observations are selected.
    pub decode: bool,
    /// First selected prediction index, inclusive.
    pub first_prediction: u64,
    /// Exclusive end of the prediction range; absent means the request limit.
    pub end_prediction: Option<u64>,
    /// Capture every Nth prediction relative to first_prediction; must be positive.
    pub every: u64,
}

impl Default for CaptureSchedule {
    fn default() -> Self {
        Self {
            prefill: true,
            decode: true,
            first_prediction: 0,
            end_prediction: None,
            every: 1,
        }
    }
}

impl CaptureSchedule {
    /// Returns selected prediction count and last index within a bounded run.
    pub fn count_and_last(
        &self,
        phase: CapturePhase,
        maximum: u64,
    ) -> Result<Option<(u64, u64)>, CaptureError> {
        self.count_and_last_from(phase, 0, maximum)
    }

    /// Counts the remaining absolute schedule without moving its frequency
    /// origin when a continuation resumes at `next_prediction`.
    pub fn count_and_last_from(
        &self,
        phase: CapturePhase,
        next_prediction: u64,
        maximum: u64,
    ) -> Result<Option<(u64, u64)>, CaptureError> {
        if self.every == 0 {
            return Err(CaptureError::Invalid("zero capture frequency".into()));
        }
        if phase == CapturePhase::Prefill {
            return Ok(
                (next_prediction == 0 && maximum > 0 && self.includes(phase, 0)).then_some((1, 0)),
            );
        }
        if !self.decode {
            return Ok(None);
        }
        let end = self.end_prediction.unwrap_or(maximum).min(maximum);
        let lower = self.first_prediction.max(1).max(next_prediction);
        if lower >= end {
            return Ok(None);
        }
        let offset = (lower - self.first_prediction).div_ceil(self.every);
        let first = add(self.first_prediction, mul(offset, self.every)?)?;
        if first >= end {
            return Ok(None);
        }
        let steps = (end - 1 - first) / self.every;
        Ok(Some((add(steps, 1)?, add(first, mul(steps, self.every)?)?)))
    }

    /// Counts eligible prediction coordinates for independently invoked phases.
    /// A coordinate may be invoked repeatedly; this is not an execution count.
    pub fn count_coordinates(
        &self,
        phase: CapturePhase,
        next: u64,
        maximum: u64,
    ) -> Result<Option<(u64, u64)>, CaptureError> {
        if self.every == 0 {
            return Err(CaptureError::Invalid("zero capture frequency".into()));
        }
        if !match phase {
            CapturePhase::Prefill => self.prefill,
            CapturePhase::Decode => self.decode,
        } {
            return Ok(None);
        }
        let end = self.end_prediction.unwrap_or(maximum).min(maximum);
        let lower = self.first_prediction.max(next);
        if lower >= end {
            return Ok(None);
        }
        let offset = (lower - self.first_prediction).div_ceil(self.every);
        let first = add(self.first_prediction, mul(offset, self.every)?)?;
        if first >= end {
            return Ok(None);
        }
        let steps = (end - 1 - first) / self.every;
        Ok(Some((add(steps, 1)?, add(first, mul(steps, self.every)?)?)))
    }

    /// Whether a phase and prediction are selected by this schedule.
    pub fn includes(&self, phase: CapturePhase, prediction: u64) -> bool {
        (match phase {
            CapturePhase::Prefill => self.prefill,
            CapturePhase::Decode => self.decode,
        }) && prediction >= self.first_prediction
            && self.end_prediction.is_none_or(|end| prediction < end)
            && self.every != 0
            && (prediction - self.first_prediction).is_multiple_of(self.every)
    }
}

/// Half-open, positive-stride slice of a catalog axis. Unmentioned axes are whole.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptureSlice {
    /// Exact semantic axis name from the discovery catalog.
    pub axis: String,
    /// Inclusive element offset.
    pub start: u64,
    /// Exclusive element offset.
    pub end: u64,
    /// Positive step between selected elements.
    pub stride: u64,
}

/// A transform is applied after axis selection. Full tensors require explicit opt-in.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CaptureTransform {
    /// Raw selected expert units with original token/route/expert/coefficient
    /// identity. Only valid at a declared sparse routed-unit boundary.
    RoutedUnits,
    /// Row-major prefix; omitted values are explicitly reported as truncated.
    Preview {
        /// Maximum row-major values to materialize.
        max_elements: u64,
    },
    /// All values of a nonempty list of axis slices, preserving integer IDs.
    Slice,
    /// All values, still subject to every budget.
    FullTensor,
    /// Finite-only statistics with explicit non-finite counts.
    Summary,
    /// Fixed finite increasing edges. Bins are [lo, hi), last bin includes hi.
    Histogram {
        /// Finite, strictly increasing bin boundaries.
        edges: Vec<f32>,
    },
    /// Highest raw logits for the current prediction; no sampling/RNG is performed.
    TopCandidates {
        /// Maximum candidate count, bounded by the vocabulary extent.
        count: u64,
    },
    /// Selected raw scores and full-model-vocabulary reductions for the last row.
    TokenScores {
        /// Unique model output IDs; at most 64. No vocabulary slicing or filtering.
        token_ids: Vec<u32>,
    },
}

/// Portable transformation categories used by capability discovery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureTransformKind {
    /// Sparse routed-unit values and original participation coordinates.
    RoutedUnits,
    /// Bounded row-major tensor prefix.
    Preview,
    /// Axis-selected tensor values.
    Slice,
    /// Explicitly requested complete tensor values.
    FullTensor,
    /// Finite/non-finite counts and finite-only statistics.
    Summary,
    /// Bounded fixed-edge histogram.
    Histogram,
    /// Bounded raw model-score candidates before sampler processing.
    TopCandidates,
    /// Selected-token score, strongest alternative, full log probability and rank.
    TokenScores,
}

impl CaptureTransform {
    /// Transformation category for capability matching.
    pub fn kind(&self) -> CaptureTransformKind {
        match self {
            Self::RoutedUnits => CaptureTransformKind::RoutedUnits,
            Self::Preview { .. } => CaptureTransformKind::Preview,
            Self::Slice => CaptureTransformKind::Slice,
            Self::FullTensor => CaptureTransformKind::FullTensor,
            Self::Summary => CaptureTransformKind::Summary,
            Self::Histogram { .. } => CaptureTransformKind::Histogram,
            Self::TopCandidates { .. } => CaptureTransformKind::TopCandidates,
            Self::TokenScores { .. } => CaptureTransformKind::TokenScores,
        }
    }
}

/// Selected backend facts; absent transformations are unsupported, never emulated
/// with an unbounded host copy.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptureCapabilities {
    /// Transformations implemented before host materialization.
    pub transformations: Vec<CaptureTransformKind>,
    /// Maximum admitted number of histogram bins.
    pub max_histogram_bins: u64,
    /// Execution, precision, and memory-accounting conditions.
    pub conditions: Vec<String>,
}

/// One exact path, schedule, axis selection, and transformation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaptureSelection {
    /// Unique caller identity, allowing several transforms at one observation point.
    pub id: String,
    /// Exact observation selector in the retained catalog.
    pub path: String,
    /// Phase, frequency, and prediction-range selection.
    pub schedule: CaptureSchedule,
    /// Semantic axis selection applied before the transform.
    pub slices: Vec<CaptureSlice>,
    /// Requested native transformation.
    pub transform: CaptureTransform,
}

/// Action taken when a value reservation would exceed an admitted limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureLimitPolicy {
    /// Fail before performing the prohibited retention or copy.
    Fail,
    /// Omit the value and emit a structured budget reason.
    Skip,
}

/// Upper bound on one step or the whole run. Retention includes capture-owned
/// source storage/dependencies and logical temporary arrays. Host bytes include
/// every transferred scalar and host result buffer, including intermediate reductions.
/// Encoded bytes cover UTF-8 JSON capture records, including metadata and escaping.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptureUsage {
    /// Number of successfully reserved value transformations, excluding diagnostic-only records.
    pub captures: u64,
    /// Conservative logical source/backing and temporary native storage reservation.
    pub retained_bytes: u64,
    /// Conservative host-buffer and materialization reservation, including intermediate scalars.
    pub host_bytes: u64,
    /// Conservative UTF-8 JSON capture-record reservation.
    pub encoded_bytes: u64,
}

impl CaptureUsage {
    /// Multiplies a conservative reservation by a bounded occurrence count.
    pub fn checked_mul(self, count: u64) -> Result<Self, CaptureError> {
        Ok(Self {
            captures: mul(self.captures, count)?,
            retained_bytes: mul(self.retained_bytes, count)?,
            host_bytes: mul(self.host_bytes, count)?,
            encoded_bytes: mul(self.encoded_bytes, count)?,
        })
    }
    /// Adds reservations with checked arithmetic in every dimension.
    pub fn checked_add(self, other: Self) -> Result<Self, CaptureError> {
        Ok(Self {
            captures: add(self.captures, other.captures)?,
            retained_bytes: add(self.retained_bytes, other.retained_bytes)?,
            host_bytes: add(self.host_bytes, other.host_bytes)?,
            encoded_bytes: add(self.encoded_bytes, other.encoded_bytes)?,
        })
    }
    /// First accounting dimension exceeding its limit, in stable diagnostic order.
    pub fn exceeded(self, limit: Self) -> Option<CaptureBudget> {
        if self.captures > limit.captures {
            Some(CaptureBudget::Captures)
        } else if self.retained_bytes > limit.retained_bytes {
            Some(CaptureBudget::Retention)
        } else if self.host_bytes > limit.host_bytes {
            Some(CaptureBudget::Host)
        } else if self.encoded_bytes > limit.encoded_bytes {
            Some(CaptureBudget::Encoded)
        } else {
            None
        }
    }
}

/// Independent per-step and cumulative limits with explicit failure policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureLimits {
    /// Limits reset at each prefill or decode step.
    pub per_step: CaptureUsage,
    /// Limits summed across all steps in the run.
    pub cumulative: CaptureUsage,
    /// Whether a value-budget miss fails execution or emits an explicit skip.
    pub on_limit: CaptureLimitPolicy,
}

/// Serializable request; only admission produces execution authority.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapturePlan {
    /// Wire schema version; unsupported versions are rejected.
    pub schema_version: u32,
    /// Empty means none. There is deliberately no implicit capture-all mode.
    pub selections: Vec<CaptureSelection>,
    /// Mandatory storage, materialization, and export limits.
    pub limits: CaptureLimits,
}

impl CapturePlan {
    /// Creates an explicit capture-none plan with zero capture budgets.
    pub fn none() -> Self {
        Self {
            schema_version: CAPTURE_SCHEMA_VERSION,
            selections: Vec::new(),
            limits: CaptureLimits {
                per_step: CaptureUsage::default(),
                cumulative: CaptureUsage::default(),
                on_limit: CaptureLimitPolicy::Fail,
            },
        }
    }

    /// Validates selectors, capabilities, geometry, and static constraints, returning an immutable proof.
    pub fn admit(
        self,
        catalog: &ObservationCatalog,
        support: &ObservationSupportReport,
        capabilities: &CaptureCapabilities,
        request: CaptureRequestShape,
    ) -> Result<AdmittedCapturePlan, CaptureError> {
        self.admit_geometry(
            catalog,
            support,
            capabilities,
            request,
            None,
            CaptureTextOrigin::default(),
        )
    }

    /// Admits ordinary text capture with an explicit cached opening prefix.
    /// New prompt length and request-local prediction coordinates are unchanged.
    /// The origin is geometry only; it grants no cache, execution or observation
    /// authority. Zero origin preserves the existing ordinary admission identity.
    pub fn admit_with_text_origin(
        self,
        catalog: &ObservationCatalog,
        support: &ObservationSupportReport,
        capabilities: &CaptureCapabilities,
        request: CaptureRequestShape,
        origin: CaptureTextOrigin,
    ) -> Result<AdmittedCapturePlan, CaptureError> {
        self.admit_geometry(catalog, support, capabilities, request, None, origin)
    }

    /// Admits independently shaped forwards under one cumulative capture budget.
    /// Exact geometry must be supplied for each invocation before execution.
    pub fn admit_invocations(
        self,
        catalog: &ObservationCatalog,
        support: &ObservationSupportReport,
        capabilities: &CaptureCapabilities,
        bounds: CaptureInvocationBounds,
    ) -> Result<AdmittedCapturePlan, CaptureError> {
        self.admit_geometry(
            catalog,
            support,
            capabilities,
            bounds.request(),
            Some(bounds),
            CaptureTextOrigin::default(),
        )
    }

    /// Copies this borrowed declaration through the closed source-copy worker.
    /// The caller retains the funding owner until this destination or its failed
    /// construction has retired. This produces no admission or execution grant.
    pub fn copy_with_funding(
        &self,
        funding: &crate::HostMetadataFunding,
    ) -> Result<Self, CaptureError> {
        admission::allocation::Allocation(Some(funding)).plan(self)
    }

    /// Admits the same declaration with prospective metadata reservations.
    /// The caller retains the actual account with the result and any failure;
    /// source buffers supplied by the caller remain caller-owned until copied.
    pub fn admit_with_funding(
        self,
        catalog: &ObservationCatalog,
        support: &ObservationSupportReport,
        capabilities: &CaptureCapabilities,
        request: CaptureRequestShape,
        funding: &crate::HostMetadataFunding,
    ) -> Result<AdmittedCapturePlan, CaptureError> {
        admission::admit_geometry(
            self,
            catalog,
            support,
            capabilities,
            request,
            None,
            CaptureTextOrigin::default(),
            admission::allocation::Allocation(Some(funding)),
        )
    }
    /// Same funded admission with an exact cached opening origin.
    pub fn admit_with_text_origin_and_funding(
        self,
        catalog: &ObservationCatalog,
        support: &ObservationSupportReport,
        capabilities: &CaptureCapabilities,
        request: CaptureRequestShape,
        origin: CaptureTextOrigin,
        funding: &crate::HostMetadataFunding,
    ) -> Result<AdmittedCapturePlan, CaptureError> {
        admission::admit_geometry(
            self,
            catalog,
            support,
            capabilities,
            request,
            None,
            origin,
            admission::allocation::Allocation(Some(funding)),
        )
    }
    /// Same funded admission for independent invocation geometry.
    pub fn admit_invocations_with_funding(
        self,
        catalog: &ObservationCatalog,
        support: &ObservationSupportReport,
        capabilities: &CaptureCapabilities,
        bounds: CaptureInvocationBounds,
        funding: &crate::HostMetadataFunding,
    ) -> Result<AdmittedCapturePlan, CaptureError> {
        admission::admit_geometry(
            self,
            catalog,
            support,
            capabilities,
            bounds.request(),
            Some(bounds),
            CaptureTextOrigin::default(),
            admission::allocation::Allocation(Some(funding)),
        )
    }

    fn admit_geometry(
        self,
        catalog: &ObservationCatalog,
        support: &ObservationSupportReport,
        capabilities: &CaptureCapabilities,
        request: CaptureRequestShape,
        invocation_bounds: Option<CaptureInvocationBounds>,
        text_origin: CaptureTextOrigin,
    ) -> Result<AdmittedCapturePlan, CaptureError> {
        admission::admit_geometry(
            self,
            catalog,
            support,
            capabilities,
            request,
            invocation_bounds,
            text_origin,
            admission::allocation::Allocation(None),
        )
    }
}

/// Request geometry used for admission. This initial protocol is ordinary committed
/// text generation; media, speculative and rank-partitioned runs need separate support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptureRequestShape {
    /// Number of sequences in the request.
    pub batch: u64,
    /// Number of new prefill positions per sequence, excluding any cached prefix.
    pub prompt_tokens: u64,
    /// Maximum number of generated predictions, including the prefill prediction.
    pub max_predictions: u64,
}

impl CaptureRequestShape {
    /// Checks the architecture-admitted decoder input before executing prefill.
    /// Media positions count after assembly, not as raw patches or text IDs alone.
    pub fn validate_prefill(self, batch: u64, sequence: u64) -> Result<(), CaptureError> {
        if batch == 0 || sequence == 0 || batch != self.batch || sequence != self.prompt_tokens {
            return Err(CaptureError::Invalid(format!(
                "prepared prefill geometry [{batch}, {sequence}] differs from admitted [{}, {}]",
                self.batch, self.prompt_tokens,
            )));
        }
        Ok(())
    }

    /// Zero-origin physical geometry of an ordinary prompt or cached forward.
    /// For an admitted plan, use AdmittedCapturePlan::geometry_at so an explicit
    /// cached opening prefix cannot be discarded.
    pub fn invocation_shape(
        self,
        phase: CapturePhase,
        prediction: u64,
    ) -> Result<CaptureInvocationShape, CaptureError> {
        Ok(CaptureInvocationShape {
            batch: self.batch,
            sequence: if phase == CapturePhase::Prefill {
                self.prompt_tokens
            } else {
                1
            },
            context: Some(add(self.prompt_tokens, prediction)?),
        })
    }
    /// Checks zero-origin request axes, including partially known shapes.
    /// Admitted plans should validate their geometry_at result instead.
    pub fn validate_actual(
        self,
        point: &ObservationPoint,
        phase: CapturePhase,
        prediction: u64,
        shape: &[u64],
    ) -> Result<(), CaptureError> {
        self.invocation_shape(phase, prediction)?
            .validate_actual(point, shape)
    }
    /// Resolves zero-origin symbolic axes. For an admitted plan with a cached
    /// prefix, use its estimate_shape or geometry_at result instead.
    pub fn resolve(
        self,
        point: &ObservationPoint,
        phase: CapturePhase,
        prediction: u64,
    ) -> Result<Option<Vec<u64>>, CaptureError> {
        self.invocation_shape(phase, prediction)?.resolve(point)
    }
}

/// Immutable admission proof. Deserialization cannot forge admission.
#[derive(Debug, Clone)]
pub struct AdmittedCapturePlan {
    plan: CapturePlan,
    points: Vec<ObservationPoint>,
    request: CaptureRequestShape,
    invocation_bounds: Option<CaptureInvocationBounds>,
    text_origin: CaptureTextOrigin,
    identity: String,
}

impl AdmittedCapturePlan {
    /// Stable digest of the plan, selected catalog semantics, request shape and
    /// explicit nonzero text origin (or the separate invocation authority).
    pub fn identity(&self) -> &str {
        &self.identity
    }
    /// Borrows the validated plan.
    pub fn plan(&self) -> &CapturePlan {
        &self.plan
    }
    /// Borrows selected catalog points in selection order.
    pub fn points(&self) -> &[ObservationPoint] {
        &self.points
    }
    /// Returns new-input and local prediction limits. This deliberately omits
    /// the separate cached origin; use geometry_at/estimate_shape for axes.
    pub fn request(&self) -> CaptureRequestShape {
        self.request
    }
    /// Explicit invocation bounds; absence retains ordinary prompt/decode geometry.
    pub fn invocation_bounds(&self) -> Option<CaptureInvocationBounds> {
        self.invocation_bounds
    }
    /// Ordinary text opening prefix. None means this plan instead requires
    /// its separately admitted explicit invocation authority.
    pub fn text_origin(&self) -> Option<CaptureTextOrigin> {
        self.invocation_bounds.is_none().then_some(self.text_origin)
    }
    /// Resolves one forward's physical axes under this exact geometry authority.
    /// An invocation plan requires explicit axes; an ordinary request rejects them.
    pub fn geometry_at(
        &self,
        phase: CapturePhase,
        prediction: u64,
        invocation: Option<CaptureInvocationShape>,
    ) -> Result<CaptureInvocationShape, CaptureError> {
        match (self.invocation_bounds, invocation) {
            (Some(bounds), Some(shape)) => {
                bounds.validate(shape, prediction)?;
                Ok(shape)
            }
            (None, None) => self
                .text_origin
                .invocation_shape(self.request, phase, prediction),
            _ => Err(CaptureError::Invalid(
                "capture invocation authority/geometry mismatch".into(),
            )),
        }
    }
    /// Revalidates this same geometry authority against current selected discovery.
    pub fn readmit(&self, discovery: &CaptureDiscovery) -> Result<Self, CaptureError> {
        self.plan.clone().admit_geometry(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            self.request,
            self.invocation_bounds,
            self.text_origin,
        )
    }
    /// Static maximum shape for an enabled schedule, preserving ordinary identities.
    pub fn estimate_shape(
        &self,
        point: &ObservationPoint,
        phase: CapturePhase,
        prediction: u64,
    ) -> Result<Option<Vec<u64>>, CaptureError> {
        match self.invocation_bounds {
            Some(bounds) => bounds.maximum()?.resolve(point),
            None => self
                .text_origin
                .invocation_shape(self.request, phase, prediction)?
                .resolve(point),
        }
    }
    /// Whether this plan captures no observation points.
    pub fn is_empty(&self) -> bool {
        self.plan.selections.is_empty()
    }
}

mod partition;
pub use partition::{
    CaptureContiguousProjectionError, CaptureContiguousProjectionPlan,
    CaptureCoordinateProjectionPlan, CaptureFragmentGeometry, CaptureSlicePartition,
};
mod partition_wire;
pub use partition_wire::{
    BorrowedPartitionCaptureFragmentRecord, BorrowedPartitionCaptureProducerRecord,
    PARTITION_CAPTURE_SCHEMA_VERSION, PartitionCaptureCombination, PartitionCaptureContext,
    PartitionCaptureContributionRecord, PartitionCaptureEvidence, PartitionCaptureFragmentRecord,
    PartitionCaptureInvocationWindow, PartitionCaptureProducerRecord, PartitionCaptureRegion,
};

/// Numeric axis slices resolved against an actual tensor before any retention/copy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedCaptureSlice {
    /// Inclusive numeric offsets in storage-axis order.
    pub starts: Vec<u64>,
    /// Exclusive numeric offsets in storage-axis order.
    pub ends: Vec<u64>,
    /// Positive numeric strides in storage-axis order.
    pub strides: Vec<u64>,
    /// Resulting selected tensor extents in storage order.
    pub shape: Vec<u64>,
}

mod slice_destination;
pub use slice_destination::{CaptureSliceDestinationError, resolve_slice_into};

/// Checks axis selection against actual extents without touching tensor payloads.
pub fn resolve_slice(
    point: &ObservationPoint,
    selection: &CaptureSelection,
    shape: &[u64],
) -> Result<ResolvedCaptureSlice, CaptureError> {
    resolve_slice_with(
        point,
        selection,
        shape,
        admission::allocation::Allocation(None),
    )
}
fn resolve_slice_with(
    point: &ObservationPoint,
    selection: &CaptureSelection,
    shape: &[u64],
    allocation: admission::allocation::Allocation<'_>,
) -> Result<ResolvedCaptureSlice, CaptureError> {
    let mut output = ResolvedCaptureSlice {
        starts: allocation.vector(shape.len())?,
        ends: allocation.vector(shape.len())?,
        strides: allocation.vector(shape.len())?,
        shape: allocation.vector(shape.len())?,
    };
    output.starts.resize(shape.len(), 0);
    output.ends.extend_from_slice(shape);
    output.strides.resize(shape.len(), 1);
    output.shape.extend_from_slice(shape);
    resolve_slice_into(
        point.axes.as_deref(),
        &selection.slices,
        shape,
        &mut output.starts,
        &mut output.ends,
        &mut output.strides,
        &mut output.shape,
    )
    .map_err(|cause| cause.legacy_with(&selection.slices, allocation))?;
    Ok(output)
}

/// Values are converted to F32 before statistics; finite-only aggregates are then
/// accumulated in F64. Integer raw captures remain exact; integer statistics may
/// round. Empty/all-nonfinite inputs have None aggregates, never fabricated zeros.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaptureSummary {
    /// Total element count, including non-finite values.
    pub elements: u64,
    /// Number of finite values after F32 conversion.
    pub finite: u64,
    /// Combined NaN and infinity count after F32 conversion.
    pub non_finite: u64,
    /// Number of NaNs after F32 conversion.
    pub nan: u64,
    /// Number of positive infinities after F32 conversion.
    pub positive_infinity: u64,
    /// Number of negative infinities after F32 conversion.
    pub negative_infinity: u64,
    /// Minimum finite value; absent when there are no finite values.
    pub min: Option<f64>,
    /// Maximum finite value; absent when there are no finite values.
    pub max: Option<f64>,
    /// Finite-only arithmetic mean; absent when there are no finite values.
    pub mean: Option<f64>,
    /// Finite-only root mean square; absent when there are no finite values.
    pub rms: Option<f64>,
}

/// Bounded finite-value histogram with explicit excluded-value counts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaptureHistogram {
    /// Finite, strictly increasing bin edges.
    pub edges: Vec<f32>,
    /// Counts for adjacent edge pairs; the last interval includes its upper endpoint.
    pub counts: Vec<u64>,
    /// Finite values strictly below the first edge.
    pub below: u64,
    /// Finite values strictly above the last edge.
    pub above: u64,
    /// NaNs and infinities excluded from bins and finite under/overflow counts.
    pub non_finite: u64,
}

/// Bounded portable output from one native transformation.
///
/// Raw and shared tensors with the same values compare equal; ownership and
/// custody are not part of semantic equality. Floating NaNs retain their usual
/// unequal behavior. Equality provides no storage identity or funding proof.
#[derive(Debug, Clone)]
pub enum CapturePayload {
    /// Raw or sliced caller-owned values; integer identifiers retain exact precision.
    Tensor(TensorObservation),
    /// Read-only aliases of an already owned tensor and its retained custody.
    ///
    /// Serializes as the same `kind: tensor` payload. Deserialization always
    /// produces the raw `Tensor` variant and never reconstructs funding.
    SharedTensor(crate::SharedTensorObservation),
    /// Finite/non-finite counts and finite-only statistics.
    Summary(CaptureSummary),
    /// Bounded fixed-edge histogram.
    Histogram(CaptureHistogram),
    /// Native top-k extraction with an explicit score-processing stage.
    Candidates(CaptureCandidates),
    /// Full-distribution reductions; probabilities are never inferred from top-k.
    TokenScores(CaptureTokenScores),
    /// Selected routed units with explicit participation; absent experts are not zeros.
    RoutedUnits(RoutedUnitCapture),
}

/// One selected token evaluated against the complete raw model distribution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaptureTokenScore {
    /// Exact selected ID and raw F32 score, with separate sampling-domain membership.
    pub target: CaptureCandidate,
    /// `score - log(sum(exp(all model scores)))`, using stable full-vocabulary reduction.
    pub log_probability: f64,
    /// One plus the count of strictly greater scores. Ties share competition rank.
    pub rank: u64,
    /// Highest-scoring different ID, absent only for a vocabulary of one.
    /// Equal maxima follow native argmax order.
    pub strongest_alternative: Option<CaptureCandidate>,
}

/// Bounded selected scores for the final sequence row, before sampler processing.
/// Normalization includes every model output ID, including tokenizer holes/padding.
/// Domain membership is evidence only; it does not alter this distribution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaptureTokenScores {
    /// Explicit raw score stage; nonlinear architecture output transforms have already run.
    pub stage: CandidateScoreStage,
    /// Original/effective position at this hook, not at earlier component hooks.
    pub source: CandidateLogitsSource,
    /// Actual complete model vocabulary width.
    pub vocabulary: u64,
    /// Stable log partition over the entire model vocabulary.
    pub log_partition: f64,
    /// Results in requested ID order.
    pub scores: Vec<CaptureTokenScore>,
    /// Optional tokenizer/constraint information, separate from normalization.
    pub domain: Option<CandidateDomain>,
}

/// The processing stage represented by a candidate score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateScoreStage {
    /// Raw model logits before token filtering, penalties, temperature, top-k/p,
    /// Mirostat processing or normalization. These are not probabilities.
    RawLogitsBeforeSampling,
}

/// Whether raw candidates precede or follow mutation at the logits hook.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateLogitsSource {
    /// Before interventions at this hook; earlier hooks may already have changed it.
    #[default]
    Original,
    /// After interventions at this hook, before ordinary sampler processing.
    Effective,
}

/// One exact vocabulary identity and finite raw model score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaptureCandidate {
    /// Canonical tokenizer vocabulary ID.
    pub token_id: u32,
    /// F32 score at the declared processing stage.
    pub score: f32,
    /// Membership in the effective tokenizer/constraint domain before a forced
    /// choice. When domain information is unavailable, this defaults to true;
    /// consult [`CaptureCandidates::domain`] before treating it as known.
    #[serde(default = "candidate_allowed_default")]
    pub allowed: bool,
}

fn candidate_allowed_default() -> bool {
    true
}

/// Exact sampling-domain summary for the captured logits row, before forcing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateDomain {
    /// Number of model output IDs allowed by tokenizer validity and constraints.
    pub allowed_tokens: u64,
    /// Actual model output width, including any tokenizer holes or padding.
    pub vocabulary: u64,
    /// A semantic constraint excludes at least one otherwise tokenizer-valid ID
    /// in this output vocabulary. Sampler truncation and forcing are excluded.
    pub constrained: bool,
}

/// Closed pre-forcing predicate for sampling observation. This is descriptive
/// semantic data, never an arbitrary callback or numerical source permission.
#[derive(Debug, Clone, Copy)]
pub enum CaptureTokenFilter<'a> {
    /// An ordinary concrete filter, including an already intersected allow set.
    Fixed(&'a crate::TokenFilter),
    /// Actual borrowed packed semantic mask intersected with tokenizer validity.
    Packed(crate::PackedTokenFilter<'a>),
    /// The actual source-derived forbidden predicate at its provisional prefix.
    Forbidden(crate::speculative::ForbiddenControllerDecision<'a>),
}
impl CaptureTokenFilter<'_> {
    /// Exact allowed-ID predicate before prospective forcing or truncation.
    pub fn allows(self, token: u32) -> bool {
        match self {
            Self::Fixed(filter) => filter.allows(token),
            Self::Packed(filter) => filter.allows(token),
            Self::Forbidden(decision) => decision.allows(token),
        }
    }
}
impl<'a> From<&'a crate::TokenFilter> for CaptureTokenFilter<'a> {
    fn from(filter: &'a crate::TokenFilter) -> Self {
        Self::Fixed(filter)
    }
}
impl<'a> From<&'a crate::SharedTokenFilter> for CaptureTokenFilter<'a> {
    fn from(filter: &'a crate::SharedTokenFilter) -> Self {
        Self::Fixed(filter)
    }
}

/// Borrowed exact filters for observation at one sampling decision.
/// No grammar queries, native values or forced-token overrides are retained.
#[derive(Debug, Clone, Copy)]
pub struct CaptureTokenDomain<'a> {
    /// The exact intersection used for sampling before any forced-token override.
    pub filter: CaptureTokenFilter<'a>,
    /// The tokenizer-valid domain before semantic restrictions.
    pub tokenizer_validity: &'a crate::TokenFilter,
}

impl CaptureTokenDomain<'_> {
    /// Summarizes only IDs in the actual model output, matching filter padding
    /// and truncation semantics without constructing another vocabulary mask.
    pub fn summary(&self, vocabulary: u32) -> CandidateDomain {
        let mut allowed_tokens = 0;
        let mut constrained = false;
        for token in 0..vocabulary {
            let allowed = self.filter.allows(token);
            allowed_tokens += u64::from(allowed);
            constrained |= !allowed && self.tokenizer_validity.allows(token);
        }
        CandidateDomain {
            allowed_tokens,
            vocabulary: u64::from(vocabulary),
            constrained,
        }
    }
}

/// Bounded highest-score candidates for the last row of the current model logits.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaptureCandidates {
    /// Explicit score-processing stage; never inferred to be a probability.
    pub stage: CandidateScoreStage,
    /// Explicit intervention position. Older records describe original logits.
    #[serde(default)]
    pub source: CandidateLogitsSource,
    /// Descending scores; equal-score ordering follows the native sorter.
    pub candidates: Vec<CaptureCandidate>,
    /// Exact domain information, or unknown for older records/controllers that
    /// cannot expose the pre-override domain. Unknown candidates use `allowed:
    /// true`; that fallback is not evidence of permission or a probability.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub domain: Option<CandidateDomain>,
}

/// Independently enforced accounting dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureBudget {
    /// Number of value transformations.
    Captures,
    /// Logical native capture storage.
    Retention,
    /// Host buffers and materialization.
    Host,
    /// UTF-8 JSON record size.
    Encoded,
}

/// Measured data and diagnostic outcomes remain distinct.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CaptureOutcome {
    /// The requested result is present.
    Captured,
    /// An explicit preview omitted part of the selected tensor.
    Truncated {
        /// Number of values in the selected tensor.
        available_elements: u64,
        /// Number of materialized preview values.
        emitted_elements: u64,
    },
    /// Schedule or budget explicitly omitted this result.
    Skipped {
        /// Structured omission reason.
        reason: CaptureSkipReason,
    },
    /// The selected point was not emitted by execution.
    Missing,
    /// Capture execution failed.
    Failed {
        /// Structured category independent of the bounded human diagnostic.
        reason: CaptureFailureReason,
        /// Failure description.
        message: String,
    },
}

/// Portable capture-failure categories; diagnostic text is separately bounded.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CaptureFailureReason {
    /// The requested reservation exceeded an admitted budget.
    Limit {
        /// Rejected accounting dimension.
        budget: CaptureBudget,
        /// Whether the run-total rather than step limit was exceeded.
        cumulative: bool,
    },
    /// Backend cannot implement the requested operation on this value.
    Unsupported,
    /// Shape, arithmetic, or protocol constraints were invalid.
    Invalid,
    /// The reserved native transformation failed.
    Native,
}

/// Structured explanations for an intentionally omitted value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CaptureSkipReason {
    /// The selected point does not belong to this explicitly selected invocation.
    NotInvoked,
    /// The current phase or prediction is not selected.
    Schedule,
    /// A value reservation exceeded its budget.
    Limit {
        /// Rejected accounting dimension.
        budget: CaptureBudget,
        /// True for a run-total limit; false for a step limit.
        cumulative: bool,
    },
}

/// One result linked back to a catalog point. The enclosing generation record owns
/// run/session identity, prediction token, phase, positions, and timing.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct CaptureRecord {
    /// Wire schema version; unsupported versions are rejected.
    pub schema_version: u32,
    /// Caller identity of the selected transformation.
    pub selection_id: String,
    /// Exact observation selector in the retained catalog.
    pub path: String,
    /// Logical architecture node associated with this observation.
    pub node_id: String,
    /// Position relative to intervention at this path.
    pub position: crate::ObservationPosition,
    /// Actual source extents, absent if this point was never reached.
    pub source_shape: Option<Vec<u64>>,
    /// Actual tensor precision before slicing or host conversion. Floating payloads
    /// may be exported as F32 even when this is F16/BF16. Absent when unavailable,
    /// including a deferred observation that was never generated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_dtype: Option<crate::checkpoint::TensorDtype>,
    /// Actual extents after semantic axis slicing.
    pub selected_shape: Option<Vec<u64>>,
    /// Distinguishes captured, truncated, skipped, missing, and failed values.
    pub outcome: CaptureOutcome,
    /// Captured data; absent for skipped, missing, or failed values.
    pub payload: Option<CapturePayload>,
    /// Conservative reservation charged before capture. Not allocator telemetry.
    pub charged: CaptureUsage,
}

/// A reservation-only view of capture accounting. Mechanisms cannot reset a step,
/// replace a ledger, or recover previously consumed credits through this contract.
pub trait CaptureReservation {
    /// Reserves work before allocation, evaluation, or transport.
    fn reserve(&mut self, usage: CaptureUsage) -> Result<Option<CaptureSkipReason>, CaptureError>;

    /// Prepays a move-only allowance. Dropping unused credits never refunds the
    /// parent. A skipped parent reservation grants no child authority.
    fn reserve_quota(&mut self, limit: CaptureUsage) -> Result<CaptureQuota, CaptureError> {
        match self.reserve(limit)? {
            None => (),
            Some(CaptureSkipReason::Limit { budget, cumulative }) => {
                return Err(CaptureError::Limit { budget, cumulative });
            }
            Some(_) => {
                return Err(CaptureError::Invalid(
                    "invalid quota reservation outcome".into(),
                ));
            }
        }
        Ok(CaptureQuota {
            limit,
            used: CaptureUsage::default(),
        })
    }
}

/// Move-only prepaid capture credits. Sub-reservations consume this allowance;
/// they never charge or reset the parent ledger a second time. There is no reset,
/// serialization, cloning, or refund operation, including after failed work.
#[derive(Debug)]
pub struct CaptureQuota {
    limit: CaptureUsage,
    used: CaptureUsage,
}

impl CaptureQuota {
    /// Full nonrefundable charge made to the parent at admission.
    pub const fn limit(&self) -> CaptureUsage {
        self.limit
    }
    /// Credits consumed by mechanisms or further move-only allowances.
    pub const fn used(&self) -> CaptureUsage {
        self.used
    }
}

impl CaptureReservation for CaptureQuota {
    fn reserve(&mut self, usage: CaptureUsage) -> Result<Option<CaptureSkipReason>, CaptureError> {
        let used = self.used.checked_add(usage)?;
        if let Some(budget) = used.exceeded(self.limit) {
            return Err(CaptureError::Limit {
                budget,
                cumulative: false,
            });
        }
        self.used = used;
        Ok(None)
    }
}

/// Monotone per-run ledger. Reservations cannot be refunded after work begins.
#[derive(Debug)]
pub struct CaptureLedger {
    limits: CaptureLimits,
    step: CaptureUsage,
    total: CaptureUsage,
}

impl CaptureLedger {
    /// Creates a fresh run ledger with the admitted limits.
    pub fn new(plan: &AdmittedCapturePlan) -> Self {
        Self {
            limits: plan.plan.limits.clone(),
            step: CaptureUsage::default(),
            total: CaptureUsage::default(),
        }
    }
    /// Starts an independently admitted child ledger with the usage already
    /// consumed at its fork boundary. Inherited usage counts against the child's
    /// cumulative limits, but not its next step. Later parent work is separate.
    pub fn with_inherited_usage(
        plan: &AdmittedCapturePlan,
        inherited: CaptureUsage,
    ) -> Result<Self, CaptureError> {
        if let Some(budget) = inherited.exceeded(plan.plan.limits.cumulative) {
            return Err(CaptureError::Limit {
                budget,
                cumulative: true,
            });
        }
        Ok(Self {
            limits: plan.plan.limits.clone(),
            step: CaptureUsage::default(),
            total: inherited,
        })
    }
    /// Resets step reservations while preserving cumulative accounting.
    pub fn begin_step(&mut self) {
        self.step = CaptureUsage::default();
    }
    /// Returns current step reservations.
    pub fn step(&self) -> CaptureUsage {
        self.step
    }
    /// Returns cumulative reservations.
    pub fn total(&self) -> CaptureUsage {
        self.total
    }
    /// Charges a conservative reservation before native work; skipped reservations are not charged.
    pub fn reserve(
        &mut self,
        usage: CaptureUsage,
    ) -> Result<Option<CaptureSkipReason>, CaptureError> {
        let step = self.step.checked_add(usage)?;
        let total = self.total.checked_add(usage)?;
        let exceeded = step
            .exceeded(self.limits.per_step)
            .map(|b| (b, false))
            .or_else(|| total.exceeded(self.limits.cumulative).map(|b| (b, true)));
        if let Some((budget, cumulative)) = exceeded {
            return match self.limits.on_limit {
                CaptureLimitPolicy::Fail => Err(CaptureError::Limit { budget, cumulative }),
                CaptureLimitPolicy::Skip => {
                    Ok(Some(CaptureSkipReason::Limit { budget, cumulative }))
                }
            };
        }
        self.step = step;
        self.total = total;
        Ok(None)
    }
}

impl CaptureReservation for CaptureLedger {
    fn reserve(&mut self, usage: CaptureUsage) -> Result<Option<CaptureSkipReason>, CaptureError> {
        CaptureLedger::reserve(self, usage)
    }
}

/// Invalid, unsupported, or oversized capture requests.
#[derive(Debug, Clone, thiserror::Error)]
pub enum CaptureError {
    /// Fixed prospective declaration-storage refusal.
    #[error(transparent)]
    AdmissionStorage(#[from] CaptureAdmissionStorageError),
    /// Fixed intervention declaration refusal, without allocated diagnostics.
    #[error(transparent)]
    Intervention(#[from] crate::intervention::InterventionDeclarationError),
    /// Ordinary error and complete diagnostic storage retained independently of a frame.
    #[error(transparent)]
    Retained(RetainedCaptureError),
    /// Plan or runtime tensor constraints are inconsistent.
    #[error("invalid capture plan or tensor: {0}")]
    Invalid(String),
    /// The selected backend or execution cannot guarantee the requested operation.
    #[error("capture operation unsupported: {0}")]
    Unsupported(String),
    /// The exact selector is absent from the retained catalog.
    #[error("capture path absent from catalog: {0}")]
    MissingPath(String),
    /// Checked size arithmetic overflowed.
    #[error("capture size arithmetic overflow")]
    Overflow,
    /// A per-step or cumulative reservation exceeded its limit.
    #[error("capture {budget:?} limit exceeded (cumulative: {cumulative})")]
    Limit {
        /// Rejected accounting dimension.
        budget: CaptureBudget,
        /// True for a run-total limit; false for a step limit.
        cumulative: bool,
    },
}

/// Computes tensor element count using checked multiplication.
pub fn elements(shape: &[u64]) -> Result<u64, CaptureError> {
    // Even a shape containing zero must not conceal overflow in another extent.
    let nonzero = shape
        .iter()
        .filter(|dimension| **dimension != 0)
        .try_fold(1, |count, dimension| mul(count, *dimension))?;
    Ok(if shape.contains(&0) { 0 } else { nonzero })
}
/// Adds two size quantities with overflow rejection.
pub fn add(a: u64, b: u64) -> Result<u64, CaptureError> {
    a.checked_add(b).ok_or(CaptureError::Overflow)
}
/// Multiplies two size quantities with overflow rejection.
pub fn mul(a: u64, b: u64) -> Result<u64, CaptureError> {
    a.checked_mul(b).ok_or(CaptureError::Overflow)
}

/// Backend mechanism for capture. The estimator is side-effect-free and must
/// conservatively cover all collector-owned native storage (including retained
/// backing arrays and dependencies), host buffers/transfers, and JSON payloads.
/// `transform` may run only after successful reservation. It must not retain the
/// tensor beyond the call or fall back to copying an unsliced source to the host.
pub trait CaptureBackend {
    /// Validate the actual ordinary transform source without allocation/evaluation.
    fn validate_capture_prefill_transform_source(
        &self,
        _tensor: &Self::Tensor,
        _fragment: &CapturePrefillTransformFragment<'_, '_>,
    ) -> Option<Result<crate::checkpoint::TensorDtype, Self::Error>> {
        None
    }
    /// Cold logical allowance for one exact physical transform, including descriptors.
    fn estimate_capture_prefill_transform(
        &self,
        _fragment: &CapturePrefillTransformFragment<'_, '_>,
    ) -> Result<CaptureUsage, CaptureError> {
        Err(CaptureError::Unsupported(
            "ordinary prefill transform unavailable".into(),
        ))
    }
    /// Produce only host payload after prepaid native work under existing recovery.
    fn capture_prefill_transform(
        &mut self,
        _tensor: &Self::Tensor,
        _fragment: &CapturePrefillTransformFragment<'_, '_>,
    ) -> Option<Result<CapturePayload, Self::Error>> {
        None
    }
    /// Cold logical allowance for the actual terminal-row candidate operation.
    fn estimate_capture_prefill_candidates(
        &self,
        _geometry: &CaptureCandidateGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        Err(CaptureError::Unsupported(
            "ordinary prefill candidates unavailable".into(),
        ))
    }
    /// Validate and read the actual terminal source using the current sampler domain.
    fn capture_prefill_candidates(
        &mut self,
        _tensor: &Self::Tensor,
        _geometry: &CaptureCandidateGeometry<'_>,
    ) -> Option<Result<CaptureCandidates, Self::Error>> {
        None
    }
    /// Cold logical allowance for the actual terminal-row token-score operation.
    fn estimate_capture_prefill_token_scores(
        &self,
        _geometry: &CaptureTokenScoreGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        Err(CaptureError::Unsupported(
            "ordinary prefill token-scores unavailable".into(),
        ))
    }
    /// Validate and read the actual terminal source using the current sampler domain.
    fn capture_prefill_token_scores(
        &mut self,
        _tensor: &Self::Tensor,
        _geometry: &CaptureTokenScoreGeometry<'_>,
    ) -> Option<Result<CaptureTokenScores, Self::Error>> {
        None
    }
    /// Backend-native tensor type; never crosses the host-record boundary.
    type Tensor;
    /// Backend transformation error.
    type Error: std::error::Error + 'static;
    /// Cold full-invocation bound for one partitioned sparse fragment, including
    /// worst-case incoming routes from nonexporting source peers.
    fn estimate_partition_routed_units(
        &self,
        _request: &PartitionRoutedUnitCaptureRequest<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        Err(CaptureError::Unsupported(
            "partitioned routed-unit collector unavailable".into(),
        ))
    }
    /// Performs one prepaid native chunk, preserving original source coordinates
    /// and selecting only this fragment's global units from actual local columns.
    fn capture_partition_routed_units(
        &mut self,
        _source: &PartitionRoutedUnitCaptureSource<'_, Self::Tensor>,
        _request: &PartitionRoutedUnitCaptureRequest<'_>,
    ) -> Option<Result<RoutedUnitCapture, Self::Error>> {
        None
    }
    /// Cold full-invocation reservation for selected routed units and route
    /// metadata. Runtime reserves once before any chunk is evaluated or copied.
    fn estimate_routed_units(
        &self,
        _shape: &[u64],
        _selection: &CaptureSelection,
        _slice: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        Err(CaptureError::Unsupported(
            "routed-unit collector unavailable".into(),
        ))
    }
    /// Performs one prepaid native chunk. Returning None declares a missing
    /// collector; no fallback may copy an unsliced expert-dense tensor.
    fn capture_routed_units(
        &mut self,
        _source: &RoutedUnitCaptureSource<'_, Self::Tensor>,
        _geometry: RoutedUnitGeometry,
        _slice: &ResolvedCaptureSlice,
    ) -> Option<Result<RoutedUnitCapture, Self::Error>> {
        None
    }
    /// Inspect an actual floating decoder source against a closed physical
    /// prefill fragment. No native work, shape allocation or handle retention.
    fn validate_capture_prefill_source(
        &self,
        _tensor: &Self::Tensor,
        _fragment: &CapturePrefillFragment<'_, '_>,
    ) -> Option<Result<crate::checkpoint::TensorDtype, Self::Error>> {
        None
    }
    /// Existing logical native/host work quota for this exact physical fragment.
    /// No result encoding or additional logical capture is implied.
    fn estimate_capture_prefill_fragment(
        &self,
        _fragment: &CapturePrefillFragment<'_, '_>,
    ) -> Result<CaptureUsage, CaptureError> {
        Err(CaptureError::Unsupported(
            "ordinary prefill fragments unavailable".into(),
        ))
    }
    /// Eagerly copy only the prepaid physical selection; the caller retains the
    /// ordinary native recovery and the final logical destination separately.
    fn capture_prefill_fragment(
        &mut self,
        _tensor: &Self::Tensor,
        _fragment: &CapturePrefillFragment<'_, '_>,
    ) -> Option<Result<crate::TensorObservation, Self::Error>> {
        None
    }
    /// Reads tensor shape without evaluation, copying, or retaining the tensor.
    fn shape(&self, tensor: &Self::Tensor) -> Result<Vec<u64>, Self::Error>;
    /// Reads actual source precision without evaluation, copying, or retention.
    /// Unknown precision must remain absent, rather than inferred from host output.
    fn source_dtype(&self, _tensor: &Self::Tensor) -> Option<crate::checkpoint::TensorDtype> {
        None
    }
    /// Cold bound for completing an ordinary source's dependencies on every
    /// executing partition, including replicas that export no fragment. Includes
    /// retained source storage and completion resources, but no host tensor copy.
    /// The default is suitable only for eager backends. Lazy backends must override
    /// this and `prepare_partition_source` and reject unsupported wait policies.
    fn estimate_partition_source(
        &self,
        _shape: &[u64],
        _wait: crate::BoundedCompletionWait,
    ) -> Result<CaptureUsage, CaptureError> {
        Ok(CaptureUsage::default())
    }
    /// Completes prepaid ordinary source dependencies without exporting values.
    /// For generated capture, runtime supplies the ordinary prototype and never
    /// invokes a nonexporting replica's factory. On error or deadline, native
    /// resources and enclosing submission authority must survive until safe
    /// completion, terminal failure or teardown. A polling error is not completion.
    fn prepare_partition_source(
        &mut self,
        _tensor: &Self::Tensor,
        _wait: crate::BoundedCompletionWait,
    ) -> Result<crate::BoundedCompletionOutcome, Self::Error> {
        Ok(crate::BoundedCompletionOutcome::Completed)
    }
    /// Computes a conservative reservation without evaluating or retaining the tensor.
    fn estimate(
        &self,
        tensor: &Self::Tensor,
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError>;
    /// Prices the generated tensor's transform using its cold source facts.
    /// The prototype supplies geometry only. Dtype-dependent estimators must
    /// override this default; creation storage is charged separately by runtime.
    fn estimate_generated(
        &self,
        prototype: &Self::Tensor,
        _source: &GeneratedCaptureSource,
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        self.estimate(prototype, selection, slice)
    }
    /// Performs the reserved transformation without retaining native handles after return.
    fn transform(
        &mut self,
        tensor: &Self::Tensor,
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CapturePayload, Self::Error>;
}

/// Outcome of the enclosing model forward, distinct from each capture operation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureStepOutcome {
    /// A low-level caller supplied no transaction evidence.
    #[default]
    Untracked,
    /// The model forward transaction committed. Subsequent sampling can still fail.
    Committed,
    /// The model forward transaction aborted. Earlier observations may have completed,
    /// but they do not describe a committed prediction or authorize state reuse.
    Aborted,
}

/// One drained step's bounded records, including diagnostics from aborted forwards.
/// Timing is measured separately from native forward/sampling time; synchronous
/// callback costs are included in end-to-end time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapturedStep {
    /// Explicit model transaction outcome, including an untracked low-level call.
    pub outcome: CaptureStepOutcome,
    /// Forward phase that produced these captures.
    pub phase: CapturePhase,
    /// Explicit physical geometry for an independently admitted invocation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub invocation: Option<CaptureInvocationShape>,
    /// Run-relative prediction index; zero is predicted by prefill.
    pub prediction_index: u64,
    /// At most one record for each admitted selection.
    pub records: Vec<CaptureRecord>,
    /// Verified global ownership and forward provenance for partitioned records.
    /// Evidence is published only after the enclosing model transaction commits.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub partitions: Vec<PartitionCaptureEvidence>,
    /// Attributed intervention outcomes and optional evidence sharing these budgets.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub interventions: Vec<crate::intervention::InterventionRecord>,
    /// Reservations charged to this forward step.
    pub step_usage: CaptureUsage,
    /// Reservations charged since the start of this run.
    pub cumulative_usage: CaptureUsage,
    /// Wall time spent in capture transforms and capture JSON accounting.
    pub capture_seconds: f64,
}

/// Retained discovery and exact source identity of the loaded session. Admission
/// uses these facts rather than accepting an unrelated caller-supplied catalog.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptureDiscovery {
    /// Content-exact identity of the prepared physical source graph.
    pub artifact_identity: String,
    /// Architecture-declared points retained from preparation.
    pub catalog: ObservationCatalog,
    /// Support for the execution actually realized by the session.
    pub support: ObservationSupportReport,
}

mod revalidation;
pub use revalidation::CaptureRevalidationError;

mod candidate_geometry;
mod tensor_geometry;
pub use candidate_geometry::CaptureCandidateGeometry;
mod token_score_geometry;
pub use tensor_geometry::{
    CaptureHistogramGeometry, CaptureRoutedPrefillFragment, CaptureRoutedPrefillPlan,
    CaptureRoutedTokenWindow, CaptureRoutedUnitsGeometry, CaptureSummaryGeometry,
    CaptureTensorGeometry, CaptureTensorGeometryError,
};
pub use token_score_geometry::CaptureTokenScoreGeometry;

mod shared_tensor_wire;
pub use shared_tensor_wire::CaptureTensorWire;
