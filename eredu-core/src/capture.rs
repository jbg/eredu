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
mod tensor_wire;

use crate::{
    ObservationCatalog, ObservationPoint, ObservationSupportReport, ObservationSupportStatus,
    SymbolicDimension, TensorObservation,
};

/// Wire version for capture plans and records.
pub const CAPTURE_SCHEMA_VERSION: u32 = 1;

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
}

/// Portable transformation categories used by capability discovery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureTransformKind {
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
}

impl CaptureTransform {
    /// Transformation category for capability matching.
    pub fn kind(&self) -> CaptureTransformKind {
        match self {
            Self::Preview { .. } => CaptureTransformKind::Preview,
            Self::Slice => CaptureTransformKind::Slice,
            Self::FullTensor => CaptureTransformKind::FullTensor,
            Self::Summary => CaptureTransformKind::Summary,
            Self::Histogram { .. } => CaptureTransformKind::Histogram,
            Self::TopCandidates { .. } => CaptureTransformKind::TopCandidates,
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
    /// Whether physical allocator/workspace limits can be enforced.
    pub physical_native_limit: bool,
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
pub struct CaptureLimits {
    /// Limits reset at each prefill or decode step.
    pub per_step: CaptureUsage,
    /// Limits summed across all steps in the run.
    pub cumulative: CaptureUsage,
    /// Optional physical allocator ceiling. Rejected if the backend cannot prove it.
    pub physical_native_bytes: Option<u64>,
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
                physical_native_bytes: None,
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
        if self.schema_version != CAPTURE_SCHEMA_VERSION
            || catalog.schema_version != crate::DISCOVERY_SCHEMA_VERSION
            || support.schema_version != crate::DISCOVERY_SCHEMA_VERSION
        {
            return Err(CaptureError::Invalid("unsupported schema version".into()));
        }
        if self.limits.physical_native_bytes.is_some() && !capabilities.physical_native_limit {
            return Err(CaptureError::Unsupported(
                "physical native allocator/workspace bound".into(),
            ));
        }
        if request.batch == 0 || request.prompt_tokens == 0 || request.max_predictions == 0 {
            return Err(CaptureError::Invalid(
                "batch, prompt, and prediction limits must be positive".into(),
            ));
        }
        add(request.prompt_tokens, request.max_predictions)?;
        mul(request.batch, request.prompt_tokens)?;
        let mut ids = std::collections::BTreeSet::new();
        let mut points = Vec::new();
        for selection in &self.selections {
            if selection.id.is_empty() || !ids.insert(selection.id.as_str()) {
                return Err(CaptureError::Invalid(
                    "capture IDs must be nonempty and unique".into(),
                ));
            }
            let point = catalog
                .get(&selection.path)
                .ok_or_else(|| CaptureError::MissingPath(selection.path.clone()))?;
            if !capabilities
                .transformations
                .contains(&selection.transform.kind())
            {
                return Err(CaptureError::Unsupported(format!(
                    "{:?}",
                    selection.transform.kind()
                )));
            }
            if selection.schedule.every == 0
                || selection
                    .schedule
                    .end_prediction
                    .is_some_and(|end| end <= selection.schedule.first_prediction)
            {
                return Err(CaptureError::Invalid("invalid capture schedule".into()));
            }
            let phase_support = support
                .points
                .iter()
                .find(|p| p.path == selection.path)
                .ok_or_else(|| {
                    CaptureError::Unsupported(format!("no selected support for {}", selection.path))
                })?;
            for (enabled, status) in [
                (selection.schedule.prefill, &phase_support.prefill),
                (selection.schedule.decode, &phase_support.decode),
            ] {
                if enabled && !matches!(status, ObservationSupportStatus::Supported) {
                    return Err(CaptureError::Unsupported(format!(
                        "{}: {status:?}",
                        selection.path
                    )));
                }
            }
            if matches!(selection.transform, CaptureTransform::Slice) && selection.slices.is_empty()
            {
                return Err(CaptureError::Invalid(
                    "slice capture requires an explicit axis slice; use FullTensor to opt in"
                        .into(),
                ));
            }
            if let CaptureTransform::Histogram { edges } = &selection.transform {
                if edges.len() < 2
                    || (edges.len() - 1) as u64 > capabilities.max_histogram_bins
                    || edges.iter().any(|edge| !edge.is_finite())
                    || edges.windows(2).any(|w| w[0] >= w[1])
                {
                    return Err(CaptureError::Invalid(
                        "histogram edges must be finite, increasing, and within the bin limit"
                            .into(),
                    ));
                }
            }
            if let CaptureTransform::TopCandidates { count } = selection.transform {
                if count == 0
                    || selection.path != crate::MODEL_LOGITS_OBSERVATION_PATH
                    || !selection.slices.is_empty()
                {
                    return Err(CaptureError::Invalid("candidate capture requires positive count, unsliced model.logits, and single-sequence execution".into()));
                }
                if request.batch != 1 {
                    return Err(CaptureError::Unsupported(
                        "candidate capture requires batch one".into(),
                    ));
                }
                if let Some(SymbolicDimension::Known(vocabulary)) = point
                    .axes
                    .as_ref()
                    .and_then(|axes| axes.last())
                    .map(|a| &a.dimension)
                {
                    if count > *vocabulary as u64 {
                        return Err(CaptureError::Invalid(
                            "candidate count exceeds vocabulary".into(),
                        ));
                    }
                }
            }
            let mut axes = std::collections::BTreeSet::new();
            for slice in &selection.slices {
                if slice.stride == 0 || slice.start > slice.end || !axes.insert(&slice.axis) {
                    return Err(CaptureError::Invalid(
                        "invalid or duplicate axis slice".into(),
                    ));
                }
                if !point
                    .axes
                    .as_ref()
                    .is_some_and(|axes| axes.iter().any(|axis| axis.name == slice.axis))
                {
                    return Err(CaptureError::Invalid(format!(
                        "unknown axis {}",
                        slice.axis
                    )));
                }
            }
            // Resolve every known shape before execution, and defer only genuinely
            // runtime-dependent dimensions. Unknown never becomes a zero extent.
            for phase in [CapturePhase::Prefill, CapturePhase::Decode] {
                if let Some((count, last)) = selection
                    .schedule
                    .count_and_last(phase, request.max_predictions)?
                {
                    let first = last - mul(count - 1, selection.schedule.every)?;
                    for prediction in [first, last] {
                        for slice in &selection.slices {
                            let axis = point
                                .axes
                                .as_ref()
                                .and_then(|axes| axes.iter().find(|axis| axis.name == slice.axis))
                                .expect("axis was validated");
                            if request
                                .extent(&axis.dimension, phase, prediction)?
                                .is_some_and(|extent| slice.end > extent)
                            {
                                return Err(CaptureError::Invalid(format!(
                                    "slice {} exceeds known request extent",
                                    slice.axis
                                )));
                            }
                        }
                        if let Some(shape) = request.resolve(point, phase, prediction)? {
                            resolve_slice(point, selection, &shape)?;
                        }
                    }
                }
            }
            points.push(point.clone());
        }
        // Identity includes catalog semantics and request shape, not just caller labels.
        let bytes = serde_json::to_vec(&(&self, &points, request))
            .map_err(|e| CaptureError::Invalid(e.to_string()))?;
        let identity = Sha256::digest(bytes)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        Ok(AdmittedCapturePlan {
            plan: self,
            points,
            request,
            identity,
        })
    }
}

/// Request geometry used for admission. This initial protocol is ordinary committed
/// text generation; media, speculative and rank-partitioned runs need separate support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptureRequestShape {
    /// Number of sequences in the request.
    pub batch: u64,
    /// Number of prefill positions per sequence.
    pub prompt_tokens: u64,
    /// Maximum number of generated predictions, including the prefill prediction.
    pub max_predictions: u64,
}

impl CaptureRequestShape {
    fn extent(
        self,
        dimension: &SymbolicDimension,
        phase: CapturePhase,
        prediction: u64,
    ) -> Result<Option<u64>, CaptureError> {
        let sequence = if phase == CapturePhase::Prefill {
            self.prompt_tokens
        } else {
            1
        };
        Ok(match dimension {
            SymbolicDimension::Known(n) => {
                Some(u64::try_from(*n).map_err(|_| CaptureError::Overflow)?)
            }
            SymbolicDimension::Batch => Some(self.batch),
            SymbolicDimension::Sequence => Some(sequence),
            SymbolicDimension::TokenRows => Some(mul(self.batch, sequence)?),
            SymbolicDimension::Context => Some(add(self.prompt_tokens, prediction)?),
            SymbolicDimension::MediaPositions | SymbolicDimension::Unknown => None,
        })
    }
    /// Checks every known or request-resolved axis, even when another axis is unknown.
    pub fn validate_actual(
        self,
        point: &ObservationPoint,
        phase: CapturePhase,
        prediction: u64,
        shape: &[u64],
    ) -> Result<(), CaptureError> {
        let Some(axes) = &point.axes else {
            if shape.len() > 32 {
                return Err(CaptureError::Unsupported(
                    "capture rank exceeds the 32-axis metadata bound".into(),
                ));
            }
            return elements(shape).map(|_| ());
        };
        if axes.len() != shape.len() {
            return Err(CaptureError::Invalid(
                "runtime rank differs from the catalog".into(),
            ));
        }
        for (axis, actual) in axes.iter().zip(shape) {
            let expected = self.extent(&axis.dimension, phase, prediction)?;
            if expected.is_some_and(|expected| expected != *actual) {
                return Err(CaptureError::Invalid(format!(
                    "runtime extent for {} differs from catalog/request",
                    axis.name
                )));
            }
        }
        elements(shape).map(|_| ())
    }

    /// Resolves symbolic axes against the request; an unknown extent returns None.
    pub fn resolve(
        self,
        point: &ObservationPoint,
        phase: CapturePhase,
        prediction: u64,
    ) -> Result<Option<Vec<u64>>, CaptureError> {
        let Some(axes) = &point.axes else {
            return Ok(None);
        };
        let mut shape = Vec::with_capacity(axes.len());
        for axis in axes {
            let Some(extent) = self.extent(&axis.dimension, phase, prediction)? else {
                return Ok(None);
            };
            shape.push(extent);
        }
        elements(&shape)?;
        Ok(Some(shape))
    }
}

/// Immutable admission proof. Deserialization cannot forge admission.
#[derive(Debug, Clone)]
pub struct AdmittedCapturePlan {
    plan: CapturePlan,
    points: Vec<ObservationPoint>,
    request: CaptureRequestShape,
    identity: String,
}

impl AdmittedCapturePlan {
    /// Stable digest of the plan, selected catalog semantics, and request shape.
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
    /// Returns admitted request geometry.
    pub fn request(&self) -> CaptureRequestShape {
        self.request
    }
    /// Whether this plan captures no observation points.
    pub fn is_empty(&self) -> bool {
        self.plan.selections.is_empty()
    }
}

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

/// Checks axis selection against actual extents without touching tensor payloads.
pub fn resolve_slice(
    point: &ObservationPoint,
    selection: &CaptureSelection,
    shape: &[u64],
) -> Result<ResolvedCaptureSlice, CaptureError> {
    if point
        .axes
        .as_ref()
        .is_some_and(|axes| axes.len() != shape.len())
    {
        return Err(CaptureError::Invalid(
            "runtime tensor rank differs from catalog".into(),
        ));
    }
    let mut output = ResolvedCaptureSlice {
        starts: vec![0; shape.len()],
        ends: shape.to_vec(),
        strides: vec![1; shape.len()],
        shape: shape.to_vec(),
    };
    for slice in &selection.slices {
        let axis = point
            .axes
            .as_ref()
            .and_then(|axes| axes.iter().position(|axis| axis.name == slice.axis))
            .ok_or_else(|| CaptureError::Invalid(format!("unknown axis {}", slice.axis)))?;
        if slice.stride == 0 || slice.start > slice.end || slice.end > shape[axis] {
            return Err(CaptureError::Invalid(format!(
                "slice {} exceeds runtime extent",
                slice.axis
            )));
        }
        output.starts[axis] = slice.start;
        output.ends[axis] = slice.end;
        output.strides[axis] = slice.stride;
        output.shape[axis] = (slice.end - slice.start).div_ceil(slice.stride);
    }
    elements(&output.shape)?;
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum CapturePayload {
    /// Raw or sliced tensor values; integer identifiers retain exact precision.
    Tensor(#[serde(with = "tensor_wire")] TensorObservation),
    /// Finite/non-finite counts and finite-only statistics.
    Summary(CaptureSummary),
    /// Bounded fixed-edge histogram.
    Histogram(CaptureHistogram),
    /// Native top-k extraction with an explicit score-processing stage.
    Candidates(CaptureCandidates),
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

/// Borrowed exact filters for observation at one sampling decision.
/// No grammar queries, native values or forced-token overrides are retained.
#[derive(Debug, Clone, Copy)]
pub struct CaptureTokenDomain<'a> {
    /// The exact intersection used for sampling before any forced-token override.
    pub filter: &'a crate::TokenFilter,
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
    /// Actual extents after semantic axis slicing.
    pub selected_shape: Option<Vec<u64>>,
    /// Distinguishes captured, truncated, skipped, missing, and failed values.
    pub outcome: CaptureOutcome,
    /// Captured data; absent for skipped, missing, or failed values.
    pub payload: Option<CapturePayload>,
    /// Conservative reservation charged before capture. Not allocator telemetry.
    pub charged: CaptureUsage,
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

/// Invalid, unsupported, or oversized capture requests.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CaptureError {
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
    /// Backend-native tensor type; never crosses the host-record boundary.
    type Tensor;
    /// Backend transformation error.
    type Error: std::error::Error + 'static;
    /// Reads tensor shape without evaluation, copying, or retaining the tensor.
    fn shape(&self, tensor: &Self::Tensor) -> Result<Vec<u64>, Self::Error>;
    /// Computes a conservative reservation without evaluating or retaining the tensor.
    fn estimate(
        &self,
        tensor: &Self::Tensor,
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError>;
    /// Performs the reserved transformation without retaining native handles after return.
    fn transform(
        &mut self,
        tensor: &Self::Tensor,
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CapturePayload, Self::Error>;
}

/// One completed step's bounded records. Timing is measured separately from native
/// forward/sampling time; synchronous callback costs are included in end-to-end time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapturedStep {
    /// Forward phase that produced these captures.
    pub phase: CapturePhase,
    /// Run-relative prediction index; zero is predicted by prefill.
    pub prediction_index: u64,
    /// At most one record for each admitted selection.
    pub records: Vec<CaptureRecord>,
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
