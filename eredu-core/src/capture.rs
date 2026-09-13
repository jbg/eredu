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
mod invocation;
#[cfg(test)]
mod prepared_geometry_tests;
pub use invocation::{CaptureInvocationBounds, CaptureInvocationShape};
mod routed;
mod tensor_wire;
pub use routed::*;

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
    /// Conservative native storage bound for creating the source and temporaries.
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
        self.admit_geometry(catalog, support, capabilities, request, None)
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
        bounds.maximum()?;
        self.admit_geometry(
            catalog,
            support,
            capabilities,
            bounds.request(),
            Some(bounds),
        )
    }

    fn admit_geometry(
        self,
        catalog: &ObservationCatalog,
        support: &ObservationSupportReport,
        capabilities: &CaptureCapabilities,
        request: CaptureRequestShape,
        invocation_bounds: Option<CaptureInvocationBounds>,
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
        if invocation_bounds.is_none() {
            add(request.prompt_tokens, request.max_predictions)?;
        }
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
            let sparse = matches!(
                point.value_type,
                crate::ObservationValueType::RoutedUnits { .. }
            );
            if sparse != matches!(selection.transform, CaptureTransform::RoutedUnits) {
                return Err(CaptureError::Unsupported(
                    "routed-unit boundaries require the routed_units transform".into(),
                ));
            }
            if let crate::ObservationValueType::RoutedUnits { geometry, .. } = &point.value_type {
                geometry.components()?;
            }
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
                if enabled
                    && !matches!(
                        status,
                        ObservationSupportStatus::Supported
                            | ObservationSupportStatus::Conditional(_)
                    )
                {
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
            if let CaptureTransform::TokenScores { token_ids } = &selection.transform {
                let unique: std::collections::BTreeSet<_> = token_ids.iter().collect();
                if token_ids.is_empty()
                    || token_ids.len() > 64
                    || unique.len() != token_ids.len()
                    || selection.path != crate::MODEL_LOGITS_OBSERVATION_PATH
                    || !selection.slices.is_empty()
                {
                    return Err(CaptureError::Invalid(
                        "token scoring requires 1..=64 unique IDs and unsliced model.logits".into(),
                    ));
                }
                if request.batch != 1 {
                    return Err(CaptureError::Unsupported(
                        "token scoring requires batch one".into(),
                    ));
                }
                if let Some(SymbolicDimension::Known(vocabulary)) = point
                    .axes
                    .as_ref()
                    .and_then(|axes| axes.last())
                    .map(|axis| &axis.dimension)
                {
                    if token_ids.iter().any(|id| *id as usize >= *vocabulary) {
                        return Err(CaptureError::Invalid(
                            "selected score ID exceeds model vocabulary".into(),
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
                let range = if invocation_bounds.is_some() {
                    selection
                        .schedule
                        .count_coordinates(phase, 0, request.max_predictions)?
                } else {
                    selection
                        .schedule
                        .count_and_last(phase, request.max_predictions)?
                };
                if let Some((count, last)) = range {
                    let first = last - mul(count - 1, selection.schedule.every)?;
                    for prediction in [first, last] {
                        for slice in &selection.slices {
                            let axis = point
                                .axes
                                .as_ref()
                                .and_then(|axes| axes.iter().find(|axis| axis.name == slice.axis))
                                .expect("axis was validated");
                            let geometry = match invocation_bounds {
                                Some(bounds) => bounds.maximum()?,
                                None => request.invocation_shape(phase, prediction)?,
                            };
                            if geometry
                                .extent(&axis.dimension)?
                                .is_some_and(|extent| slice.end > extent)
                            {
                                return Err(CaptureError::Invalid(format!(
                                    "slice {} exceeds known request extent",
                                    slice.axis
                                )));
                            }
                        }
                        let geometry = match invocation_bounds {
                            Some(bounds) => bounds.maximum()?,
                            None => request.invocation_shape(phase, prediction)?,
                        };
                        if let Some(shape) = geometry.resolve(point)? {
                            resolve_slice(point, selection, &shape)?;
                        }
                    }
                }
            }
            points.push(point.clone());
        }
        // Identity includes catalog semantics and request shape, not just caller labels.
        let bytes = match invocation_bounds {
            Some(bounds) => serde_json::to_vec(&("invocation", &self, &points, bounds)),
            None => serde_json::to_vec(&(&self, &points, request)),
        }
        .map_err(|e| CaptureError::Invalid(e.to_string()))?;
        let identity = Sha256::digest(bytes)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        Ok(AdmittedCapturePlan {
            plan: self,
            points,
            request,
            invocation_bounds,
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

    /// Physical geometry of an ordinary prompt or single-row cached forward.
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
    /// Checks every known or request-resolved axis, including partially known shapes.
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
    /// Resolves symbolic axes against ordinary request geometry.
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
    /// Explicit invocation bounds; absence retains ordinary prompt/decode geometry.
    pub fn invocation_bounds(&self) -> Option<CaptureInvocationBounds> {
        self.invocation_bounds
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
            (None, None) => self.request.invocation_shape(phase, prediction),
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
            None => self.request.resolve(point, phase, prediction),
        }
    }
    /// Whether this plan captures no observation points.
    pub fn is_empty(&self) -> bool {
        self.plan.selections.is_empty()
    }
}

mod partition;
pub use partition::{CaptureFragmentGeometry, CaptureSlicePartition};
mod partition_wire;
pub use partition_wire::{
    PartitionCaptureCombination, PartitionCaptureContext, PartitionCaptureContributionRecord,
    PartitionCaptureEvidence, PartitionCaptureFragmentRecord, PartitionCaptureProducerRecord,
    PartitionCaptureRegion, PARTITION_CAPTURE_SCHEMA_VERSION,
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
                ))
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
    /// A low-level caller or older serialized record supplied no transaction evidence.
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
    /// Actual model transaction outcome; absence in older records is untracked.
    #[serde(default)]
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
