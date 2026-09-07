//! Immutable, session-bound intervention plans. Tensor selection and prediction
//! scheduling are independent. Operations execute in their declared list order;
//! routing operations with potentially overlapping schedules are rejected.

use crate::{capture::*, ObservationPoint, ObservationSupportStatus, TensorAxis};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

/// Independent wire version for intervention declarations, plans, and outcomes.
pub const INTERVENTION_SCHEMA_VERSION: u32 = 1;
/// Hard bound on operations, including inactive operations and their diagnostics.
pub const MAX_INTERVENTION_OPERATIONS: usize = 64;
/// Hard bound on compact JSON plan bytes, including replacement values.
pub const MAX_INTERVENTION_PLAN_BYTES: u64 = 1024 * 1024;
/// Hard bound on unencoded replacement, mask, bias, and ID storage.
pub const MAX_INTERVENTION_PAYLOAD_BYTES: u64 = 512 * 1024;

/// Exact activation storage type. Host payloads never authorize a silent cast.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterventionDtype {
    /// IEEE binary32.
    Float32,
    /// IEEE binary16, encoded as exact bits in host payloads.
    Float16,
    /// Bfloat16, encoded as exact bits in host payloads.
    Bfloat16,
}

/// Complete row-major host values. A preview is never a replacement payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "dtype", content = "values", rename_all = "snake_case")]
pub enum InterventionValues {
    /// Finite IEEE binary32 values.
    Float32(Vec<f32>),
    /// Finite IEEE binary16 bit patterns.
    Float16(Vec<u16>),
    /// Finite bfloat16 bit patterns.
    Bfloat16(Vec<u16>),
}

impl InterventionValues {
    /// Exact dtype encoded by these values.
    pub fn dtype(&self) -> InterventionDtype {
        match self {
            Self::Float32(_) => InterventionDtype::Float32,
            Self::Float16(_) => InterventionDtype::Float16,
            Self::Bfloat16(_) => InterventionDtype::Bfloat16,
        }
    }
    /// Number of complete elements.
    pub fn len(&self) -> usize {
        match self {
            Self::Float32(v) => v.len(),
            Self::Float16(v) | Self::Bfloat16(v) => v.len(),
        }
    }
    /// Whether there are no values.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn validate(&self) -> Result<(), CaptureError> {
        let finite = match self {
            Self::Float32(v) => v.iter().all(|v| v.is_finite()),
            Self::Float16(v) => v.iter().all(|v| v & 0x7c00 != 0x7c00),
            Self::Bfloat16(v) => v.iter().all(|v| v & 0x7f80 != 0x7f80),
        };
        require(finite, "intervention values must be finite")
    }
    fn bytes(&self) -> Result<u64, CaptureError> {
        mul(
            self.len() as u64,
            if self.dtype() == InterventionDtype::Float32 {
                4
            } else {
                2
            },
        )
    }
}

/// Complete typed tensor with exact selected extents; broadcasting is prohibited.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterventionTensor {
    /// Row-major extents, including all selected singleton axes.
    pub shape: Vec<u64>,
    /// Complete values in the exact target dtype.
    pub values: InterventionValues,
}

/// Distinct stages of an architecture's router. These are not interchangeable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutingScoreStage {
    /// Projection output, including the learned projection bias, before scoring.
    RawLogits,
    /// Scoring transform output, before selection-only corrections.
    TransformedScores,
    /// Scores plus learned selection corrections; affects ranking only.
    RankingScores,
}

/// Architecture-declared score transform, independent of native implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutingScoring {
    /// Softmax across the entire expert axis, then gather selected scores.
    Softmax,
    /// Select logits, then softmax across selected slots only.
    SelectedSoftmax,
    /// Elementwise sigmoid before selection.
    Sigmoid,
    /// Elementwise square root of softplus before selection.
    SqrtSoftplus,
}

/// Exact global routed-expert namespace and architecture coefficient policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterventionRoutingPolicy {
    /// Routed IDs are in `0..expert_count`; shared experts are outside this namespace.
    pub expert_count: u32,
    /// Exact number of distinct selected IDs per token row.
    pub top_k: u32,
    /// Score transform used by the ordinary router.
    pub scoring: RoutingScoring,
    /// Whether gathered scores are normalized before scaling.
    pub normalize_selected: bool,
    /// Denominator epsilon used by the architecture.
    pub normalization_epsilon: f32,
    /// Architecture's final coefficient multiplier.
    pub coefficient_scale: f32,
    /// Number of equal contiguous expert partitions used by group selection.
    pub groups: u32,
    /// Maximum eligible groups per token.
    pub selected_groups: u32,
    /// Whether a learned per-expert multiplier is applied after normalization.
    pub learned_coefficient_scale: bool,
    /// Separately executed shared experts, unchanged by routed operations.
    pub shared_experts: u32,
}

/// Execution boundary at which a target becomes mutable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterventionStage {
    /// After ordinary observation, before the activation is consumed downstream.
    Activation,
    /// After ordinary logits observation, before the existing sampler pipeline.
    LogitsBeforeSampling,
    /// Router score/ID/coefficient control before any expert provider dispatch.
    RoutingBeforeDispatch,
}

/// Operation categories used for exact capability matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterventionKind {
    /// Set selected activation elements to zero.
    Zero,
    /// Multiply selected activation elements by a finite scalar.
    Scale,
    /// Keep or zero individual selected activation elements.
    Mask,
    /// Replace every selected element with a complete typed payload.
    Replace,
    /// Add a complete typed payload to selected elements, including logit bias.
    Add,
    /// Set selected vocabulary IDs to negative infinity before ordinary sampling.
    MaskLogits,
    /// Remove routed experts from ranking eligibility and reselect.
    ExcludeExperts,
    /// Preserve IDs and set specified coefficients to zero, without renormalizing.
    ZeroExpertContribution,
    /// Apply bias at an explicitly supported router score stage.
    BiasRoutingScores,
    /// Dispatch exact per-token IDs, obtaining weights from the ordinary router.
    ForceExperts,
}

/// A genuine architecture intervention point, separately declared from observations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterventionPoint {
    /// Exact execution target; routing targets identify the routed module.
    pub path: String,
    /// Existing logical architecture node identity.
    pub node_id: String,
    /// Exact mutable stage.
    pub stage: InterventionStage,
    /// Semantic axes in storage order. Routing uses `[token, selected_expert]`.
    pub axes: Vec<TensorAxis>,
    /// Exact allowed activation dtypes; empty for routing control.
    pub dtypes: Vec<InterventionDtype>,
    /// Operations structurally implemented at this point.
    pub operations: Vec<InterventionKind>,
    /// Bias stages implemented by this router; empty for activations.
    pub score_stages: Vec<RoutingScoreStage>,
    /// Per-phase support after combining architecture and loaded-session facts.
    pub prefill: ObservationSupportStatus,
    /// Per-phase support after combining architecture and loaded-session facts.
    pub decode: ObservationSupportStatus,
    /// Additional shape/dtype/execution conditions presented to applications.
    pub conditions: Vec<String>,
    /// Present only for pre-dispatch routing points.
    pub routing: Option<InterventionRoutingPolicy>,
}

impl InterventionPoint {
    /// Geometry adapter for shared capture selection and request validation.
    pub fn observation_geometry(&self) -> ObservationPoint {
        ObservationPoint {
            path: self.path.clone(),
            node_id: self.node_id.clone(),
            meaning: String::new(),
            value_type: crate::ObservationValueType::Tensor,
            dtype: crate::ObservationDtype::Floating,
            axes: Some(self.axes.clone()),
            prefill: true,
            decode: true,
            requirements: vec![],
            position: crate::ObservationPosition::BeforeIntervention,
            retained_bytes: None,
            host_bytes: None,
        }
    }
}

/// Loaded-session intervention support, retaining exact prepared-source identity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterventionDiscovery {
    /// Intervention wire version.
    pub schema_version: u32,
    /// Content-exact identity of the prepared physical source graph.
    pub artifact_identity: String,
    /// Identity assigned when a backend session is realized. Cold declarations
    /// have no session identity and cannot admit nonempty intervention plans.
    #[serde(default)]
    pub session_identity: Option<String>,
    /// Genuine targets with explicit supported, unsupported, or unverified phases.
    pub points: Vec<InterventionPoint>,
}

/// Creates a fresh identity when a backend publishes a loaded intervention session.
/// It is provenance, not a secret or a substitute for completion authority.
pub fn new_intervention_session_identity() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    format!(
        "intervention-session-{}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos()),
        NEXT.fetch_add(1, Ordering::Relaxed)
    )
}

/// Side-effect-free backend facts; absent operations or dtypes are unsupported.
#[derive(Debug, Clone, Default)]
pub struct InterventionMechanisms {
    /// Implemented native operation categories, independent of model family.
    pub operations: Vec<InterventionKind>,
    /// Exact floating dtypes supported by native activation arithmetic.
    pub dtypes: Vec<InterventionDtype>,
    /// Router stages whose biases the backend implements.
    pub score_stages: Vec<RoutingScoreStage>,
}

/// One typed mutation. No variant contains native handles or executable code.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InterventionAction {
    /// Zero selected elements; retains dtype and shape.
    Zero {
        /// Required runtime storage dtype.
        dtype: InterventionDtype,
    },
    /// Native multiplication; the finite scalar must be representable in the dtype.
    Scale {
        /// Required runtime storage dtype.
        dtype: InterventionDtype,
        /// Finite scalar, with native dtype rounding.
        factor: f32,
    },
    /// False entries become zero; shape must exactly equal the selected region.
    Mask {
        /// Required runtime storage dtype.
        dtype: InterventionDtype,
        /// Complete selected extents.
        shape: Vec<u64>,
        /// Row-major keep bits.
        keep: Vec<bool>,
    },
    /// Whole-value replacement or patching according to the explicit axis slices.
    Replace {
        /// Complete exact-dtype selected values.
        tensor: InterventionTensor,
    },
    /// Elementwise addition; supports typed logit bias without a second sampler.
    Add {
        /// Complete exact-dtype selected values.
        tensor: InterventionTensor,
    },
    /// Negative-infinity mask at final logits; selection must retain full vocabulary.
    MaskLogits {
        /// Required runtime storage dtype.
        dtype: InterventionDtype,
        /// Unique canonical vocabulary IDs.
        token_ids: Vec<u32>,
    },
    /// Eligibility exclusion before the architecture's existing selection algorithm.
    /// Gathered scores retain ordinary normalization and scaling semantics.
    ExcludeExperts {
        /// Unique global routed-expert IDs.
        expert_ids: Vec<u32>,
    },
    /// Suppress selected contributions without reselection or renormalization.
    /// Remaining coefficients retain their original magnitude. Expert computation
    /// is not promised to be avoided. Shared experts remain unchanged.
    ZeroExpertContribution {
        /// Unique global routed-expert IDs.
        expert_ids: Vec<u32>,
    },
    /// Per-expert bias replicated explicitly over the selected token rows.
    BiasRoutingScores {
        /// Precise scoring boundary.
        stage: RoutingScoreStage,
        /// Unique global routed-expert IDs.
        expert_ids: Vec<u32>,
        /// One finite additive bias per listed expert.
        biases: Vec<f32>,
    },
    /// Replace selected IDs before dispatch. Scores/weights are gathered from the
    /// architecture's router, preserving its normalization, scaling and learned
    /// multipliers. There is no caller coefficient override.
    ForceExperts {
        /// Exact `[selected_token_rows, top_k]` extents.
        shape: [u64; 2],
        /// Complete row-major global routed-expert IDs.
        expert_ids: Vec<u32>,
    },
}

impl InterventionAction {
    /// Revalidates complete host parameters against an exact native selected region.
    pub fn validate_activation_region(
        &self,
        dtype: InterventionDtype,
        shape: &[u64],
    ) -> Result<(), CaptureError> {
        require(
            self.dtype() == Some(dtype),
            "intervention runtime dtype differs from exact declared dtype",
        )?;
        require(
            !shape.is_empty() && shape.len() <= 32 && !shape.contains(&0),
            "invalid activation region shape",
        )?;
        validate_activation_parameters(self, shape.last().and_then(|n| u32::try_from(*n).ok()))?;
        validate_payload_shape(self, shape)
    }
    /// Exact operation category.
    pub fn kind(&self) -> InterventionKind {
        match self {
            Self::Zero { .. } => InterventionKind::Zero,
            Self::Scale { .. } => InterventionKind::Scale,
            Self::Mask { .. } => InterventionKind::Mask,
            Self::Replace { .. } => InterventionKind::Replace,
            Self::Add { .. } => InterventionKind::Add,
            Self::MaskLogits { .. } => InterventionKind::MaskLogits,
            Self::ExcludeExperts { .. } => InterventionKind::ExcludeExperts,
            Self::ZeroExpertContribution { .. } => InterventionKind::ZeroExpertContribution,
            Self::BiasRoutingScores { .. } => InterventionKind::BiasRoutingScores,
            Self::ForceExperts { .. } => InterventionKind::ForceExperts,
        }
    }
    /// Exact activation dtype, absent for architecture-owned routing arithmetic.
    pub fn dtype(&self) -> Option<InterventionDtype> {
        match self {
            Self::Zero { dtype }
            | Self::Scale { dtype, .. }
            | Self::Mask { dtype, .. }
            | Self::MaskLogits { dtype, .. } => Some(*dtype),
            Self::Replace { tensor } | Self::Add { tensor } => Some(tensor.values.dtype()),
            _ => None,
        }
    }
    fn payload_bytes(&self) -> Result<u64, CaptureError> {
        match self {
            Self::Replace { tensor } | Self::Add { tensor } => {
                add(mul(tensor.shape.len() as u64, 8)?, tensor.values.bytes()?)
            }
            Self::Mask { shape, keep, .. } => add(mul(shape.len() as u64, 8)?, keep.len() as u64),
            Self::MaskLogits { token_ids, .. } => mul(token_ids.len() as u64, 4),
            Self::ExcludeExperts { expert_ids }
            | Self::ZeroExpertContribution { expert_ids }
            | Self::ForceExperts { expert_ids, .. } => mul(expert_ids.len() as u64, 4),
            Self::BiasRoutingScores {
                expert_ids, biases, ..
            } => mul(add(expert_ids.len() as u64, biases.len() as u64)?, 4),
            _ => Ok(0),
        }
    }
}

/// Bounded before/after evidence. Both sides are charged to the shared capture
/// ledger. Routing evidence includes selected IDs and coefficients, not a second
/// unmodified model forward pass.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InterventionEvidence {
    /// Only the attributed application outcome is recorded.
    None,
    /// Bounded row-major values per side/route field.
    Preview {
        /// Positive maximum number of elements per field.
        max_elements: u64,
    },
    /// Finite statistics of activation values on each side.
    Summary,
}

/// One exact operation. List order is deterministic for activation composition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterventionOperation {
    /// Unique, nonempty caller identity, at most 128 UTF-8 bytes.
    pub id: String,
    /// Exact path from loaded-session intervention discovery.
    pub target: String,
    /// Prediction schedule; does not select prompt tensor rows.
    pub schedule: CaptureSchedule,
    /// Explicit within-tensor semantic slices. Unmentioned axes remain whole.
    pub slices: Vec<CaptureSlice>,
    /// Typed operation and complete bounded payload.
    pub action: InterventionAction,
    /// Requested bounded evidence, charged in addition to ordinary captures.
    pub evidence: InterventionEvidence,
}

/// Serializable request. Only admission produces an executable immutable plan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterventionPlan {
    /// Intervention wire version.
    pub schema_version: u32,
    /// Activation operations compose in list order; routing overlaps are rejected.
    pub operations: Vec<InterventionOperation>,
}

impl InterventionPlan {
    /// Empty plan preserves the ordinary path and sampling/RNG behavior.
    pub fn none() -> Self {
        Self {
            schema_version: INTERVENTION_SCHEMA_VERSION,
            operations: vec![],
        }
    }

    /// Validates all known constraints without creating or evaluating native values.
    /// The facade supplies its loaded session identity, never an application catalog.
    pub fn admit(
        self,
        discovery: &InterventionDiscovery,
        request: CaptureRequestShape,
        session_id: &str,
    ) -> Result<AdmittedInterventionPlan, CaptureError> {
        require(
            self.schema_version == INTERVENTION_SCHEMA_VERSION
                && discovery.schema_version == INTERVENTION_SCHEMA_VERSION,
            "unsupported intervention schema",
        )?;
        require(
            !session_id.is_empty() && !discovery.artifact_identity.is_empty(),
            "missing intervention session/source identity",
        )?;
        require(
            discovery
                .session_identity
                .as_ref()
                .is_some_and(|id| !id.is_empty()),
            "intervention admission requires a realized backend session",
        )?;
        require(
            request.batch > 0 && request.prompt_tokens > 0 && request.max_predictions > 0,
            "empty intervention request geometry",
        )?;
        add(request.prompt_tokens, request.max_predictions)?;
        mul(request.batch, request.prompt_tokens)?;
        require(
            self.operations.len() <= MAX_INTERVENTION_OPERATIONS,
            "too many intervention operations",
        )?;
        let mut payload_bytes = 0;
        let mut ids = BTreeSet::new();
        let mut points = Vec::new();
        for operation in &self.operations {
            require(
                !operation.id.is_empty() && operation.id.len() <= 128 && ids.insert(&operation.id),
                "intervention IDs must be unique and contain 1..=128 bytes",
            )?;
            require(
                operation.target.len() <= 1024,
                "intervention target exceeds metadata bound",
            )?;
            payload_bytes = add(payload_bytes, operation.action.payload_bytes()?)?;
            require(
                payload_bytes <= MAX_INTERVENTION_PAYLOAD_BYTES,
                "intervention payload exceeds hard bound",
            )?;
            let mut matching = discovery
                .points
                .iter()
                .filter(|p| p.path == operation.target);
            let point = matching
                .next()
                .ok_or_else(|| CaptureError::MissingPath(operation.target.clone()))?;
            require(
                matching.next().is_none(),
                "ambiguous intervention target declaration",
            )?;
            validate_operation(operation, point, request)?;
            points.push(point.clone());
        }
        for (index, (operation, point)) in self.operations.iter().zip(&points).enumerate() {
            if point.routing.is_none() {
                continue;
            }
            for previous in &self.operations[..index] {
                if previous.target != operation.target {
                    continue;
                }
                for phase in [CapturePhase::Prefill, CapturePhase::Decode] {
                    let a = previous
                        .schedule
                        .count_and_last(phase, request.max_predictions)?;
                    let b = operation
                        .schedule
                        .count_and_last(phase, request.max_predictions)?;
                    if let (Some((ac, al)), Some((bc, bl))) = (a, b) {
                        let af = al - (ac - 1) * previous.schedule.every;
                        let bf = bl - (bc - 1) * operation.schedule.every;
                        require(
                            al < bf || bl < af,
                            "routing operations have potentially overlapping schedules",
                        )?;
                    }
                }
            }
        }
        let mut encoded = PlanCounter(0);
        serde_json::to_writer(&mut encoded, &self)
            .map_err(|_| CaptureError::Invalid("intervention plan exceeds encoded bound".into()))?;
        let digest = Sha256::digest(
            serde_json::to_vec(&(
                &self,
                &points,
                request,
                &discovery.artifact_identity,
                &discovery.session_identity,
                session_id,
            ))
            .map_err(|e| CaptureError::Invalid(e.to_string()))?,
        );
        let identity = format!(
            "intervention-v1-{}",
            digest
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        );
        Ok(AdmittedInterventionPlan {
            plan: self,
            points,
            request,
            identity,
            artifact_identity: discovery.artifact_identity.clone(),
            session_id: session_id.into(),
        })
    }
}

struct PlanCounter(u64);
impl std::io::Write for PlanCounter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0 = self
            .0
            .checked_add(bytes.len() as u64)
            .filter(|n| *n <= MAX_INTERVENTION_PLAN_BYTES)
            .ok_or_else(|| std::io::Error::other("intervention plan bound"))?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Immutable admission proof. It cannot be deserialized or edited during a run.
#[derive(Debug, Clone)]
pub struct AdmittedInterventionPlan {
    plan: InterventionPlan,
    points: Vec<InterventionPoint>,
    request: CaptureRequestShape,
    identity: String,
    artifact_identity: String,
    session_id: String,
}

impl AdmittedInterventionPlan {
    /// Stable plan/semantics/request/source/session digest.
    pub fn identity(&self) -> &str {
        &self.identity
    }
    /// Exact prepared-source identity at admission.
    pub fn artifact_identity(&self) -> &str {
        &self.artifact_identity
    }
    /// Exact loaded facade session at admission.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }
    /// Immutable operations in composition order.
    pub fn plan(&self) -> &InterventionPlan {
        &self.plan
    }
    /// Selected declarations in operation order.
    pub fn points(&self) -> &[InterventionPoint] {
        &self.points
    }
    /// Admitted prompt and prediction geometry.
    pub fn request(&self) -> CaptureRequestShape {
        self.request
    }
    /// Whether the ordinary path can omit intervention machinery.
    pub fn is_empty(&self) -> bool {
        self.plan.operations.is_empty()
    }
    /// Validates request-dependent shape, exact dtype and payload geometry before
    /// applying one scheduled operation. This does not assert prior cache rollback.
    pub fn validate_actual(
        &self,
        index: usize,
        phase: CapturePhase,
        prediction: u64,
        shape: &[u64],
        dtype: Option<InterventionDtype>,
    ) -> Result<ResolvedCaptureSlice, CaptureError> {
        let operation = self
            .plan
            .operations
            .get(index)
            .ok_or_else(|| CaptureError::Invalid("invalid admitted operation index".into()))?;
        require(
            prediction < self.request.max_predictions
                && operation.schedule.includes(phase, prediction),
            "inactive intervention operation",
        )?;
        require(
            operation.action.dtype() == dtype,
            "intervention runtime dtype differs from exact declared dtype",
        )?;
        let point = &self.points[index];
        let geometry = point.observation_geometry();
        self.request
            .validate_actual(&geometry, phase, prediction, shape)?;
        let slice = operation.resolve_slice(point, shape)?;
        validate_payload_shape(&operation.action, &slice.shape)?;
        Ok(slice)
    }
}

impl InterventionOperation {
    /// Resolves semantic selection with the same half-open/stride rules as capture.
    pub fn resolve_slice(
        &self,
        point: &InterventionPoint,
        shape: &[u64],
    ) -> Result<ResolvedCaptureSlice, CaptureError> {
        resolve_slice(
            &point.observation_geometry(),
            &CaptureSelection {
                id: self.id.clone(),
                path: self.target.clone(),
                schedule: self.schedule.clone(),
                slices: self.slices.clone(),
                transform: CaptureTransform::FullTensor,
            },
            shape,
        )
    }
}

fn validate_operation(
    operation: &InterventionOperation,
    point: &InterventionPoint,
    request: CaptureRequestShape,
) -> Result<(), CaptureError> {
    require(
        point.node_id.len() <= 1024 && !point.node_id.is_empty(),
        "invalid intervention node identity",
    )?;
    require(
        !point.axes.is_empty() && point.axes.len() <= 32,
        "intervention requires declared bounded tensor rank",
    )?;
    require(
        point.operations.contains(&operation.action.kind()),
        "operation unsupported at intervention point",
    )?;
    require(
        operation.schedule.every != 0
            && operation
                .schedule
                .end_prediction
                .is_none_or(|end| end > operation.schedule.first_prediction),
        "invalid intervention schedule",
    )?;
    let mut axes = BTreeSet::new();
    for axis in &point.axes {
        require(
            axes.insert(&axis.name),
            "ambiguous intervention axis declaration",
        )?;
    }
    let mut selected = BTreeSet::new();
    for slice in &operation.slices {
        require(
            selected.insert(&slice.axis) && axes.contains(&slice.axis),
            "duplicate or unknown intervention axis",
        )?;
        require(
            slice.stride > 0 && slice.start < slice.end,
            "intervention slices must be nonempty with positive stride",
        )?;
        if point.routing.is_some() {
            require(
                slice.axis == "token",
                "routing permits token-row selection only",
            )?;
        }
        if matches!(operation.action, InterventionAction::MaskLogits { .. }) {
            require(
                slice.axis != "vocabulary",
                "logit masks require the complete vocabulary axis",
            )?;
        }
    }
    if let Some(dtype) = operation.action.dtype() {
        require(
            point.routing.is_none() && point.dtypes.contains(&dtype),
            "activation dtype unsupported at target",
        )?;
    } else {
        require(
            point.routing.is_some(),
            "routing action requires a pre-dispatch routing point",
        )?;
    }
    match operation.evidence {
        InterventionEvidence::Preview { max_elements } => require(
            max_elements > 0 && max_elements <= 4096,
            "intervention preview requires 1..=4096 elements",
        )?,
        InterventionEvidence::Summary => require(
            point.routing.is_none(),
            "routing evidence requires an ID/coefficient preview",
        )?,
        InterventionEvidence::None => (),
    }
    validate_action(&operation.action, point)?;
    for (phase, status) in [
        (CapturePhase::Prefill, &point.prefill),
        (CapturePhase::Decode, &point.decode),
    ] {
        if let Some((_, last)) = operation
            .schedule
            .count_and_last(phase, request.max_predictions)?
        {
            if *status != ObservationSupportStatus::Supported {
                return Err(CaptureError::Unsupported(format!(
                    "intervention {} {phase:?}: {status:?}",
                    point.path
                )));
            }
            let geometry = point.observation_geometry();
            if let Some(shape) = request.resolve(&geometry, phase, last)? {
                let slice = operation.resolve_slice(point, &shape)?;
                validate_payload_shape(&operation.action, &slice.shape)?;
            }
        }
    }
    Ok(())
}

fn validate_activation_parameters(
    action: &InterventionAction,
    vocabulary: Option<u32>,
) -> Result<(), CaptureError> {
    use InterventionAction as A;
    match action {
        A::Scale { dtype, factor } => {
            let maximum = match dtype {
                InterventionDtype::Float16 => 65504.0,
                InterventionDtype::Bfloat16 => f32::from_bits(0x7f7f0000),
                InterventionDtype::Float32 => f32::MAX,
            };
            require(
                factor.is_finite() && factor.abs() <= maximum,
                "scale must be finite and representable in the target dtype",
            )?;
        }
        A::Replace { tensor } | A::Add { tensor } => {
            require(
                !tensor.shape.is_empty() && tensor.shape.len() <= 32 && !tensor.shape.contains(&0),
                "invalid replacement tensor rank/extents",
            )?;
            require(
                elements(&tensor.shape)? == tensor.values.len() as u64,
                "replacement tensor is incomplete or has malformed shape",
            )?;
            tensor.values.validate()?;
        }
        A::Mask { shape, keep, .. } => {
            require(
                !shape.is_empty() && shape.len() <= 32 && !shape.contains(&0),
                "invalid mask rank/extents",
            )?;
            require(
                elements(shape)? == keep.len() as u64,
                "mask shape differs from complete keep values",
            )?;
        }
        A::MaskLogits { token_ids, .. } => {
            let vocabulary = vocabulary.ok_or_else(|| {
                CaptureError::Unsupported("logit mask requires known vocabulary extent".into())
            })?;
            validate_ids(token_ids, vocabulary)?;
            require(
                token_ids.len() < vocabulary as usize,
                "logit mask cannot exclude the whole vocabulary",
            )?;
        }
        A::Zero { .. } => (),
        _ => {
            return Err(CaptureError::Invalid(
                "routing action cannot replace an activation".into(),
            ))
        }
    }
    Ok(())
}

fn validate_action(
    action: &InterventionAction,
    point: &InterventionPoint,
) -> Result<(), CaptureError> {
    use InterventionAction as A;
    if action.dtype().is_some() {
        let vocabulary = if matches!(action, A::MaskLogits { .. }) {
            require(
                point.stage == InterventionStage::LogitsBeforeSampling,
                "logit masking requires final logits",
            )?;
            point
                .axes
                .iter()
                .find(|axis| axis.name == "vocabulary")
                .and_then(|axis| match axis.dimension {
                    crate::SymbolicDimension::Known(n) => u32::try_from(n).ok(),
                    _ => None,
                })
        } else {
            None
        };
        return validate_activation_parameters(action, vocabulary);
    }

    match action {
        A::ExcludeExperts { expert_ids }
        | A::ZeroExpertContribution { expert_ids }
        | A::BiasRoutingScores { expert_ids, .. }
        | A::ForceExperts { expert_ids, .. } => {
            let policy = point
                .routing
                .as_ref()
                .ok_or_else(|| CaptureError::Invalid("missing routing policy".into()))?;
            require(
                point.stage == InterventionStage::RoutingBeforeDispatch,
                "routing control requires pre-dispatch target",
            )?;
            require(
                policy.expert_count > 0
                    && policy.top_k > 0
                    && policy.top_k <= policy.expert_count
                    && policy.groups > 0
                    && policy.expert_count % policy.groups == 0
                    && policy.selected_groups > 0
                    && policy.selected_groups <= policy.groups
                    && policy.top_k
                        <= policy.selected_groups * (policy.expert_count / policy.groups)
                    && policy.normalization_epsilon.is_finite()
                    && policy.normalization_epsilon >= 0.0
                    && policy.coefficient_scale.is_finite()
                    && policy.coefficient_scale > 0.0,
                "invalid routing declaration",
            )?;
            if let A::ForceExperts { shape, .. } = action {
                require(
                    shape[0] > 0
                        && shape[1] == policy.top_k as u64
                        && elements(shape)? == expert_ids.len() as u64,
                    "forced route shape must be [selected_token_rows, top_k]",
                )?;
                for row in expert_ids.chunks(policy.top_k as usize) {
                    validate_ids(row, policy.expert_count)?;
                    let width = policy.expert_count / policy.groups;
                    let groups: BTreeSet<_> = row.iter().map(|id| id / width).collect();
                    require(
                        groups.len() <= policy.selected_groups as usize,
                        "forced IDs exceed architecture's selected-group count",
                    )?;
                }
            } else {
                validate_ids(expert_ids, policy.expert_count)?;
            }
            if let A::ExcludeExperts { .. } = action {
                require(
                    policy.expert_count as usize - expert_ids.len() >= policy.top_k as usize,
                    "excluded experts make top-k infeasible",
                )?;
                // Any group the unchanged algorithm can choose must have enough
                // eligible members. This conservative condition prevents -inf IDs
                // from silently entering a dispatch after group selection.
                let width = policy.expert_count / policy.groups;
                let mut available = vec![width; policy.groups as usize];
                for id in expert_ids {
                    available[(id / width) as usize] -= 1;
                }
                available.sort_unstable();
                require(
                    available
                        .iter()
                        .take(policy.selected_groups as usize)
                        .sum::<u32>()
                        >= policy.top_k,
                    "exclusion cannot guarantee top-k within selected groups",
                )?;
            }
            if let A::BiasRoutingScores { stage, biases, .. } = action {
                require(
                    point.score_stages.contains(stage),
                    "router bias stage unsupported",
                )?;
                require(
                    biases.len() == expert_ids.len() && biases.iter().all(|b| b.is_finite()),
                    "router bias must have one finite value per expert",
                )?;
            }
        }
        _ => unreachable!("activation actions validated above"),
    }
    Ok(())
}

fn validate_payload_shape(
    action: &InterventionAction,
    selected: &[u64],
) -> Result<(), CaptureError> {
    let expected: Option<&[u64]> = match action {
        InterventionAction::Mask { shape, .. } => Some(shape),
        InterventionAction::Replace { tensor } | InterventionAction::Add { tensor } => {
            Some(&tensor.shape)
        }
        InterventionAction::ForceExperts { shape, .. } => Some(shape),
        _ => None,
    };
    require(!selected.contains(&0), "empty intervention selection")?;
    require(
        expected.is_none_or(|expected| expected == selected),
        "intervention payload must exactly match selected shape; broadcasting is prohibited",
    )
}

fn validate_ids(ids: &[u32], count: u32) -> Result<(), CaptureError> {
    require(!ids.is_empty(), "intervention ID list must be nonempty")?;
    let mut unique = BTreeSet::new();
    require(
        ids.iter().all(|id| *id < count && unique.insert(*id)),
        "intervention IDs are duplicate or out of range",
    )
}

fn require(condition: bool, message: &str) -> Result<(), CaptureError> {
    if condition {
        Ok(())
    } else {
        Err(CaptureError::Invalid(message.into()))
    }
}

/// Actual per-operation application outcome, independent of generation success.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InterventionOutcome {
    /// The phase/prediction was not selected by this operation's schedule.
    Inactive,
    /// The replacement/effective routes were passed downstream.
    Applied,
    /// The scheduled point was unexpectedly absent from the forward pass.
    Missing,
    /// This operation failed; earlier operations or cache work may already have run.
    Failed {
        /// Bounded human diagnostic.
        message: String,
    },
}

/// Bounded attributed event. Large plan payloads are never repeated in events.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterventionRecord {
    /// Intervention wire version.
    pub schema_version: u32,
    /// Immutable admitted plan identity.
    pub plan_id: String,
    /// Caller operation identity.
    pub operation_id: String,
    /// Exact execution target.
    pub target: String,
    /// Existing logical architecture node identity.
    pub node_id: String,
    /// Actual forward phase.
    pub phase: CapturePhase,
    /// Capture-compatible prediction index.
    pub prediction_index: u64,
    /// Actual application status.
    pub outcome: InterventionOutcome,
    /// Optional bounded pre/post activation or route-field captures. Position
    /// identifies original/effective values at this exact operation boundary.
    pub evidence: Vec<CaptureRecord>,
    /// Diagnostic reservation in the shared capture ledger; evidence owns its charges.
    pub charged: CaptureUsage,
}

/// Native activation primitives. Portable runtime code owns operation dispatch,
/// payload validation, region selection/update order, and composition.
pub trait InterventionBackend: CaptureBackend {
    /// Exact dtype without evaluating the native value.
    fn intervention_dtype(&self, tensor: &Self::Tensor) -> Result<InterventionDtype, Self::Error>;
    /// Checks backend indexing/shape limits without allocating or evaluating tensors.
    fn validate_intervention_geometry(
        &self,
        source: &[u64],
        slice: &ResolvedCaptureSlice,
    ) -> Result<(), CaptureError>;
    /// Selects the exact positive-stride region, preserving singleton dimensions.
    fn select_region(
        &mut self,
        tensor: &Self::Tensor,
        slice: &ResolvedCaptureSlice,
    ) -> Result<Self::Tensor, Self::Error>;
    /// Returns a value with only the selected region replaced; preserves the source.
    fn update_region(
        &mut self,
        tensor: &Self::Tensor,
        slice: &ResolvedCaptureSlice,
        replacement: &Self::Tensor,
    ) -> Result<Self::Tensor, Self::Error>;
    /// Native zeros with the supplied exact shape/dtype.
    fn zeros(
        &mut self,
        shape: &[u64],
        dtype: InterventionDtype,
    ) -> Result<Self::Tensor, Self::Error>;
    /// Multiplies by a scalar explicitly rounded to the value's dtype.
    fn scale(&mut self, value: &Self::Tensor, factor: f32) -> Result<Self::Tensor, Self::Error>;
    /// Keeps true elements and fills false elements with an exact native-dtype scalar.
    fn fill_masked(
        &mut self,
        value: &Self::Tensor,
        keep: &[bool],
        fill: f32,
    ) -> Result<Self::Tensor, Self::Error>;
    /// Uploads a complete tensor with exact dtype, including f16/bf16 bit patterns.
    fn realize_tensor(&mut self, tensor: &InterventionTensor) -> Result<Self::Tensor, Self::Error>;
    /// Adds tensors of identical shape and dtype, without broadcasting.
    fn add(
        &mut self,
        left: &Self::Tensor,
        right: &Self::Tensor,
    ) -> Result<Self::Tensor, Self::Error>;
    /// Fills explicit last-axis columns over all rows of this selected region.
    fn fill_columns(
        &mut self,
        value: &Self::Tensor,
        ids: &[u32],
        fill: f32,
    ) -> Result<Self::Tensor, Self::Error>;
}

/// Cold backend facts used by both admission and execution reservation. Unknown
/// estimates must return an error; they must never be represented as zero cost.
pub trait InterventionEstimator: Send + Sync {
    /// Additional native indexing constraints, beyond portable slice validation.
    fn validate_geometry(
        &self,
        source: &[u64],
        slice: &ResolvedCaptureSlice,
    ) -> Result<(), CaptureError>;
    /// Existing evidence-transform cost, excluding original-decision work.
    fn capture_usage(
        &self,
        source: &[u64],
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError>;
    /// Extra original-decision work only: logical native temporaries and additional
    /// host materialization. Diagnostic/evidence encoding is charged by the runtime.
    fn original_route_usage(
        &self,
        policy: &InterventionRoutingPolicy,
        rows: u64,
    ) -> Result<CaptureUsage, CaptureError>;
}

#[cfg(test)]
mod tests;
