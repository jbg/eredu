//! Request-specific, descriptive memory planning. These estimates never authorize
//! allocations or change execution. Native scratch, loading and allocator costs
//! remain explicit inputs; an unknown cost cannot establish a likely fit.

use std::num::NonZeroU8;

use eredu_core::{
    cache::{StateTensorDimension, StateTensorPresence},
    estimate_runtime_state, AvailableMemory, CapabilityError, InputTokenCount, ObservationKind,
    Observed, PhysicalMemorySemantics, StateMemoryLayout, StaticMemoryReport,
};
use serde::{Deserialize, Serialize};

/// Payload size, estimated interval, or known contribution with an unknown tail.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryBytes {
    /// Calculated or estimated lower end, not an allocation guarantee.
    pub lower_bytes: u64,
    /// Upper end; absent when coverage is unavailable.
    pub upper_bytes: Option<u64>,
    /// Whether this is calculated payload or an estimate.
    pub kind: ObservationKind,
    /// Source, assumption, or missing coverage.
    pub detail: String,
}

impl MemoryBytes {
    /// Calculated payload bytes, excluding undeclared overhead.
    pub fn exact(bytes: u64) -> Self {
        Self {
            lower_bytes: bytes,
            upper_bytes: Some(bytes),
            kind: ObservationKind::Exact,
            detail: "calculated payload".into(),
        }
    }

    /// An explicit estimated interval. Invalid intervals are rejected by estimation.
    pub fn estimated(lower: u64, upper: u64, reason: impl Into<String>) -> Self {
        Self {
            lower_bytes: lower,
            upper_bytes: Some(upper),
            kind: ObservationKind::Estimated,
            detail: reason.into(),
        }
    }

    /// Unavailable coverage; zero is only the known lower contribution.
    pub fn unknown(reason: impl Into<String>) -> Self {
        Self {
            lower_bytes: 0,
            upper_bytes: None,
            kind: ObservationKind::Estimated,
            detail: reason.into(),
        }
    }

    pub(crate) fn validate(&self) -> Result<(), CapabilityError> {
        if self
            .upper_bytes
            .is_some_and(|upper| upper < self.lower_bytes)
        {
            return Err(invalid("memory interval", "upper end is below lower end"));
        }
        Ok(())
    }

    pub(crate) fn add(&self, other: &Self) -> Result<Self, CapabilityError> {
        self.validate()?;
        other.validate()?;
        Ok(Self {
            lower_bytes: add(self.lower_bytes, other.lower_bytes)?,
            upper_bytes: match (self.upper_bytes, other.upper_bytes) {
                (Some(a), Some(b)) => Some(add(a, b)?),
                _ => None,
            },
            kind: if self.kind == ObservationKind::Exact && other.kind == ObservationKind::Exact {
                ObservationKind::Exact
            } else {
                ObservationKind::Estimated
            },
            detail: "sum of overlapping contributions; see phase breakdown".into(),
        })
    }

    pub(crate) fn maximum(&self, other: &Self) -> Self {
        Self {
            lower_bytes: self.lower_bytes.max(other.lower_bytes),
            upper_bytes: self
                .upper_bytes
                .zip(other.upper_bytes)
                .map(|(a, b)| a.max(b)),
            kind: ObservationKind::Estimated,
            detail: "maximum of phase peaks, not their sum".into(),
        }
    }

    pub(crate) fn additional(&self, already_resident: u64) -> Self {
        Self {
            lower_bytes: self.lower_bytes.saturating_sub(already_resident),
            upper_bytes: self.upper_bytes.map(|n| n.saturating_sub(already_resident)),
            kind: self.kind,
            detail: "additional to declared already-resident bytes".into(),
        }
    }
}

/// Disjoint physical capacity pool. Shared backing belongs to one pool once.
/// JSON uses a `kind` tag and a `device` identifier for independent devices.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "device", rename_all = "snake_case")]
pub enum MemoryDomain {
    /// Host and accelerators share one physical pool.
    Unified,
    /// Host capacity, separate from each device.
    Host,
    /// Independent device capacity, identified by backend device identifier.
    Device(String),
}

/// Compatibility input for serialized aggregate decoder forecasts.
///
/// Current preparation supplies [`crate::execution_topology::TextExecutionTopology`]
/// from ordinary module construction. Existing aggregate-only records preserve
/// their declared envelope through the same resource lifetime evaluator; they
/// do not supply enough facts to reconstruct a module topology.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceGeometry {
    /// Local residual width.
    pub hidden_size: u64,
    /// Largest simultaneously evaluated feed-forward intermediate width.
    pub intermediate_size: u64,
    /// Total local query projection width.
    pub query_width: u64,
    /// Total local width of each key/value projection.
    pub key_value_width: u64,
    /// Local query head count.
    pub query_heads: u64,
    /// Local vocabulary projection width, including any gathered result.
    pub vocabulary_size: u64,
    /// Additional gated causal depthwise-convolution workspace, when scheduled.
    /// Absent in legacy dense-attention records.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gated_convolution: Option<GatedConvolutionWorkspace>,
    /// Explicit input-score attention and its selected native workspace mechanism.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_score_attention: Option<InputScoreAttentionWorkspace>,
    /// Full float32 parameter payload allowed for mixed-width promotion during
    /// execution. Persistent parameters remain separately charged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mixed_precision_parameter_bytes: Option<u64>,
}

/// Geometry for a three-way input projection, input/output gates, and causal
/// depthwise convolution. Persistent history is accounted in the state layout.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GatedConvolutionWorkspace {
    /// Local convolution channel width.
    pub channels: u64,
    /// Positive causal kernel width.
    pub kernel_size: u64,
    /// Number of scheduled convolution layers sharing this geometry.
    pub layers: u64,
}

/// Architecture-declared explicit input-score layers. Missing native facts keep
/// their workspace unbounded rather than substituting a fused-attention model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputScoreAttentionWorkspace {
    /// Number of scheduled layers using the explicit score arithmetic.
    pub layers: u64,
    /// Selected backend facts; absence preserves an unknown upper end.
    pub mechanism: Option<InputScoreAttentionMechanism>,
}

/// Side-effect-free native facts for tiled explicit input-score attention.
/// Copies and score bytes are conservative live-buffer allowances, not payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputScoreAttentionMechanism {
    /// Query-by-key elements per head before query tiling.
    pub score_tile_elements: u64,
    /// Maximum query rows in one tile.
    pub max_query_rows: u64,
    /// Fallback expanded K/V and contiguous projection copies per query tile,
    /// used when `full_key_tiles` is absent or does not cover the key length.
    pub key_value_copies: u64,
    /// Working bytes per score element, including precision conversions.
    pub score_bytes: u64,
    /// Optional full-key layout reuse and bounded tile-graph retention. Missing
    /// facts retain the legacy per-tile copy allowance for all key lengths.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub full_key_tiles: Option<FullKeyAttentionTiles>,
}

/// Native retention contract for tiles that each attend to the entire key row.
/// Beyond this contract's key limit, the mechanism's legacy allowances apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FullKeyAttentionTiles {
    /// Largest key row covered by shared preparation and bounded evaluation.
    pub max_key_positions: u64,
    /// Expanded K/V and native contiguous layouts, in query-width payloads.
    /// This is charged once per invocation rather than once per query tile.
    pub shared_key_value_copies: u64,
    /// Largest number of unevaluated query tiles retained simultaneously.
    pub max_live_query_tiles: u64,
    /// Whole-query output payloads retained, including final concatenation.
    pub retained_output_copies: u64,
    /// When the query tile count exceeds `max_live_query_tiles`, each batch,
    /// including the last, completes synchronously and detaches evaluated input
    /// dependency graphs. This releases unowned intermediates, not owner-held
    /// residuals, state views, completed outputs, parameters or allocator cache.
    /// Absent legacy facts do not establish an upstream release boundary.
    #[serde(default)]
    pub evaluates_input_dependencies: bool,
}

/// Selected attention implementation's workspace behavior.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AttentionWorkspace {
    /// Score and probability matrices materialized in float32.
    Materialized,
    /// Conservative score-matrix upper estimate when a fused kernel's scratch
    /// has not been calibrated. Its lower end is zero, so this fallback alone
    /// cannot establish a likely shortfall.
    ScoreMatrixUpperBound,
    /// Fused attention with supplied peak scratch for the request's geometry.
    Fused {
        /// Backend scratch estimate, excluding Q/K/V payloads.
        scratch: MemoryBytes,
    },
    /// Selection or kernel scratch is unavailable.
    Unknown,
}

/// Whether cache update overlaps an old and replacement state allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheUpdateWorkspace {
    /// Updates existing storage; no additional whole-cache copy.
    InPlace,
    /// Conservatively retain a full extra persistent-state payload during update.
    CopyState,
    /// Backend cache update behavior is not known.
    Unknown,
}

/// Actual output-projection behavior selected for prefill.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogitsWorkspace {
    /// Only the final position of each invocation is projected. Nonfinal chunks
    /// still retain one row to satisfy the existing output/completion contract.
    FinalPosition,
    /// Every position in each chunk is projected.
    EveryPosition,
}

/// Backend estimate for simultaneous residual/projection/MLP intermediates.
/// Lazy execution may retain several layers' graphs at once. This is a coarse
/// planning interval, not a claim about individual tensor lifetimes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceOverlap {
    /// Upper number of equivalent one-layer linear workspaces, at least one.
    /// Absent when backend overlap is not known. The lower end is one copy.
    pub upper_live_copies: Option<u64>,
    /// Selected mechanism fact or empirical calibration assumption.
    pub detail: String,
}

impl WorkspaceOverlap {
    /// Explicit single-layer execution assumption, suitable when completion or
    /// an established backend memory schedule bounds intermediate retention.
    pub fn single_layer() -> Self {
        Self {
            upper_live_copies: Some(1),
            detail: "one live layer workspace".into(),
        }
    }

    /// Uncalibrated backend graph retention stays unavailable.
    pub fn unknown() -> Self {
        Self {
            upper_live_copies: None,
            detail: "backend intermediate retention is unavailable".into(),
        }
    }
}

/// One simultaneous rank or replica, described using its actual local geometry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionMemoryPlan {
    /// Exact architecture state layout, reused by the core state estimator.
    pub state_layout: StateMemoryLayout,
    /// Legacy aggregate geometry for archived or manually constructed requests.
    /// Current preparation supplies `execution_topology` instead.
    pub workspace: Option<WorkspaceGeometry>,
    /// Ordinary selected module topology; authoritative over legacy aggregate
    /// geometry even when its mechanism coverage is incomplete.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_topology: Option<crate::execution_topology::TextExecutionTopology>,
    /// Native explicit-score attention calibration, independent of module topology.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_score_attention_mechanism: Option<InputScoreAttentionMechanism>,
    /// Selected attention implementation.
    pub attention: AttentionWorkspace,
    /// Cache update overlap.
    pub cache_update: CacheUpdateWorkspace,
    /// Selected vocabulary projection behavior.
    pub logits: LogitsWorkspace,
    /// Simultaneous linear intermediates, independently of attention scratch,
    /// vocabulary projection and persistent/cache-copy costs.
    pub workspace_overlap: WorkspaceOverlap,
}

/// Independent application and observed-availability comparisons.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryBudget {
    /// Limit for total modeled memory in this physical pool.
    pub application_limit_bytes: Option<u64>,
    /// Point-in-time free capacity; compared with additional memory only.
    pub available_bytes: Option<u64>,
    /// Explicit reserve for other work and uncertainty, added to both comparisons.
    pub reserve_bytes: u64,
}

impl MemoryBudget {
    /// Reuses a host/unified availability observation without substituting total
    /// installed capacity when a free-capacity observation is unavailable.
    /// Device domains instead need the selected device's own availability fact.
    pub fn from_available_memory(
        available: &AvailableMemory,
        application_limit_bytes: Option<u64>,
        reserve_bytes: u64,
    ) -> Self {
        let available_bytes = match available.available_memory_bytes {
            Observed::Available { value, .. } => Some(value),
            _ => None,
        };
        Self {
            application_limit_bytes,
            available_bytes,
            reserve_bytes,
        }
    }
}

/// Descriptive costs within one physical pool. Independent domains are never summed
/// into a fictitious capacity; simultaneous executions within a domain are summed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DomainMemoryPlan {
    /// Physical capacity pool.
    pub domain: MemoryDomain,
    /// Live parameter payloads, including quantization scales/biases and replicas.
    pub resident_parameters: MemoryBytes,
    /// Already-allocated portion of this domain's modeled total, counted once
    /// per backing allocation. Deducted only for available-capacity comparisons;
    /// application limits still compare against the total peak plus reserve.
    /// On unified memory use host + device - known shared backing, or the
    /// conservative `max(host, device)` when overlap is unknown. The lower end
    /// returned by [`static_parameter_placement`] supplies that parameter baseline.
    /// Do not substitute process RSS, global allocator activity, planned disk
    /// bytes, or other allocations absent from this plan. Use zero before loading.
    pub already_resident_bytes: u64,
    /// Additional retained inputs and instrumentation in this physical pool,
    /// excluding embeddings modeled by state. Capture forecasts include bounded
    /// native transforms, host records and immutable plans here.
    pub retained_input: MemoryBytes,
    /// Concurrent materialization or transfer staging during generation.
    pub staging: MemoryBytes,
    /// Graph, allocator and kernel overhead not otherwise modeled.
    pub backend_overhead: MemoryBytes,
    /// Loading/conversion peak, including live source and destination overlap,
    /// excluding the separately supplied backend overhead (added by the estimator).
    /// Use zero for an already-loaded model; unknown prevents whole-lifecycle fit claims.
    pub loading_peak: MemoryBytes,
    /// Concurrent rank-local or replica executions; empty for a host-only domain.
    pub executions: Vec<ExecutionMemoryPlan>,
    /// Application and availability limits for this domain.
    pub budget: MemoryBudget,
}

/// One request and selected execution configuration, usable before native loading.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationMemoryRequest {
    /// Scoped retention subledger. Its payload is already covered by resident
    /// parameters and the pending conversion workspace; never add it to totals.
    /// Requests without this field use aggregate conversion accounting.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameter_conversion_retention:
        Option<crate::memory_forecast::ConversionRetentionMemoryPlan>,
    /// Validated input geometry, including media workspace when known.
    pub input: InputTokenCount,
    /// Finite output allowance, or no lifetime bound.
    pub max_output_tokens: Option<u64>,
    /// Explicit planning horizon used when output allowance is unspecified.
    pub forecast_output_tokens: u64,
    /// Logical batch size for every supplied execution.
    pub batch_size: u64,
    /// Maximum positions per prefill chunk; positive.
    pub prefill_chunk_tokens: u64,
    /// Activation and generic floating-state scalar width.
    pub scalar_bytes: NonZeroU8,
    /// Disjoint physical pools with selected costs and local geometries.
    pub domains: Vec<DomainMemoryPlan>,
}

/// Lifetime phase whose contributions overlap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryPhase {
    /// Already installed continuation state; no model invocation.
    ContinuationStart,
    /// Artifact loading and conversion before generation.
    Loading,
    /// Prompt execution, including growing persistent state.
    Prefill,
    /// Cached autoregressive execution at the output allowance or forecast horizon.
    Decode,
    /// Tentative assistant advancement, including rollback and copy overlap.
    SpeculativeDraft,
    /// Target block verification, possibly overlapping optimistic drafting.
    SpeculativeVerification,
    /// Accepted-prefix replay and canonical draft advancement.
    SpeculativeCommit,
}

/// Explainable simultaneous contributions at a phase peak.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhaseMemoryEstimate {
    /// Execution phase.
    pub phase: MemoryPhase,
    /// State positions at the evaluated frontier.
    pub positions: u64,
    /// Positions evaluated in one model invocation.
    pub query_positions: u64,
    /// Resident parameter payload, including quantization companions.
    pub parameters: MemoryBytes,
    /// KV, recurrent and other persistent execution state.
    pub persistent_state: MemoryBytes,
    /// Retained request inputs and prepared media embeddings.
    pub retained_input: MemoryBytes,
    /// Decoder/media workspace and cache-update copies.
    pub workspace: MemoryBytes,
    /// Transfer and materialization staging overlapping this phase.
    pub staging: MemoryBytes,
    /// Explicit backend/allocator overhead.
    pub backend_overhead: MemoryBytes,
    /// Sum of overlapping contributions; loading uses the supplied whole peak.
    pub total: MemoryBytes,
}

/// Planning conclusion, never an allocation guarantee or execution gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryFit {
    /// Known estimated upper end plus reserve is within every supplied limit.
    LikelyFit,
    /// Known lower end plus reserve exceeds at least one supplied limit.
    LikelyShortfall,
    /// Missing limits, unknown coverage, or an interval straddling a limit.
    InsufficientInformation,
}

/// Per-pool conclusions retain loading and generation as distinct questions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DomainMemoryEstimate {
    /// Physical pool being compared.
    pub domain: MemoryDomain,
    /// Evaluated phase peaks, including final and preceding full prefill chunks.
    pub phases: Vec<PhaseMemoryEstimate>,
    /// Maximum of generation phases.
    pub generation_peak: MemoryBytes,
    /// Maximum of loading and generation; never a sum of separate phase peaks.
    pub overall_peak: MemoryBytes,
    /// Generation peak minus declared already-resident memory.
    pub additional_generation_peak: MemoryBytes,
    /// Unbounded state payload growth per output position across these executions.
    /// Allocation granularity may make actual growth occur in steps.
    pub state_growth_bytes_per_position: u64,
    /// Conclusion for an already-loaded execution.
    pub generation_fit: MemoryFit,
    /// Conclusion including loading/conversion.
    pub fit: MemoryFit,
}

/// Request-specific result with explicit forecast and uncertainty semantics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationMemoryEstimate {
    /// Evaluated prompt-plus-output frontier.
    pub requested_positions: u64,
    /// True means this is a finite forecast, not a lifetime peak.
    pub is_forecast: bool,
    /// Physical-domain results.
    pub domains: Vec<DomainMemoryEstimate>,
    /// Combined result; one independent domain's shortfall is enough.
    pub fit: MemoryFit,
    /// Formula assumptions and interpretation of unmodeled costs.
    pub assumptions: Vec<String>,
    /// Unavailable coverage and interval sources, retained from the selected plan.
    pub uncertainties: Vec<String>,
}

fn invalid(field: &'static str, detail: &str) -> CapabilityError {
    CapabilityError::InvalidConfiguration {
        field,
        detail: detail.into(),
    }
}
fn add(a: u64, b: u64) -> Result<u64, CapabilityError> {
    a.checked_add(b).ok_or(CapabilityError::ArithmeticOverflow {
        operation: "generation memory sum",
    })
}
fn mul(a: u64, b: u64) -> Result<u64, CapabilityError> {
    a.checked_mul(b).ok_or(CapabilityError::ArithmeticOverflow {
        operation: "generation memory product",
    })
}
fn has_nonmonotonic_state(layout: &StateMemoryLayout) -> bool {
    layout.layer_layout().iter().any(|layer| {
        layer.fixed_state().iter().any(|tensor| {
            matches!(
                tensor.presence,
                StateTensorPresence::PrefixRemainderNonZero(_)
            ) || tensor
                .shape
                .iter()
                .any(|dimension| matches!(dimension, StateTensorDimension::PrefixTokensRem(_)))
        })
    })
}

fn workspace(
    execution: &ExecutionMemoryPlan,
    request: &GenerationMemoryRequest,
    positions: u64,
    query: u64,
    persistent: u64,
) -> Result<MemoryBytes, CapabilityError> {
    if execution.workspace_overlap.upper_live_copies == Some(0) {
        return Err(invalid(
            "workspace_overlap",
            "upper live copies must be positive",
        ));
    }
    crate::workspace_resources::workspace(execution, request, positions, query, persistent)
}

pub(crate) fn phase(
    plan: &DomainMemoryPlan,
    request: &GenerationMemoryRequest,
    phase: MemoryPhase,
    positions: u64,
    query: u64,
) -> Result<PhaseMemoryEstimate, CapabilityError> {
    let mut persistent = MemoryBytes::exact(0);
    let mut retained = plan.retained_input.clone();
    let mut transient = MemoryBytes::exact(0);
    for execution in &plan.executions {
        // Preserve authoritative media geometry while evaluating a particular
        // state frontier. Embeddings are retained only during prefill.
        let mut input = request.input;
        input.model_positions = positions;
        let state = estimate_runtime_state(
            &execution.state_layout,
            input,
            0,
            request.batch_size,
            request.scalar_bytes,
        )?;
        let state_bytes = add(state.fixed_state_bytes, state.context_state_bytes)?;
        let mut state_payload = MemoryBytes::exact(state_bytes);
        if has_nonmonotonic_state(&execution.state_layout) {
            // Endpoint sampling cannot bound a remainder-shaped tensor whose
            // interior peak may occur before the final request frontier.
            state_payload.upper_bytes = None;
            state_payload.kind = ObservationKind::Estimated;
            state_payload.detail = "interior peaks of remainder-shaped state unavailable".into();
        }
        persistent = persistent.add(&state_payload)?;
        if phase == MemoryPhase::Prefill {
            retained = retained.add(&MemoryBytes::exact(state.multimodal_embedding_bytes))?;
            let media = if request.input.media_positions > 0
                && request.input.media_execution_workspace_kind() != ObservationKind::Exact
            {
                MemoryBytes {
                    lower_bytes: state.media_execution_workspace_bytes,
                    upper_bytes: None,
                    kind: ObservationKind::Estimated,
                    detail: "media workspace coverage is not exact".into(),
                }
            } else {
                MemoryBytes::exact(state.media_execution_workspace_bytes)
            };
            transient = transient.add(&media)?;
        }
        transient = transient.add(&workspace(
            execution,
            request,
            positions,
            query,
            state_bytes,
        )?)?;
    }
    let mut parameters = plan.resident_parameters.clone();
    if !plan.executions.is_empty()
        && request
            .parameter_conversion_retention
            .as_ref()
            .is_some_and(|p| {
                p.groups.iter().any(|g| {
                    g.report
                        .usage
                        .value()
                        .is_none_or(|u| u.reserved_payload_bytes != 0)
                })
            })
    {
        // Reservations have no native allocation/binding attribution. They may
        // be unevaluated or already materialized outside this invocation. Do not
        // invent a second allocation, subtract arbitrary casts, or omit them at
        // a zero-query boundary.
        parameters.upper_bytes = None;
        parameters.kind = ObservationKind::Estimated;
        parameters.detail = "outstanding or unobserved conversion reservations have no publication/backing attribution; current parameter floor retained, overlap upper unknown".into();
    }
    let total = parameters
        .add(&persistent)?
        .add(&retained)?
        .add(&transient)?
        .add(&plan.staging)?
        .add(&plan.backend_overhead)?;
    Ok(PhaseMemoryEstimate {
        phase,
        positions,
        query_positions: query,
        parameters,
        persistent_state: persistent,
        retained_input: retained,
        workspace: transient,
        staging: plan.staging.clone(),
        backend_overhead: plan.backend_overhead.clone(),
        total,
    })
}

pub(crate) fn fit(
    bytes: &MemoryBytes,
    already_resident: u64,
    budget: &MemoryBudget,
) -> Result<MemoryFit, CapabilityError> {
    let total = bytes.add(&MemoryBytes::exact(budget.reserve_bytes))?;
    let additional = bytes
        .additional(already_resident)
        .add(&MemoryBytes::exact(budget.reserve_bytes))?;
    let comparisons = [
        (budget.application_limit_bytes, total),
        (budget.available_bytes, additional),
    ];
    let mut known = false;
    let mut uncertain = false;
    for (limit, need) in comparisons {
        if let Some(limit) = limit {
            known = true;
            if need.lower_bytes > limit {
                return Ok(MemoryFit::LikelyShortfall);
            }
            uncertain |= need.upper_bytes.is_none_or(|upper| upper > limit);
        }
    }
    Ok(if known && !uncertain {
        MemoryFit::LikelyFit
    } else {
        MemoryFit::InsufficientInformation
    })
}

/// Estimates overlapping request phase costs and compares every physical pool
/// independently. Work is bounded by the number of supplied ranks and state layers,
/// not by the prompt length or number of prefill chunks.
pub fn estimate_generation_memory(
    request: &GenerationMemoryRequest,
) -> Result<GenerationMemoryEstimate, CapabilityError> {
    if request.batch_size == 0
        || request.prefill_chunk_tokens == 0
        || request.input.model_positions == 0
    {
        return Err(invalid(
            "request geometry",
            "batch, prefill chunk and input positions must be positive",
        ));
    }
    if request.domains.is_empty() {
        return Err(invalid("domains", "at least one physical pool is required"));
    }
    for (i, domain) in request.domains.iter().enumerate() {
        if request.domains[..i]
            .iter()
            .any(|other| other.domain == domain.domain)
            || (request.domains.len() > 1 && domain.domain == MemoryDomain::Unified)
        {
            return Err(invalid(
                "domains",
                "physical pools must be disjoint; combine shared backing in one unified pool",
            ));
        }
    }
    let output = request
        .max_output_tokens
        .unwrap_or(request.forecast_output_tokens);
    let positions = add(request.input.model_positions, output)?;
    let chunk = request
        .prefill_chunk_tokens
        .min(request.input.model_positions);
    let remainder = request.input.model_positions % chunk;
    let final_query = if remainder == 0 { chunk } else { remainder };
    let mut domains = Vec::with_capacity(request.domains.len());
    for plan in &request.domains {
        plan.loading_peak.validate()?;
        let mut phases = Vec::new();
        // The last full chunk can exceed the final partial chunk's workspace.
        if request.input.model_positions > final_query {
            phases.push(phase(
                plan,
                request,
                MemoryPhase::Prefill,
                request.input.model_positions - final_query,
                chunk,
            )?);
        }
        phases.push(phase(
            plan,
            request,
            MemoryPhase::Prefill,
            request.input.model_positions,
            final_query,
        )?);
        if output > 0 {
            phases.push(phase(plan, request, MemoryPhase::Decode, positions, 1)?);
        }
        let generation_peak = phases
            .iter()
            .fold(MemoryBytes::exact(0), |a, p| a.maximum(&p.total));
        let loading_peak = plan.loading_peak.add(&plan.backend_overhead)?;
        let overall_peak = generation_peak.maximum(&loading_peak);
        let mut growth = 0;
        for execution in &plan.executions {
            let state = estimate_runtime_state(
                &execution.state_layout,
                request.input,
                output,
                request.batch_size,
                request.scalar_bytes,
            )?;
            growth = add(
                growth,
                mul(state.bytes_per_position_per_batch, request.batch_size)?,
            )?;
        }
        phases.push(PhaseMemoryEstimate {
            phase: MemoryPhase::Loading,
            positions: 0,
            query_positions: 0,
            parameters: MemoryBytes::exact(0),
            persistent_state: MemoryBytes::exact(0),
            retained_input: MemoryBytes::exact(0),
            workspace: MemoryBytes::exact(0),
            staging: plan.loading_peak.clone(),
            backend_overhead: plan.backend_overhead.clone(),
            total: loading_peak,
        });
        domains.push(DomainMemoryEstimate {
            domain: plan.domain.clone(),
            phases,
            additional_generation_peak: generation_peak.additional(plan.already_resident_bytes),
            generation_fit: fit(&generation_peak, plan.already_resident_bytes, &plan.budget)?,
            fit: fit(&overall_peak, plan.already_resident_bytes, &plan.budget)?,
            generation_peak,
            overall_peak,
            state_growth_bytes_per_position: growth,
        });
    }
    let overall_fit = if domains.iter().any(|d| d.fit == MemoryFit::LikelyShortfall) {
        MemoryFit::LikelyShortfall
    } else if domains.iter().all(|d| d.fit == MemoryFit::LikelyFit) {
        MemoryFit::LikelyFit
    } else {
        MemoryFit::InsufficientInformation
    };
    let uncertainties = memory_uncertainties(request, false);
    Ok(GenerationMemoryEstimate { requested_positions: positions, is_forecast: request.max_output_tokens.is_none(),
        domains, fit: overall_fit, uncertainties, assumptions: vec![
            "Planning estimates do not bound every allocation or total process memory; reserve capacity for other work.".into(),
            "Parameters, growing state, retained inputs, workspace, staging and overhead overlap within each phase; separate phase peaks are maximized.".into(),
            "Selected ordinary module topology supplies projection, attention, convolution and feed-forward geometry; reusable mechanism calibrations and explicit evaluation lifetimes compose their workspace. Legacy aggregate requests are compatibility inputs to the same resource lifetime evaluator; their earlier bounds remain preserved.".into(),
            "Sliding attention workspace uses the full evaluated context conservatively; state follows the architecture's exact cache policy and allocation granularity.".into(),
            "Rank-local executions supplied in one physical pool are treated as concurrent; shared parameter backing must be declared once and replicas separately.".into(),
        ] })
}

/// Sources shared by fresh and native-state continuation projections.
pub(crate) fn memory_uncertainties(
    request: &GenerationMemoryRequest,
    native_state_envelope: bool,
) -> Vec<String> {
    let mut uncertainties = Vec::new();
    for plan in &request.domains {
        for (name, bytes) in [
            ("parameters", &plan.resident_parameters),
            ("retained input", &plan.retained_input),
            ("staging", &plan.staging),
            ("backend overhead", &plan.backend_overhead),
            ("loading peak", &plan.loading_peak),
        ] {
            if bytes.kind != ObservationKind::Exact || bytes.upper_bytes.is_none() {
                uncertainties.push(format!("{:?} {name}: {}", plan.domain, bytes.detail));
            }
        }
        for (rank, execution) in plan.executions.iter().enumerate() {
            let prefix = format!("{:?} execution {rank}", plan.domain);
            if let Some(topology) = &execution.execution_topology {
                uncertainties.push(format!("{prefix}: generic mechanism calibration v1 uses explicit lazy-evaluation envelopes; native scratch, alignment and arithmetic promotion are planning assumptions, not allocator guarantees"));
                uncertainties.extend(
                    topology
                        .missing
                        .iter()
                        .map(|reason| format!("{prefix}: {reason}")),
                );
                for layer in &topology.layers {
                    if let crate::execution_topology::TokenMixerTopology::Unknown { reason } =
                        &layer.mixer
                    {
                        uncertainties.push(format!("{prefix}: {reason}"));
                    }
                    if let crate::execution_topology::FeedForwardTopology::Unknown { reason } =
                        &layer.feed_forward
                    {
                        uncertainties.push(format!("{prefix}: {reason}"));
                    }
                }
            } else if let Some(g) = &execution.workspace {
                if g.gated_convolution.is_some() {
                    uncertainties.push(format!("{prefix}: gated-convolution calibration v1 includes gated rows, padded input copies, kernel-width scratch and layer overlap; linear upper allows float32 promotion; persistent history is separate"));
                }
                if g.mixed_precision_parameter_bytes.is_some() {
                    uncertainties.push(format!("{prefix}: mixed floating widths allow float32 parameter casts and promoted state/replacement above nominal state storage"));
                }
                if g.input_score_attention.is_some() {
                    uncertainties.push(format!("{prefix}: input-score attention includes native query tiling, repeated expanded K/V, score conversions and layer overlap; linear upper allows float32 promotion; missing native facts preserve an unknown upper end"));
                }
            }
            if execution.workspace.is_none() && execution.execution_topology.is_none() {
                uncertainties.push(format!("{prefix}: decoder workspace geometry unavailable"));
            }
            if execution.workspace_overlap.upper_live_copies != Some(1) {
                uncertainties.push(format!(
                    "{prefix} linear workspace overlap: {}",
                    execution.workspace_overlap.detail
                ));
            }
            if !native_state_envelope && has_nonmonotonic_state(&execution.state_layout) {
                uncertainties.push(format!(
                    "{prefix}: interior peaks of remainder-shaped state unavailable"
                ));
            }
            let fused_attention = execution
                .execution_topology
                .as_ref()
                .is_none_or(|topology| {
                    topology.layers.iter().any(|layer| {
                        matches!(
                            layer.mixer,
                            crate::execution_topology::TokenMixerTopology::Attention {
                                input_scores: false,
                                softcap: false,
                                ..
                            }
                        )
                    })
                });
            match &execution.attention {
                AttentionWorkspace::Unknown if fused_attention => {
                    uncertainties.push(format!("{prefix}: attention kernel workspace unavailable"))
                }
                AttentionWorkspace::ScoreMatrixUpperBound if fused_attention => {
                    uncertainties.push(format!(
                        "{prefix}: uncalibrated fused attention uses score-matrix upper estimate"
                    ))
                }
                AttentionWorkspace::Fused { scratch }
                    if fused_attention
                        && (scratch.kind != ObservationKind::Exact
                            || scratch.upper_bytes.is_none()) =>
                {
                    uncertainties.push(format!("{prefix} attention scratch: {}", scratch.detail))
                }
                _ => {}
            }
            if execution.cache_update == CacheUpdateWorkspace::Unknown {
                uncertainties.push(format!("{prefix}: cache update overlap unavailable"));
            }
        }
    }
    uncertainties
}

/// Recomputes supported chunk candidates independently, preserving the supplied
/// mechanism facts. Backend scratch ranges must cover every candidate geometry.
pub fn estimate_prefill_candidates(
    request: &GenerationMemoryRequest,
    chunks: &[u64],
) -> Result<Vec<(u64, GenerationMemoryEstimate)>, CapabilityError> {
    chunks
        .iter()
        .map(|chunk| {
            let mut candidate = request.clone();
            candidate.prefill_chunk_tokens = *chunk;
            estimate_generation_memory(&candidate).map(|estimate| (*chunk, estimate))
        })
        .collect()
}

/// Parameter placement derived from an existing static report. Unified memory
/// implies shared capacity, not identical host/device backing. With unknown overlap
/// the interval spans max(host, device)..host+device; no global allocator counter is
/// added to parameter payloads (that would double-count live tensors).
///
/// `shared_backing_bytes` is the known intersection of the host and device
/// resident parameter observations: `Some(0)` declares distinct allocations;
/// `None` means overlap is unknown. Missing observations preserve an unknown
/// upper end. Use the returned interval's `lower_bytes` as the conservative
/// parameter contribution to [`DomainMemoryPlan::already_resident_bytes`].
/// Loaded facade forecasts, including the CLI, use this rule with unknown overlap.
pub fn static_parameter_placement(
    report: &StaticMemoryReport,
    shared_backing_bytes: Option<u64>,
) -> Result<Vec<(MemoryDomain, MemoryBytes)>, CapabilityError> {
    fn bytes(value: &Observed<u64>) -> MemoryBytes {
        match value {
            Observed::Available {
                value,
                kind,
                source,
            } => MemoryBytes {
                lower_bytes: *value,
                upper_bytes: Some(*value),
                kind: *kind,
                detail: source.clone(),
            },
            Observed::Unavailable { reason } | Observed::Unsupported { reason } => {
                MemoryBytes::unknown(reason)
            }
        }
    }
    let host = bytes(&report.current_host_resident_bytes);
    let device = bytes(&report.current_device_resident_bytes);
    match report.physical_semantics {
        PhysicalMemorySemantics::SeparateTiers => Ok(vec![
            (MemoryDomain::Host, host),
            (MemoryDomain::Device("default".into()), device),
        ]),
        PhysicalMemorySemantics::Unified => {
            let sum = host.add(&device)?;
            let payload = if let Some(shared) = shared_backing_bytes {
                if shared > host.lower_bytes.min(device.lower_bytes) {
                    return Err(invalid(
                        "shared_backing_bytes",
                        "exceeds either tier's known resident payload",
                    ));
                }
                MemoryBytes {
                    lower_bytes: sum.lower_bytes - shared,
                    upper_bytes: sum.upper_bytes.map(|n| n - shared),
                    kind: sum.kind,
                    detail: "host plus device minus declared shared backing".into(),
                }
            } else {
                MemoryBytes {
                    lower_bytes: host.lower_bytes.max(device.lower_bytes),
                    upper_bytes: sum.upper_bytes,
                    kind: ObservationKind::Estimated,
                    detail: "unified capacity; host/device backing overlap is unknown".into(),
                }
            };
            Ok(vec![(MemoryDomain::Unified, payload)])
        }
        PhysicalMemorySemantics::Unknown => Err(invalid(
            "physical_semantics",
            "host/device relationship is unknown; supply explicit physical domain plans",
        )),
    }
}

#[cfg(test)]
mod tests;
