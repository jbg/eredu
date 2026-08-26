//! Backend-neutral automatic execution planning.
//!
//! Backends report hardware, artifact resources, and candidate admission. This
//! module owns policy validation, resource budgeting, plan selection, feedback
//! matching, and the serialized planning and telemetry documents.

use crate::{
    artifact::ArtifactFormat,
    backend::{BackendProvider, ModelLoadingBackend, ModelRuntime},
    execution::{
        DevicePlan, DraftingPlan, ExecutionPlan, ExpertCachePlan, ResidencyPlan,
        DEFAULT_MAX_MAPPED_SHARDS,
    },
    speculative::SpeculativeDraft,
};
use serde::{Deserialize, Serialize};
use std::{path::PathBuf, time::Duration};

/// Schema version shared by automatic-planning and telemetry documents.
pub const AUTOMATIC_SCHEMA_VERSION: u32 = 4;

/// Confidence attached to an observed or derived value.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationKind {
    /// Derived exactly from validated metadata or an exact counter.
    Exact,
    /// An upper bound chosen to avoid understating a resource requirement.
    Conservative,
    /// A point-in-time observation which may immediately change.
    Observational,
    /// A platform or model-derived estimate.
    Estimated,
}

/// A value which remains explicit when the runtime cannot produce it.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Observed<T> {
    /// A usable value with documented provenance.
    Available {
        /// Observed value.
        value: T,
        /// Confidence and measurement semantics.
        kind: ObservationKind,
        /// Stable human-readable provenance.
        source: String,
    },
    /// The platform or artifact cannot provide this measurement.
    Unsupported {
        /// Reason the measurement is unsupported.
        reason: String,
    },
    /// The measurement is meaningful but was not available.
    Unavailable {
        /// Reason the value could not be obtained.
        reason: String,
    },
}

impl<T> Observed<T> {
    /// Creates an exact observation.
    pub fn exact(value: T, source: impl Into<String>) -> Self {
        Self::Available {
            value,
            kind: ObservationKind::Exact,
            source: source.into(),
        }
    }

    /// Creates an unavailable observation without inventing a default value.
    pub fn unavailable(reason: impl Into<String>) -> Self {
        Self::Unavailable {
            reason: reason.into(),
        }
    }

    /// Creates an unsupported observation without inventing a default value.
    pub fn unsupported(reason: impl Into<String>) -> Self {
        Self::Unsupported {
            reason: reason.into(),
        }
    }

    /// Borrows the available value, returning `None` when no value was reported.
    pub const fn value(&self) -> Option<&T> {
        match self {
            Self::Available { value, .. } => Some(value),
            Self::Unsupported { .. } | Self::Unavailable { .. } => None,
        }
    }
}

fn unobserved_embedded_draft_layers() -> Observed<usize> {
    Observed::unavailable("embedded drafting requires normalized architecture inspection")
}

/// Architecture and header-derived planning facts used before a model is loaded.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModelResourceProfile {
    /// Version of this serialized resource schema.
    pub schema_version: u32,
    /// Inspected checkpoint path.
    pub path: PathBuf,
    /// Physical checkpoint container.
    pub artifact_format: ArtifactFormat,
    /// Resolved model family, when architecture inspection succeeded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_family: Option<String>,
    /// Resolved architecture name, when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub architecture: Option<String>,
    /// Number of logical tensors exposed by the checkpoint catalog.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensor_count: Option<usize>,
    /// Number of physical checkpoint shards.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checkpoint_shards: Option<usize>,
    /// Embedded prediction depth derived from normalized architecture policy.
    #[serde(default = "unobserved_embedded_draft_layers")]
    pub embedded_draft_layers: Observed<usize>,
    /// Sum of encoded tensor payload bytes, excluding container metadata.
    pub stored_tensor_bytes: Observed<u64>,
    /// Largest encoded logical or physical tensor payload.
    pub largest_stored_tensor_bytes: Observed<u64>,
    /// Expected execution-time parameter bytes after translation or quantization.
    pub materialized_parameter_bytes: Observed<u64>,
    /// Bytes in parameters pinned outside repeated execution groups.
    pub pinned_parameter_bytes: Observed<u64>,
    /// Largest single repeated execution group.
    pub largest_execution_group_bytes: Observed<u64>,
    /// Largest adjacent pair required by dense streaming's device window.
    pub largest_adjacent_execution_groups_bytes: Observed<u64>,
    /// Total routed-expert bytes, where the architecture exposes an exact plan.
    pub expert_parameter_bytes: Observed<u64>,
}

impl ModelResourceProfile {
    /// Creates an explicitly unmeasured resource profile.
    pub fn unmeasured(path: PathBuf, artifact_format: ArtifactFormat) -> Self {
        let unavailable = || {
            Observed::unavailable("resource value requires a validated checkpoint parameter plan")
        };
        Self {
            schema_version: AUTOMATIC_SCHEMA_VERSION,
            path,
            artifact_format,
            model_family: None,
            architecture: None,
            tensor_count: None,
            checkpoint_shards: None,
            embedded_draft_layers: unobserved_embedded_draft_layers(),
            stored_tensor_bytes: Observed::unavailable(
                "checkpoint tensor catalog was not established",
            ),
            largest_stored_tensor_bytes: Observed::unavailable(
                "checkpoint tensor catalog was not established",
            ),
            materialized_parameter_bytes: unavailable(),
            pinned_parameter_bytes: unavailable(),
            largest_execution_group_bytes: unavailable(),
            largest_adjacent_execution_groups_bytes: unavailable(),
            expert_parameter_bytes: unavailable(),
        }
    }
}

/// One logical device visible to an execution backend.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct HardwareDeviceProfile {
    /// Backend-stable device identifier.
    pub id: String,
    /// Backend-defined device family.
    pub family: String,
    /// Process-local device index.
    pub index: usize,
    /// Total physical device capacity, if independently observable.
    pub total_memory_bytes: Observed<u64>,
    /// Point-in-time available device capacity, if independently observable.
    pub available_memory_bytes: Observed<u64>,
}

/// Availability and devices for one execution backend.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct HardwareBackendProfile {
    /// Backend identity.
    pub backend: crate::execution::BackendId,
    /// Whether the runtime can execute through this backend.
    pub available: bool,
    /// Reason discovery could not establish availability.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    /// Devices which discovery can enumerate without guessing.
    pub devices: Vec<HardwareDeviceProfile>,
}

/// Hardware and memory observations used as automatic-planning inputs.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct HardwareProfile {
    /// Version of this serialized hardware schema.
    pub schema_version: u32,
    /// Rust target operating-system name.
    pub operating_system: String,
    /// Rust target architecture name.
    pub architecture: String,
    /// Logical CPU parallelism available to the process.
    pub logical_cpu_count: Observed<u64>,
    /// Installed host or unified physical memory.
    pub physical_memory_bytes: Observed<u64>,
    /// Point-in-time host or unified available memory.
    pub available_memory_bytes: Observed<u64>,
    /// Whether logical host and accelerator allocations share capacity.
    pub physical_memory_semantics: HardwareMemorySemantics,
    /// Execution backends visible to the selected adapter.
    pub backends: Vec<HardwareBackendProfile>,
}

/// Serializable form of physical host/device memory semantics.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HardwareMemorySemantics {
    /// Host and device allocations share one physical capacity.
    Unified,
    /// Host and accelerator memory are physically separate.
    SeparateTiers,
    /// The relationship cannot be established.
    Unknown,
}

/// Severity of one planner explanation entry.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanExplanationLevel {
    /// Normal selection rationale.
    Decision,
    /// A limitation or risk worth surfacing to the caller.
    Warning,
    /// A candidate rejected by compatibility or resource admission.
    Rejection,
}

/// One stable, machine-routable planner explanation entry.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct PlanExplanationEntry {
    /// Severity/category of the explanation.
    pub level: PlanExplanationLevel,
    /// Stable machine-readable code.
    pub code: String,
    /// Human-readable explanation.
    pub detail: String,
}

/// Explanation accompanying a selected execution plan.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct PlanExplanation {
    /// Short description of the selected plan.
    pub summary: String,
    /// Ordered decisions, warnings, and candidate rejections.
    pub entries: Vec<PlanExplanationEntry>,
}

/// Complete automatic-planning document suitable for JSON persistence.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExecutionPlanReport {
    /// Version of this serialized planning document.
    pub schema_version: u32,
    /// Hardware observations used by the planner.
    pub hardware: HardwareProfile,
    /// Header-only model resource observations used by the planner.
    pub resources: ModelResourceProfile,
    /// Concrete selected execution settings.
    pub plan: ExecutionPlan,
    /// Ordered rationale and rejected alternatives.
    pub explanation: PlanExplanation,
}

/// Tunable, serializable automatic-planning policy.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct AutomaticPlannerPolicy {
    /// Device budget used when current device availability is unavailable.
    pub device_memory_fallback_bytes: u64,
    /// Host budget used when current host availability is unavailable.
    pub host_memory_fallback_bytes: u64,
    /// Percentage of observed free memory reserved for runtime state and drift.
    pub memory_headroom_percent: u8,
    /// Percentage of bounded residency budgets assigned to routed experts.
    pub expert_cache_share_percent: u8,
    /// Repeated execution groups retained in the layerwise device window.
    pub device_layer_window: usize,
    /// Maximum simultaneously mapped checkpoint shards or readers.
    pub max_mapped_shards: usize,
    /// Maximum proposals used when embedded MTP is available.
    pub embedded_mtp_draft_tokens: usize,
    /// Minimum generated-token count for one prior run to influence planning.
    pub minimum_feedback_tokens: usize,
}

impl Default for AutomaticPlannerPolicy {
    fn default() -> Self {
        Self {
            device_memory_fallback_bytes: 4 << 30,
            host_memory_fallback_bytes: 16 << 30,
            memory_headroom_percent: 30,
            expert_cache_share_percent: 40,
            device_layer_window: 1,
            max_mapped_shards: DEFAULT_MAX_MAPPED_SHARDS,
            embedded_mtp_draft_tokens: 3,
            minimum_feedback_tokens: 1,
        }
    }
}

/// Timings reported for one generation request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimingTelemetry {
    /// Model load duration in seconds.
    pub load_seconds: f64,
    /// Generation duration in seconds.
    pub generation_seconds: f64,
    /// Time to the first emitted token in seconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_to_first_token_seconds: Option<f64>,
    /// Complete operation duration in seconds.
    pub total_seconds: f64,
    /// Overall generated-token rate.
    pub token_rate: f64,
    /// Post-first-token decode rate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_token_rate: Option<f64>,
}

impl TimingTelemetry {
    /// Builds stable timing metrics from monotonic durations.
    pub fn new(
        load: Duration,
        generation: Duration,
        time_to_first_token: Option<Duration>,
        generated_tokens: usize,
        total: Duration,
    ) -> Self {
        fn rate(tokens: usize, elapsed: Duration) -> f64 {
            if elapsed.is_zero() {
                0.0
            } else {
                tokens as f64 / elapsed.as_secs_f64()
            }
        }
        Self {
            load_seconds: load.as_secs_f64(),
            generation_seconds: generation.as_secs_f64(),
            time_to_first_token_seconds: time_to_first_token.map(|value| value.as_secs_f64()),
            total_seconds: total.as_secs_f64(),
            token_rate: rate(generated_tokens, generation),
            decode_token_rate: time_to_first_token.map(|first| {
                rate(
                    generated_tokens.saturating_sub(1),
                    generation.saturating_sub(first),
                )
            }),
        }
    }
}

/// Backend allocator observations for one execution.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct AllocatorTelemetry {
    /// Peak active backend-managed allocation bytes.
    pub peak_bytes: u64,
    /// Active backend-managed allocation bytes at collection time.
    pub active_bytes: u64,
    /// Bytes retained by the backend allocator cache at collection time.
    pub cache_bytes: u64,
}

/// Logical bytes and transfers reported by bounded parameter residency.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResidencyTelemetry {
    /// Planned logical disk bytes.
    pub planned_disk_bytes: u64,
    /// Planned logical host bytes.
    pub planned_host_bytes: u64,
    /// Planned logical device bytes.
    pub planned_device_bytes: u64,
    /// Current logical host-resident bytes.
    pub current_host_bytes: u64,
    /// Current logical device-resident bytes.
    pub current_device_bytes: u64,
    /// Peak logical host-resident bytes.
    pub peak_host_bytes: u64,
    /// Peak logical device-resident bytes.
    pub peak_device_bytes: u64,
    /// Transfers in stable source-to-destination order.
    pub transfers: Vec<TransferTelemetry>,
}

/// One logical residency transfer counter.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct TransferTelemetry {
    /// Stable direction label.
    pub direction: String,
    /// Completed transfer count.
    pub count: u64,
    /// Logical bytes transferred.
    pub bytes: u64,
    /// Accumulated transfer time in seconds.
    pub seconds: DurationSeconds,
}

/// Floating-point duration wrapper with equality based on its bit pattern.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DurationSeconds(pub f64);

impl PartialEq for DurationSeconds {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}
impl Eq for DurationSeconds {}

/// Routed-expert cache occupancy summary.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExpertCacheTelemetry {
    /// Owned expert count.
    pub owned_experts: usize,
    /// Owned logical expert bytes.
    pub owned_bytes: u64,
    /// Current host-resident expert count.
    pub host_resident_experts: usize,
    /// Current device-resident expert count.
    pub device_resident_experts: usize,
    /// Current host allocation capacity for experts.
    pub host_resident_bytes: u64,
    /// Current logical device expert bytes.
    pub device_resident_bytes: u64,
    /// Peak host expert bytes.
    pub peak_host_resident_bytes: u64,
    /// Peak device expert bytes.
    pub peak_device_resident_bytes: u64,
}

/// Speculative-decoding observations for one request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MtpTelemetry {
    /// Stable target/assistant execution-placement topology label.
    pub execution_topology: String,
    /// Target tokens evaluated.
    pub target_tokens: usize,
    /// Assistant proposals.
    pub draft_tokens: usize,
    /// Accepted assistant proposals.
    pub accepted_tokens: usize,
    /// Proposal acceptance fraction.
    pub accept_rate: f64,
    /// Verification rounds.
    pub rounds: usize,
    /// Accepted proposal count per round.
    pub accept_lens: Vec<usize>,
    /// Emitted tokens, including terminal EOS where applicable.
    pub emitted_tokens: usize,
    /// Optimistically drafted tokens.
    pub optimistic_draft_tokens: usize,
    /// Optimistically reused tokens.
    pub reused_optimistic_tokens: usize,
    /// Optimistically discarded tokens.
    pub discarded_optimistic_tokens: usize,
    /// Whether adaptive accounting disabled further lookahead.
    pub adaptive_lookahead_disabled: bool,
    /// Host time spent in optimistic drafting.
    pub optimistic_draft_seconds: f64,
    /// Target verification in-flight wall time.
    pub verification_in_flight_seconds: f64,
}

/// Stable JSON telemetry for one completed model execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionTelemetry {
    /// Version of this serialized telemetry schema.
    pub schema_version: u32,
    /// Runtime model type.
    pub model_type: String,
    /// Concrete execution choices used by the run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plan: Option<ExecutionPlan>,
    /// Explanation of how the recorded plan was selected.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plan_explanation: Option<PlanExplanation>,
    /// Pre-load hardware observations used or available to the caller.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hardware: Option<HardwareProfile>,
    /// Header-only model resource observations for the selected load policy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resources: Option<ModelResourceProfile>,
    /// Input token count.
    pub prompt_tokens: usize,
    /// Emitted token count after terminal-token normalization.
    pub generated_tokens: usize,
    /// Stable completion reason.
    pub stop_reason: String,
    /// Load and generation timings.
    pub timing: TimingTelemetry,
    /// Backend allocator observations.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allocator: Option<AllocatorTelemetry>,
    /// Bounded ordinary-weight residency observations.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub residency: Option<ResidencyTelemetry>,
    /// Independent routed-expert cache observations.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expert_cache: Option<ExpertCacheTelemetry>,
    /// Speculative-decoding observations.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mtp: Option<MtpTelemetry>,
}

/// Owned input to one automatic planning session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct AutomaticPlanRequest {
    /// Version of this serialized request.
    pub schema_version: u32,
    /// Local model directory or GGUF checkpoint to inspect.
    pub model_path: PathBuf,
    /// Single execution device to plan for.
    pub device: DevicePlan,
    /// Completed runtime observations from earlier sessions.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub prior_telemetry: Vec<ExecutionTelemetry>,
}

impl AutomaticPlanRequest {
    /// Creates a request with no historical runtime feedback.
    pub fn new(model_path: impl Into<PathBuf>, device: DevicePlan) -> Self {
        Self {
            schema_version: AUTOMATIC_SCHEMA_VERSION,
            model_path: model_path.into(),
            device,
            prior_telemetry: Vec::new(),
        }
    }

    /// Adds completed telemetry for consideration during this planning session.
    pub fn with_prior_telemetry(
        mut self,
        telemetry: impl IntoIterator<Item = ExecutionTelemetry>,
    ) -> Self {
        self.prior_telemetry.extend(telemetry);
        self
    }
}

/// Backend candidate-admission result consumed by the neutral planner.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct CandidateAdmission {
    /// Whether the backend can materialize and execute this plan.
    pub supported: bool,
    /// Stable rejection detail when unsupported.
    pub rejection: Option<String>,
}

/// Exact bounded device-window requirement established by a backend probe.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct BoundedResidencyRequirement {
    /// Bytes pinned outside the repeated execution window.
    pub static_bytes: u64,
    /// Bytes in the required repeated execution window.
    pub window_bytes: u64,
    /// Total required bytes.
    pub required_bytes: u64,
    /// Number of adjacent repeated groups in the window.
    pub depth: usize,
}

/// High-level observations a backend supplies to the neutral planner.
pub trait AutomaticPlanningBackend {
    /// Stable identity used by execution plans for this backend adapter.
    fn backend_id(&self) -> crate::execution::BackendId;
    /// Discovers the devices and memory facts visible to this backend adapter.
    fn discover_hardware(&self) -> Result<HardwareProfile, AutomaticPlanningError>;
    /// Inspects an artifact without materializing its tensor payloads.
    fn inspect_resources(
        &self,
        model_path: &std::path::Path,
    ) -> Result<ModelResourceProfile, AutomaticPlanningError>;
    /// Checks whether this backend can load a concrete portable plan.
    fn admit_candidate(
        &self,
        model_path: &std::path::Path,
        plan: &ExecutionPlan,
    ) -> Result<CandidateAdmission, AutomaticPlanningError>;
    /// Establishes the exact bounded window needed by a non-resident plan.
    fn bounded_residency_requirement(
        &self,
        model_path: &std::path::Path,
        plan: &ExecutionPlan,
    ) -> Result<BoundedResidencyRequirement, AutomaticPlanningError>;
}

/// One target-backend instance and load policy realized from a portable execution plan.
///
/// The backend owns its selected device, execution queues, transfer queues, and
/// optional communication state. Callers pass this value directly to the
/// generic model loader instead of reconstructing backend-specific options.
pub struct ExecutionPlanTarget<B: ModelLoadingBackend> {
    backend: B,
    load_options: B::LoadOptions,
}

impl<B: ModelLoadingBackend> ExecutionPlanTarget<B> {
    /// Creates one backend-owned realization.
    ///
    /// Backend adapters call this from [`ExecutionPlanBackendFactory::realize_target`].
    /// Portable identity, device, capability, and plan validation is applied by
    /// [`realize_execution_plan_target`] before the value reaches an application.
    pub fn new(backend: B, load_options: B::LoadOptions) -> Self {
        Self {
            backend,
            load_options,
        }
    }

    /// Borrows the selected backend.
    pub const fn backend(&self) -> &B {
        &self.backend
    }

    /// Consumes the realization into the generic loader inputs.
    pub fn into_parts(self) -> (B, B::LoadOptions) {
        (self.backend, self.load_options)
    }
}

/// Proof that a target and external assistant use the same token-id vocabulary mapping.
///
/// The fingerprint is exposed only after both portable tokenizer identities have
/// been compared. Backend factories consume this proof instead of deciding
/// tokenizer compatibility themselves.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct TokenizerCompatibilityProof {
    fingerprint: [u8; 32],
}

impl TokenizerCompatibilityProof {
    /// Establishes compatibility from independently reconstructed tokenizer identities.
    pub fn prove(
        target_fingerprint: [u8; 32],
        assistant_fingerprint: [u8; 32],
    ) -> Result<Self, TokenizerCompatibilityError> {
        if target_fingerprint != assistant_fingerprint {
            return Err(TokenizerCompatibilityError);
        }
        Ok(Self {
            fingerprint: target_fingerprint,
        })
    }

    /// Returns the shared token-id vocabulary fingerprint established by this proof.
    pub const fn fingerprint(self) -> [u8; 32] {
        self.fingerprint
    }

    /// Verifies that this proof is being applied to the target it was established for.
    pub fn validate_target(
        self,
        target_fingerprint: [u8; 32],
    ) -> Result<(), TokenizerCompatibilityError> {
        if self.fingerprint != target_fingerprint {
            return Err(TokenizerCompatibilityError);
        }
        Ok(())
    }
}

/// A target and external assistant do not share the same token-id vocabulary mapping.
#[derive(Debug, Clone, Copy, Eq, PartialEq, thiserror::Error)]
#[error("assistant token-id vocabulary mapping does not match the target")]
pub struct TokenizerCompatibilityError;

/// Architecture-prepared assistant artifact and proven portable tokenizer compatibility.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ExternalDraftArtifact<P> {
    /// Inspected, backend-neutral assistant materialization plan.
    pub preparation: P,
    /// Proof that the target and external assistant share one token-id vocabulary mapping.
    pub tokenizer_compatibility: TokenizerCompatibilityProof,
}

/// Backend-owned drafting resources realized for one complete execution plan.
pub enum RealizedDrafting<D> {
    /// Ordinary target-only decoding.
    Disabled,
    /// Draft heads embedded in the prepared target model.
    Embedded,
    /// Separately prepared assistant owned by the selected backend.
    External(D),
}

impl<D> RealizedDrafting<D> {
    /// Borrows the request-level draft selection when speculative execution is enabled.
    pub fn as_speculative_draft(&mut self) -> Option<SpeculativeDraft<'_, D>> {
        match self {
            Self::Disabled => None,
            Self::Embedded => Some(SpeculativeDraft::Embedded),
            Self::External(drafter) => Some(SpeculativeDraft::External(drafter)),
        }
    }

    /// Returns whether this plan owns a separately prepared assistant.
    pub const fn is_external(&self) -> bool {
        matches!(self, Self::External(_))
    }
}

/// Creates an executable whole-model backend from a portable execution plan.
///
/// This deliberately operates above tensor primitives. An implementation maps
/// one complete [`DevicePlan`] and [`ExecutionPlan`] to an owned backend and
/// its opaque load policy. Core then verifies backend identity, selected-device
/// identity, structural plan invariants, and fail-closed capabilities.
pub trait ExecutionPlanBackendFactory: AutomaticPlanningBackend {
    /// Backend implementation created for the selected model/session.
    type Backend: ModelLoadingBackend;
    /// Architecture-owned preparation consumed by assistant materialization.
    type DrafterPreparation;
    /// Backend-owned separately prepared assistant type.
    type Drafter;

    /// Backend hook which owns device/queue construction and plan translation.
    ///
    /// Applications should call [`realize_execution_plan_target`] so portable
    /// validation cannot be bypassed accidentally.
    fn realize_target(
        &self,
        plan: &ExecutionPlan,
    ) -> Result<ExecutionPlanTarget<Self::Backend>, AutomaticPlanningError>;

    /// Realizes the plan's complete drafting mode against a prepared target session.
    ///
    /// `external_artifact` is present exactly for [`DraftingPlan::External`].
    /// It is assembled by the portable facade, which owns architecture
    /// inspection and tokenizer loading, while the backend owns only assistant
    /// materialization, placement, and architecture compatibility validation.
    fn realize_drafting(
        &self,
        plan: &ExecutionPlan,
        target: &ModelRuntime<Self::Backend>,
        external_artifact: Option<ExternalDraftArtifact<Self::DrafterPreparation>>,
    ) -> Result<RealizedDrafting<Self::Drafter>, AutomaticPlanningError>;
}

/// Validates and realizes the target portion of a portable execution plan.
pub fn realize_execution_plan_target<F: ExecutionPlanBackendFactory>(
    factory: &F,
    plan: &ExecutionPlan,
) -> Result<ExecutionPlanTarget<F::Backend>, AutomaticPlanningError> {
    let expected_backend = factory.backend_id();
    if plan.device.backend != expected_backend {
        return Err(AutomaticPlanningError::Invalid(format!(
            "execution plan selects backend {} but factory owns {}",
            plan.device.backend, expected_backend
        )));
    }
    plan.validate_structure()
        .map_err(|error| AutomaticPlanningError::Invalid(error.to_string()))?;

    let realization = factory.realize_target(plan)?;
    let descriptor = realization.backend().descriptor();
    if descriptor.name != expected_backend.as_str() {
        return Err(AutomaticPlanningError::Invalid(format!(
            "factory identity {} does not match realized backend {}",
            expected_backend, descriptor.name
        )));
    }
    let devices =
        realization
            .backend()
            .devices()
            .map_err(|error| AutomaticPlanningError::Backend {
                operation: "realize_execution_plan_devices",
                message: error.to_string(),
            })?;
    let capabilities = devices
        .iter()
        .find_map(|(device, capabilities)| {
            (device.id == plan.device.device).then_some(capabilities)
        })
        .ok_or_else(|| {
            AutomaticPlanningError::Invalid(format!(
                "realized backend {} does not expose selected device {}",
                expected_backend, plan.device.device
            ))
        })?;
    plan.validate_device_capabilities(capabilities)
        .map_err(|error| AutomaticPlanningError::Invalid(error.to_string()))?;
    Ok(realization)
}

/// Validates and realizes the drafting portion of a portable execution plan.
pub fn realize_execution_plan_drafting<F: ExecutionPlanBackendFactory>(
    factory: &F,
    plan: &ExecutionPlan,
    target: &ModelRuntime<F::Backend>,
    external_artifact: Option<ExternalDraftArtifact<F::DrafterPreparation>>,
) -> Result<RealizedDrafting<F::Drafter>, AutomaticPlanningError> {
    match (&plan.drafting, external_artifact.as_ref()) {
        (DraftingPlan::External { .. }, None) => {
            return Err(AutomaticPlanningError::Invalid(
                "external drafting requires proven tokenizer compatibility".into(),
            ));
        }
        (DraftingPlan::Disabled | DraftingPlan::Embedded { .. }, Some(_)) => {
            return Err(AutomaticPlanningError::Invalid(
                "tokenizer compatibility was supplied for a plan without an external assistant"
                    .into(),
            ));
        }
        _ => {}
    }
    let drafting = factory.realize_drafting(plan, target, external_artifact)?;
    let matches_plan = matches!(
        (&plan.drafting, &drafting),
        (DraftingPlan::Disabled, RealizedDrafting::Disabled)
            | (DraftingPlan::Embedded { .. }, RealizedDrafting::Embedded)
            | (DraftingPlan::External { .. }, RealizedDrafting::External(_))
    );
    if !matches_plan {
        return Err(AutomaticPlanningError::Invalid(
            "backend factory realized a drafting mode different from the execution plan".into(),
        ));
    }
    Ok(drafting)
}

/// Failure produced by portable planning or its selected backend adapter.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum AutomaticPlanningError {
    /// A portable request or policy invariant is invalid.
    #[error("automatic planning error: {0}")]
    Invalid(String),
    /// A selected backend observation or admission operation failed.
    #[error("automatic planning backend failed during {operation}: {message}")]
    Backend {
        /// Stable high-level operation name.
        operation: &'static str,
        /// Backend-provided context.
        message: String,
    },
}

/// Backend-neutral automatic planner.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize, Default)]
pub struct AutomaticPlanner {
    policy: AutomaticPlannerPolicy,
}

impl AutomaticPlanner {
    /// Creates a planner with an explicit, serializable policy.
    pub fn new(policy: AutomaticPlannerPolicy) -> Self {
        Self { policy }
    }

    /// Returns the policy used for subsequent planning calls.
    pub fn policy(&self) -> &AutomaticPlannerPolicy {
        &self.policy
    }

    /// Selects a plan using only portable policy and backend observations.
    pub fn plan<B: AutomaticPlanningBackend>(
        &self,
        backend: &B,
        request: &AutomaticPlanRequest,
    ) -> Result<ExecutionPlanReport, AutomaticPlanningError> {
        validate_request(request, &self.policy)?;
        let backend_id = backend.backend_id();
        if request.device.backend != backend_id {
            return Err(AutomaticPlanningError::Invalid(format!(
                "selected planning backend {} cannot plan device owned by {}",
                backend_id, request.device.backend
            )));
        }
        let hardware = backend.discover_hardware()?;
        validate_device(&hardware, &request.device)?;
        let mut resources = backend.inspect_resources(&request.model_path)?;
        let selected_device =
            selected_device(&hardware, &request.device).expect("validated device is present");
        let device_capacity = memory_basis(
            observed_u64(&selected_device.available_memory_bytes),
            observed_u64(&selected_device.total_memory_bytes)
                .or_else(|| observed_u64(&hardware.physical_memory_bytes)),
            hardware.physical_memory_semantics,
        );
        let host_capacity = memory_basis(
            observed_u64(&hardware.available_memory_bytes),
            observed_u64(&hardware.physical_memory_bytes),
            hardware.physical_memory_semantics,
        );
        let device_budget = budget(
            device_capacity,
            self.policy.device_memory_fallback_bytes,
            self.policy.memory_headroom_percent,
        );
        let host_budget = budget(
            host_capacity,
            self.policy.host_memory_fallback_bytes,
            self.policy.memory_headroom_percent,
        );
        let model_bytes = observed_u64(&resources.materialized_parameter_bytes)
            .or_else(|| observed_u64(&resources.stored_tensor_bytes));
        let candidates = base_candidates(
            request.device.clone(),
            device_budget,
            host_budget,
            &self.policy,
        );
        let resident = backend.admit_candidate(&request.model_path, &candidates[0])?;
        let mut layerwise = backend.admit_candidate(&request.model_path, &candidates[1])?;
        let mut disk = backend.admit_candidate(&request.model_path, &candidates[2])?;
        let resident_fits = model_bytes.is_some_and(|bytes| bytes <= device_budget);
        let layerwise_host_fits = model_bytes.is_some_and(|bytes| {
            if hardware.physical_memory_semantics == HardwareMemorySemantics::Unified {
                bytes <= host_budget.saturating_mul(2)
            } else {
                bytes <= host_budget
            }
        });
        if !resident_fits || !resident.supported {
            apply_bounded_probe(
                backend,
                &request.model_path,
                &candidates[1],
                device_budget,
                &mut layerwise,
                &mut resources,
                false,
            )?;
            apply_bounded_probe(
                backend,
                &request.model_path,
                &candidates[2],
                device_budget,
                &mut disk,
                &mut resources,
                true,
            )?;
        }
        let selected =
            if resident_fits && resident.supported {
                0
            } else if layerwise_host_fits && layerwise.supported {
                1
            } else if disk.supported {
                2
            } else {
                return Err(AutomaticPlanningError::Invalid(format!(
                "no loadable single-device policy: resident: {}; layerwise: {}; disk-streamed: {}",
                rejection(&resident), rejection(&layerwise), rejection(&disk)
            )));
            };
        let mut plan = candidates[selected].clone();
        let mut entries = vec![PlanExplanationEntry {
            level: PlanExplanationLevel::Decision,
            code: "single_device_scope".into(),
            detail: format!(
                "automatic planning is restricted to {}:{} with {}% memory headroom",
                request.device.backend, request.device.device, self.policy.memory_headroom_percent
            ),
        }];
        if selected > 0 {
            entries.push(PlanExplanationEntry {
                level: PlanExplanationLevel::Rejection,
                code: "fully_resident_not_admitted".into(),
                detail: resident
                    .rejection
                    .unwrap_or_else(|| "the model exceeds the device memory budget".into()),
            });
        }
        if selected > 1 {
            entries.push(PlanExplanationEntry {
                level: PlanExplanationLevel::Rejection,
                code: "layerwise_not_admitted".into(),
                detail: layerwise
                    .rejection
                    .unwrap_or_else(|| "the model exceeds the host-backed admission budget".into()),
            });
        }
        let mut summary = match selected {
            0 => "selected fully resident execution for the lowest expected latency".to_string(),
            1 => "selected host-backed layerwise execution with a validated bounded device window"
                .to_string(),
            _ => "selected bounded dense disk streaming because resident and layerwise admission failed"
                .to_string(),
        };

        if selected > 0 {
            let expert_plan = with_expert_cache(plan.clone(), &self.policy);
            let expert = backend.admit_candidate(&request.model_path, &expert_plan)?;
            if expert.supported {
                plan = expert_plan;
                entries.push(PlanExplanationEntry {
                    level: PlanExplanationLevel::Decision,
                    code: "expert_cache_selected".into(),
                    detail: "the backend admitted independent routed-expert caching".into(),
                });
            }
        }

        let embedded_layers = resources.embedded_draft_layers.value().copied();
        if embedded_layers.is_some_and(|layers| layers > 0) {
            plan.drafting = DraftingPlan::Embedded {
                max_draft_tokens: self.policy.embedded_mtp_draft_tokens,
                lookahead: true,
                adaptive_lookahead: true,
            };
            entries.push(PlanExplanationEntry {
                level: PlanExplanationLevel::Decision,
                code: "embedded_mtp_selected".into(),
                detail: "checkpoint metadata advertises embedded prediction layers".into(),
            });
        }

        if let Some((feedback, samples, median)) = select_feedback_plan(
            backend,
            request,
            &hardware,
            &resources,
            &self.policy,
            embedded_layers,
        )? {
            plan = feedback;
            summary = format!(
                "selected a previously observed plan at {median:.2} median decode tokens/s"
            );
            entries.push(PlanExplanationEntry {
                level: PlanExplanationLevel::Decision,
                code: "prior_telemetry_selected".into(),
                detail: format!("selected using {samples} matching runtime sample(s)"),
            });
        }

        Ok(ExecutionPlanReport {
            schema_version: AUTOMATIC_SCHEMA_VERSION,
            hardware,
            resources,
            plan,
            explanation: PlanExplanation { summary, entries },
        })
    }
}

fn observed_u64(value: &Observed<u64>) -> Option<u64> {
    value.value().copied()
}

fn validate_request(
    request: &AutomaticPlanRequest,
    policy: &AutomaticPlannerPolicy,
) -> Result<(), AutomaticPlanningError> {
    if request.schema_version != AUTOMATIC_SCHEMA_VERSION {
        return Err(AutomaticPlanningError::Invalid(format!(
            "automatic request schema {} does not match supported schema {}",
            request.schema_version, AUTOMATIC_SCHEMA_VERSION
        )));
    }
    if policy.device_memory_fallback_bytes == 0 || policy.host_memory_fallback_bytes == 0 {
        return Err(AutomaticPlanningError::Invalid(
            "automatic fallback memory budgets must be greater than zero".into(),
        ));
    }
    if policy.memory_headroom_percent >= 100
        || policy.expert_cache_share_percent == 0
        || policy.expert_cache_share_percent >= 100
        || policy.device_layer_window == 0
        || policy.max_mapped_shards == 0
        || policy.embedded_mtp_draft_tokens == 0
        || policy.minimum_feedback_tokens == 0
    {
        return Err(AutomaticPlanningError::Invalid(
            "automatic percentage and count policy values are outside their valid ranges".into(),
        ));
    }
    Ok(())
}

fn selected_device<'a>(
    hardware: &'a HardwareProfile,
    device: &DevicePlan,
) -> Option<&'a HardwareDeviceProfile> {
    hardware
        .backends
        .iter()
        .find(|backend| backend.backend == device.backend && backend.available)
        .and_then(|backend| backend.devices.iter().find(|item| item.id == device.device))
}

fn validate_device(
    hardware: &HardwareProfile,
    device: &DevicePlan,
) -> Result<(), AutomaticPlanningError> {
    selected_device(hardware, device)
        .map(|_| ())
        .ok_or_else(|| {
            AutomaticPlanningError::Invalid(format!(
                "hardware discovery did not report available {} device {}",
                device.backend, device.device
            ))
        })
}

fn memory_basis(
    available: Option<u64>,
    physical: Option<u64>,
    semantics: HardwareMemorySemantics,
) -> Option<u64> {
    available.or_else(|| {
        (semantics == HardwareMemorySemantics::Unified)
            .then_some(physical)
            .flatten()
    })
}

fn budget(available: Option<u64>, fallback: u64, headroom_percent: u8) -> u64 {
    available
        .map(|bytes| bytes.saturating_mul(u64::from(100 - headroom_percent)) / 100)
        .unwrap_or(fallback)
        .max(1)
}

fn base_candidates(
    device: DevicePlan,
    device_budget: u64,
    host_budget: u64,
    policy: &AutomaticPlannerPolicy,
) -> [ExecutionPlan; 3] {
    let mut resident = ExecutionPlan::fully_resident(device);
    resident.max_mapped_shards = policy.max_mapped_shards;
    let mut layerwise = resident.clone();
    layerwise.residency = ResidencyPlan::LayerwiseHost {
        device_layer_window: policy.device_layer_window,
        device_budget_bytes: Some(device_budget),
        host_budget_bytes: Some(host_budget),
    };
    let mut disk = resident.clone();
    disk.residency = ResidencyPlan::DenseDiskStream {
        device_budget_bytes: device_budget,
        host_budget_bytes: host_budget,
        host_lookahead: usize::from(host_budget > 0) * 2,
        background_queue: usize::from(host_budget > 0) * 2,
    };
    [resident, layerwise, disk]
}

fn apply_bounded_probe<B: AutomaticPlanningBackend>(
    backend: &B,
    path: &std::path::Path,
    plan: &ExecutionPlan,
    budget: u64,
    admission: &mut CandidateAdmission,
    resources: &mut ModelResourceProfile,
    adjacent: bool,
) -> Result<(), AutomaticPlanningError> {
    if !admission.supported {
        return Ok(());
    }
    let requirement = backend.bounded_residency_requirement(path, plan)?;
    if requirement.required_bytes > budget {
        admission.supported = false;
        admission.rejection = Some(format!(
            "device budget {budget} bytes cannot contain {} pinned static bytes plus the depth-{} device window ({} bytes, {} total)",
            requirement.static_bytes,
            requirement.depth,
            requirement.window_bytes,
            requirement.required_bytes
        ));
    }
    resources.pinned_parameter_bytes =
        Observed::exact(requirement.static_bytes, "validated backend parameter plan");
    if adjacent {
        resources.largest_adjacent_execution_groups_bytes =
            Observed::exact(requirement.window_bytes, "validated backend parameter plan");
    } else {
        resources.largest_execution_group_bytes =
            Observed::exact(requirement.window_bytes, "validated backend parameter plan");
    }
    Ok(())
}

fn rejection(admission: &CandidateAdmission) -> &str {
    admission.rejection.as_deref().unwrap_or("not admitted")
}

fn with_expert_cache(mut plan: ExecutionPlan, policy: &AutomaticPlannerPolicy) -> ExecutionPlan {
    let split = |bytes: u64, percent: u8| bytes.saturating_mul(u64::from(percent)) / 100;
    let ordinary_share = 100 - policy.expert_cache_share_percent;
    let (device_budget, host_budget) = match &mut plan.residency {
        ResidencyPlan::FullyResident => (
            policy.device_memory_fallback_bytes,
            policy.host_memory_fallback_bytes,
        ),
        ResidencyPlan::LayerwiseHost {
            device_budget_bytes,
            host_budget_bytes,
            ..
        } => {
            let device = device_budget_bytes.unwrap_or(policy.device_memory_fallback_bytes);
            let host = host_budget_bytes.unwrap_or(policy.host_memory_fallback_bytes);
            *device_budget_bytes = Some(split(device, ordinary_share).max(1));
            *host_budget_bytes = Some(split(host, ordinary_share).max(1));
            (device, host)
        }
        ResidencyPlan::DenseDiskStream {
            device_budget_bytes,
            host_budget_bytes,
            ..
        } => {
            let (device, host) = (*device_budget_bytes, *host_budget_bytes);
            *device_budget_bytes = split(device, ordinary_share).max(1);
            *host_budget_bytes = split(host, ordinary_share).max(1);
            (device, host)
        }
    };
    let scratch = (1_u64 << 30).min(device_budget.max(1));
    plan.expert_cache = Some(ExpertCachePlan {
        device_budget_bytes: Some(split(device_budget, policy.expert_cache_share_percent).max(1)),
        host_budget_bytes: Some(split(host_budget, policy.expert_cache_share_percent).max(1)),
        scratch_bytes: scratch,
        prefill_bank_bytes: scratch,
        eviction_policy: crate::residency::CacheEvictionPolicy::LeastRecentlyUsed,
    });
    plan
}

fn select_feedback_plan<B: AutomaticPlanningBackend>(
    backend: &B,
    request: &AutomaticPlanRequest,
    hardware: &HardwareProfile,
    resources: &ModelResourceProfile,
    policy: &AutomaticPlannerPolicy,
    embedded_layers: Option<usize>,
) -> Result<Option<(ExecutionPlan, usize, f64)>, AutomaticPlanningError> {
    let mut groups: Vec<(ExecutionPlan, Vec<f64>)> = Vec::new();
    for telemetry in &request.prior_telemetry {
        let (Some(plan), Some(prior_hardware), Some(prior_resources)) = (
            telemetry.plan.as_ref(),
            telemetry.hardware.as_ref(),
            telemetry.resources.as_ref(),
        ) else {
            continue;
        };
        if telemetry.schema_version != AUTOMATIC_SCHEMA_VERSION
            || telemetry.generated_tokens < policy.minimum_feedback_tokens
            || plan.device != request.device
            || prior_resources.path != resources.path
            || prior_resources.artifact_format != resources.artifact_format
            || prior_resources.model_family != resources.model_family
            || prior_hardware.operating_system != hardware.operating_system
            || prior_hardware.architecture != hardware.architecture
            || (matches!(plan.drafting, DraftingPlan::Embedded { .. })
                && embedded_layers == Some(0))
        {
            continue;
        }
        let rate = telemetry
            .timing
            .decode_token_rate
            .filter(|value| value.is_finite() && *value > 0.0)
            .or_else(|| {
                (telemetry.timing.token_rate.is_finite() && telemetry.timing.token_rate > 0.0)
                    .then_some(telemetry.timing.token_rate)
            });
        let Some(rate) = rate else { continue };
        if let Some((_, rates)) = groups.iter_mut().find(|(candidate, _)| candidate == plan) {
            rates.push(rate);
        } else {
            groups.push((plan.clone(), vec![rate]));
        }
    }
    let mut accepted = Vec::new();
    for (plan, mut rates) in groups {
        if !backend
            .admit_candidate(&request.model_path, &plan)?
            .supported
        {
            continue;
        }
        rates.sort_by(f64::total_cmp);
        let middle = rates.len() / 2;
        let median = if rates.len() % 2 == 0 {
            (rates[middle - 1] + rates[middle]) / 2.0
        } else {
            rates[middle]
        };
        accepted.push((plan, rates.len(), median));
    }
    Ok(accepted
        .into_iter()
        .max_by(|left, right| left.2.total_cmp(&right.2)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::BackendId;

    struct MockPlanningBackend {
        model_bytes: u64,
        embedded_layers: usize,
    }

    impl Default for MockPlanningBackend {
        fn default() -> Self {
            Self {
                model_bytes: 2 << 30,
                embedded_layers: 0,
            }
        }
    }

    impl AutomaticPlanningBackend for MockPlanningBackend {
        fn backend_id(&self) -> BackendId {
            BackendId::new("mock").unwrap()
        }

        fn discover_hardware(&self) -> Result<HardwareProfile, AutomaticPlanningError> {
            Ok(HardwareProfile {
                schema_version: AUTOMATIC_SCHEMA_VERSION,
                operating_system: "test".into(),
                architecture: "mock".into(),
                logical_cpu_count: Observed::exact(8, "fixture"),
                physical_memory_bytes: Observed::exact(32 << 30, "fixture"),
                available_memory_bytes: Observed::exact(24 << 30, "fixture"),
                physical_memory_semantics: HardwareMemorySemantics::SeparateTiers,
                backends: vec![HardwareBackendProfile {
                    backend: BackendId::new("mock").unwrap(),
                    available: true,
                    detail: None,
                    devices: vec![HardwareDeviceProfile {
                        id: "gpu:0".into(),
                        family: "gpu".into(),
                        index: 0,
                        total_memory_bytes: Observed::exact(16 << 30, "fixture"),
                        available_memory_bytes: Observed::exact(12 << 30, "fixture"),
                    }],
                }],
            })
        }

        fn inspect_resources(
            &self,
            path: &std::path::Path,
        ) -> Result<ModelResourceProfile, AutomaticPlanningError> {
            let mut profile =
                ModelResourceProfile::unmeasured(path.into(), ArtifactFormat::SafeTensors);
            profile.model_family = Some("llama".into());
            profile.embedded_draft_layers =
                Observed::exact(self.embedded_layers, "normalized architecture fixture");
            profile.stored_tensor_bytes = Observed::exact(self.model_bytes, "fixture");
            profile.materialized_parameter_bytes = Observed::exact(self.model_bytes, "fixture");
            Ok(profile)
        }

        fn admit_candidate(
            &self,
            _path: &std::path::Path,
            _plan: &ExecutionPlan,
        ) -> Result<CandidateAdmission, AutomaticPlanningError> {
            Ok(CandidateAdmission {
                supported: true,
                rejection: None,
            })
        }

        fn bounded_residency_requirement(
            &self,
            _path: &std::path::Path,
            _plan: &ExecutionPlan,
        ) -> Result<BoundedResidencyRequirement, AutomaticPlanningError> {
            Ok(BoundedResidencyRequirement {
                static_bytes: 1 << 20,
                window_bytes: 2 << 20,
                required_bytes: 3 << 20,
                depth: 1,
            })
        }
    }

    #[test]
    fn neutral_planner_selects_a_mock_backend_session_plan() {
        let request = AutomaticPlanRequest::new("model", DevicePlan::new("mock", "gpu:0").unwrap());
        let report = AutomaticPlanner::default()
            .plan(&MockPlanningBackend::default(), &request)
            .unwrap();
        assert_eq!(report.plan.device.backend.as_str(), "mock");
        assert_eq!(report.plan.residency, ResidencyPlan::FullyResident);
    }

    #[test]
    fn neutral_planner_selects_bounded_residency_and_embedded_drafting() {
        let request = AutomaticPlanRequest::new("model", DevicePlan::new("mock", "gpu:0").unwrap());
        let report = AutomaticPlanner::default()
            .plan(
                &MockPlanningBackend {
                    model_bytes: 10 << 30,
                    embedded_layers: 2,
                },
                &request,
            )
            .unwrap();
        assert!(matches!(
            report.plan.residency,
            ResidencyPlan::LayerwiseHost { .. }
        ));
        assert!(matches!(
            report.plan.drafting,
            DraftingPlan::Embedded { .. }
        ));
        assert_eq!(
            observed_u64(&report.resources.pinned_parameter_bytes),
            Some(1 << 20)
        );
    }

    #[test]
    fn selected_backend_identity_fails_closed() {
        let request =
            AutomaticPlanRequest::new("model", DevicePlan::new("other", "gpu:0").unwrap());
        assert!(matches!(
            AutomaticPlanner::default().plan(&MockPlanningBackend::default(), &request),
            Err(AutomaticPlanningError::Invalid(message))
                if message.contains("cannot plan device")
        ));
    }

    #[test]
    fn documents_round_trip_without_an_accelerator_runtime() {
        let request = AutomaticPlanRequest::new("model", DevicePlan::new("mock", "gpu:0").unwrap());
        let encoded = serde_json::to_vec(&request).unwrap();
        assert_eq!(
            serde_json::from_slice::<AutomaticPlanRequest>(&encoded).unwrap(),
            request
        );
        let unavailable = serde_json::to_value(Observed::<u64>::unavailable("unknown")).unwrap();
        assert!(unavailable.get("value").is_none());
    }

    #[test]
    fn tokenizer_compatibility_requires_identical_vocabularies() {
        let fingerprint = [7; 32];
        let proof = TokenizerCompatibilityProof::prove(fingerprint, fingerprint).unwrap();
        assert_eq!(proof.fingerprint(), fingerprint);
        assert_eq!(proof.validate_target(fingerprint), Ok(()));
        assert_eq!(
            proof.validate_target([8; 32]),
            Err(TokenizerCompatibilityError)
        );
        assert_eq!(
            TokenizerCompatibilityProof::prove(fingerprint, [8; 32]),
            Err(TokenizerCompatibilityError)
        );
    }

    #[test]
    fn zero_duration_rates_are_finite() {
        let timing = TimingTelemetry::new(
            Duration::ZERO,
            Duration::ZERO,
            Some(Duration::ZERO),
            3,
            Duration::ZERO,
        );
        assert_eq!(timing.token_rate, 0.0);
        assert_eq!(timing.decode_token_rate, Some(0.0));
    }
}
