//! Hardware, resource, execution-plan, and telemetry schemas for automatic tuning.
//!
//! These types deliberately distinguish exact observations from estimates and
//! unavailable data. Automatic planners must not turn a missing device-memory
//! query or an unknown materialized weight size into a zero-byte value.

use std::{
    fs,
    path::{Path, PathBuf},
    time::Duration,
};

use safemlx::{Device, DeviceType, Stream};
use serde::{Deserialize, Serialize};

use super::{
    available_memory, inspect_model, load_model_with_options, ArtifactKind, CapabilityValue,
    InspectionSeverity, ModelInspectionOptions, ModelKind, ModelLoadOptions,
    PhysicalMemorySemantics,
};
use crate::runtime::{
    checkpoint::quantization::{AffineQuantization, WeightQuantization},
    execution::layerwise::{
        LayerwiseLoadOptions, LayerwiseModelError, NonExpertWeightResidency, WeightResidency,
    },
    residency::{dense_stream::DenseDiskStreamLoadOptions, expert_cache::ExpertCacheLoadOptions},
};
use crate::{
    core::{
        residency::{MemoryTier, OffloadConfig, TransferDirection},
        BackendId, DevicePlan, DraftingPlan, ExecutionPlan, ExpertCachePlan, MtpStats,
        ResidencyPlan, WeightTransformationPlan,
    },
    error::Error,
};

/// Schema version shared by automatic-planning and telemetry documents.
pub const AUTOMATIC_SCHEMA_VERSION: u32 = crate::core::EXECUTION_PLAN_SCHEMA_VERSION;

fn mlx_backend_id() -> BackendId {
    BackendId::new("mlx").expect("the MLX backend identifier is valid")
}

#[cfg(test)]
fn mlx_device_plan(family: &str, index: usize) -> DevicePlan {
    DevicePlan::new("mlx", format!("{family}:{index}"))
        .expect("MLX device identifiers are non-empty")
}

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
}

impl Observed<u64> {
    fn from_capability(value: &CapabilityValue<u64>) -> Self {
        match value {
            CapabilityValue::Available {
                value,
                kind,
                source,
            } => Self::Available {
                value: *value,
                kind: match kind {
                    super::MeasurementKind::Exact => ObservationKind::Exact,
                    super::MeasurementKind::Conservative => ObservationKind::Conservative,
                    super::MeasurementKind::Observational => ObservationKind::Observational,
                    super::MeasurementKind::Estimated => ObservationKind::Estimated,
                },
                source: (*source).into(),
            },
            CapabilityValue::Unsupported { reason } => Self::unsupported(reason.clone()),
            CapabilityValue::Unavailable { reason } => Self::unavailable(reason.clone()),
        }
    }
}

/// Header-only model resource accounting used before a model is loaded.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModelResourceProfile {
    /// Version of this serialized resource schema.
    pub schema_version: u32,
    /// Inspected checkpoint path.
    pub path: PathBuf,
    /// Physical checkpoint container.
    pub artifact_kind: ArtifactKind,
    /// Resolved model family, when architecture inspection succeeded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_kind: Option<ModelKind>,
    /// Resolved architecture name, when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub architecture: Option<String>,
    /// Number of logical tensors exposed by the checkpoint catalog.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensor_count: Option<usize>,
    /// Number of physical checkpoint shards.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checkpoint_shards: Option<usize>,
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
    pub(crate) fn empty(path: PathBuf, artifact_kind: ArtifactKind) -> Self {
        let unavailable = || {
            Observed::unavailable("resource value requires a validated checkpoint parameter plan")
        };
        Self {
            schema_version: AUTOMATIC_SCHEMA_VERSION,
            path,
            artifact_kind,
            model_kind: None,
            architecture: None,
            tensor_count: None,
            checkpoint_shards: None,
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

/// One logical device visible to SafeMLX.
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
    pub backend: BackendId,
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
    /// Execution backends visible to this SafeMLX build.
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

impl From<PhysicalMemorySemantics> for HardwareMemorySemantics {
    fn from(value: PhysicalMemorySemantics) -> Self {
        match value {
            PhysicalMemorySemantics::Unified => Self::Unified,
            PhysicalMemorySemantics::SeparateTiers => Self::SeparateTiers,
            PhysicalMemorySemantics::Unknown => Self::Unknown,
        }
    }
}

/// Discovers hardware facts available through the current SafeMLX build.
///
/// Missing accelerator enumeration or VRAM APIs remain explicit unavailable
/// values. In particular, system RAM is never reported as CUDA device memory.
pub fn discover_hardware() -> HardwareProfile {
    let logical_cpu_count = std::thread::available_parallelism().map_or_else(
        |error| Observed::unavailable(error.to_string()),
        |count| Observed::exact(count.get() as u64, "std::thread::available_parallelism"),
    );
    let (physical_memory_bytes, available_memory_bytes, semantics) = match available_memory() {
        Ok(memory) => (
            Observed::from_capability(&memory.physical_memory_bytes),
            Observed::from_capability(&memory.available_memory_bytes),
            memory.physical_semantics.into(),
        ),
        Err(error) => (
            Observed::unavailable(error.to_string()),
            Observed::unavailable(error.to_string()),
            HardwareMemorySemantics::Unknown,
        ),
    };

    let mut devices = vec![HardwareDeviceProfile {
        id: "cpu:0".into(),
        family: "cpu".into(),
        index: 0,
        total_memory_bytes: physical_memory_bytes.clone(),
        available_memory_bytes: available_memory_bytes.clone(),
    }];
    let mut discovery_details = Vec::new();

    #[cfg(target_os = "macos")]
    {
        let (available, detail) = match safemlx::metal::is_available() {
            Ok(available) => (available, None),
            Err(error) => (false, Some(error.to_string())),
        };
        let (device_total, device_available) = if semantics == HardwareMemorySemantics::Unified {
            (
                physical_memory_bytes.clone(),
                available_memory_bytes.clone(),
            )
        } else {
            (
                Observed::unavailable(
                    "SafeMLX does not expose discrete Metal device-memory discovery",
                ),
                Observed::unavailable(
                    "SafeMLX does not expose discrete Metal device-memory discovery",
                ),
            )
        };
        if available {
            devices.push(HardwareDeviceProfile {
                id: "metal:0".into(),
                family: "metal".into(),
                index: 0,
                total_memory_bytes: device_total,
                available_memory_bytes: device_available,
            });
        } else if let Some(detail) = detail {
            discovery_details.push(format!("Metal: {detail}"));
        }
    }

    #[cfg(feature = "cuda")]
    {
        let (available, detail) = match safemlx::cuda::is_available() {
            Ok(available) => (available, None),
            Err(error) => (false, Some(error.to_string())),
        };
        if available {
            devices.push(HardwareDeviceProfile {
                id: "cuda:0".into(),
                family: "cuda".into(),
                index: 0,
                total_memory_bytes: Observed::unavailable(
                    "SafeMLX does not yet expose CUDA device-memory discovery",
                ),
                available_memory_bytes: Observed::unavailable(
                    "SafeMLX does not yet expose CUDA device-memory discovery",
                ),
            });
        } else if let Some(detail) = detail {
            discovery_details.push(format!("CUDA: {detail}"));
        }
    }

    let backends = vec![HardwareBackendProfile {
        backend: mlx_backend_id(),
        available: true,
        detail: (!discovery_details.is_empty()).then(|| discovery_details.join("; ")),
        devices,
    }];

    HardwareProfile {
        schema_version: AUTOMATIC_SCHEMA_VERSION,
        operating_system: std::env::consts::OS.into(),
        architecture: std::env::consts::ARCH.into(),
        logical_cpu_count,
        physical_memory_bytes,
        available_memory_bytes,
        physical_memory_semantics: semantics,
        backends,
    }
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
///
/// Keeping inputs beside the selected plan makes cached decisions auditable
/// and gives applications one versioned payload to exchange with a tuner.
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

/// Tunable bounds used by [`AutomaticPlanner`].
///
/// The defaults implement the conservative single-device policy used by the
/// example CLI. Applications may persist this value beside planning requests
/// so later policy changes remain explicit and reproducible.
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
            max_mapped_shards: crate::core::DEFAULT_MAX_MAPPED_SHARDS,
            embedded_mtp_draft_tokens: 3,
            minimum_feedback_tokens: 1,
        }
    }
}

/// Owned input to one automatic planning session.
///
/// `prior_telemetry` is deliberately the same stable telemetry document
/// emitted by the runtime. A host application can persist completed runs and
/// submit them on a later launch without translating them into planner-private
/// scores or cache records.
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

/// Stable automatic planner facade intended for embedding applications.
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

    /// Inspects a model and selects one currently loadable execution plan.
    ///
    /// Matching prior telemetry is grouped by exact execution plan and ranked
    /// by median post-first-token decode rate, falling back to overall token
    /// rate. A historical plan is considered only when its model and stable
    /// hardware identity still match, it remains within current memory bounds,
    /// and header-only inspection still admits its load policy.
    pub fn plan(&self, request: &AutomaticPlanRequest) -> Result<ExecutionPlanReport, Error> {
        plan_automatic_execution_with_policy(request, &self.policy)
    }
}

/// Plans one execution using [`AutomaticPlannerPolicy::default`].
pub fn plan_automatic_execution(
    request: &AutomaticPlanRequest,
) -> Result<ExecutionPlanReport, Error> {
    AutomaticPlanner::default().plan(request)
}

/// Converts a serialized execution plan into loader options.
///
/// Device selection and speculative decoding streams remain caller-owned;
/// this conversion applies weight transformation and residency choices.
pub fn execution_plan_load_options(plan: &ExecutionPlan) -> Result<ModelLoadOptions, Error> {
    if plan.schema_version != AUTOMATIC_SCHEMA_VERSION {
        return Err(Error::AutomaticPlanning(format!(
            "execution plan schema {} does not match supported schema {}",
            plan.schema_version, AUTOMATIC_SCHEMA_VERSION
        )));
    }
    if plan.topology.world_size() != 1 {
        return Err(Error::AutomaticPlanning(
            "single-device automatic plans require a 1x1x1 parallel topology".into(),
        ));
    }
    let mut load = match plan.weight_transformation {
        WeightTransformationPlan::PreserveCheckpoint => ModelLoadOptions::default(),
        WeightTransformationPlan::Affine { bits, group_size } => {
            ModelLoadOptions::with_quantization(AffineQuantization::new(group_size, bits)?)
        }
        WeightTransformationPlan::MxFp4 => {
            ModelLoadOptions::with_quantization(WeightQuantization::MxFp4)
        }
    };
    let residency = match &plan.residency {
        ResidencyPlan::FullyResident => NonExpertWeightResidency::FullyResident,
        ResidencyPlan::LayerwiseHost {
            device_layer_window,
            device_budget_bytes,
            host_budget_bytes,
        } => NonExpertWeightResidency::LayerwiseHost(LayerwiseLoadOptions {
            offload: OffloadConfig::new(
                *device_budget_bytes,
                *host_budget_bytes,
                *device_layer_window,
            )?,
            max_mapped_shards: plan.max_mapped_shards,
            ..LayerwiseLoadOptions::default()
        }),
        ResidencyPlan::DenseDiskStream {
            device_budget_bytes,
            host_budget_bytes,
            host_lookahead,
            background_queue,
        } => {
            let mut options = DenseDiskStreamLoadOptions::new(
                *device_budget_bytes,
                *host_budget_bytes,
                *host_lookahead,
                *background_queue,
            )?;
            options.max_mapped_shards = plan.max_mapped_shards;
            NonExpertWeightResidency::DenseDiskStream(options)
        }
    };
    let residency = if let Some(expert) = &plan.expert_cache {
        let experts = OffloadConfig::new(expert.device_budget_bytes, expert.host_budget_bytes, 1)?;
        WeightResidency::with_expert_cache(
            residency,
            ExpertCacheLoadOptions::new(experts, expert.scratch_bytes, expert.prefill_bank_bytes)?,
        )
    } else {
        match residency {
            NonExpertWeightResidency::FullyResident => WeightResidency::fully_resident(),
            NonExpertWeightResidency::LayerwiseHost(options) => {
                WeightResidency::layerwise_host(options)
            }
            NonExpertWeightResidency::DenseDiskStream(options) => {
                WeightResidency::dense_disk_stream(options)
            }
        }
    };
    load = load.with_weight_residency(residency);
    Ok(load)
}

/// Timings reported for one CLI or server generation request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimingTelemetry {
    /// Model load duration in seconds.
    pub load_seconds: f64,
    /// Generation duration in seconds.
    pub generation_seconds: f64,
    /// Time to the first emitted token in seconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_to_first_token_seconds: Option<f64>,
    /// Complete process operation duration in seconds.
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
        let token_rate = rate(generated_tokens, generation);
        let decode_token_rate = time_to_first_token.map(|first| {
            rate(
                generated_tokens.saturating_sub(1),
                generation.saturating_sub(first),
            )
        });
        Self {
            load_seconds: load.as_secs_f64(),
            generation_seconds: generation.as_secs_f64(),
            time_to_first_token_seconds: time_to_first_token.map(|value| value.as_secs_f64()),
            total_seconds: total.as_secs_f64(),
            token_rate,
            decode_token_rate,
        }
    }
}

fn rate(tokens: usize, elapsed: Duration) -> f64 {
    if elapsed.is_zero() {
        0.0
    } else {
        tokens as f64 / elapsed.as_secs_f64()
    }
}

/// MLX allocator observations for one execution.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct AllocatorTelemetry {
    /// Peak active MLX-managed allocation bytes.
    pub peak_bytes: u64,
    /// Active MLX-managed allocation bytes at collection time.
    pub active_bytes: u64,
    /// Bytes retained by MLX's allocator cache at collection time.
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

impl From<&MtpStats> for MtpTelemetry {
    fn from(stats: &MtpStats) -> Self {
        Self {
            execution_topology: stats.execution_topology.to_string(),
            target_tokens: stats.target_tokens,
            draft_tokens: stats.draft_tokens,
            accepted_tokens: stats.accepted_tokens,
            accept_rate: stats.accept_rate(),
            rounds: stats.rounds,
            accept_lens: stats.accept_lens.clone(),
            emitted_tokens: stats.emitted_tokens,
            optimistic_draft_tokens: stats.optimistic_draft_tokens,
            reused_optimistic_tokens: stats.reused_optimistic_tokens,
            discarded_optimistic_tokens: stats.discarded_optimistic_tokens,
            adaptive_lookahead_disabled: stats.adaptive_lookahead_disabled,
            optimistic_draft_seconds: stats.optimistic_draft_time.as_secs_f64(),
            verification_in_flight_seconds: stats.verification_in_flight_time.as_secs_f64(),
        }
    }
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
    /// Emitted token count after CLI terminal-token normalization.
    pub generated_tokens: usize,
    /// Stable completion reason.
    pub stop_reason: String,
    /// Load and generation timings.
    pub timing: TimingTelemetry,
    /// MLX allocator observations.
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

#[derive(Debug)]
struct AutomaticCandidateAdmission {
    supported: bool,
    rejection: Option<String>,
}

fn automatic_observed_u64(value: &Observed<u64>) -> Option<u64> {
    match value {
        Observed::Available { value, .. } => Some(*value),
        Observed::Unsupported { .. } | Observed::Unavailable { .. } => None,
    }
}

fn validate_automatic_policy(policy: &AutomaticPlannerPolicy) -> Result<(), Error> {
    if policy.device_memory_fallback_bytes == 0 || policy.host_memory_fallback_bytes == 0 {
        return Err(Error::AutomaticPlanning(
            "automatic fallback memory budgets must be greater than zero".into(),
        ));
    }
    if policy.memory_headroom_percent >= 100 {
        return Err(Error::AutomaticPlanning(
            "automatic memory headroom must be less than 100 percent".into(),
        ));
    }
    if policy.expert_cache_share_percent == 0 || policy.expert_cache_share_percent >= 100 {
        return Err(Error::AutomaticPlanning(
            "automatic expert-cache share must be between 1 and 99 percent".into(),
        ));
    }
    if policy.device_layer_window == 0
        || policy.max_mapped_shards == 0
        || policy.embedded_mtp_draft_tokens == 0
        || policy.minimum_feedback_tokens == 0
    {
        return Err(Error::AutomaticPlanning(
            "automatic count and token policy values must be greater than zero".into(),
        ));
    }
    Ok(())
}

fn automatic_budget(available: Option<u64>, fallback: u64, headroom_percent: u8) -> u64 {
    available
        .map(|bytes| bytes.saturating_mul(u64::from(100 - headroom_percent)) / 100)
        .unwrap_or(fallback)
        .max(1)
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum AutomaticMemoryBasis {
    Available,
    UnifiedPhysicalCapacity,
    PolicyFallback,
}

fn automatic_memory_basis(
    available: Option<u64>,
    physical: Option<u64>,
    semantics: HardwareMemorySemantics,
) -> (Option<u64>, AutomaticMemoryBasis) {
    if let Some(available) = available {
        (Some(available), AutomaticMemoryBasis::Available)
    } else if semantics == HardwareMemorySemantics::Unified {
        physical.map_or((None, AutomaticMemoryBasis::PolicyFallback), |physical| {
            (
                Some(physical),
                AutomaticMemoryBasis::UnifiedPhysicalCapacity,
            )
        })
    } else {
        (None, AutomaticMemoryBasis::PolicyFallback)
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct AutomaticBoundedRequirement {
    static_bytes: u64,
    window_bytes: u64,
    required_bytes: u64,
    depth: usize,
}

fn automatic_probe_device(device: &DevicePlan) -> Result<Device, Error> {
    if device.backend.as_str() != "mlx" {
        return Err(Error::AutomaticPlanning(format!(
            "MLX automatic planning cannot probe backend {}",
            device.backend
        )));
    }
    let (family, index) = device.device.split_once(':').ok_or_else(|| {
        Error::AutomaticPlanning(format!(
            "MLX device identifier {:?} must use family:index syntax",
            device.device
        ))
    })?;
    let index = index.parse::<usize>().map_err(|_| {
        Error::AutomaticPlanning(format!(
            "MLX device identifier {:?} has an invalid index",
            device.device
        ))
    })?;
    let index = i32::try_from(index).map_err(|_| {
        Error::AutomaticPlanning(format!(
            "automatic device index {index} exceeds the MLX device-index range"
        ))
    })?;
    let kind = match family {
        "cpu" => DeviceType::Cpu,
        "metal" | "cuda" | "gpu" => DeviceType::Gpu,
        other => {
            return Err(Error::AutomaticPlanning(format!(
                "MLX automatic planning does not recognize device family {other:?}"
            )))
        }
    };
    Ok(Device::new(kind, index))
}

fn probe_automatic_bounded_requirement(
    model_path: &Path,
    plan: &ExecutionPlan,
) -> Result<Result<AutomaticBoundedRequirement, String>, Error> {
    let mut probe = plan.clone();
    probe.expert_cache = None;
    match &mut probe.residency {
        ResidencyPlan::LayerwiseHost {
            device_budget_bytes,
            ..
        } => *device_budget_bytes = Some(1),
        ResidencyPlan::DenseDiskStream {
            device_budget_bytes,
            ..
        } => *device_budget_bytes = 1,
        ResidencyPlan::FullyResident => {
            return Ok(Err(
                "fully resident execution has no bounded device-window requirement".into(),
            ));
        }
    }
    let stream = Stream::new_with_device(&automatic_probe_device(&probe.device)?);
    let weights_stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    match load_model_with_options(
        model_path,
        execution_plan_load_options(&probe)?,
        &stream,
        &weights_stream,
    ) {
        Err(Error::LayerwiseModel(LayerwiseModelError::DeviceBudgetTooSmall {
            static_bytes,
            window_bytes,
            depth,
            required,
            ..
        })) => Ok(Ok(AutomaticBoundedRequirement {
            static_bytes,
            window_bytes,
            required_bytes: required,
            depth,
        })),
        Err(error) => Ok(Err(error.to_string())),
        Ok(_) => Ok(Ok(AutomaticBoundedRequirement {
            static_bytes: 0,
            window_bytes: 0,
            required_bytes: 1,
            depth: 0,
        })),
    }
}

fn apply_automatic_bounded_requirement(
    admission: &mut AutomaticCandidateAdmission,
    requirement: Result<AutomaticBoundedRequirement, String>,
    budget: u64,
) -> Option<AutomaticBoundedRequirement> {
    if !admission.supported {
        return None;
    }
    match requirement {
        Ok(requirement) if requirement.required_bytes <= budget => Some(requirement),
        Ok(requirement) => {
            admission.supported = false;
            admission.rejection = Some(format!(
                "device budget {budget} bytes cannot contain {} pinned static bytes plus the depth-{} device window ({} bytes, {} total)",
                requirement.static_bytes,
                requirement.depth,
                requirement.window_bytes,
                requirement.required_bytes,
            ));
            Some(requirement)
        }
        Err(rejection) => {
            admission.supported = false;
            admission.rejection = Some(rejection);
            None
        }
    }
}

fn selected_device_profile<'a>(
    hardware: &'a HardwareProfile,
    device: &DevicePlan,
) -> Option<&'a HardwareDeviceProfile> {
    hardware
        .backends
        .iter()
        .find(|backend| backend.backend == device.backend && backend.available)
        .and_then(|backend| {
            backend
                .devices
                .iter()
                .find(|candidate| candidate.id == device.device)
        })
}

fn validate_automatic_device(hardware: &HardwareProfile, device: &DevicePlan) -> Result<(), Error> {
    let backend = hardware
        .backends
        .iter()
        .find(|backend| backend.backend == device.backend)
        .ok_or_else(|| {
            Error::AutomaticPlanning(format!(
                "hardware discovery did not report the selected {} backend",
                device.backend
            ))
        })?;
    if !backend.available {
        return Err(Error::AutomaticPlanning(format!(
            "selected {} backend is unavailable: {}",
            device.backend,
            backend.detail.as_deref().unwrap_or("no detail reported")
        )));
    }
    if !backend
        .devices
        .iter()
        .any(|candidate| candidate.id == device.device)
    {
        return Err(Error::AutomaticPlanning(format!(
            "hardware discovery did not report {} device {}",
            device.backend, device.device
        )));
    }
    Ok(())
}

fn automatic_model_bytes(resources: &ModelResourceProfile) -> Option<(u64, &'static str)> {
    automatic_observed_u64(&resources.materialized_parameter_bytes)
        .map(|bytes| (bytes, "materialized parameter estimate"))
        .or_else(|| {
            automatic_observed_u64(&resources.stored_tensor_bytes)
                .map(|bytes| (bytes, "stored checkpoint tensor bytes"))
        })
}

fn automatic_base_candidates(
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

fn inspect_automatic_candidate(
    model_path: &Path,
    plan: &ExecutionPlan,
) -> Result<AutomaticCandidateAdmission, Error> {
    let report = inspect_model(
        model_path,
        ModelInspectionOptions {
            load: execution_plan_load_options(plan)?,
            chat_request: None,
        },
    )?;
    let supported = report.is_loadable();
    let rejection = (!supported).then(|| {
        report
            .issues
            .iter()
            .find(|issue| issue.severity == InspectionSeverity::Error)
            .map(|issue| issue.detail.clone())
            .unwrap_or_else(|| "checkpoint inspection did not admit this load policy".into())
    });
    Ok(AutomaticCandidateAdmission {
        supported,
        rejection,
    })
}

fn automatic_embedded_mtp_count(value: &serde_json::Value) -> Option<u64> {
    match value {
        serde_json::Value::Object(object) => {
            for key in ["mtp_num_hidden_layers", "num_nextn_predict_layers"] {
                if let Some(count) = object.get(key).and_then(serde_json::Value::as_u64) {
                    return Some(count);
                }
            }
            object.values().find_map(automatic_embedded_mtp_count)
        }
        serde_json::Value::Array(values) => values.iter().find_map(automatic_embedded_mtp_count),
        _ => None,
    }
}

fn automatic_embedded_mtp_layers(
    model_path: &Path,
    kind: Option<ModelKind>,
) -> Result<Option<usize>, Error> {
    if !matches!(
        kind,
        Some(
            ModelKind::DeepSeekV3
                | ModelKind::Inkling
                | ModelKind::NemotronH
                | ModelKind::Qwen3Next
                | ModelKind::Qwen35
        )
    ) {
        return Ok(Some(0));
    }
    if !model_path.is_dir() {
        return Ok(None);
    }
    let config: serde_json::Value =
        serde_json::from_slice(&fs::read(model_path.join("config.json"))?)?;
    automatic_embedded_mtp_count(&config)
        .map(|count| {
            usize::try_from(count).map_err(|_| {
                Error::AutomaticPlanning("embedded MTP layer count exceeds usize".into())
            })
        })
        .transpose()
        .map(|count| Some(count.unwrap_or(0)))
}

fn automatic_with_expert_cache(
    mut plan: ExecutionPlan,
    policy: &AutomaticPlannerPolicy,
) -> ExecutionPlan {
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
            let device = *device_budget_bytes;
            let host = *host_budget_bytes;
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
    });
    plan
}

fn choose_automatic_residency(
    resident_fits: bool,
    layerwise_fits: bool,
    disk_fits: bool,
    resident_supported: bool,
    layerwise_supported: bool,
    disk_supported: bool,
) -> Option<usize> {
    if resident_fits && resident_supported {
        Some(0)
    } else if layerwise_fits && layerwise_supported {
        Some(1)
    } else if disk_fits && disk_supported {
        Some(2)
    } else {
        None
    }
}

fn automatic_plan_resource_admitted(
    hardware: &HardwareProfile,
    resources: &ModelResourceProfile,
    plan: &ExecutionPlan,
    policy: &AutomaticPlannerPolicy,
) -> bool {
    let selected_device = selected_device_profile(hardware, &plan.device);
    let observed_device =
        selected_device.and_then(|device| automatic_observed_u64(&device.available_memory_bytes));
    let observed_host = automatic_observed_u64(&hardware.available_memory_bytes);
    let physical_device = selected_device
        .and_then(|device| automatic_observed_u64(&device.total_memory_bytes))
        .or_else(|| automatic_observed_u64(&hardware.physical_memory_bytes));
    let physical_host = automatic_observed_u64(&hardware.physical_memory_bytes);
    let (device_capacity, _) = automatic_memory_basis(
        observed_device,
        physical_device,
        hardware.physical_memory_semantics,
    );
    let (host_capacity, _) = automatic_memory_basis(
        observed_host,
        physical_host,
        hardware.physical_memory_semantics,
    );
    let device_limit = automatic_budget(
        device_capacity,
        policy.device_memory_fallback_bytes,
        policy.memory_headroom_percent,
    );
    let host_limit = automatic_budget(
        host_capacity,
        policy.host_memory_fallback_bytes,
        policy.memory_headroom_percent,
    );
    if matches!(plan.residency, ResidencyPlan::FullyResident) {
        return automatic_model_bytes(resources).is_some_and(|(bytes, _)| bytes <= device_limit);
    }
    let (mut device_required, mut host_required) = match plan.residency {
        ResidencyPlan::LayerwiseHost {
            device_budget_bytes: Some(device),
            host_budget_bytes: Some(host),
            ..
        } => (device, host),
        ResidencyPlan::DenseDiskStream {
            device_budget_bytes,
            host_budget_bytes,
            ..
        } => (device_budget_bytes, host_budget_bytes),
        ResidencyPlan::LayerwiseHost { .. } | ResidencyPlan::FullyResident => return false,
    };
    if let Some(expert) = &plan.expert_cache {
        let (Some(device), Some(host)) = (expert.device_budget_bytes, expert.host_budget_bytes)
        else {
            return false;
        };
        device_required = device_required.saturating_add(device);
        host_required = host_required.saturating_add(host);
    }
    if hardware.physical_memory_semantics == HardwareMemorySemantics::Unified {
        let unified_limit = device_limit.max(host_limit);
        device_required <= unified_limit && host_required <= unified_limit
    } else {
        device_required <= device_limit && host_required <= host_limit
    }
}

fn canonical_path(path: &Path) -> PathBuf {
    fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

fn automatic_telemetry_matches(
    telemetry: &ExecutionTelemetry,
    hardware: &HardwareProfile,
    resources: &ModelResourceProfile,
    device: DevicePlan,
    minimum_tokens: usize,
) -> bool {
    if telemetry.schema_version != AUTOMATIC_SCHEMA_VERSION
        || telemetry.generated_tokens < minimum_tokens
    {
        return false;
    }
    let (Some(plan), Some(prior_hardware), Some(prior_resources)) = (
        telemetry.plan.as_ref(),
        telemetry.hardware.as_ref(),
        telemetry.resources.as_ref(),
    ) else {
        return false;
    };
    if plan.schema_version != AUTOMATIC_SCHEMA_VERSION
        || plan.device != device
        || plan.topology.world_size() != 1
        || matches!(plan.drafting, DraftingPlan::External { .. })
    {
        return false;
    }
    let current_total = selected_device_profile(hardware, &device)
        .and_then(|item| automatic_observed_u64(&item.total_memory_bytes));
    let prior_total = selected_device_profile(prior_hardware, &device)
        .and_then(|item| automatic_observed_u64(&item.total_memory_bytes));
    canonical_path(&prior_resources.path) == canonical_path(&resources.path)
        && prior_resources.artifact_kind == resources.artifact_kind
        && prior_resources.model_kind == resources.model_kind
        && prior_resources.architecture == resources.architecture
        && prior_resources.tensor_count == resources.tensor_count
        && prior_resources.checkpoint_shards == resources.checkpoint_shards
        && automatic_observed_u64(&prior_resources.stored_tensor_bytes)
            == automatic_observed_u64(&resources.stored_tensor_bytes)
        && prior_hardware.operating_system == hardware.operating_system
        && prior_hardware.architecture == hardware.architecture
        && prior_hardware.physical_memory_semantics == hardware.physical_memory_semantics
        && automatic_observed_u64(&prior_hardware.physical_memory_bytes)
            == automatic_observed_u64(&hardware.physical_memory_bytes)
        && prior_total == current_total
}

fn automatic_feedback_rate(telemetry: &ExecutionTelemetry) -> Option<f64> {
    telemetry
        .timing
        .decode_token_rate
        .filter(|rate| rate.is_finite() && *rate > 0.0)
        .or_else(|| {
            let rate = telemetry.timing.token_rate;
            (rate.is_finite() && rate > 0.0).then_some(rate)
        })
}

fn automatic_median(mut values: Vec<f64>) -> Option<f64> {
    values.retain(|value| value.is_finite());
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Some(if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    })
}

fn automatic_feedback_plan(
    model_path: &Path,
    request: &AutomaticPlanRequest,
    hardware: &HardwareProfile,
    resources: &ModelResourceProfile,
    policy: &AutomaticPlannerPolicy,
    embedded_mtp_layers: Option<usize>,
) -> Result<Option<(ExecutionPlan, usize, f64)>, Error> {
    let mut groups: Vec<(ExecutionPlan, Vec<f64>)> = Vec::new();
    for telemetry in &request.prior_telemetry {
        if !automatic_telemetry_matches(
            telemetry,
            hardware,
            resources,
            request.device.clone(),
            policy.minimum_feedback_tokens,
        ) {
            continue;
        }
        let (Some(plan), Some(rate)) =
            (telemetry.plan.as_ref(), automatic_feedback_rate(telemetry))
        else {
            continue;
        };
        if matches!(plan.drafting, DraftingPlan::Embedded { .. }) && embedded_mtp_layers == Some(0)
        {
            continue;
        }
        if let Some((_, rates)) = groups.iter_mut().find(|(candidate, _)| candidate == plan) {
            rates.push(rate);
        } else {
            groups.push((plan.clone(), vec![rate]));
        }
    }
    let mut admitted = Vec::new();
    for (plan, rates) in groups {
        if !automatic_plan_resource_admitted(hardware, resources, &plan, policy)
            || !matches!(
                inspect_automatic_candidate(model_path, &plan),
                Ok(AutomaticCandidateAdmission {
                    supported: true,
                    ..
                })
            )
        {
            continue;
        }
        let samples = rates.len();
        if let Some(median) = automatic_median(rates) {
            admitted.push((plan, samples, median));
        }
    }
    Ok(admitted
        .into_iter()
        .max_by(|left, right| left.2.total_cmp(&right.2)))
}

fn plan_automatic_execution_with_policy(
    request: &AutomaticPlanRequest,
    policy: &AutomaticPlannerPolicy,
) -> Result<ExecutionPlanReport, Error> {
    if request.schema_version != AUTOMATIC_SCHEMA_VERSION {
        return Err(Error::AutomaticPlanning(format!(
            "automatic request schema {} does not match supported schema {}",
            request.schema_version, AUTOMATIC_SCHEMA_VERSION
        )));
    }
    validate_automatic_policy(policy)?;
    let hardware = discover_hardware();
    validate_automatic_device(&hardware, &request.device)?;
    let mut resources =
        inspect_model(&request.model_path, ModelInspectionOptions::default())?.resources;
    let selected_device = selected_device_profile(&hardware, &request.device);
    let device_available =
        selected_device.and_then(|device| automatic_observed_u64(&device.available_memory_bytes));
    let host_available = automatic_observed_u64(&hardware.available_memory_bytes);
    let device_physical = selected_device
        .and_then(|device| automatic_observed_u64(&device.total_memory_bytes))
        .or_else(|| automatic_observed_u64(&hardware.physical_memory_bytes));
    let host_physical = automatic_observed_u64(&hardware.physical_memory_bytes);
    let (device_capacity, device_memory_basis) = automatic_memory_basis(
        device_available,
        device_physical,
        hardware.physical_memory_semantics,
    );
    let (host_capacity, _) = automatic_memory_basis(
        host_available,
        host_physical,
        hardware.physical_memory_semantics,
    );
    let device_budget = automatic_budget(
        device_capacity,
        policy.device_memory_fallback_bytes,
        policy.memory_headroom_percent,
    );
    let host_budget = automatic_budget(
        host_capacity,
        policy.host_memory_fallback_bytes,
        policy.memory_headroom_percent,
    );
    let model_bytes = automatic_model_bytes(&resources);
    let candidates =
        automatic_base_candidates(request.device.clone(), device_budget, host_budget, policy);
    let resident = inspect_automatic_candidate(&request.model_path, &candidates[0])?;
    let mut layerwise = inspect_automatic_candidate(&request.model_path, &candidates[1])?;
    let mut disk = inspect_automatic_candidate(&request.model_path, &candidates[2])?;
    let mut entries = vec![PlanExplanationEntry {
        level: PlanExplanationLevel::Decision,
        code: "single_device_scope".into(),
        detail: format!(
            "automatic planning is restricted to {}:{} with {}% memory headroom",
            request.device.backend, request.device.device, policy.memory_headroom_percent
        ),
    }];
    let resident_fits = match model_bytes {
        Some((bytes, model_basis)) => {
            let (level, code, memory_detail) = match device_memory_basis {
                AutomaticMemoryBasis::Available => (
                    PlanExplanationLevel::Decision,
                    "resource_basis",
                    format!(
                        "a {device_budget}-byte planning budget derived from {} currently available bytes",
                        device_capacity.expect("available memory basis has a capacity")
                    ),
                ),
                AutomaticMemoryBasis::UnifiedPhysicalCapacity => (
                    PlanExplanationLevel::Warning,
                    "available_memory_unavailable",
                    format!(
                        "a {device_budget}-byte planning budget derived from {} bytes of unified physical memory after reserving {}% because live availability was unavailable",
                        device_capacity.expect("physical memory basis has a capacity"),
                        policy.memory_headroom_percent
                    ),
                ),
                AutomaticMemoryBasis::PolicyFallback => (
                    PlanExplanationLevel::Warning,
                    "device_memory_unavailable",
                    format!(
                        "the {}-byte policy fallback because device and unified physical memory were unavailable",
                        policy.device_memory_fallback_bytes
                    ),
                ),
            };
            entries.push(PlanExplanationEntry {
                level,
                code: code.into(),
                detail: format!("used {model_basis} ({bytes} bytes) against {memory_detail}"),
            });
            bytes <= device_budget
        }
        None => {
            entries.push(PlanExplanationEntry {
                level: PlanExplanationLevel::Warning,
                code: "model_footprint_unavailable".into(),
                detail: "no materialized or stored checkpoint size was available; bounded loading is preferred".into(),
            });
            false
        }
    };
    let layerwise_host_fits = model_bytes.is_some_and(|(bytes, _)| {
        if hardware.physical_memory_semantics == HardwareMemorySemantics::Unified {
            bytes <= host_budget.saturating_mul(2)
        } else {
            bytes <= host_budget
        }
    });
    let mut disk_fits = false;
    if !(resident_fits && resident.supported) {
        let layerwise_budget = match &candidates[1].residency {
            ResidencyPlan::LayerwiseHost {
                device_budget_bytes: Some(budget),
                ..
            } => *budget,
            _ => 0,
        };
        if layerwise.supported {
            let probe = probe_automatic_bounded_requirement(&request.model_path, &candidates[1])?;
            if let Some(requirement) =
                apply_automatic_bounded_requirement(&mut layerwise, probe, layerwise_budget)
            {
                resources.pinned_parameter_bytes = Observed::exact(
                    requirement.static_bytes,
                    "validated layerwise parameter plan",
                );
                resources.largest_execution_group_bytes = Observed::exact(
                    requirement.window_bytes,
                    "validated depth-1 layerwise parameter plan",
                );
            }
        }
        let disk_budget = match &candidates[2].residency {
            ResidencyPlan::DenseDiskStream {
                device_budget_bytes,
                ..
            } => *device_budget_bytes,
            _ => 0,
        };
        if disk.supported {
            let probe = probe_automatic_bounded_requirement(&request.model_path, &candidates[2])?;
            if let Some(requirement) =
                apply_automatic_bounded_requirement(&mut disk, probe, disk_budget)
            {
                resources.pinned_parameter_bytes = Observed::exact(
                    requirement.static_bytes,
                    "validated dense-stream parameter plan",
                );
                resources.largest_adjacent_execution_groups_bytes = Observed::exact(
                    requirement.window_bytes,
                    "validated depth-2 dense-stream parameter plan",
                );
            }
        }
        disk_fits = disk.supported;
    }
    let layerwise_fits = layerwise_host_fits && layerwise.supported;
    let selected = choose_automatic_residency(
        resident_fits,
        layerwise_fits,
        disk_fits,
        resident.supported,
        layerwise.supported,
        disk.supported,
    )
    .ok_or_else(|| {
        Error::AutomaticPlanning(format!(
            "no loadable single-device policy: resident: {}; layerwise: {}; disk-streamed: {}",
            resident.rejection.as_deref().unwrap_or("not admitted"),
            layerwise.rejection.as_deref().unwrap_or("not admitted"),
            disk.rejection.as_deref().unwrap_or("not admitted")
        ))
    })?;
    if selected != 0 {
        entries.push(PlanExplanationEntry {
            level: PlanExplanationLevel::Rejection,
            code: if resident_fits {
                "resident_unsupported"
            } else {
                "resident_exceeds_quick_budget"
            }
            .into(),
            detail: resident.rejection.clone().unwrap_or_else(|| {
                "fully resident execution exceeds the policy memory budget".into()
            }),
        });
    }
    if selected == 2 {
        entries.push(PlanExplanationEntry {
            level: PlanExplanationLevel::Rejection,
            code: "layerwise_not_admitted".into(),
            detail: layerwise.rejection.clone().unwrap_or_else(|| {
                "the checkpoint exceeds the policy host-backed admission budget".into()
            }),
        });
    }
    let (mut plan, mut summary) = match selected {
        0 => (
            candidates[0].clone(),
            "selected fully resident execution for the lowest expected latency".to_string(),
        ),
        1 => (
            candidates[1].clone(),
            "selected host-backed layerwise execution with a validated bounded device window"
                .to_string(),
        ),
        2 => (
            candidates[2].clone(),
            "selected bounded dense disk streaming because resident and layerwise admission failed"
                .into(),
        ),
        _ => unreachable!("automatic residency candidate index is bounded"),
    };
    entries.push(PlanExplanationEntry {
        level: PlanExplanationLevel::Decision,
        code: match plan.residency {
            ResidencyPlan::FullyResident => "fully_resident_selected",
            ResidencyPlan::LayerwiseHost { .. } => "layerwise_host_selected",
            ResidencyPlan::DenseDiskStream { .. } => "dense_disk_stream_selected",
        }
        .into(),
        detail: summary.clone(),
    });
    if !matches!(plan.residency, ResidencyPlan::FullyResident) {
        let expert_plan = automatic_with_expert_cache(plan.clone(), policy);
        let mut expert = inspect_automatic_candidate(&request.model_path, &expert_plan)?;
        let bounded_required = match &plan.residency {
            ResidencyPlan::LayerwiseHost { .. } => {
                automatic_observed_u64(&resources.pinned_parameter_bytes).zip(
                    automatic_observed_u64(&resources.largest_execution_group_bytes),
                )
            }
            ResidencyPlan::DenseDiskStream { .. } => {
                automatic_observed_u64(&resources.pinned_parameter_bytes).zip(
                    automatic_observed_u64(&resources.largest_adjacent_execution_groups_bytes),
                )
            }
            ResidencyPlan::FullyResident => None,
        }
        .and_then(|(static_bytes, window_bytes)| static_bytes.checked_add(window_bytes));
        let ordinary_budget = match &expert_plan.residency {
            ResidencyPlan::LayerwiseHost {
                device_budget_bytes,
                ..
            } => *device_budget_bytes,
            ResidencyPlan::DenseDiskStream {
                device_budget_bytes,
                ..
            } => Some(*device_budget_bytes),
            ResidencyPlan::FullyResident => None,
        };
        if expert.supported
            && bounded_required
                .zip(ordinary_budget)
                .is_some_and(|(required, budget)| required > budget)
        {
            expert.supported = false;
            expert.rejection = Some(format!(
                "expert-cache budget splitting leaves {} ordinary device bytes, below the validated {}-byte non-expert requirement",
                ordinary_budget.expect("compared ordinary budget is present"),
                bounded_required.expect("compared bounded requirement is present")
            ));
        }
        if expert.supported {
            plan = expert_plan;
            entries.push(PlanExplanationEntry {
                level: PlanExplanationLevel::Decision,
                code: "expert_cache_selected".into(),
                detail: format!(
                    "the checkpoint admits routed-expert caching; {}% of each bounded tier budget was assigned to experts",
                    policy.expert_cache_share_percent
                ),
            });
        } else {
            entries.push(PlanExplanationEntry {
                level: PlanExplanationLevel::Rejection,
                code: "expert_cache_not_selected".into(),
                detail: expert
                    .rejection
                    .unwrap_or_else(|| "the checkpoint did not admit routed-expert caching".into()),
            });
        }
    }
    let embedded_mtp_layers =
        automatic_embedded_mtp_layers(&request.model_path, resources.model_kind)?;
    match embedded_mtp_layers {
        Some(layers) if layers > 0 => {
            plan.drafting = DraftingPlan::Embedded {
                max_draft_tokens: policy.embedded_mtp_draft_tokens,
                lookahead: true,
                adaptive_lookahead: true,
            };
            entries.push(PlanExplanationEntry {
                level: PlanExplanationLevel::Decision,
                code: "embedded_mtp_selected".into(),
                detail: format!(
                    "validated configuration advertises {layers} embedded prediction layer(s); enabled {}-token adaptive lookahead",
                    policy.embedded_mtp_draft_tokens
                ),
            });
        }
        Some(_) => entries.push(PlanExplanationEntry {
            level: PlanExplanationLevel::Decision,
            code: "embedded_mtp_absent".into(),
            detail: "the checkpoint configuration does not advertise embedded prediction layers"
                .into(),
        }),
        None => entries.push(PlanExplanationEntry {
            level: PlanExplanationLevel::Warning,
            code: "embedded_mtp_unobservable".into(),
            detail: "embedded MTP capability was not observable from this checkpoint container"
                .into(),
        }),
    }
    if let Some((feedback_plan, samples, median_rate)) = automatic_feedback_plan(
        &request.model_path,
        request,
        &hardware,
        &resources,
        policy,
        embedded_mtp_layers,
    )? {
        plan = feedback_plan;
        summary = format!(
            "selected a previously observed plan at {median_rate:.2} median decode tokens/s"
        );
        entries.push(PlanExplanationEntry {
            level: PlanExplanationLevel::Decision,
            code: "prior_telemetry_selected".into(),
            detail: format!(
                "selected the fastest matching, currently admitted plan using {samples} prior runtime sample(s)"
            ),
        });
    } else if !request.prior_telemetry.is_empty() {
        entries.push(PlanExplanationEntry {
            level: PlanExplanationLevel::Warning,
            code: "prior_telemetry_not_applicable".into(),
            detail: format!(
                "none of {} prior telemetry document(s) matched this model, hardware, device, and current admission state",
                request.prior_telemetry.len()
            ),
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

impl ResidencyTelemetry {
    /// Collects bounded ordinary-weight residency telemetry from a loaded model.
    pub fn collect(model: &super::LoadedModel) -> Result<Option<Self>, crate::error::Error> {
        let Some(report) = model.residency_report()? else {
            return Ok(None);
        };
        let offload = report.offload();
        let planned = offload.planned_bytes();
        let current = offload.resident_bytes();
        let peak = offload.peak_resident_bytes();
        let transfers = TransferDirection::ALL
            .into_iter()
            .map(|direction| {
                let metrics = offload.transfer(direction);
                TransferTelemetry {
                    direction: transfer_direction_name(direction).into(),
                    count: metrics.count(),
                    bytes: metrics.bytes(),
                    seconds: DurationSeconds(metrics.duration().as_secs_f64()),
                }
            })
            .collect();
        Ok(Some(Self {
            planned_disk_bytes: planned.get(MemoryTier::Disk),
            planned_host_bytes: planned.get(MemoryTier::Host),
            planned_device_bytes: planned.get(MemoryTier::Device),
            current_host_bytes: current.get(MemoryTier::Host),
            current_device_bytes: current.get(MemoryTier::Device),
            peak_host_bytes: peak.get(MemoryTier::Host),
            peak_device_bytes: peak.get(MemoryTier::Device),
            transfers,
        }))
    }
}

impl ExpertCacheTelemetry {
    /// Collects independent routed-expert cache occupancy from a loaded model.
    pub fn collect(model: &super::LoadedModel) -> Result<Option<Self>, crate::error::Error> {
        Ok(model.expert_cache_report()?.map(|report| Self {
            owned_experts: report.owned_experts,
            owned_bytes: report.owned_bytes,
            host_resident_experts: report.host_resident_experts,
            device_resident_experts: report.device_resident_experts,
            host_resident_bytes: report.host_resident_bytes,
            device_resident_bytes: report.device_resident_bytes,
            peak_host_resident_bytes: report.peak_host_resident_bytes,
            peak_device_resident_bytes: report.peak_device_resident_bytes,
        }))
    }
}

fn transfer_direction_name(direction: TransferDirection) -> &'static str {
    match direction {
        TransferDirection::DeviceToHost => "device_to_host",
        TransferDirection::DeviceToDisk => "device_to_disk",
        TransferDirection::HostToDevice => "host_to_device",
        TransferDirection::HostToDisk => "host_to_disk",
        TransferDirection::DiskToDevice => "disk_to_device",
        TransferDirection::DiskToHost => "disk_to_host",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feedback_fixture(
        path: PathBuf,
    ) -> (HardwareProfile, ModelResourceProfile, ExecutionTelemetry) {
        let hardware = discover_hardware();
        let device = mlx_device_plan("cpu", 0);
        let mut resources = ModelResourceProfile::empty(path, ArtifactKind::SafeTensorsDirectory);
        resources.model_kind = Some(ModelKind::Llama);
        resources.architecture = Some("llama".into());
        resources.tensor_count = Some(4);
        resources.checkpoint_shards = Some(1);
        resources.stored_tensor_bytes = Observed::exact(1024, "fixture");
        let plan = ExecutionPlan::fully_resident(device);
        let telemetry = ExecutionTelemetry {
            schema_version: AUTOMATIC_SCHEMA_VERSION,
            model_type: "Llama".into(),
            plan: Some(plan),
            plan_explanation: None,
            hardware: Some(hardware.clone()),
            resources: Some(resources.clone()),
            prompt_tokens: 8,
            generated_tokens: 16,
            stop_reason: "length".into(),
            timing: TimingTelemetry {
                load_seconds: 1.0,
                generation_seconds: 2.0,
                time_to_first_token_seconds: Some(0.25),
                total_seconds: 3.0,
                token_rate: 8.0,
                decode_token_rate: Some(10.0),
            },
            allocator: None,
            residency: None,
            expert_cache: None,
            mtp: None,
        };
        (hardware, resources, telemetry)
    }

    #[test]
    fn unavailable_observations_do_not_serialize_as_zero() {
        let value = Observed::<u64>::unavailable("not observable");
        let json = serde_json::to_value(value).unwrap();
        assert_eq!(json["status"], "unavailable");
        assert_eq!(json["reason"], "not observable");
        assert!(json.get("value").is_none());
    }

    #[test]
    fn execution_plan_round_trips_through_json() {
        let plan = ExecutionPlan::fully_resident(mlx_device_plan("metal", 0));
        let json = serde_json::to_string(&plan).unwrap();
        assert_eq!(serde_json::from_str::<ExecutionPlan>(&json).unwrap(), plan);
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

    #[test]
    fn hardware_discovery_always_reports_cpu() {
        let profile = discover_hardware();
        assert!(profile.backends.iter().any(|backend| {
            backend.backend == mlx_backend_id()
                && backend.available
                && backend.devices.iter().any(|device| device.id == "cpu:0")
        }));
    }

    #[test]
    fn unified_physical_capacity_replaces_missing_live_availability() {
        assert_eq!(
            automatic_memory_basis(None, Some(128 << 30), HardwareMemorySemantics::Unified),
            (
                Some(128 << 30),
                AutomaticMemoryBasis::UnifiedPhysicalCapacity
            )
        );
        assert_eq!(
            automatic_memory_basis(
                None,
                Some(128 << 30),
                HardwareMemorySemantics::SeparateTiers
            ),
            (None, AutomaticMemoryBasis::PolicyFallback)
        );
    }

    #[test]
    fn bounded_admission_rejects_static_plus_window_over_budget() {
        let mut admission = AutomaticCandidateAdmission {
            supported: true,
            rejection: None,
        };
        let requirement = AutomaticBoundedRequirement {
            static_bytes: 5_524_521_984,
            window_bytes: 1_935_777_792,
            required_bytes: 7_460_299_776,
            depth: 2,
        };
        assert_eq!(
            apply_automatic_bounded_requirement(&mut admission, Ok(requirement), 4_294_967_296),
            Some(requirement)
        );
        assert!(!admission.supported);
        assert!(admission
            .rejection
            .as_deref()
            .is_some_and(|detail| detail.contains("7460299776 total")));
    }

    #[test]
    fn automatic_residency_never_uses_a_resource_rejected_fallback() {
        assert_eq!(
            choose_automatic_residency(false, false, false, true, true, true),
            None
        );
        assert_eq!(
            choose_automatic_residency(false, false, true, true, true, true),
            Some(2)
        );
    }

    #[test]
    fn automatic_request_and_policy_are_stable_json_documents() {
        let (_, _, telemetry) = feedback_fixture("model".into());
        let request = AutomaticPlanRequest::new("model", mlx_device_plan("cpu", 0))
            .with_prior_telemetry([telemetry]);
        let request_json = serde_json::to_string(&request).unwrap();
        assert_eq!(
            serde_json::from_str::<AutomaticPlanRequest>(&request_json).unwrap(),
            request
        );

        let planner = AutomaticPlanner::default();
        let policy_json = serde_json::to_string(planner.policy()).unwrap();
        assert_eq!(
            serde_json::from_str::<AutomaticPlannerPolicy>(&policy_json).unwrap(),
            *planner.policy()
        );
    }

    #[test]
    fn execution_plan_loader_conversion_rejects_non_singleton_topology() {
        let mut plan = ExecutionPlan::fully_resident(mlx_device_plan("cpu", 0));
        assert!(execution_plan_load_options(&plan).is_ok());
        plan.topology = crate::core::topology::ParallelTopology::new(2, 1, 1, 1).unwrap();
        assert!(matches!(
            execution_plan_load_options(&plan),
            Err(Error::AutomaticPlanning(_))
        ));
    }

    #[test]
    fn prior_telemetry_requires_matching_model_and_hardware_identity() {
        let path = std::env::temp_dir().join("safemlx-automatic-feedback-fixture");
        let (hardware, resources, mut telemetry) = feedback_fixture(path.clone());
        let device = mlx_device_plan("cpu", 0);
        assert!(automatic_telemetry_matches(
            &telemetry,
            &hardware,
            &resources,
            device.clone(),
            1
        ));
        telemetry.resources.as_mut().unwrap().path = path.join("different");
        assert!(!automatic_telemetry_matches(
            &telemetry, &hardware, &resources, device, 1
        ));
    }

    #[test]
    fn feedback_prefers_decode_rate_and_uses_median() {
        let (_, _, mut telemetry) = feedback_fixture("model".into());
        assert_eq!(automatic_feedback_rate(&telemetry), Some(10.0));
        telemetry.timing.decode_token_rate = None;
        assert_eq!(automatic_feedback_rate(&telemetry), Some(8.0));
        assert_eq!(automatic_median(vec![30.0, 10.0, 20.0]), Some(20.0));
        assert_eq!(automatic_median(vec![40.0, 10.0]), Some(25.0));
    }
}
