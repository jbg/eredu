//! Hardware, resource, execution-plan, and telemetry schemas for automatic tuning.
//!
//! These types deliberately distinguish exact observations from estimates and
//! unavailable data. Automatic planners must not turn a missing device-memory
//! query or an unknown materialized weight size into a zero-byte value.

use std::{path::PathBuf, time::Duration};

use serde::{Deserialize, Serialize};

use super::{available_memory, ArtifactKind, CapabilityValue, ModelKind, PhysicalMemorySemantics};
use crate::runtime::{
    generation::speculative::MtpStats,
    residency::policy::{MemoryTier, TransferDirection},
};

/// Schema version shared by automatic-planning and telemetry documents.
pub const AUTOMATIC_SCHEMA_VERSION: u32 = 1;

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

/// Execution backend represented in a hardware or execution plan.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    /// Host CPU execution.
    Cpu,
    /// GPU execution where the build does not identify a Metal or CUDA backend.
    Gpu,
    /// Apple Metal GPU execution.
    Metal,
    /// NVIDIA CUDA GPU execution.
    Cuda,
}

/// One logical device visible to SafeMLX.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct HardwareDeviceProfile {
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
    pub kind: BackendKind,
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

    let mut backends = vec![HardwareBackendProfile {
        kind: BackendKind::Cpu,
        available: true,
        detail: None,
        devices: vec![HardwareDeviceProfile {
            index: 0,
            total_memory_bytes: physical_memory_bytes.clone(),
            available_memory_bytes: available_memory_bytes.clone(),
        }],
    }];

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
        backends.push(HardwareBackendProfile {
            kind: BackendKind::Metal,
            available,
            detail,
            devices: available
                .then(|| HardwareDeviceProfile {
                    index: 0,
                    total_memory_bytes: device_total,
                    available_memory_bytes: device_available,
                })
                .into_iter()
                .collect(),
        });
    }

    #[cfg(feature = "cuda")]
    {
        let (available, detail) = match safemlx::cuda::is_available() {
            Ok(available) => (available, None),
            Err(error) => (false, Some(error.to_string())),
        };
        backends.push(HardwareBackendProfile {
            kind: BackendKind::Cuda,
            available,
            detail,
            devices: if available {
                vec![HardwareDeviceProfile {
                    index: 0,
                    total_memory_bytes: Observed::unavailable(
                        "SafeMLX does not yet expose CUDA device-memory discovery",
                    ),
                    available_memory_bytes: Observed::unavailable(
                        "SafeMLX does not yet expose CUDA device-memory discovery",
                    ),
                }]
            } else {
                Vec::new()
            },
        });
    }

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

/// Process-local device selected by an execution plan.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct DevicePlan {
    /// Execution backend.
    pub backend: BackendKind,
    /// Process-local device index.
    pub index: usize,
}

/// Cartesian parallel topology selected by a plan.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct ParallelismPlan {
    /// Tensor-parallel rank count.
    pub tensor: usize,
    /// Pipeline-parallel rank count.
    pub pipeline: usize,
    /// Expert-parallel rank count.
    pub expert: usize,
}

impl Default for ParallelismPlan {
    fn default() -> Self {
        Self {
            tensor: 1,
            pipeline: 1,
            expert: 1,
        }
    }
}

/// Static weight placement selected by an execution plan.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum ResidencyPlan {
    /// Retain all selected weights on the execution device.
    FullyResident,
    /// Retain repeated groups on the host and promote a bounded device window.
    LayerwiseHost {
        /// Maximum repeated groups resident on the device.
        device_layer_window: usize,
        /// Logical device parameter budget.
        #[serde(skip_serializing_if = "Option::is_none")]
        device_budget_bytes: Option<u64>,
        /// Charged host-transfer budget.
        #[serde(skip_serializing_if = "Option::is_none")]
        host_budget_bytes: Option<u64>,
    },
    /// Stream repeated groups through disk, host, and device caches.
    DenseDiskStream {
        /// Finite logical device budget.
        device_budget_bytes: u64,
        /// Finite charged host budget.
        host_budget_bytes: u64,
        /// Protected host lookahead.
        host_lookahead: usize,
        /// Background materialization queue capacity.
        background_queue: usize,
    },
}

/// Optional independent routed-expert cache selected by a plan.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExpertCachePlan {
    /// Logical device expert-cache budget.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_budget_bytes: Option<u64>,
    /// Charged host expert-cache budget.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub host_budget_bytes: Option<u64>,
    /// Hard compact-bank scratch bound.
    pub scratch_bytes: u64,
    /// Soft prefill compact-bank target.
    pub prefill_bank_bytes: u64,
}

/// Speculative decoding selected by an execution plan.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum DraftingPlan {
    /// Ordinary target-only decoding.
    Disabled,
    /// Use checkpoint-embedded prediction heads.
    Embedded {
        /// Maximum proposals per verification round.
        max_draft_tokens: usize,
        /// Whether same-request optimistic lookahead is enabled.
        lookahead: bool,
        /// Whether deterministic adaptive lookahead is enabled.
        adaptive_lookahead: bool,
    },
    /// Use an explicitly supplied external assistant.
    External {
        /// Assistant artifact path or identifier.
        model: String,
        /// Stream/device placement used for assistant execution.
        placement: DraftPlacementPlan,
        /// Maximum proposals per verification round.
        max_draft_tokens: usize,
        /// Whether same-request optimistic lookahead is enabled.
        lookahead: bool,
        /// Whether deterministic adaptive lookahead is enabled.
        adaptive_lookahead: bool,
    },
}

/// External assistant placement selected by an execution plan.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum DraftPlacementPlan {
    /// Reuse the target execution stream.
    Target,
    /// Create a distinct stream on an explicit device.
    Device {
        /// Explicit process-local assistant device.
        device: DevicePlan,
    },
}

/// Optional load-time transformation applied to checkpoint weights.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum WeightTransformationPlan {
    /// Preserve checkpoint-native weight encodings.
    PreserveCheckpoint,
    /// Convert eligible weights to grouped affine quantization while loading.
    Affine {
        /// Quantized bits per weight.
        bits: i32,
        /// Adjacent weights sharing quantization parameters.
        group_size: i32,
    },
    /// Convert eligible weights to MXFP4 while loading.
    MxFp4,
}

/// A concrete, serializable set of runtime execution choices.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExecutionPlan {
    /// Version of this serialized plan schema.
    pub schema_version: u32,
    /// Main process-local execution device.
    pub device: DevicePlan,
    /// Distributed topology shape.
    pub parallelism: ParallelismPlan,
    /// Ordinary static-weight placement.
    pub residency: ResidencyPlan,
    /// Optional transformation applied while checkpoint weights are loaded.
    pub weight_transformation: WeightTransformationPlan,
    /// Maximum number of checkpoint shards or readers retained simultaneously.
    pub max_mapped_shards: usize,
    /// Independent routed-expert cache, when enabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expert_cache: Option<ExpertCachePlan>,
    /// Speculative decoding configuration.
    pub drafting: DraftingPlan,
    /// Process-global allocator-cache limit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mlx_cache_limit_bytes: Option<u64>,
}

impl ExecutionPlan {
    /// Creates the minimal fully-resident, target-only plan for one device.
    pub fn fully_resident(device: DevicePlan) -> Self {
        Self {
            schema_version: AUTOMATIC_SCHEMA_VERSION,
            device,
            parallelism: ParallelismPlan::default(),
            residency: ResidencyPlan::FullyResident,
            weight_transformation: WeightTransformationPlan::PreserveCheckpoint,
            max_mapped_shards: crate::runtime::checkpoint::store::DEFAULT_MAX_MAPPED_SHARDS,
            expert_cache: None,
            drafting: DraftingPlan::Disabled,
            mlx_cache_limit_bytes: None,
        }
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
    /// Stable stream-topology label.
    pub stream_topology: String,
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
            stream_topology: stats.stream_topology.to_string(),
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
        let plan = ExecutionPlan::fully_resident(DevicePlan {
            backend: BackendKind::Metal,
            index: 0,
        });
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
        assert!(profile
            .backends
            .iter()
            .any(|backend| backend.kind == BackendKind::Cpu && backend.available));
    }
}
