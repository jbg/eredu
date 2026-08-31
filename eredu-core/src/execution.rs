//! Portable execution-plan and telemetry schemas.

use crate::{
    backend::{DeviceCapabilities, SessionCapabilities},
    residency::CacheEvictionPolicy,
    topology::ParallelTopology,
};
use serde::{Deserialize, Serialize};

/// Schema version shared by execution-plan documents.
pub const EXECUTION_PLAN_SCHEMA_VERSION: u32 = 4;

/// Default bound for simultaneously open checkpoint payload sources.
pub const DEFAULT_MAX_CACHED_SHARDS: usize = 4;

/// Stable, extensible identity of an execution backend.
///
/// Core deliberately does not enumerate implementations. Values such as
/// Concrete implementations are registered by their backend adapters.
#[derive(Debug, Clone, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct BackendId(String);

impl BackendId {
    /// Creates a non-empty backend identifier.
    pub fn new(value: impl Into<String>) -> Result<Self, ExecutionPlanError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(ExecutionPlanError::EmptyBackendId);
        }
        Ok(Self(value))
    }

    /// Returns the stable adapter-defined identifier.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for BackendId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for BackendId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Self::new(String::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

/// Backend and device selected for one complete model session.
#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub struct DevicePlan {
    /// Backend adapter identity.
    pub backend: BackendId,
    /// Backend-stable device identifier.
    pub device: String,
}

impl<'de> Deserialize<'de> for DevicePlan {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RawDevicePlan {
            backend: BackendId,
            device: String,
        }

        let raw = RawDevicePlan::deserialize(deserializer)?;
        Self::new(raw.backend.0, raw.device).map_err(serde::de::Error::custom)
    }
}

impl DevicePlan {
    /// Creates a validated backend/device selection.
    pub fn new(
        backend: impl Into<String>,
        device: impl Into<String>,
    ) -> Result<Self, ExecutionPlanError> {
        let device = device.into();
        if device.trim().is_empty() {
            return Err(ExecutionPlanError::EmptyDeviceId);
        }
        Ok(Self {
            backend: BackendId::new(backend)?,
            device,
        })
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
    /// Deterministic eviction ordering for independently cached experts.
    pub eviction_policy: CacheEvictionPolicy,
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
        /// Backend/device placement used for assistant execution.
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
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum DraftPlacementPlan {
    /// Reuse the target execution context.
    Target,
    /// Use an explicit backend/device selection.
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

/// A concrete, backend-neutral set of model/session execution choices.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExecutionPlan {
    /// Version of this serialized plan schema.
    pub schema_version: u32,
    /// Backend and process-local device selected for the whole session.
    pub device: DevicePlan,
    /// Distributed Cartesian topology.
    pub topology: ParallelTopology,
    /// Ordinary static-weight placement.
    pub residency: ResidencyPlan,
    /// Optional transformation applied while checkpoint weights are loaded.
    pub weight_transformation: WeightTransformationPlan,
    /// Maximum number of checkpoint shards or readers retained simultaneously.
    pub max_cached_shards: usize,
    /// Independent routed-expert cache, when enabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expert_cache: Option<ExpertCachePlan>,
    /// Speculative decoding configuration.
    pub drafting: DraftingPlan,
    /// Capabilities which the selected device must provide.
    pub required_device_capabilities: DeviceCapabilities,
    /// Capabilities which the exact prepared session must provide.
    pub required_session_capabilities: SessionCapabilities,
}

impl ExecutionPlan {
    /// Creates the minimal fully-resident, target-only plan for one device.
    pub fn fully_resident(device: DevicePlan) -> Self {
        Self {
            schema_version: EXECUTION_PLAN_SCHEMA_VERSION,
            device,
            topology: ParallelTopology::new(1, 1, 1, 1).expect("the singleton topology is valid"),
            residency: ResidencyPlan::FullyResident,
            weight_transformation: WeightTransformationPlan::PreserveCheckpoint,
            max_cached_shards: DEFAULT_MAX_CACHED_SHARDS,
            expert_cache: None,
            drafting: DraftingPlan::Disabled,
            required_device_capabilities: DeviceCapabilities {
                exact_completion: true,
                ..DeviceCapabilities::default()
            },
            required_session_capabilities: SessionCapabilities::default(),
        }
    }

    /// Validates portable plan invariants and fail-closed capabilities.
    pub fn validate_device_capabilities(
        &self,
        available: &DeviceCapabilities,
    ) -> Result<(), ExecutionPlanError> {
        self.validate_structure()?;
        for (required, supported, name) in [
            (
                self.required_device_capabilities.exact_completion,
                available.exact_completion,
                "exact_completion",
            ),
            (
                self.required_device_capabilities.transfers,
                available.transfers,
                "transfers",
            ),
            (
                self.required_device_capabilities.collectives,
                available.collectives,
                "collectives",
            ),
        ] {
            if required && !supported {
                return Err(ExecutionPlanError::Capability(name));
            }
        }
        Ok(())
    }

    /// Validates exact prepared-session requirements independently.
    pub fn validate_session_capabilities(
        &self,
        available: &SessionCapabilities,
    ) -> Result<(), ExecutionPlanError> {
        self.validate_structure()?;
        self.required_session_capabilities
            .validate(available)
            .map_err(|error| ExecutionPlanError::Capability(error.capability()))
    }

    /// Validates schema, topology, and portable resource invariants without a backend.
    pub fn validate_structure(&self) -> Result<(), ExecutionPlanError> {
        if self.schema_version != EXECUTION_PLAN_SCHEMA_VERSION {
            return Err(ExecutionPlanError::Schema(self.schema_version));
        }
        if self.max_cached_shards == 0 {
            return Err(ExecutionPlanError::ZeroMappedShards);
        }
        match &self.drafting {
            DraftingPlan::Disabled => {}
            DraftingPlan::Embedded {
                max_draft_tokens, ..
            } => {
                if *max_draft_tokens == 0 {
                    return Err(ExecutionPlanError::ZeroDraftTokens);
                }
            }
            DraftingPlan::External {
                model,
                max_draft_tokens,
                ..
            } => {
                if model.trim().is_empty() {
                    return Err(ExecutionPlanError::EmptyDraftModel);
                }
                if *max_draft_tokens == 0 {
                    return Err(ExecutionPlanError::ZeroDraftTokens);
                }
            }
        }
        ParallelTopology::new(
            self.topology.tensor,
            self.topology.pipeline,
            self.topology.expert,
            self.topology.data,
        )
        .map_err(|error| ExecutionPlanError::Topology(error.to_string()))?;
        Ok(())
    }
}

/// Execution plan validation error.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum ExecutionPlanError {
    /// Backend identifier is empty.
    #[error("execution-plan backend identifier must not be empty")]
    EmptyBackendId,
    /// Device identifier is empty.
    #[error("execution-plan device identifier must not be empty")]
    EmptyDeviceId,
    /// Unsupported schema version.
    #[error("unsupported execution-plan schema version {0}")]
    Schema(u32),
    /// Required capability is absent.
    #[error("execution plan requires unavailable capability {0}")]
    Capability(&'static str),
    /// Topology is invalid.
    #[error("execution-plan topology is invalid: {0}")]
    Topology(String),
    /// The checkpoint source bound is zero.
    #[error("execution-plan max_cached_shards must be greater than zero")]
    ZeroMappedShards,
    /// An external assistant artifact path or identifier is empty.
    #[error("execution-plan external draft model must not be empty")]
    EmptyDraftModel,
    /// Speculative execution has no proposal capacity.
    #[error("execution-plan max_draft_tokens must be greater than zero")]
    ZeroDraftTokens,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_round_trips_with_extensible_backend_identity() {
        let plan = ExecutionPlan::fully_resident(DevicePlan::new("iree", "vulkan:2").unwrap());
        let encoded = serde_json::to_vec(&plan).unwrap();
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&encoded).unwrap()["schema_version"],
            4
        );
        assert_eq!(
            serde_json::from_slice::<ExecutionPlan>(&encoded).unwrap(),
            plan
        );
    }

    #[test]
    fn plan_capabilities_fail_closed() {
        let mut plan = ExecutionPlan::fully_resident(DevicePlan::new("mlx", "metal:0").unwrap());
        assert_eq!(
            plan.validate_device_capabilities(&DeviceCapabilities::default()),
            Err(ExecutionPlanError::Capability("exact_completion"))
        );

        plan.required_session_capabilities.activation_inspection = true;
        assert_eq!(
            plan.validate_session_capabilities(&SessionCapabilities::default()),
            Err(ExecutionPlanError::Capability("activation_inspection"))
        );
        assert!(plan
            .validate_device_capabilities(&DeviceCapabilities {
                exact_completion: true,
                transfers: false,
                collectives: false,
            })
            .is_ok());
        assert!(plan
            .validate_session_capabilities(&SessionCapabilities {
                activation_inspection: true,
                ..SessionCapabilities::default()
            })
            .is_ok());
    }

    #[test]
    fn backend_and_device_identifiers_fail_closed_during_deserialization() {
        assert!(serde_json::from_str::<DevicePlan>(r#"{"backend":"","device":"cpu:0"}"#).is_err());
        assert!(serde_json::from_str::<DevicePlan>(r#"{"backend":"mlx","device":""}"#).is_err());
    }

    #[test]
    fn speculative_plan_structure_fails_closed() {
        let mut plan = ExecutionPlan::fully_resident(DevicePlan::new("mock", "gpu:0").unwrap());
        plan.drafting = DraftingPlan::Embedded {
            max_draft_tokens: 0,
            lookahead: false,
            adaptive_lookahead: false,
        };
        assert_eq!(
            plan.validate_structure(),
            Err(ExecutionPlanError::ZeroDraftTokens)
        );

        plan.drafting = DraftingPlan::External {
            model: "  ".into(),
            placement: DraftPlacementPlan::Target,
            max_draft_tokens: 1,
            lookahead: false,
            adaptive_lookahead: false,
        };
        assert_eq!(
            plan.validate_structure(),
            Err(ExecutionPlanError::EmptyDraftModel)
        );
    }
}
