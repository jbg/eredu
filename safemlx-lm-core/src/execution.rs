//! Portable execution-plan, resource, and telemetry schemas.

use crate::{backend::BackendCapabilities, model::ModelIdentity, residency::ResidencyPlan};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Backend-neutral plan selecting one backend/device for a model session.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExecutionPlan {
    /// Stable schema version, currently one.
    pub schema_version: u32,
    /// Planned model identity.
    pub model: ModelIdentity,
    /// Backend implementation name.
    pub backend: String,
    /// Backend-stable device identifier.
    pub device: String,
    /// Capabilities required by the plan.
    pub required_capabilities: BackendCapabilities,
    /// Logical residency plan.
    pub residency: ResidencyPlan,
}

impl ExecutionPlan {
    /// Validates schema, residency, and fail-closed capability satisfaction.
    pub fn validate(&self, available: &BackendCapabilities) -> Result<(), ExecutionPlanError> {
        if self.schema_version != 1 {
            return Err(ExecutionPlanError::Schema(self.schema_version));
        }
        self.residency
            .validate()
            .map_err(|error| ExecutionPlanError::Residency(error.to_string()))?;
        for (required, supported, name) in [
            (
                self.required_capabilities.exact_completion,
                available.exact_completion,
                "exact_completion",
            ),
            (
                self.required_capabilities.transfers,
                available.transfers,
                "transfers",
            ),
            (
                self.required_capabilities.collectives,
                available.collectives,
                "collectives",
            ),
            (
                self.required_capabilities.persistent_cache,
                available.persistent_cache,
                "persistent_cache",
            ),
        ] {
            if required && !supported {
                return Err(ExecutionPlanError::Capability(name));
            }
        }
        Ok(())
    }
}

/// Portable observations for one complete model/session run.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ExecutionTelemetry {
    /// Prompt tokens consumed.
    pub prompt_tokens: u64,
    /// Decode tokens committed.
    pub decode_tokens: u64,
    /// Time to first committed token in microseconds.
    pub time_to_first_token_us: Option<u64>,
    /// Total elapsed microseconds.
    pub elapsed_us: u64,
    /// Peak logical resource bytes by stable resource name.
    pub peak_resource_bytes: BTreeMap<String, u64>,
    /// Backend-specific numeric observations with stable explicit keys.
    pub backend_counters: BTreeMap<String, u64>,
}

/// Execution plan validation error.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum ExecutionPlanError {
    /// Unsupported schema version.
    #[error("unsupported execution-plan schema version {0}")]
    Schema(u32),
    /// Required capability is absent.
    #[error("execution plan requires unavailable capability {0}")]
    Capability(&'static str),
    /// Residency plan is invalid.
    #[error("execution-plan residency is invalid: {0}")]
    Residency(String),
}
