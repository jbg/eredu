//! Shared forecast calibration and loaded execution observations.

use crate::memory_estimation::*;
use eredu_core::{
    AvailableMemory, CapabilityError, ModelCapabilityBackend, ModelRuntime, StaticMemoryReport,
};
use serde::{Deserialize, Serialize};
use std::num::NonZeroU8;

mod loaded;
pub use loaded::*;
mod continuation;
pub use continuation::*;
mod capture;
pub use capture::{apply_capture_memory_bound, CaptureMemoryPlan};
mod speculative;
pub use speculative::*;

/// Portable forecast failures retain native sources and typed policy errors.
#[derive(Debug, thiserror::Error)]
pub enum GenerationForecastError {
    /// Invalid or unavailable portable geometry/accounting.
    #[error(transparent)]
    Capability(#[from] CapabilityError),
    /// The requested boundary cannot be observed without advancing execution.
    #[error("continuation forecast unavailable: {0}")]
    UnsupportedContinuation(String),
    /// Invalid resolved generation settings.
    #[error(transparent)]
    Generation(#[from] eredu_core::generation::GenerationError),
    /// Native observation failure with its original source.
    #[error(transparent)]
    Backend(#[from] eredu_core::BackendFailure),
}

/// Versioned planning assumptions, independently overrideable by calibrated consumers.
#[derive(Debug, Clone)]
pub struct ForecastCalibration {
    /// Selected attention scratch assumption.
    pub attention: AttentionWorkspace,
    /// Selected cache replacement overlap.
    pub cache_update: CacheUpdateWorkspace,
    /// Override the default one-set-per-layer plus 25 percent overlap envelope.
    pub workspace_overlap: Option<WorkspaceOverlap>,
    /// Graph/driver allowance in addition to allocator retention, not a process bound.
    pub graph_driver_bytes: u64,
}

impl Default for ForecastCalibration {
    fn default() -> Self {
        Self {
            attention: AttentionWorkspace::ScoreMatrixUpperBound,
            cache_update: CacheUpdateWorkspace::CopyState,
            workspace_overlap: None,
            graph_driver_bytes: 64 * 1024 * 1024,
        }
    }
}

impl ForecastCalibration {
    /// Combines observed allocator retention with the labeled driver allowance.
    pub fn allocator_overhead(&self, limit: u64, retained: u64) -> MemoryBytes {
        match limit.max(retained).checked_add(self.graph_driver_bytes) {
            Some(upper) => MemoryBytes::estimated(0, upper, format!(
                "forecast calibration v1: allocator-cache limit {limit} bytes, retained cache {retained} bytes; their maximum plus {} bytes graph/driver allowance; planning assumption, not a total-process bound", self.graph_driver_bytes)),
            None => MemoryBytes::unknown("allocator-cache and graph/driver allowance overflow"),
        }
    }

    /// Applies one shared calibration to every execution in a physical pool.
    pub fn apply(&self, request: &mut GenerationMemoryRequest) -> Result<(), CapabilityError> {
        for domain in &mut request.domains {
            for execution in &mut domain.executions {
                let layers = execution.state_layout.layer_layout().len() as u64;
                let upper = layers
                    .checked_add(layers.div_ceil(4))
                    .ok_or(CapabilityError::ArithmeticOverflow {
                        operation: "forecast layer overlap",
                    })?
                    .max(1);
                execution.attention = self.attention.clone();
                execution.cache_update = self.cache_update;
                execution.workspace_overlap = self.workspace_overlap.clone().unwrap_or(WorkspaceOverlap {
                    upper_live_copies: Some(upper),
                    detail: "forecast calibration v1: one linear activation set per local state layer plus 25% scratch; calibrated on SmolLM-135M Metal original/4-bit and dense LFM2.5-1.2B Metal BF16, not a universal bound".into(),
                });
            }
        }
        Ok(())
    }
}

/// Architecture-selected geometry retained at preparation, without reopening sources.
#[derive(Debug, Clone)]
pub struct LoadedMemoryGeometry {
    /// Architecture-owned selected persistent-state layout.
    pub state_layout: eredu_core::StateMemoryLayout,
    /// Architecture workspace geometry, absent for uncovered equations.
    pub workspace: Option<WorkspaceGeometry>,
    /// Actual selected floating-state element width.
    pub scalar_bytes: NonZeroU8,
    /// Whether parameters stay resident rather than materializing during generation.
    pub fully_resident: bool,
    /// Retained architecture coverage explanations.
    pub assumptions: Vec<String>,
}

/// Native observations; portable policy performs the physical-pool accounting.
#[derive(Clone)]
pub struct LoadedMemoryProfile {
    /// Geometry retained by architecture-owned preparation.
    pub geometry: LoadedMemoryGeometry,
    /// Current model parameter residency; never inferred from process counters.
    pub parameters: StaticMemoryReport,
    /// Point-in-time observed capacity.
    pub available: AvailableMemory,
    /// Whether execution consumes the host capacity pool.
    pub host_execution: bool,
    /// Current native cache policy, including an explicit unavailable observation.
    pub allocator_cache_limit: eredu_core::Observed<u64>,
}

/// The selected executor's actual prefill contract for this prompt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForecastExecutionContract {
    /// Absent when bounded ordinary prefill is supported.
    pub full_pass_reason: Option<String>,
    /// Rows actually projected by one prefill invocation.
    pub logits: LogitsWorkspace,
}

/// Backend facts needed to forecast an already loaded request. No inference is submitted.
pub trait GenerationForecastBackend: ModelCapabilityBackend {
    /// Complete ordinary capture sizing, including source creation where needed.
    /// None retains the admitted ceilings. No submission or reservation is allowed.
    fn capture_memory_projection(
        _runtime: &ModelRuntime<Self>,
        _capture: &eredu_core::capture::AdmittedCapturePlan,
        _intervention: Option<&eredu_core::intervention::AdmittedInterventionPlan>,
        _first_prediction: u64,
    ) -> Result<Option<crate::capture::CaptureUsageProjection>, GenerationForecastError> {
        Ok(None)
    }

    /// Whether capture/intervention native temporaries are completed and released
    /// before the next prediction. This excludes allocator cache, accounted for
    /// separately. The conservative default uses the cumulative retained limit.
    fn capture_transforms_complete_per_step(_runtime: &ModelRuntime<Self>) -> bool {
        false
    }

    /// Observes retained geometry and current native memory facts. Existing
    /// mutable state must be projected or leave the workspace upper end unknown;
    /// a continuation must never silently be treated as a fresh request.
    fn loaded_memory_profile(
        runtime: &ModelRuntime<Self>,
    ) -> Result<LoadedMemoryProfile, GenerationForecastError>;
    /// Reports the actual prompt and instrumentation contract without submission.
    fn forecast_execution_contract(
        runtime: &ModelRuntime<Self>,
        prompt: Option<&Self::Prompt>,
        instrumented: bool,
    ) -> ForecastExecutionContract;
}
