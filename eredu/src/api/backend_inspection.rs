//! Shared inspection and memory forecasting through the selected backend.

use std::path::Path;

use eredu_core::{BackendFailure, DevicePlan, ExecutionPlan, InputTokenCount};

use super::{ForecastCalibration, GenerationForecastError, GenerationMemoryOptions, MemoryBytes};

pub use eredu_architectures::BackendInspection;

/// Inspects an artifact against the selected backend and plan without loading it.
pub fn inspect_execution(
    backend: &impl BackendInspection,
    path: impl AsRef<Path>,
    plan: &ExecutionPlan,
) -> Result<eredu_architectures::ModelInspectionOutcome, BackendFailure> {
    backend.inspect_execution(path.as_ref(), plan)
}

impl GenerationMemoryOptions {
    /// Observes the selected backend and derives memory options for a device.
    /// Availability is a point-in-time observation. Allocation policy is unchanged.
    pub fn for_backend_device(
        backend: &impl BackendInspection,
        input: InputTokenCount,
        device: &DevicePlan,
    ) -> Result<Self, GenerationForecastError> {
        let host_execution = backend.device_uses_host_memory(device).map_err(|error| {
            eredu_core::CapabilityError::InvalidConfiguration {
                field: "device",
                detail: error.to_string(),
            }
        })?;
        let hardware = backend.discover_hardware().map_err(|error| {
            eredu_core::CapabilityError::InvalidConfiguration {
                field: "hardware",
                detail: error.to_string(),
            }
        })?;
        let mut options = Self::for_hardware_device(input, &hardware, device, host_execution)?;
        options.backend_overhead = observed_allocator_overhead(backend);
        Ok(options)
    }

    /// Uses explicit placement and observes the selected backend's allocator overhead.
    pub fn for_backend(
        backend: &impl BackendInspection,
        input: InputTokenCount,
        placement: super::GenerationMemoryPlacement,
    ) -> Self {
        let mut options = Self::new(input, placement);
        options.backend_overhead = observed_allocator_overhead(backend);
        options
    }
}

fn observed_allocator_overhead(backend: &impl BackendInspection) -> MemoryBytes {
    match backend.allocator_observation() {
        Ok(Some((policy, memory))) => {
            let mut overhead = ForecastCalibration::default()
                .allocator_overhead(policy.limit_bytes, memory.cache_bytes);
            overhead
                .detail
                .push_str(&format!("; native policy provenance: {:?}", policy.source));
            overhead
        }
        Ok(None) => MemoryBytes::unknown("backend allocator overhead is unavailable"),
        Err(error) => MemoryBytes::unknown(format!("allocator-cache observation failed: {error}")),
    }
}
