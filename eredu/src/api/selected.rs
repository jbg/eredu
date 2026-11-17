//! MLX runtime configuration, inspection, and diagnostics for generic facade models.

#[cfg(all(feature = "metal", target_vendor = "apple"))]
use std::path::PathBuf;
use std::time::Duration;

pub use eredu_core::{AllocatorCachePolicyReport, AllocatorCachePolicySource};

/// Observes the cache limit and native provenance atomically without configuring it.
pub fn local_allocator_cache_policy(
) -> Result<AllocatorCachePolicyReport, eredu_core::BackendFailure> {
    eredu_backend_mlx::allocator_cache_policy().map_err(|error| {
        eredu_core::BackendFailure::from_error(error).with_operation("allocator-cache policy query")
    })
}

/// Reads the process-global allocator-cache limit in bytes without changing it.
/// This snapshot may initialize the native allocator and can change later.
pub fn local_allocator_cache_limit() -> Result<usize, eredu_core::BackendFailure> {
    eredu_backend_mlx::allocator_cache_limit().map_err(|error| {
        eredu_core::BackendFailure::from_error(error).with_operation("allocator-cache limit query")
    })
}

/// Sets the process-global allocator-cache limit and returns the previous value
/// for explicit restoration. Coordinate changes with other runtime users.
pub fn set_local_allocator_cache_limit(bytes: usize) -> Result<usize, eredu_core::BackendFailure> {
    eredu_backend_mlx::set_allocator_cache_limit(bytes).map_err(|error| {
        eredu_core::BackendFailure::from_error(error)
            .with_operation("allocator-cache configuration")
    })
}

impl super::BackendInspection for eredu_backend_mlx::MlxBackendFactory {
    fn inspect_execution(
        &self,
        path: &std::path::Path,
        plan: &crate::ExecutionPlan,
    ) -> Result<eredu_architectures::ModelInspectionOutcome, BackendFailure> {
        let request = self
            .load_request_for_plan(plan)
            .map_err(BackendFailure::from_error)?;
        eredu_backend_mlx::native::inspect_model_preparation(
            path,
            eredu_backend_mlx::native::MlxInspectionOptions::new(request),
        )
        .map_err(|error| BackendFailure::from_error(error).with_operation("model inspection"))
    }

    fn device_uses_host_memory(&self, device: &crate::DevicePlan) -> Result<bool, BackendFailure> {
        self.device_uses_host_memory(device)
            .map_err(BackendFailure::from_error)
    }

    fn allocator_observation(
        &self,
    ) -> Result<Option<(AllocatorCachePolicyReport, crate::AllocatorTelemetry)>, BackendFailure>
    {
        Ok(Some((
            local_allocator_cache_policy()?,
            allocator_telemetry()?,
        )))
    }
}

use super::{DevicePlanError, ExpertCacheBenchmarkError};
use eredu_core::BackendFailure;

impl super::LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>> {
    /// Synchronizes the model and samples allocator counters.
    pub fn allocator_telemetry(&self) -> Result<crate::AllocatorTelemetry, BackendFailure> {
        self.synchronize()
            .map_err(|error| BackendFailure::from_error(error).with_operation("synchronization"))?;
        allocator_telemetry()
    }

    /// Returns portable sparse expert-cache telemetry when available.
    pub fn expert_cache_telemetry(
        &self,
    ) -> Result<Option<crate::ExpertCacheTelemetry>, BackendFailure> {
        self.runtime
            .session()
            .parameter_bank_report()
            .map(|report| {
                report
                    .as_ref()
                    .map(eredu_backend_mlx::parameter_bank_telemetry)
            })
            .map_err(|error| {
                BackendFailure::from_error(error).with_operation("expert-cache telemetry")
            })
    }

    /// Returns portable weight-residency telemetry when available.
    pub fn residency_telemetry(&self) -> Result<Option<crate::ResidencyTelemetry>, BackendFailure> {
        self.runtime
            .session()
            .residency_report()
            .map_err(|error| {
                BackendFailure::from_error(error).with_operation("residency telemetry")
            })
            .map(|report| report.as_ref().map(eredu_runtime::residency_telemetry))
    }
}

/// A facade-level device class for the selected local backend.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum LocalDevice {
    /// Host CPU execution.
    Cpu,
    /// The selected native accelerator at this zero-based index.
    Accelerator(u32),
}

/// Selects accelerator zero when this build includes a native accelerator
/// family, or the CPU for CPU-only builds.
pub const fn default_local_device() -> LocalDevice {
    if compiled_accelerator_family().is_some() {
        LocalDevice::Accelerator(0)
    } else {
        LocalDevice::Cpu
    }
}

/// Process-global allocator policy, selected before native model realization.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum LocalAllocatorCachePolicy {
    /// Cap an untouched native default at 1/32 of available memory, up to 2 GiB.
    /// Missing memory observations use 256 MiB; smaller native defaults remain.
    /// Explicit settings and earlier initialization always take precedence.
    #[default]
    Automatic,
    /// Preserve the untouched native default across subsequent automatic startup.
    /// This must be selected before the first model is realized or policy initialized.
    PreserveNative,
    /// Explicitly set the process-global byte limit, overriding earlier policy.
    Fixed(usize),
}

/// Process-global configuration for the selected local runtime.
#[derive(Debug, Clone, Default)]
pub struct LocalRuntimeConfiguration {
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    accelerator_library_path: Option<PathBuf>,
    allocator_cache_policy: LocalAllocatorCachePolicy,
}

impl LocalRuntimeConfiguration {
    /// Overrides the native accelerator kernel-library path.
    ///
    /// Embedded Apple applications use this when their bundled library cannot
    /// be found through the runtime's default search path.
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    pub fn with_accelerator_library(mut self, path: impl Into<PathBuf>) -> Self {
        self.accelerator_library_path = Some(path.into());
        self
    }

    /// Sets the selected runtime's process-global allocator-cache limit.
    pub const fn with_allocator_cache_limit(mut self, bytes: usize) -> Self {
        self.allocator_cache_policy = LocalAllocatorCachePolicy::Fixed(bytes);
        self
    }

    /// Selects automatic, preserved-native or explicit cache policy. Initialization
    /// is process-global; coordinate it with other MLX users before loading models.
    pub const fn with_allocator_cache_policy(mut self, policy: LocalAllocatorCachePolicy) -> Self {
        self.allocator_cache_policy = policy;
        self
    }
}

/// Applies process-global configuration before creating a local model session.
pub fn configure_local_runtime(
    configuration: &LocalRuntimeConfiguration,
) -> Result<(), BackendFailure> {
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    if let Some(path) = &configuration.accelerator_library_path {
        eredu_backend_mlx::set_accelerator_library_path(path).map_err(|error| {
            BackendFailure::from_error(error).with_operation("runtime configuration")
        })?;
    }
    match configuration.allocator_cache_policy {
        LocalAllocatorCachePolicy::Fixed(bytes) => {
            set_local_allocator_cache_limit(bytes)?;
        }
        policy => {
            eredu_backend_mlx::initialize_allocator_cache_policy(
                policy == LocalAllocatorCachePolicy::PreserveNative,
            )
            .map_err(|error| {
                BackendFailure::from_error(error).with_operation("allocator-cache initialization")
            })?;
        }
    }
    Ok(())
}

/// Creates a portable plan device for the selected local backend.
pub fn local_device_plan(device: LocalDevice) -> Result<crate::DevicePlan, DevicePlanError> {
    let device = match device {
        LocalDevice::Cpu => "cpu:0".to_owned(),
        LocalDevice::Accelerator(index) => {
            let family =
                compiled_accelerator_family().ok_or(DevicePlanError::AcceleratorNotCompiled)?;
            format!("{family}:{index}")
        }
    };
    Ok(crate::DevicePlan::new("mlx", device)
        .expect("the selected local backend and generated device identifier are non-empty"))
}

const fn compiled_accelerator_family() -> Option<&'static str> {
    if cfg!(feature = "cuda") {
        Some("cuda")
    } else if cfg!(all(feature = "metal", target_vendor = "apple")) {
        Some("metal")
    } else {
        None
    }
}

/// Resets the selected runtime's allocator high-water mark.
pub fn reset_local_allocator_peak() -> Result<(), BackendFailure> {
    eredu_backend_mlx::reset_allocator_peak().map_err(|error| {
        BackendFailure::from_error(error).with_operation("allocator peak reset")
    })?;
    Ok(())
}

fn allocator_telemetry() -> Result<crate::AllocatorTelemetry, BackendFailure> {
    let memory = eredu_backend_mlx::allocator_memory()
        .map_err(|error| BackendFailure::from_error(error).with_operation("allocator telemetry"))?;
    Ok(crate::AllocatorTelemetry {
        peak_bytes: memory.peak_bytes(),
        active_bytes: memory.active_bytes(),
        cache_bytes: memory.cached_bytes(),
    })
}

/// One measured phase of a selected-backend expert-cache benchmark.
#[derive(Debug, Clone, Copy)]
pub struct LocalExpertCacheBenchmarkSample {
    /// End-to-end phase latency after exact completion.
    pub elapsed: Duration,
    /// Route rows requested by the router.
    pub requested_routes: u64,
    /// Distinct logical experts requested after coalescing.
    pub distinct_experts: u64,
    /// Duplicate requests eliminated before materialization.
    pub coalesced_duplicates: u64,
    /// Temporary compact banks built.
    pub compact_banks: u64,
    /// Temporary compact-bank bytes built.
    pub compact_bank_bytes: u64,
    /// Host-cache hits.
    pub host_hits: u64,
    /// Host-cache misses.
    pub host_misses: u64,
    /// Host-cache evictions.
    pub host_evictions: u64,
    /// Device-cache hits.
    pub device_hits: u64,
    /// Device-cache misses.
    pub device_misses: u64,
    /// Device-cache evictions.
    pub device_evictions: u64,
    /// Host-resident expert count after the phase.
    pub host_resident_experts: usize,
    /// Host-resident expert bytes after the phase.
    pub host_resident_bytes: u64,
    /// Device-resident expert count after the phase.
    pub device_resident_experts: usize,
    /// Device-resident expert bytes after the phase.
    pub device_resident_bytes: u64,
}

/// Cold prefill, repeated prefill, and cached decode measurements.
#[derive(Debug, Clone, Copy)]
pub struct LocalExpertCacheBenchmark {
    /// Prefill after resetting the model session.
    pub cold_prefill: LocalExpertCacheBenchmarkSample,
    /// A second prefill after resetting only model state.
    pub repeated_prefill: LocalExpertCacheBenchmarkSample,
    /// One decode using the state produced by repeated prefill.
    pub cached_decode: LocalExpertCacheBenchmarkSample,
}

#[derive(Clone, Copy)]
struct ExpertSnapshot {
    prefill: eredu_backend_mlx::backend::runtime::residency::parameter_bank::BankPassStatistics,
    decode: eredu_backend_mlx::backend::runtime::residency::parameter_bank::BankPassStatistics,
    host_resident_experts: usize,
    host_resident_bytes: u64,
    device_resident_experts: usize,
    device_resident_bytes: u64,
}

fn expert_snapshot(
    model: &super::LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>,
) -> Result<ExpertSnapshot, ExpertCacheBenchmarkError> {
    let report = model
        .runtime
        .session()
        .parameter_bank_report()
        .map_err(|error| {
            BackendFailure::from_error(error).with_operation("expert-cache telemetry")
        })?
        .ok_or(ExpertCacheBenchmarkError::ExpertCacheUnavailable)?;
    Ok(ExpertSnapshot {
        prefill: *report.bulk(),
        decode: *report.incremental(),
        host_resident_experts: report.host_resident_entries(),
        host_resident_bytes: report.host_resident_bytes(),
        device_resident_experts: report.device_resident_entries(),
        device_resident_bytes: report.device_resident_bytes(),
    })
}

fn benchmark_sample(
    elapsed: Duration,
    before: eredu_backend_mlx::backend::runtime::residency::parameter_bank::BankPassStatistics,
    after: eredu_backend_mlx::backend::runtime::residency::parameter_bank::BankPassStatistics,
    occupancy: ExpertSnapshot,
) -> LocalExpertCacheBenchmarkSample {
    LocalExpertCacheBenchmarkSample {
        elapsed,
        requested_routes: after
            .requested_selections()
            .saturating_sub(before.requested_selections()),
        distinct_experts: after
            .distinct_entries()
            .saturating_sub(before.distinct_entries()),
        coalesced_duplicates: after
            .coalesced_duplicates()
            .saturating_sub(before.coalesced_duplicates()),
        compact_banks: after.compact_banks().saturating_sub(before.compact_banks()),
        compact_bank_bytes: after
            .compact_bank_bytes()
            .saturating_sub(before.compact_bank_bytes()),
        host_hits: after.host().hits().saturating_sub(before.host().hits()),
        host_misses: after.host().misses().saturating_sub(before.host().misses()),
        host_evictions: after
            .host()
            .evictions()
            .saturating_sub(before.host().evictions()),
        device_hits: after.device().hits().saturating_sub(before.device().hits()),
        device_misses: after
            .device()
            .misses()
            .saturating_sub(before.device().misses()),
        device_evictions: after
            .device()
            .evictions()
            .saturating_sub(before.device().evictions()),
        host_resident_experts: occupancy.host_resident_experts,
        host_resident_bytes: occupancy.host_resident_bytes,
        device_resident_experts: occupancy.device_resident_experts,
        device_resident_bytes: occupancy.device_resident_bytes,
    }
}

fn validate_expert_cache_benchmark_prompt(
    token_ids: &[u32],
) -> Result<(), ExpertCacheBenchmarkError> {
    if token_ids.is_empty() {
        return Err(ExpertCacheBenchmarkError::EmptyPrompt);
    }
    Ok(())
}

/// Benchmarks selected-backend expert-cache reuse without exposing tensors or streams.
pub fn benchmark_local_expert_cache(
    model: &mut super::LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>,
    token_ids: &[u32],
) -> Result<LocalExpertCacheBenchmark, ExpertCacheBenchmarkError> {
    validate_expert_cache_benchmark_prompt(token_ids)?;
    let prompt = <eredu_backend_mlx::backend::MlxBackend as eredu_core::TextGenerationBackend>::prepare_text_prompt(
        model.runtime.backend(),
        token_ids.to_vec(),
    )
    .map_err(|error| BackendFailure::from_error(error).with_operation("expert-cache prompt preparation"))?;

    let before_cold = expert_snapshot(model)?;
    model.runtime.session_mut().reset().map_err(|error| {
        BackendFailure::from_error(error).with_operation("expert-cache session reset")
    })?;
    let started = std::time::Instant::now();
    let logits = model
        .runtime
        .prefill(prompt.clone())
        .map_err(|error| {
            BackendFailure::from_error(error).with_operation("expert-cache prefill submission")
        })?
        .wait()
        .map_err(|error| {
            BackendFailure::from_error(error).with_operation("expert-cache prefill completion")
        })?
        .into_logits()
        .ok_or(ExpertCacheBenchmarkError::LogitsUnavailable)?;
    drop(logits);
    let after_cold = expert_snapshot(model)?;
    let cold_prefill = benchmark_sample(
        started.elapsed(),
        before_cold.prefill,
        after_cold.prefill,
        after_cold,
    );

    model.runtime.session_mut().reset().map_err(|error| {
        BackendFailure::from_error(error).with_operation("expert-cache session reset")
    })?;
    let started = std::time::Instant::now();
    let logits = model
        .runtime
        .prefill(prompt)
        .map_err(|error| {
            BackendFailure::from_error(error).with_operation("expert-cache prefill submission")
        })?
        .wait()
        .map_err(|error| {
            BackendFailure::from_error(error).with_operation("expert-cache prefill completion")
        })?
        .into_logits()
        .ok_or(ExpertCacheBenchmarkError::LogitsUnavailable)?;
    drop(logits);
    let after_repeated = expert_snapshot(model)?;
    let repeated_prefill = benchmark_sample(
        started.elapsed(),
        after_cold.prefill,
        after_repeated.prefill,
        after_repeated,
    );

    let started = std::time::Instant::now();
    let output = {
        let (backend, session) = model.runtime.parts_mut();
        session
            .submit_token_decode(backend, token_ids[token_ids.len() - 1])
            .map_err(|error| {
                BackendFailure::from_error(error).with_operation("expert-cache decode submission")
            })?
    }
    .wait()
    .map_err(|error| {
        BackendFailure::from_error(error).with_operation("expert-cache decode completion")
    })?;
    let logits = output
        .into_logits()
        .ok_or(ExpertCacheBenchmarkError::LogitsUnavailable)?;
    drop(logits);
    let after_decode = expert_snapshot(model)?;
    let cached_decode = benchmark_sample(
        started.elapsed(),
        after_repeated.decode,
        after_decode.decode,
        after_decode,
    );

    Ok(LocalExpertCacheBenchmark {
        cold_prefill,
        repeated_prefill,
        cached_decode,
    })
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::{validate_expert_cache_benchmark_prompt, ExpertCacheBenchmarkError};

    #[test]
    fn empty_benchmark_prompt_is_a_facade_input_error() {
        assert!(matches!(
            validate_expert_cache_benchmark_prompt(&[]),
            Err(ExpertCacheBenchmarkError::EmptyPrompt)
        ));
        validate_expert_cache_benchmark_prompt(&[1]).unwrap();
    }
}

/// Releases unused local allocator buffers without changing the configured policy.
/// Active tensors and unresolved native submissions remain protected.
pub fn trim_local_allocator_cache() -> Result<(), eredu_core::BackendFailure> {
    eredu_backend_mlx::trim_allocator_cache().map_err(|error| {
        eredu_core::BackendFailure::from_error(error).with_operation("allocator-cache trimming")
    })
}
