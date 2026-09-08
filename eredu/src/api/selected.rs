//! MLX runtime configuration, inspection, and diagnostics for generic facade models.

use std::path::Path;
#[cfg(all(feature = "metal", target_vendor = "apple"))]
use std::path::PathBuf;
use std::time::Duration;

/// Discovers hardware available to the MLX backend.
pub fn discover_local_hardware() -> eredu_core::HardwareProfile {
    eredu_backend_mlx::discover_hardware()
}

/// Failure reported by MLX runtime configuration, inspection, and diagnostics.
///
/// The diagnostic message retains the backend's original context. Generic model
/// loading and generation use their backend-parameterized error types.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[error("selected local backend failed during {operation}: {message}")]
pub struct LocalBackendError {
    operation: &'static str,
    message: String,
}

impl LocalBackendError {
    fn new(operation: &'static str, error: impl std::fmt::Display) -> Self {
        Self {
            operation,
            message: error.to_string(),
        }
    }

    /// Facade operation that failed.
    pub const fn operation(&self) -> &'static str {
        self.operation
    }

    /// Backend diagnostic without exposing its native error type.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl super::LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>> {
    /// Resets this model's session state.
    pub fn reset(&mut self) -> Result<(), LocalBackendError> {
        self.runtime
            .session_mut()
            .reset()
            .map_err(|error| LocalBackendError::new("session reset", error))
    }

    /// Waits for all work submitted by this model.
    pub fn synchronize(&self) -> Result<(), LocalBackendError> {
        self.runtime
            .backend()
            .synchronize()
            .map_err(|error| LocalBackendError::new("synchronization", error))
    }

    /// Synchronizes the model and samples allocator counters.
    pub fn allocator_telemetry(&self) -> Result<crate::AllocatorTelemetry, LocalBackendError> {
        self.synchronize()?;
        allocator_telemetry()
    }

    /// Returns portable sparse expert-cache telemetry when available.
    pub fn expert_cache_telemetry(
        &self,
    ) -> Result<Option<crate::ExpertCacheTelemetry>, LocalBackendError> {
        self.runtime
            .session()
            .parameter_bank_report()
            .map(|report| {
                report
                    .as_ref()
                    .map(eredu_backend_mlx::parameter_bank_telemetry)
            })
            .map_err(|error| LocalBackendError::new("expert-cache telemetry", error))
    }

    /// Returns portable weight-residency telemetry when available.
    pub fn residency_telemetry(
        &self,
    ) -> Result<Option<crate::ResidencyTelemetry>, LocalBackendError> {
        self.runtime
            .session()
            .residency_report()
            .map_err(|error| LocalBackendError::new("residency telemetry", error))
            .map(|report| report.as_ref().map(eredu_runtime::residency_telemetry))
    }
}

/// Facade-owned policy for loading a model on the selected local backend.
///
/// Distributed native device bindings are intentionally absent. Application
/// clients select portable placement and topology through an
/// [`crate::ExecutionPlan`].
#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct LocalLoadOptions {
    normalized: eredu_runtime::NormalizedLoadRequest,
}

impl LocalLoadOptions {
    /// Creates load options that quantize eligible dense weights on load.
    pub fn with_quantization(quantization: crate::QuantizationRequest) -> Self {
        Self {
            normalized: eredu_runtime::NormalizedLoadRequest::with_quantization(quantization),
        }
    }

    /// Selects fully resident or bounded layer execution for checkpoint weights.
    pub fn with_weight_residency(mut self, residency: eredu_runtime::WeightResidency) -> Self {
        self.normalized = self.normalized.with_weight_residency(residency);
        self
    }

    /// Requires capabilities from the exact inspected and realized session.
    pub fn with_required_session_capabilities(
        mut self,
        capabilities: crate::SessionCapabilities,
    ) -> Self {
        self.normalized = self
            .normalized
            .with_required_session_capabilities(capabilities);
        self
    }

    /// Requested dense-weight transformation, if any.
    pub const fn quantization(&self) -> Option<crate::QuantizationRequest> {
        self.normalized.quantization()
    }

    /// Selected immutable-weight residency policy.
    pub const fn weight_residency(&self) -> eredu_runtime::WeightResidency {
        self.normalized.weight_residency()
    }

    /// Capabilities required from the realized session.
    pub const fn required_session_capabilities(&self) -> crate::SessionCapabilities {
        self.normalized.required_session_capabilities()
    }

    /// Portable drafting policy retained from explicit or planned loading.
    pub const fn drafting(&self) -> eredu_runtime::DraftingLoadRequest {
        self.normalized.drafting()
    }

    fn into_backend(self) -> eredu_backend_mlx::MlxLoadRequest {
        eredu_backend_mlx::MlxLoadRequest::from_normalized(self.normalized)
    }

    fn from_backend(
        options: eredu_backend_mlx::MlxLoadRequest,
    ) -> Result<Self, crate::AutomaticPlanningError> {
        if options.normalized().has_parallel_execution() {
            return Err(crate::AutomaticPlanningError::Invalid(
                "selected-local inspection options cannot contain a native parallel context; use a portable execution plan"
                    .into(),
            ));
        }
        Ok(Self {
            normalized: options.normalized().clone(),
        })
    }
}

/// Facade-owned options for selected-local-backend model inspection.
#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct LocalInspectionOptions {
    /// The exact facade loading policy that admission should validate.
    load: LocalLoadOptions,
}

impl LocalInspectionOptions {
    /// Creates inspection options for one exact facade load request.
    pub const fn new(load: LocalLoadOptions) -> Self {
        Self { load }
    }

    /// Returns the load request whose feasibility is being inspected.
    pub fn load(&self) -> LocalLoadOptions {
        self.load.clone()
    }

    /// Derives inspection options from a portable execution plan.
    pub fn for_execution_plan(
        factory: &eredu_backend_mlx::MlxBackendFactory,
        plan: &crate::ExecutionPlan,
    ) -> Result<Self, crate::AutomaticPlanningError> {
        let options = factory.load_request_for_plan(plan)?;
        Ok(Self::new(LocalLoadOptions::from_backend(options)?))
    }
}

/// Inspects a model using facade-owned options and errors.
pub fn inspect_local_model(
    path: impl AsRef<Path>,
    options: LocalInspectionOptions,
) -> Result<crate::ModelInspectionReport, LocalBackendError> {
    eredu_backend_mlx::native::inspect_model(
        path,
        eredu_backend_mlx::native::MlxInspectionOptions::new(options.load().into_backend()),
    )
    .map_err(|error| LocalBackendError::new("model inspection", error))
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

/// Failure to map a facade device choice to the selected local backend.
#[derive(Debug, Clone, Copy, Eq, PartialEq, thiserror::Error)]
pub enum LocalDevicePlanError {
    /// This build contains the MLX adapter but no native accelerator family.
    #[error("no local accelerator family is compiled for this target")]
    AcceleratorNotCompiled,
}

/// Process-global configuration for the selected local runtime.
#[derive(Debug, Clone, Default)]
pub struct LocalRuntimeConfiguration {
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    accelerator_library_path: Option<PathBuf>,
    allocator_cache_limit: Option<usize>,
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
        self.allocator_cache_limit = Some(bytes);
        self
    }
}

/// Applies process-global configuration before creating a local model session.
pub fn configure_local_runtime(
    configuration: &LocalRuntimeConfiguration,
) -> Result<(), LocalBackendError> {
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    if let Some(path) = &configuration.accelerator_library_path {
        eredu_backend_mlx::set_accelerator_library_path(path)
            .map_err(|error| LocalBackendError::new("runtime configuration", error))?;
    }
    if let Some(bytes) = configuration.allocator_cache_limit {
        eredu_backend_mlx::set_allocator_cache_limit(bytes)
            .map_err(|error| LocalBackendError::new("allocator configuration", error))?;
    }
    Ok(())
}

/// Creates a portable plan device for the selected local backend.
pub fn local_device_plan(device: LocalDevice) -> Result<crate::DevicePlan, LocalDevicePlanError> {
    let device = match device {
        LocalDevice::Cpu => "cpu:0".to_owned(),
        LocalDevice::Accelerator(index) => {
            let family = compiled_accelerator_family()
                .ok_or(LocalDevicePlanError::AcceleratorNotCompiled)?;
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
pub fn reset_local_allocator_peak() -> Result<(), LocalBackendError> {
    eredu_backend_mlx::reset_allocator_peak()
        .map_err(|error| LocalBackendError::new("allocator peak reset", error))?;
    Ok(())
}

fn allocator_telemetry() -> Result<crate::AllocatorTelemetry, LocalBackendError> {
    let memory = eredu_backend_mlx::allocator_memory()
        .map_err(|error| LocalBackendError::new("allocator telemetry", error))?;
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

/// Failure while running the facade-owned expert-cache benchmark workflow.
#[derive(Debug, thiserror::Error)]
pub enum LocalExpertCacheBenchmarkError {
    /// The benchmark needs a non-empty prompt for prefill and cached decode.
    #[error("expert-cache benchmark requires at least one prompt token")]
    EmptyPrompt,
    /// The selected model does not expose sparse expert-cache telemetry.
    #[error("sparse expert-cache benchmark requires an expert-cache model")]
    ExpertCacheUnavailable,
    /// The local rank did not produce logits needed to complete a benchmark phase.
    #[error("expert-cache benchmark requires logits on the local rank")]
    LogitsUnavailable,
    /// The selected backend failed while preparing or executing the benchmark.
    #[error(transparent)]
    Backend(#[from] LocalBackendError),
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
) -> Result<ExpertSnapshot, LocalExpertCacheBenchmarkError> {
    let report = model
        .runtime
        .session()
        .parameter_bank_report()
        .map_err(|error| LocalBackendError::new("expert-cache telemetry", error))?
        .ok_or(LocalExpertCacheBenchmarkError::ExpertCacheUnavailable)?;
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
) -> Result<(), LocalExpertCacheBenchmarkError> {
    if token_ids.is_empty() {
        return Err(LocalExpertCacheBenchmarkError::EmptyPrompt);
    }
    Ok(())
}

/// Benchmarks selected-backend expert-cache reuse without exposing tensors or streams.
pub fn benchmark_local_expert_cache(
    model: &mut super::LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>,
    token_ids: &[u32],
) -> Result<LocalExpertCacheBenchmark, LocalExpertCacheBenchmarkError> {
    validate_expert_cache_benchmark_prompt(token_ids)?;
    let prompt = <eredu_backend_mlx::backend::MlxBackend as eredu_core::TextGenerationBackend>::prepare_text_prompt(
        model.runtime.backend(),
        token_ids.to_vec(),
    )
    .map_err(|error| LocalBackendError::new("expert-cache prompt preparation", error))?;

    let before_cold = expert_snapshot(model)?;
    model
        .runtime
        .session_mut()
        .reset()
        .map_err(|error| LocalBackendError::new("expert-cache session reset", error))?;
    let started = std::time::Instant::now();
    let logits = model
        .runtime
        .prefill(prompt.clone())
        .map_err(|error| LocalBackendError::new("expert-cache prefill submission", error))?
        .wait()
        .map_err(|error| LocalBackendError::new("expert-cache prefill completion", error))?
        .into_logits()
        .ok_or(LocalExpertCacheBenchmarkError::LogitsUnavailable)?;
    drop(logits);
    let after_cold = expert_snapshot(model)?;
    let cold_prefill = benchmark_sample(
        started.elapsed(),
        before_cold.prefill,
        after_cold.prefill,
        after_cold,
    );

    model
        .runtime
        .session_mut()
        .reset()
        .map_err(|error| LocalBackendError::new("expert-cache session reset", error))?;
    let started = std::time::Instant::now();
    let logits = model
        .runtime
        .prefill(prompt)
        .map_err(|error| LocalBackendError::new("expert-cache prefill submission", error))?
        .wait()
        .map_err(|error| LocalBackendError::new("expert-cache prefill completion", error))?
        .into_logits()
        .ok_or(LocalExpertCacheBenchmarkError::LogitsUnavailable)?;
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
            .map_err(|error| LocalBackendError::new("expert-cache decode submission", error))?
    }
    .wait()
    .map_err(|error| LocalBackendError::new("expert-cache decode completion", error))?;
    let logits = output
        .into_logits()
        .ok_or(LocalExpertCacheBenchmarkError::LogitsUnavailable)?;
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
mod tests {
    use super::{
        default_local_device, local_device_plan, validate_expert_cache_benchmark_prompt,
        LocalDevice, LocalDevicePlanError, LocalExpertCacheBenchmarkError,
    };

    #[test]
    fn empty_benchmark_prompt_is_a_facade_input_error() {
        assert!(matches!(
            validate_expert_cache_benchmark_prompt(&[]),
            Err(LocalExpertCacheBenchmarkError::EmptyPrompt)
        ));
        validate_expert_cache_benchmark_prompt(&[1]).unwrap();
    }

    #[test]
    fn local_accelerator_plan_names_the_compiled_family() {
        let plan = local_device_plan(LocalDevice::Accelerator(3));
        if cfg!(feature = "cuda") {
            assert_eq!(plan.unwrap().device(), "cuda:3");
        } else if cfg!(all(feature = "metal", target_vendor = "apple")) {
            assert_eq!(plan.unwrap().device(), "metal:3");
        } else {
            assert_eq!(plan, Err(LocalDevicePlanError::AcceleratorNotCompiled));
        }
    }

    #[test]
    fn default_device_uses_an_available_accelerator_or_cpu() {
        let expected = if cfg!(any(
            feature = "cuda",
            all(feature = "metal", target_vendor = "apple")
        )) {
            LocalDevice::Accelerator(0)
        } else {
            LocalDevice::Cpu
        };
        let device = default_local_device();
        assert_eq!(device, expected);
        assert!(local_device_plan(device).is_ok());
    }
}
