//! MLX runtime configuration, inspection, and diagnostics for generic facade models.

use std::path::Path;
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

/// Discovers hardware available to the MLX backend.
/// Available memory is a point-in-time estimate, not an allocation guarantee.
pub fn discover_local_hardware() -> eredu_core::HardwareProfile {
    eredu_backend_mlx::discover_hardware()
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

impl super::GenerationMemoryOptions {
    /// Creates cold-forecast options for the exact selected local device.
    ///
    /// Observes local hardware once, derives placement and available execution
    /// and host capacity, and samples allocator overhead. Application limits and
    /// reserves remain unset. Missing capacity/overhead stays unknown; an invalid
    /// or undiscovered device is rejected. This does not load a model, allocate
    /// tensors, or change allocator policy; diagnostic queries may initialize
    /// the native runtime. Availability is a point-in-time observation.
    pub fn for_local_device(
        input: eredu_core::InputTokenCount,
        device: &crate::DevicePlan,
    ) -> Result<Self, super::GenerationForecastError> {
        let host_execution = eredu_backend_mlx::MlxBackendFactory::default()
            .device_uses_host_memory(device)
            .map_err(|error| eredu_core::CapabilityError::InvalidConfiguration {
                field: "device",
                detail: error.to_string(),
            })?;
        let hardware = discover_local_hardware();
        let mut options = Self::for_hardware_device(input, &hardware, device, host_execution)?;
        options.backend_overhead = observed_local_allocator_overhead();
        Ok(options)
    }

    /// Creates local-backend options with an observed allocator-cache allowance.
    /// Use [`Self::for_local_device`] to also derive placement and availability
    /// from a selected device. This explicit-placement constructor is unchanged.
    ///
    /// Overhead uses the larger of the current cache limit and retained cache,
    /// plus a 64 MiB graph/driver planning allowance. This is an estimated bound,
    /// not a total-process limit. Other execution mechanisms remain explicit.
    /// Query failure or an unrepresentable bound leaves overhead unknown.
    ///
    /// This diagnostic may initialize the native allocator; cold inspection
    /// itself does not query native resources. Recreate options after changing
    /// the process-global cache limit. Callers may override `backend_overhead`.
    pub fn for_local_backend(
        input: eredu_core::InputTokenCount,
        placement: super::GenerationMemoryPlacement,
    ) -> Self {
        let mut options = Self::new(input, placement);
        options.backend_overhead = observed_local_allocator_overhead();
        options
    }
}

fn observed_local_allocator_overhead() -> super::MemoryBytes {
    match local_allocator_cache_policy()
        .and_then(|policy| allocator_telemetry().map(|memory| (policy, memory.cache_bytes)))
    {
        Ok((policy, cached)) => {
            let mut overhead = local_allocator_overhead(policy.limit_bytes, cached);
            overhead
                .detail
                .push_str(&format!("; native policy provenance: {:?}", policy.source));
            overhead
        }
        Err(error) => super::MemoryBytes::unknown(format!(
            "local allocator-cache observation failed: {error}"
        )),
    }
}

fn local_allocator_overhead(limit: u64, cached: u64) -> super::MemoryBytes {
    super::ForecastCalibration::default().allocator_overhead(limit, cached)
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
    /// Selects model-scoped retained conversion payload at load time. `None`
    /// uses the finite 256 MiB managed default for eligible resident executions.
    /// This is independent of allocator caching and does not limit temporary casts.
    /// Zero normalizes to disabled; unlimited reuse requires an explicit request.
    /// Host-layerwise, disk-streamed and explicit device ceilings disable retention.
    /// Inspect [`super::LoadedModel::parameter_conversion_retention`] for effective
    /// policy and live usage. Reset preserves conversions; settled trimming releases
    /// optional claims without reloading weights or changing the selected policy.
    pub fn with_parameter_conversion_retention(
        mut self,
        policy: Option<eredu_core::residency::ParameterConversionRetentionPolicy>,
    ) -> Self {
        self.normalized = self.normalized.with_parameter_conversion_retention(policy);
        self
    }

    /// Explicit retention request, or `None` for the managed default.
    pub const fn parameter_conversion_retention(
        &self,
    ) -> Option<eredu_core::residency::ParameterConversionRetentionPolicy> {
        self.normalized.parameter_conversion_retention()
    }

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
) -> Result<crate::ModelInspectionReport, BackendFailure> {
    eredu_backend_mlx::native::inspect_model(
        path,
        eredu_backend_mlx::native::MlxInspectionOptions::new(options.load().into_backend()),
    )
    .map_err(|error| BackendFailure::from_error(error).with_operation("model inspection"))
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
    /// Cap an untouched native default at 256 MiB, preserving smaller defaults.
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
mod tests {
    use super::{
        default_local_device, local_device_plan, validate_expert_cache_benchmark_prompt,
        DevicePlanError, ExpertCacheBenchmarkError, LocalDevice,
    };

    #[test]
    fn local_overhead_covers_retained_cache_and_checks_overflow() {
        let allowance = 64 * 1024 * 1024;
        assert_eq!(
            super::local_allocator_overhead(0, 0).upper_bytes,
            Some(allowance)
        );
        assert_eq!(
            super::local_allocator_overhead(1024, 4096).upper_bytes,
            Some(4096 + allowance)
        );
        assert_eq!(
            super::local_allocator_overhead(8192, 4096).upper_bytes,
            Some(8192 + allowance)
        );
        assert_eq!(
            super::local_allocator_overhead(0, u64::MAX).upper_bytes,
            None
        );
    }

    #[test]
    fn empty_benchmark_prompt_is_a_facade_input_error() {
        assert!(matches!(
            validate_expert_cache_benchmark_prompt(&[]),
            Err(ExpertCacheBenchmarkError::EmptyPrompt)
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
            assert_eq!(plan, Err(DevicePlanError::AcceleratorNotCompiled));
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
