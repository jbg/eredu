//! Application-facing adapter for the execution backend selected by `eredu` features.
//!
//! Native device, stream, tensor, random-state, and allocator handles stay
//! behind this module. Applications configure and operate the selected local
//! backend through portable plans and facade-owned diagnostics.

#[cfg(feature = "metal")]
use std::path::PathBuf;
use std::time::{Duration, Instant};

use eredu_core::{RealtimeBackend as _, RealtimeModelLoadingBackend as _};

/// Discovers hardware available to the selected local backend.
pub use eredu_backend_mlx::discover_hardware as discover_local_hardware;
/// Converts a selected-backend expert-cache report into portable telemetry.
pub use eredu_backend_mlx::expert_cache_telemetry as local_expert_cache_telemetry;
/// Inspects a model using the selected local backend.
pub use eredu_backend_mlx::inspect_model as inspect_local_model;
/// Converts speculative statistics into portable telemetry.
pub use eredu_backend_mlx::mtp_telemetry as local_mtp_telemetry;
/// Converts a selected-backend residency report into portable telemetry.
pub use eredu_backend_mlx::residency_telemetry as local_residency_telemetry;
/// Backend selected for local model execution.
///
/// Native streams remain private to the selected backend:
///
/// ```compile_fail
/// fn native_stream(backend: &eredu::api::LocalBackend<'_>) {
///     let _ = backend.stream();
/// }
/// ```
pub use eredu_backend_mlx::MlxBackend as LocalBackend;
/// Automatic planner and execution-plan factory for the selected local backend.
pub use eredu_backend_mlx::MlxBackendFactory as LocalBackendFactory;
/// Error reported by the selected local backend.
pub use eredu_backend_mlx::MlxError as LocalBackendError;
/// Options for selected-local-backend model inspection.
pub use eredu_backend_mlx::MlxInspectionOptions as LocalInspectionOptions;
/// Load policy accepted by the selected local backend.
pub use eredu_backend_mlx::ModelLoadOptions as LocalLoadOptions;
/// Scoped opt-in for selected-backend MTP component timing.
pub use eredu_backend_mlx::MtpComponentTimingGuard as LocalMtpComponentTimingGuard;

/// A facade-level device class for the selected local backend.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum LocalDevice {
    /// Host CPU execution.
    Cpu,
    /// The selected native accelerator at this zero-based index.
    Accelerator(u32),
}

/// Failure to map a facade device choice to the selected local backend.
#[derive(Debug, Clone, Copy, Eq, PartialEq, thiserror::Error)]
pub enum LocalDevicePlanError {
    /// This build contains the MLX adapter but no native accelerator family.
    #[error("no local accelerator family is compiled for this target")]
    AcceleratorNotCompiled,
}

/// Factory for local realtime model loading and execution.
#[derive(Debug, Clone, Copy)]
pub struct LocalRealtimeBackendFactory {
    device: LocalDevice,
}

impl LocalRealtimeBackendFactory {
    /// Selects the device used for realtime model execution.
    pub const fn new(device: LocalDevice) -> Self {
        Self { device }
    }

    /// Loads an architecture-prepared realtime model with the default policy.
    pub fn load(
        &self,
        preparation: crate::RealtimePreparationPlan,
    ) -> Result<LocalRealtimeModel, LocalBackendError> {
        self.load_with_options(preparation, LocalLoadOptions::default())
    }

    /// Loads an architecture-prepared realtime model with explicit policy.
    pub fn load_with_options(
        &self,
        preparation: crate::RealtimePreparationPlan,
        options: LocalLoadOptions,
    ) -> Result<LocalRealtimeModel, LocalBackendError> {
        let device = local_device_plan(self.device)
            .map_err(|error| LocalBackendError::AutomaticPlanning(error.to_string()))?;
        let backend = eredu_backend_mlx::create_realtime_backend(&device)?;
        let model = backend.materialize_realtime_model(preparation, options)?;
        Ok(LocalRealtimeModel {
            inner: eredu_core::RealtimeModel::new(backend, model),
        })
    }
}

impl Default for LocalRealtimeBackendFactory {
    fn default() -> Self {
        Self::new(LocalDevice::Cpu)
    }
}

/// Realtime coordination failure from the selected local backend.
pub type LocalRealtimeError = eredu_core::RealtimeError<LocalBackendError>;

/// A selected local realtime backend and its loaded model.
///
/// Native models, tensors, streams, and backend traits remain private. Use a
/// [`LocalRealtimeScheduler`] to submit portable [`crate::RealtimeInputFrame`]
/// values and observe portable [`crate::RealtimeOutputFrame`] values.
pub struct LocalRealtimeModel {
    inner: eredu_core::RealtimeModel<eredu_backend_mlx::MlxRealtimeAdapter>,
}

impl LocalRealtimeModel {
    /// Name of the selected execution backend.
    pub fn backend_name(&self) -> &str {
        self.inner.backend().name()
    }

    /// Portable codec-token geometry for this model.
    pub fn speech_config(&self) -> crate::RealtimeSpeechConfig {
        self.inner.speech_config()
    }
}

/// Request-local selected-backend state released from a realtime scheduler.
pub struct LocalRealtimeSession {
    inner: eredu_core::RealtimeSession<eredu_backend_mlx::MlxRealtimeAdapter>,
}

impl LocalRealtimeSession {
    /// Committed batch dimension, when at least one frame was accepted.
    pub fn batch_size(&self) -> Option<usize> {
        self.inner.batch_size()
    }
}

/// One completed, portable realtime transition.
pub struct LocalRealtimeCompletedStep {
    work: crate::WorkId,
    output: crate::RealtimeOutputFrame,
}

impl LocalRealtimeCompletedStep {
    /// Scheduler-assigned work identity.
    pub const fn work(&self) -> crate::WorkId {
        self.work
    }

    /// Borrows the completed host token frame.
    pub const fn output(&self) -> &crate::RealtimeOutputFrame {
        &self.output
    }

    /// Consumes this completion into its work identity and host token frame.
    pub fn into_parts(self) -> (crate::WorkId, crate::RealtimeOutputFrame) {
        (self.work, self.output)
    }
}

/// Fair bounded realtime scheduler for a [`LocalRealtimeModel`].
///
/// This facade materializes portable host token frames before execution and
/// observes completed backend outputs before returning them to the caller.
pub struct LocalRealtimeScheduler {
    inner: eredu_core::RealtimeScheduler<eredu_backend_mlx::MlxRealtimeAdapter>,
}

impl LocalRealtimeScheduler {
    /// Creates an empty scheduler bound to one loaded local model.
    pub fn new(
        model: &LocalRealtimeModel,
        limits: crate::SchedulerLimits,
    ) -> Result<Self, LocalRealtimeError> {
        Ok(Self {
            inner: eredu_core::RealtimeScheduler::new(&model.inner, limits)?,
        })
    }

    /// Registers a request with fresh selected-backend state.
    pub fn register_request(
        &mut self,
        model: &LocalRealtimeModel,
        request: crate::RequestId,
        sampling: crate::RealtimeSampling,
    ) -> Result<(), LocalRealtimeError> {
        self.inner.register_request(&model.inner, request, sampling)
    }

    /// Registers a previously released request session.
    pub fn register_request_with_session(
        &mut self,
        model: &LocalRealtimeModel,
        request: crate::RequestId,
        session: LocalRealtimeSession,
    ) -> Result<(), LocalRealtimeError> {
        self.inner
            .register_request_with_session(&model.inner, request, session.inner)
    }

    /// Materializes and enqueues one portable host token frame.
    pub fn enqueue(
        &mut self,
        model: &LocalRealtimeModel,
        request: crate::RequestId,
        frame: crate::RealtimeInputFrame,
    ) -> Result<crate::WorkId, LocalRealtimeError> {
        self.enqueue_with_deadline(model, request, frame, None)
    }

    /// Materializes and enqueues one frame with an absolute deadline.
    pub fn enqueue_with_deadline(
        &mut self,
        model: &LocalRealtimeModel,
        request: crate::RequestId,
        frame: crate::RealtimeInputFrame,
        deadline: Option<Instant>,
    ) -> Result<crate::WorkId, LocalRealtimeError> {
        let input = model
            .inner
            .backend()
            .materialize_input(model.inner.model(), &frame)
            .map_err(eredu_core::RealtimeError::Backend)?;
        self.inner
            .enqueue_with_deadline(&model.inner, request, input, deadline)
    }

    /// Atomically materializes and enqueues ordered portable host token frames.
    pub fn enqueue_batch(
        &mut self,
        model: &LocalRealtimeModel,
        request: crate::RequestId,
        frames: Vec<crate::RealtimeInputFrame>,
    ) -> Result<Vec<crate::WorkId>, LocalRealtimeError> {
        let inputs = frames
            .iter()
            .map(|frame| {
                model
                    .inner
                    .backend()
                    .materialize_input(model.inner.model(), frame)
                    .map_err(eredu_core::RealtimeError::Backend)
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.inner.enqueue_batch(&model.inner, request, inputs)
    }

    /// Advances one unbounded fair scheduling turn and observes completed frames.
    pub fn run_queued(
        &mut self,
        model: &mut LocalRealtimeModel,
    ) -> Result<Vec<LocalRealtimeCompletedStep>, LocalRealtimeError> {
        let completed = self.inner.run_queued(&mut model.inner)?;
        observe_realtime_steps(model, completed)
    }

    /// Advances at most `max_frames` transitions and observes completed frames.
    pub fn run_bounded(
        &mut self,
        model: &mut LocalRealtimeModel,
        max_frames: usize,
    ) -> Result<Vec<LocalRealtimeCompletedStep>, LocalRealtimeError> {
        let completed = self.inner.run_bounded(&mut model.inner, max_frames)?;
        observe_realtime_steps(model, completed)
    }

    /// Completes one request and drops its backend session.
    pub fn finish_request(&mut self, request: crate::RequestId) -> Result<(), LocalRealtimeError> {
        self.inner.finish_request(request)
    }

    /// Cancels one request and discards queued frames.
    pub fn cancel_request(&mut self, request: crate::RequestId) -> Result<(), LocalRealtimeError> {
        self.inner.cancel_request(request)
    }

    /// Releases an idle request for persistence or resumption.
    pub fn release_request(
        &mut self,
        request: crate::RequestId,
    ) -> Result<LocalRealtimeSession, LocalRealtimeError> {
        Ok(LocalRealtimeSession {
            inner: self.inner.release_request(request)?,
        })
    }

    /// Removes a terminal identity for explicit reuse.
    pub fn forget_terminal_request(
        &mut self,
        request: crate::RequestId,
    ) -> Result<crate::RequestStatus, LocalRealtimeError> {
        self.inner.forget_terminal_request(request)
    }

    /// Lifecycle state for a known request.
    pub fn request_status(&self, request: crate::RequestId) -> Option<crate::RequestStatus> {
        self.inner.request_status(request)
    }

    /// Queued frame count for one request.
    pub fn queued_for_request(&self, request: crate::RequestId) -> usize {
        self.inner.queued_for_request(request)
    }

    /// Replaces sampling controls for an idle active request.
    pub fn set_request_sampling(
        &mut self,
        model: &LocalRealtimeModel,
        request: crate::RequestId,
        sampling: crate::RealtimeSampling,
    ) -> Result<(), LocalRealtimeError> {
        self.inner
            .set_request_sampling(&model.inner, request, sampling)
    }

    /// Generic occupancy and lifecycle telemetry.
    pub fn report(&self) -> crate::SchedulerReport {
        self.inner.report()
    }

    /// Configured bounds and observed backend capabilities.
    pub fn capabilities(&self) -> crate::SchedulerCapabilities {
        self.inner.capabilities()
    }
}

fn observe_realtime_steps(
    model: &LocalRealtimeModel,
    completed: Vec<eredu_core::RealtimeCompletedStep<eredu_backend_mlx::native::MlxRealtimeOutput>>,
) -> Result<Vec<LocalRealtimeCompletedStep>, LocalRealtimeError> {
    completed
        .into_iter()
        .map(|step| {
            let (work, output) = step.into_parts();
            let output = model
                .inner
                .backend()
                .observe_output(&output)
                .map_err(eredu_core::RealtimeError::Backend)?;
            Ok(LocalRealtimeCompletedStep { work, output })
        })
        .collect()
}

/// Process-global configuration for the selected local runtime.
#[derive(Debug, Clone, Default)]
pub struct LocalRuntimeConfiguration {
    #[cfg(feature = "metal")]
    accelerator_library_path: Option<PathBuf>,
    allocator_cache_limit: Option<usize>,
}

impl LocalRuntimeConfiguration {
    /// Overrides the native accelerator kernel-library path.
    ///
    /// Embedded Apple applications use this when their bundled library cannot
    /// be found through the runtime's default search path.
    #[cfg(feature = "metal")]
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
    #[cfg(feature = "metal")]
    if let Some(path) = &configuration.accelerator_library_path {
        eredu_backend_mlx::native::metal::set_metallib_path(path)?;
    }
    if let Some(bytes) = configuration.allocator_cache_limit {
        eredu_backend_mlx::native::memory::set_cache_limit(bytes)?;
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

/// Waits for work submitted to the selected local backend.
pub fn synchronize_local_backend(backend: &LocalBackend<'_>) -> Result<(), LocalBackendError> {
    backend.synchronize()
}

/// Resets the selected runtime's allocator high-water mark.
pub fn reset_local_allocator_peak() -> Result<(), LocalBackendError> {
    eredu_backend_mlx::native::memory::reset_peak_memory()?;
    Ok(())
}

/// Synchronizes the selected backend and samples its allocator counters.
pub fn local_allocator_telemetry(
    backend: &LocalBackend<'_>,
) -> Result<crate::AllocatorTelemetry, LocalBackendError> {
    synchronize_local_backend(backend)?;
    Ok(crate::AllocatorTelemetry {
        peak_bytes: eredu_backend_mlx::native::memory::peak_memory()? as u64,
        active_bytes: eredu_backend_mlx::native::memory::active_memory()? as u64,
        cache_bytes: eredu_backend_mlx::native::memory::cache_memory()? as u64,
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
    prefill: eredu_backend_mlx::ExpertPassStatistics,
    decode: eredu_backend_mlx::ExpertPassStatistics,
    host_resident_experts: usize,
    host_resident_bytes: u64,
    device_resident_experts: usize,
    device_resident_bytes: u64,
}

fn expert_snapshot(
    runtime: &eredu_core::ModelRuntime<LocalBackend<'static>>,
) -> Result<ExpertSnapshot, LocalExpertCacheBenchmarkError> {
    let report = runtime
        .session()
        .expert_cache_report()?
        .ok_or(LocalExpertCacheBenchmarkError::ExpertCacheUnavailable)?;
    Ok(ExpertSnapshot {
        prefill: report.prefill,
        decode: report.decode,
        host_resident_experts: report.host_resident_experts,
        host_resident_bytes: report.host_resident_bytes,
        device_resident_experts: report.device_resident_experts,
        device_resident_bytes: report.device_resident_bytes,
    })
}

fn benchmark_sample(
    elapsed: Duration,
    before: eredu_backend_mlx::ExpertPassStatistics,
    after: eredu_backend_mlx::ExpertPassStatistics,
    occupancy: ExpertSnapshot,
) -> LocalExpertCacheBenchmarkSample {
    LocalExpertCacheBenchmarkSample {
        elapsed,
        requested_routes: after
            .requested_routes
            .saturating_sub(before.requested_routes),
        distinct_experts: after
            .distinct_experts
            .saturating_sub(before.distinct_experts),
        coalesced_duplicates: after
            .coalesced_duplicates
            .saturating_sub(before.coalesced_duplicates),
        compact_banks: after.compact_banks.saturating_sub(before.compact_banks),
        compact_bank_bytes: after
            .compact_bank_bytes
            .saturating_sub(before.compact_bank_bytes),
        host_hits: after.host.hits.saturating_sub(before.host.hits),
        host_misses: after.host.misses.saturating_sub(before.host.misses),
        host_evictions: after.host.evictions.saturating_sub(before.host.evictions),
        device_hits: after.device.hits.saturating_sub(before.device.hits),
        device_misses: after.device.misses.saturating_sub(before.device.misses),
        device_evictions: after
            .device
            .evictions
            .saturating_sub(before.device.evictions),
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
    runtime: &mut eredu_core::ModelRuntime<LocalBackend<'static>>,
    token_ids: &[u32],
) -> Result<LocalExpertCacheBenchmark, LocalExpertCacheBenchmarkError> {
    validate_expert_cache_benchmark_prompt(token_ids)?;
    let prompt = <LocalBackend<'static> as eredu_core::TextGenerationBackend>::prepare_text_prompt(
        runtime.backend(),
        token_ids.to_vec(),
    )?;

    let before_cold = expert_snapshot(runtime)?;
    runtime.session_mut().reset()?;
    let started = std::time::Instant::now();
    let logits = runtime
        .prefill(prompt.clone())?
        .wait()?
        .into_logits()
        .ok_or(LocalExpertCacheBenchmarkError::LogitsUnavailable)?;
    drop(logits);
    let after_cold = expert_snapshot(runtime)?;
    let cold_prefill = benchmark_sample(
        started.elapsed(),
        before_cold.prefill,
        after_cold.prefill,
        after_cold,
    );

    runtime.session_mut().reset()?;
    let started = std::time::Instant::now();
    let logits = runtime
        .prefill(prompt)?
        .wait()?
        .into_logits()
        .ok_or(LocalExpertCacheBenchmarkError::LogitsUnavailable)?;
    drop(logits);
    let after_repeated = expert_snapshot(runtime)?;
    let repeated_prefill = benchmark_sample(
        started.elapsed(),
        after_cold.prefill,
        after_repeated.prefill,
        after_repeated,
    );

    let started = std::time::Instant::now();
    let output = {
        let (backend, session) = runtime.parts_mut();
        session.submit_token_decode(backend, token_ids[token_ids.len() - 1])?
    }
    .wait()?;
    let logits = output
        .into_logits()
        .ok_or(LocalExpertCacheBenchmarkError::LogitsUnavailable)?;
    drop(logits);
    let after_decode = expert_snapshot(runtime)?;
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
        local_device_plan, validate_expert_cache_benchmark_prompt, LocalDevice,
        LocalDevicePlanError, LocalExpertCacheBenchmarkError,
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
            assert_eq!(plan.unwrap().device, "cuda:3");
        } else if cfg!(all(feature = "metal", target_vendor = "apple")) {
            assert_eq!(plan.unwrap().device, "metal:3");
        } else {
            assert_eq!(plan, Err(LocalDevicePlanError::AcceleratorNotCompiled));
        }
    }
}
