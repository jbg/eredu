//! Application-facing adapter for the execution backend selected by `eredu` features.
//!
//! This surface intentionally exposes only the native types currently needed to
//! configure and operate a local model session. Backend implementation modules,
//! architecture composition, and reusable MLX infrastructure remain owned by
//! `eredu-backend-mlx`.

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
/// Routed-expert counters for one execution-path class.
pub use eredu_backend_mlx::ExpertPassStatistics as LocalExpertPassStatistics;
/// Routed-expert counters for one residency tier.
pub use eredu_backend_mlx::ExpertTierStatistics as LocalExpertTierStatistics;
/// Raw input part accepted by the selected local backend.
pub use eredu_backend_mlx::InputPart as LocalInputPart;
/// Stateful Mirostat V2 sampler for raw local generation.
pub use eredu_backend_mlx::MirostatV2Sampler as LocalMirostatV2Sampler;
/// Backend selected for local model execution.
pub use eredu_backend_mlx::MlxBackend as LocalBackend;
/// Automatic planner and execution-plan factory for the selected local backend.
pub use eredu_backend_mlx::MlxBackendFactory as LocalBackendFactory;
/// Options for selected-local-backend model inspection.
pub use eredu_backend_mlx::MlxInspectionOptions as LocalInspectionOptions;
/// Prepared raw input owned by the selected local backend.
pub use eredu_backend_mlx::MlxModelInput as LocalPreparedModelInput;
/// Raw model input accepted by the selected local backend.
pub use eredu_backend_mlx::ModelInput as LocalModelInput;
/// Load policy accepted by the selected local backend.
pub use eredu_backend_mlx::ModelLoadOptions as LocalLoadOptions;
/// Scoped opt-in for selected-backend MTP component timing.
pub use eredu_backend_mlx::MtpComponentTimingGuard as LocalMtpComponentTimingGuard;
/// Sampler contract used by raw local generation.
pub use eredu_backend_mlx::Sampler as LocalSampler;
