//! Application-facing adapter for the execution backend selected by `eredu` features.
//!
//! This surface intentionally exposes only the native types currently needed to
//! configure and operate a local model session. Backend implementation modules,
//! architecture composition, and reusable MLX infrastructure remain owned by
//! `eredu-backend-mlx`.

/// Stateful Mirostat V2 sampler for raw local generation.
pub use eredu_backend_mlx::backend::mlx::runtime::generation::sampler::MirostatV2Sampler as LocalMirostatV2Sampler;
/// Sampler contract used by raw local generation.
pub use eredu_backend_mlx::backend::mlx::runtime::generation::sampler::Sampler as LocalSampler;
/// Raw input part accepted by the selected local backend.
pub use eredu_backend_mlx::backend::mlx::runtime::media::input::InputPart as LocalInputPart;
/// Raw model input accepted by the selected local backend.
pub use eredu_backend_mlx::backend::mlx::runtime::media::input::ModelInput as LocalModelInput;
/// Routed-expert counters for one execution-path class.
pub use eredu_backend_mlx::backend::mlx::runtime::residency::expert_cache::ExpertPassStatistics as LocalExpertPassStatistics;
/// Routed-expert counters for one residency tier.
pub use eredu_backend_mlx::backend::mlx::runtime::residency::expert_cache::ExpertTierStatistics as LocalExpertTierStatistics;
/// Backend selected for local model execution.
pub use eredu_backend_mlx::backend::mlx::MlxBackend as LocalBackend;
/// Load policy accepted by the selected local backend.
pub use eredu_backend_mlx::backend::mlx::ModelLoadOptions as LocalLoadOptions;
/// Discovers hardware available to the selected local backend.
pub use eredu_backend_mlx::composition::mlx::automatic::discover_hardware as discover_local_hardware;
/// Converts a selected-backend expert-cache report into portable telemetry.
pub use eredu_backend_mlx::composition::mlx::automatic::expert_cache_telemetry as local_expert_cache_telemetry;
/// Converts speculative statistics into portable telemetry.
pub use eredu_backend_mlx::composition::mlx::automatic::mtp_telemetry as local_mtp_telemetry;
/// Converts a selected-backend residency report into portable telemetry.
pub use eredu_backend_mlx::composition::mlx::automatic::residency_telemetry as local_residency_telemetry;
/// Automatic planner and execution-plan factory for the selected local backend.
pub use eredu_backend_mlx::composition::mlx::automatic::MlxBackendFactory as LocalBackendFactory;
/// Inspects a model using the selected local backend.
pub use eredu_backend_mlx::composition::mlx::inspect_model as inspect_local_model;
/// Scoped opt-in for selected-backend MTP component timing.
pub use eredu_backend_mlx::composition::mlx::speculative::MtpComponentTimingGuard as LocalMtpComponentTimingGuard;
/// Options for selected-local-backend model inspection.
pub use eredu_backend_mlx::composition::mlx::MlxInspectionOptions as LocalInspectionOptions;
/// Prepared raw input owned by the selected local backend.
pub use eredu_backend_mlx::composition::mlx::MlxModelInput as LocalPreparedModelInput;
