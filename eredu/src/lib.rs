#![doc = include_str!("../README.md")]
#![warn(missing_docs)]
#![cfg_attr(test, allow(dead_code))]
// Backend execution boundaries intentionally pass complete runtime context;
// concrete models remain unboxed; builder helpers can return typed builders;
// explicit drops delimit provider borrows; and MLX completions are process-local.
#![allow(
    clippy::arc_with_non_send_sync,
    clippy::drop_non_drop,
    clippy::large_enum_variant,
    clippy::new_ret_no_self,
    clippy::too_many_arguments,
    clippy::type_complexity
)]

#[cfg(all(feature = "metal", feature = "cuda"))]
compile_error!(
    "the `metal` and `cuda` MLX backend features are mutually exclusive; disable default features before enabling `cuda`"
);

/// High-level model loading, dispatch, and request APIs.
pub mod api;
/// Backend-independent chat and committed-generation orchestration.
pub mod runtime;

// Internal bindings keep implementation paths concise without adding public
// aliases for contracts owned by dependency crates.
#[cfg(feature = "mlx")]
use eredu_architectures::moshi::RealtimePreparationPlan;
#[cfg(test)]
use eredu_architectures::{GgufArchitecture, ModelKind};
#[cfg(feature = "mlx")]
use eredu_core::scheduler::{
    RequestId, RequestStatus, SchedulerCapabilities, SchedulerLimits, SchedulerReport, WorkId,
};
#[cfg(feature = "mlx")]
use eredu_core::{
    AllocatorTelemetry, AutomaticPlanRequest, AutomaticPlanner, AutomaticPlanningError, DevicePlan,
    ExecutionPlan, ExecutionPlanReport, ExpertCacheTelemetry, ModelInspectionReport,
    QuantizationRequest, RealtimeInputFrame, RealtimeOutputFrame, RealtimeSampling,
    RealtimeSpeechConfig, ResidencyTelemetry, SessionCapabilities, SpeculativeDecodingTelemetry,
};
