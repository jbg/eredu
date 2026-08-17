//! Backend-neutral language-model contracts and orchestration.
//!
//! This crate deliberately contains no tensor runtime. Backends own tensors,
//! streams, executable models, caches, and completion primitives; core owns
//! validation, lifecycle state, scheduling, and portable schemas.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

/// Validated decoder attention schedules.
pub mod attention;
/// High-level execution-backend contract.
pub mod backend;
/// Aggregate ownership and admission for backend-managed live caches.
pub mod cache;
/// Neutral checkpoint tensor descriptions and validation.
pub mod checkpoint;
/// Backend-neutral distributed scheduler consensus.
pub mod consensus;
/// Portable execution plans, capabilities, and telemetry.
pub mod execution;
/// Backend-independent generation lifecycle and output events.
pub mod generation;
/// Stable model and artifact identities.
pub mod model;
/// Weight-residency ownership, capacity, and resource planning.
pub mod residency;
/// Transactional fair work scheduler.
pub mod scheduler;
/// Parallel topology and placement planning.
pub mod topology;

pub use attention::{AttentionPolicy, LayerSchedule, LayerScheduleError};
pub use backend::{
    Backend, BackendCapabilities, BackendDescriptor, BackendError, BackendSession, Completion,
    DeviceDescriptor, PreparedModel, Submission,
};
