//! Backend-independent facade orchestration.

/// Chat preparation, output dialects, and constrained decoding.
pub mod chat;
pub(crate) mod generation;
pub(crate) mod model_loading;
/// Portable selection and execution wiring for speculative generation.
pub mod speculative;
pub(crate) mod voicechat;
