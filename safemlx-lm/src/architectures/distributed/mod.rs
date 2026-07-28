//! Architecture-dispatched distributed model adapters.

/// Executable expert-parallel model adapters.
pub mod expert;
/// Executable pipeline-parallel model adapters.
pub mod pipeline;
/// Executable tensor-parallel model adapters.
pub mod tensor;
