//! MLX token-sampling primitives.

mod backend;
mod original;
pub(crate) use original::control_bytes as original_sampling_control_bytes;
pub(crate) use original::{OriginalSamplingBackend, OriginalSamplingContext};

pub use backend::MlxSamplingBackend;

/// Named controls of the existing immutable processing primitives. Their
/// dynamic masks/history buffers are supplied by the actual Workspace report.
pub(crate) fn processing_control_bytes() -> Option<usize> {
    backend::filter_control_bytes()?.checked_add(backend::penalty_control_bytes()?)?
        .checked_add(backend::mirostat_control_bytes()?)
}

pub(crate) use backend::apply_token_mask;

/// Fixed transports of the unchanged TopK/TopP/MinP equation workers.
pub(crate) fn sampling_filter_control_bytes() -> Option<usize> { backend::filter_control_bytes() }
