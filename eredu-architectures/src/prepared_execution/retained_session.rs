//! Rank construction facts and the portable retained session contract.

use eredu_core::SessionCapabilities;

pub use eredu_runtime::executable_binding::retained_session::PreparedRetainedTextSession;

/// Fully constructed rank and the retained facts used by the outer loader.
pub struct RealizedRank<R> {
    /// Typed portable text session.
    pub session: R,
    /// Selected raw processor when applicable.
    pub raw_processor: Option<crate::processor_execution::PreparedProcessor>,
    /// Realized session capabilities.
    pub capabilities: SessionCapabilities,
    /// Width of the selected floating state.
    pub floating_state_bytes: std::num::NonZeroU8,
}
