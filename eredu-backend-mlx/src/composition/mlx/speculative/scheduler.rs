//! MLX resource adaptation for neutral speculative scheduling.

use std::{cell::Cell, marker::PhantomData, rc::Rc, time::Duration};

use eredu_core::{SpeculativeStats, SpeculativeTelemetry};

/// Component timings accumulated by an architecture-specific speculative backend.
#[derive(Debug, Clone, Copy, Default)]
pub struct SpeculativeComponentTimings {
    /// Committed-context encoding and assistant K/V projection.
    pub draft_context: Duration,
    /// Assistant proposal-block execution.
    pub draft_assistant: Duration,
    /// Raw vocabulary-head projection.
    pub draft_head: Duration,
    /// Target verification execution.
    pub target_verification: Duration,
}

thread_local! {
    static COMPONENT_TIMING_ENABLED: Cell<bool> = const { Cell::new(false) };
}

/// Scoped opt-in for device-timeline speculative component profiling.
///
/// Schedulers created while this guard is alive collect architecture-level
/// timings. Timestamp boundaries add profiling overhead, so normal generation
/// leaves them disabled. The guard is intentionally bound to its creating
/// thread and restores the previous setting when dropped.
pub struct SpeculativeComponentTimingGuard {
    previous: bool,
    _thread_bound: PhantomData<Rc<()>>,
}

impl SpeculativeComponentTimingGuard {
    /// Enables component profiling for schedulers created on this thread.
    pub fn enable() -> Self {
        let previous = COMPONENT_TIMING_ENABLED.replace(true);
        Self {
            previous,
            _thread_bound: PhantomData,
        }
    }
}

impl Drop for SpeculativeComponentTimingGuard {
    fn drop(&mut self) {
        COMPONENT_TIMING_ENABLED.set(self.previous);
    }
}

pub(crate) fn component_timing_enabled() -> bool {
    COMPONENT_TIMING_ENABLED.get()
}

impl SpeculativeComponentTimings {
    fn add_to(self, stats: &mut SpeculativeStats) {
        stats.add_component_timings(
            self.draft_context,
            self.draft_assistant,
            self.draft_head,
            self.target_verification,
        );
    }
}

impl SpeculativeTelemetry for SpeculativeComponentTimings {
    fn record(self, stats: &mut SpeculativeStats) {
        self.add_to(stats);
    }
}
