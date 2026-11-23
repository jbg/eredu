//! Opt-in speculative component timing and telemetry.

use std::{cell::Cell, marker::PhantomData, rc::Rc, time::Duration};

use crate::{SpeculativeStats, SpeculativeTelemetry};

/// Component wall times measured by an eager speculative executor.
#[derive(Debug, Clone, Copy, Default)]
pub struct SpeculativeComponentTimings {
    /// Committed-context encoding and assistant K/V projection.
    pub draft_context: Duration,
    /// Assistant proposal-block execution.
    pub draft_assistant: Duration,
    /// Target verification execution.
    pub target_verification: Duration,
}

thread_local! {
    static COMPONENT_TIMING_ENABLED: Cell<bool> = const { Cell::new(false) };
}

/// Scoped opt-in for speculative component profiling.
///
/// Schedulers created while this guard is alive can collect eager component
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

/// Reports whether the current execution thread has opted into component profiling.
pub fn component_timing_enabled() -> bool {
    COMPONENT_TIMING_ENABLED.get()
}

impl SpeculativeComponentTimings {
    fn add_to(self, stats: &mut SpeculativeStats) {
        stats.add_component_timings(
            self.draft_context,
            self.draft_assistant,
            self.target_verification,
        );
    }
}

impl SpeculativeTelemetry for SpeculativeComponentTimings {
    fn record(self, stats: &mut SpeculativeStats) {
        if [
            self.draft_context,
            self.draft_assistant,
            self.target_verification,
        ]
        .into_iter()
        .any(|time| !time.is_zero())
        {
            self.add_to(stats);
        }
    }
}
