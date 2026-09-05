thread_local! {
    static EAGER_TIMING_PROFILING: Cell<bool> = const { Cell::new(false) };
}

/// Scoped opt-in profiling mode for expert-parallel phase timings.
///
/// MLX executes lazily, so ordinary phase timings primarily describe graph
/// submission. While this guard is alive, expert-parallel code materializes
/// phase outputs before stopping each timer. This makes the measurements useful
/// for benchmarks, at the cost of extra synchronization and changed scheduling.
#[must_use]
pub struct ExpertParallelTimingGuard {
    previous: bool,
}

impl Drop for ExpertParallelTimingGuard {
    fn drop(&mut self) {
        EAGER_TIMING_PROFILING.with(|enabled| enabled.set(self.previous));
    }
}

/// Enables device-complete expert-parallel phase timings for the current thread.
pub fn profile_expert_parallel_timings() -> ExpertParallelTimingGuard {
    let previous = EAGER_TIMING_PROFILING.with(|enabled| {
        let previous = enabled.get();
        enabled.set(true);
        previous
    });
    ExpertParallelTimingGuard { previous }
}

/// Returns whether eager expert-parallel phase timing is enabled on this thread.
pub fn timing_profiling_enabled() -> bool {
    EAGER_TIMING_PROFILING.with(Cell::get)
}

/// Materializes a phase's outputs when eager timing is enabled.
pub fn materialize_timing_phase<'a>(
    outputs: impl IntoIterator<Item = &'a Array>,
) -> safemlx::error::Result<()> {
    if timing_profiling_enabled() {
        eval(outputs)?;
    }
    Ok(())
}
