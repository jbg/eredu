# Asynchronous device timing

`transforms::async_eval_timed(outputs, stream)` submits a lazy MLX graph between
two stream-ordered timestamp boundaries and returns a `TimedEvaluation`. The
submission call never waits for the measured work. Applications can continue
constructing or submitting target and draft graphs, poll with `try_elapsed`,
and call `elapsed` only when they actually need the duration.

```rust,ignore
let context_time = async_eval_timed([&context], &stream)?;
let transformer_time = async_eval_timed([&transformer], &stream)?;
let vocabulary_time = async_eval_timed([&vocabulary], &stream)?;
let verification_time = async_eval_timed([&verification], &stream)?;

// No timing call above waited for execution. Resolve after all phases and any
// overlapping target/draft work have been submitted.
for timing in [
    context_time,
    transformer_time,
    vocabulary_time,
    verification_time,
] {
    println!("{:?}", timing.elapsed()?);
}
```

## Interval semantics

The output graph must be rooted on exactly the stream passed to
`async_eval_timed`. A different stream or device is rejected before submission.
Dependencies may execute on other streams; waits encoded on the measured stream
are honored without a host wait. Unrelated work already queued on the measured
stream is outside the starting boundary. Treatment of dependency waits and idle
gaps after that boundary is backend-specific:

This is execution-timeline time, not Rust graph-construction or completion
callback wall time. Backend details are:

- Metal commits any pre-marker command buffer without waiting, collects native
  `GPUStartTime`/`GPUEndTime` for every measured command buffer, and sums their
  active intervals. Queue latency before a command buffer starts and gaps when
  no measured command buffer is active are excluded. A wait encoded within an
  active command buffer is part of that buffer's interval. Large phases
  spanning multiple command buffers are supported.
- CUDA flushes pending CUDA graph nodes at each boundary and records
  timing-enabled CUDA events on the native stream. Resolution uses
  `cudaEventElapsedTime`, so stream waits and idle gaps between the events are
  included.
- CPU queues `steady_clock` markers through the same stream scheduler as CPU
  primitives. Scheduler waits and idle gaps between the markers are included;
  it does not time the host calls which enqueue the graph.

Empty or already-materialized output sets have a defined zero duration. A
substantial workload should report a positive duration, but extremely small
work can be below a backend's timestamp granularity.

## Completion, errors, and ownership

`TimedEvaluation::event()` exposes the ordinary completion event for device-side
dependencies. `try_elapsed()` returns `None` until both the graph and native
timestamp bookkeeping are complete. `elapsed()` waits for both. Repeated
queries return one cached `Duration`.

Asynchronous backend failures are retained by the completion event and are
returned before timing is resolved. Dropping a timed token before completion is
safe: native command buffers, CUDA completion work, or CPU scheduler tasks keep
the timestamp state alive until it can be released.

Untimed `async_eval` and `async_eval_with_event` calls do not create timestamp
state or add markers. Their only shared-path cost is an inactive pointer check
when a backend command buffer is committed.

See `safemlx/examples/timed_inference_phases.rs` for a concise phase example and
`safemlx/examples/timing_benchmark.rs` for the submission-overhead probe.
