# Completion events

Current state as of 2026-08-09: SafeMLX exposes backend-independent,
single-shot completion events through its patched MLX 0.32.0 source, MLX-C,
`safemlx-sys`, and the safe `safemlx` API.

`transforms::async_eval_with_event(outputs)` is the producer operation. It
submits the requested lazy MLX graphs and returns an owning `Event`; constructing
array operations alone does not record or enqueue them. `Event::synchronize`
blocks the host, `Event::is_complete` is a nonblocking monotonic query, and
`Stream::wait_event` inserts a backend-side dependency for work submitted later
on a compatible stream. Producer and consumer devices must match exactly.
Empty output sets and sets which are already host-available return an
already-complete identity-free event.

Events are reusable for observation and multiple waits, but they are not
re-recordable timelines. MLX backend queues retain their implementation, so
dropping a C or Rust handle cannot invalidate in-flight producers or queued
consumers. CPU worker failures and Metal command-buffer failures are retained;
CUDA record, query, wait, and synchronize statuses are checked. Repeated host
waits and completed queries continue to report a retained error, while a failed
dependency poisons its consumer stream.

The MLX changes live in
`safemlx-sys/src/mlx-c/patches/mlx-completion-events.patch`. The patch extends
MLX's existing `Event`, scheduler, Metal shared-event, and CUDA event machinery
and is kept separate from MLX-C and Rust wrapper changes so it can be proposed
upstream. The patch is applied idempotently to pinned MLX 0.32.0 by the vendored
MLX-C CMake build.

`safemlx-lm` immutable weight residency and expert acquisition use the event
API through caller-owned `ResidentTransfer` guards. Dense layerwise and
pipeline execution additionally use a dedicated same-device transfer stream
and a fixed current-plus-next completion-lease window. Paged cache movement,
MTP same-device handoffs, expert-cache acquisition, checkpoint materialization,
and bounded conversion tiles likewise retain their exact event ownership.

Distributed execution exposes `DistributedCompletion<T>`. Direct Cartesian
pipeline sends and receives return this value, while `PipelineStageCompletion`
and `PipelineMicrobatchOutput` delegate to it for stage transport, cache
updates, final logits, and lane barriers. These are rank-local backend events;
they submit and observe MLX distributed operations but are not cross-process
event handles. Distributed pipeline code contains no whole-stream completion
waits. Host boundaries use exact events, while downstream compatible streams
use backend-ordered waits.

CUDA compilation is expected in Linux and Windows CUDA CI. Runtime tests remain
opt-in and require a CUDA-capable runner. The explicit Metal two-stream test is
ignored in the ordinary suite and must be run on a Metal host with:

```console
cargo test -p safemlx --test events \
  metal_two_stream_handoff_is_gpu_ordered_without_host_synchronization \
  -- --ignored --exact --nocapture
```

The safemlx-lm distributed handoff test is also explicit:

```console
cargo test -p safemlx-lm \
  distributed_completion_metal_wait_does_not_block_the_host \
  -- --ignored --exact --nocapture
```
