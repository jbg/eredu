# Sparse expert cache synchronization

Sparse expert caching has one unavoidable data dependency: exact checkpoint
experts cannot be selected until the router result is available. The cache
keeps this boundary bounded by reducing the route tensor on-device to a demand
histogram with one element per global expert plus an invalid-route scalar. Only
that metadata is read by the host. Original route rows remain on-device and are
rewritten through a global-to-compact lookup table, preserving route order and
weights.

Cold and warm acquisitions are processed as one residency batch. Capacity for
every missing unit is reserved before materialization, requested units are
temporarily protected from eviction, and mmap leases plus source arrays remain
owned by a caller-owned `ResidentTransfer`. Cached host bindings are immutable
typed host-transfer buffers, not MLX arrays. Promotion submits one backend copy
per binding and publishes one aggregate backend-independent completion event.
The execution stream waits on that aggregate event before compact-bank
construction, without synchronizing the host. The transfer retains mmap
leases, source arrays, host buffers, and their per-binding events until exact
aggregate completion; dropping it early waits for that event rather than
synchronizing either entire stream.

The shared residency state records only an in-flight generation, not an MLX
event or source handles. This keeps the event on its supported host thread and
removes the former unsafe shared pending-source object. A concurrent
synchronous acquisition waits on a condition variable until the caller-owned
transfer publishes success or rolls back failure. Event errors propagate from
explicit synchronization and poison dependent stream work through MLX's normal
completion semantics.

The remaining host waits are exact completion-event observations:

- the host must observe the bounded route-demand metadata before it can select
  checkpoint ranges and make eviction decisions;
- event-backed materializations must complete before their mappings can be
  released safely; and
- the final expert output is evaluated before its expert leases are released.

MLX documents lazy computation and recommends evaluating related outputs
together because every evaluation has fixed overhead:
[MLX lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html).
SafeMLX's patched MLX C API exposes asynchronous evaluation with an owning
completion event, exact host query/wait, and same-device stream waits:
[MLX C transforms](https://ml-explore.github.io/mlx-c/build/html/transforms.html)
and [MLX C streams](https://ml-explore.github.io/mlx-c/build/html/stream.html).

This ownership refactor does not yet add predictive expert prefetch or schedule
independent shared-expert computation during transfer. Exact routing must still
load every selected expert and never substitute another. Those scheduling
optimizations can now use the existing event-backed transfer contract without
reintroducing a second pending-source path.
