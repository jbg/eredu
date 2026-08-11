# Cancellation and bounded execution

SafeMLX cancellation is cooperative at scheduler submission boundaries. It
does not claim to interrupt an executing Metal kernel or a committed Metal
command buffer. The current MLX/Metal event API can query, wait for, and order
an exact completion; it does not expose physical cancellation of work already
submitted to Metal.

## Transition lifecycle

The architecture-neutral `FairScheduler` is authoritative for every decoder
pipeline microbatch and Moshi/PersonaPlex frame:

- `Queued`: accepted, but no descriptor or state branch exists.
- `Prepared`: the descriptor and a semantic state branch exist, but no backend
  work has been issued.
- `Submitted`: the branch, output, retained arrays and leases, request/work
  identity, deadline, disposition, and exact backend completion are owned by
  one transition.
- `Completing`: the exact completion succeeded and publication is being
  resolved.
- `Committed`: the branch became canonical and its output became visible.
- `Abandoned`: cancellation won before publication. Submitted resources remain
  retained until the exact completion resolves, then branch-local deltas are
  rolled back and dropped.
- `Failed`: preparation, submission, asynchronous execution, rollback, or
  commit failed.

Queued cancellation does not construct a descriptor. Prepared cancellation
drops the branch without encoding or submission. Submitted cancellation marks
the transition abandoned and returns without waiting for it or unrelated
stream work. A completion racing cancellation has one winner: completion
polled first can commit; cancellation recorded first prevents publication.

Request-local execution failures fail that request and leave unrelated
requests schedulable. The whole scheduler is poisoned only when descriptor,
completion, cancellation, or failure consensus proves that shared distributed
operation ordering is unsafe.

## State transactions

`SemanticStateTransaction` branches semantic request state and supplies the
only commit boundary. Branches share immutable MLX array backing and copy
mutable metadata; complete model caches are not deep-copied. Pipeline cache
branches use the same checkpoint/restore operations as speculative execution.
Resident tails and state slots are structurally cloned. Paged KV, compressed
latent, and embedded-predictor cache deltas are removed from their shared
residency managers when an abandoned exact completion releases the branch.

Realtime branches include temporal and depth caches, delayed-stream frames,
text and audio samplers, and the request PRNG state. None of those objects is
canonical until the output event succeeds. Abandoned sampled tokens and
delayed frames are never returned.

## Bounds and the physical backend interval

`SchedulerLimits` configures:

- maximum accepted work and active requests;
- maximum newly submitted work per turn;
- maximum submitted transitions globally and per request; and
- the maximum program-defined execution slice represented by one transition.

Pipeline work reports its microbatch sequence length as its slice size.
Realtime work reports one frame. Autoregressive state rejects parallel branches
even when the numerical per-request bound is larger. A program may permit
multiple branches only when its branch deltas are independently mergeable.

Cancellation latency has two components. Before submission it is bounded by
one scheduler turn and no backend work is issued. After submission the
unavoidable non-preemptible interval is exactly from that transition's backend
submission until its exact completion event resolves. Submission-turn and
slice bounds limit how much additional work can enter that interval; they do
not shorten an executing kernel or committed command buffer.

## Distributed ordering

Distributed schedule descriptors are compared before point-to-point or
collective payload operations begin. Cancellation and deadline dispositions
use topology-scoped exact consensus. Completion polling compares the same
in-flight work identities and publishes only after all ranks report successful
exact completion. An abandoned distributed transition retains transport
endpoints and cache/resource leases until every rank can release it safely.
No cancellation path performs whole-stream synchronization.

## Capabilities and telemetry

`SchedulerCapabilities` reports configured bounds, observed completion
backends, physical-preemption support, and the exact non-preemptible interval.
For current Metal completions, `executing_work_physically_preemptible` is
`false`.

`SchedulerReport` separates queued, prepared, submitted, completing,
committed, abandoned, and failed work. It also reports current and peak
in-flight occupancy, cancellation before and after submission, resources held
by abandoned work, abandoned release count, deadline expiry, configured turn
and slice bounds, and last/maximum observed cancellation-to-release latency.
