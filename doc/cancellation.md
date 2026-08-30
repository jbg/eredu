# Cancellation and bounded execution

Eredu schedulers cancel cooperatively at submission and publication
boundaries. They do not claim to interrupt an executing kernel or another
operation already committed to a backend device or distributed transport.

## Work lifecycle

A scheduled transition moves through these states:

1. `Queued`: accepted, but no request-state branch or backend descriptor exists.
2. `Prepared`: a branch and descriptor exist, but no backend work was issued.
3. `Submitted`: the transition owns its branch, output arrays, leases, and
   exact completion event.
4. `Completing`: backend execution succeeded and publication is being resolved.
5. `Committed`, `Abandoned`, or `Failed`: the branch is published, rolled back,
   or rejected.

Queued cancellation avoids preparation. Prepared cancellation drops the branch
without submission. Submitted cancellation marks the transition abandoned and
returns without waiting; the transition keeps its arrays and resource leases
until its exact completion resolves, then rolls back the branch. Abandoned
output is never published.

If completion and cancellation race, the first recorded disposition wins:
completion may publish only when it wins before cancellation.

## Transactional request state

`SemanticStateTransaction` is the commit boundary for decoder and realtime
requests. A branch shares immutable array backing where safe and copies mutable
metadata. Cache deltas, sampler state, PRNG state, delayed audio frames, and
semantic events remain branch-local until commit. Cancellation therefore
returns exactly the already committed prefix.

Request-local failures fail that request. The scheduler is poisoned only when
descriptor, completion, cancellation, or distributed failure consensus shows
that shared operation ordering can no longer be trusted.

## Bounds and latency

`SchedulerLimits` bounds accepted work, active requests, new submissions per
turn, submitted transitions, and the program-defined work slice represented by
one transition. Decoder work reports its microbatch sequence length; realtime
work reports one frame.

Before submission, cancellation latency is bounded by scheduler progress and no
backend work is issued. After submission, the non-preemptible interval lasts
from backend submission until that transition's completion event resolves.
Queue and slice limits constrain how much work enters this interval; they do
not shorten already submitted work.

Distributed transitions compare descriptors before transport or collective
payload operations. Cancellation, failure, and completion use topology-scoped
consensus so every rank retains resources and publishes the same disposition.

`SchedulerCapabilities` reports configured bounds and whether a backend offers
physical preemption. `SchedulerReport` separates lifecycle counts, occupancy,
cancellation before and after submission, resources held by abandoned work,
deadline expiry, and cancellation-to-release latency.
