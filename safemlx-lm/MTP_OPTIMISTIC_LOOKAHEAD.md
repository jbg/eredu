# Bonus-preserving optimistic MTP lookahead

This note defines the state invariants used by the canonical MTP scheduler.
They are part of the implementation contract: an optimistic branch may be
promoted only when every invariant below holds.

## Stream topology and dependency boundaries

`MtpExecutionStreams` classifies execution as:

- `Single`: target and draft operations use one ordered stream; same-request
  lookahead is not submitted because it cannot overlap target work.
- `SameDeviceSplit`: distinct streams use the same device. Target-produced
  arrays are evaluated and the producing stream is synchronized at a true
  dependency boundary, then draft execution reuses their MLX array handles and
  physical storage. No `Array::copy` is performed.
- `CrossDeviceSplit`: target and draft streams use different devices. The same
  dependency synchronization is followed by the physical copies required to
  move target state, draft distributions, and stochastic draft roots.

For either split topology, target verification is submitted with `async_eval`
before optional draft work. The scheduler is deliberately single-threaded:
MLX operations are enqueued sequentially by the host onto distinct streams,
while the device runtimes may execute independent command queues concurrently.
The target stream is observed only during verification resolution. The draft
stream is synchronized before its distributions are consumed by the target or
its branch state is promoted.

Two streams on one GPU do not imply a performance win. Target and assistant
kernels contend for the same compute and memory bandwidth, so same-device
lookahead is opt-in and should be compared with `with_lookahead(false)`.
Correctness, RNG assignment, cache state, and output are independent of that
performance result.

## Cache and state conventions

The target cache contains only tokens that have been target inputs. It trails
the generated output by one token: the last emitted token is the leading input
of the next verification. If a verification starts from target checkpoint `C`
with inputs

```text
[last_committed_output, proposal_0, ..., proposal_n]
```

then full proposal acceptance commits all of those inputs. The target bonus is
sampled from the final extra target distribution but is not in that input
array, so it is emitted uncached. The next target verification therefore starts
with that bonus.

A fully accepted bonus-emitting verification commits every submitted input and
does not call cache truncation. When rejection requires partial rollback, a
chunked KV cache changes its logical length and offset but retains the same
backing arrays and capacity. The next target update overwrites the abandoned
speculative suffix in place.

A draft block owns, as one unit:

- its ordered proposal tokens;
- the exact processed `q` distribution used to sample each token, at the same
  index; and
- the assistant frontier after producing the block.

The assistant frontier follows the backend's autoregressive cache convention:
the state has processed every token before the final proposal, and the final
proposal is the pending `last_token` input for the next draft step. Thus the
state and ordered token block together identify one exact assistant prefix.

An optimistic branch is nested in the in-flight verification from which it was
created. While that transaction is in flight, neither the request output nor
the verified proposal block can change. The branch stores an exact copy of the
entire assumed generated-token prefix and is valid only when it byte-for-byte
equals the canonical prefix and the verification accepts the entire rooted
proposal block. Prefix divergence is an invariant error; the scheduler never
falls back to reusing potentially stale state.

## Full-acceptance transitions

Let `P` be the fully accepted current proposal block, let target bonus `B` be
sampled from the verification's extra target distribution, and let the
optimistic block be:

```text
[(X0, q0), (X1, q1), ..., (Xm, qm)]
```

The target sampler processes and commits `B` exactly as it does without
lookahead. The target cache commits the verification inputs through `P`, while
`B` remains uncached.

### Match: `B == X0`

`X0` is consumed by the target bonus. It is not a reused proposal and is never
submitted for speculative acceptance. The scheduler removes `(X0, q0)` as one
paired operation and promotes:

```text
[(X1, q1), ..., (Xm, qm)]
```

The retained distributions are not recomputed. Each remains valid because its
sampled prefix contains `X0`, which exactly equals canonical `B`, and all
earlier assumed tokens are the fully accepted `P`.

When at least one token remains, the already-advanced assistant frontier is
promoted with the shortened block. It processed `X0` while producing `X1`, so
the frontier and retained block represent exactly:

```text
canonical prefix + P + B + retained optimistic proposals
```

under the pending-final-token cache convention above. When no token remains,
the branch is consumed and dropped; the request returns to canonical drafting
from target state and no empty verification is submitted.

Consuming `X0` shortens the block by one. Before submission, the canonical
draft phase extends the block from its promoted assistant frontier up to the
same proposal capacity that non-lookahead execution would use. Retained
token/distribution pairs are not recomputed. Restoring this verification
boundary also restores the exact target-PRNG draw grouping used without
lookahead.

### Mismatch: `B != X0`

The entire optimistic branch is dropped atomically. Its assistant frontier,
tokens, processed distributions, and speculative diagnostics never become
canonical. Draft sampling uses position-addressed substreams, so speculative
work has not advanced canonical draft randomness. Draft logit processing ran
on a sampler clone whose capability contract guarantees history-pure behavior,
so it has not mutated the canonical sampler. Target cache/state, target PRNG,
sampler commits, history, output, and callbacks therefore equal execution with
lookahead disabled immediately after emitting `B`.

### Terminal bonus

If `B` is EOS or reaches `max_tokens`, it is committed and emitted, and the
whole continuation is dropped. No branch state is promoted. Optimistic work is
not started when the current verified block leaves capacity only for the
target bonus.

## RNG invariants

The request key is split once into disjoint target and draft roots. Target
acceptance, residual, and bonus sampling use only the target stream.

Draft sampling is addressed by generated-token position, not by scheduler
operation order. The proposal for logical output position `k` always receives
the deterministic split subkey derived from `(draft_root, k)`. The first
optimistic token occupies the possible bonus position. On a bonus match that
position is consumed; retained token `Xi` already used the same substream that
ordinary post-bonus drafting would use at its logical output position. On
mismatch no draft cursor needs rewinding. The canonical refill described above
uses later position keys and preserves ordinary verification boundaries.
Consequently scheduler interleaving, lookahead branch-slot availability,
promotion, and discard cannot change the draft or target random stream assigned
to canonical execution.

## Disable and adaptive policy

`MtpSchedulerOptions::with_lookahead(false)` prevents all same-request
optimistic branches. It does not select a separate speculative loop: the same
canonical scheduler continues through draft, verification, resolution, bonus
sampling, cache commit, sampler commit, callbacks, and statistics. This is the
reference path for equivalent A/B tests.

Adaptive disabling is also output-neutral. After
`adaptive_lookahead_min_blocks` resolved branches (four by default), the
scheduler permanently disables future branch creation for that request when
either no proposal token has been reused or:

```text
reused_optimistic_tokens < discarded_optimistic_tokens
```

Retained proposals are work that canonical execution did not recompute;
discarded proposals are work that produced no reuse. The first matching branch
token is excluded because it is consumed as a target bonus, not reused as a
proposal. The accounting uses committed per-request counters only and never
changes branch validity, target sampling, target PRNG state, canonical draft
substreams, or already submitted work. `adaptive_lookahead_disabled` reports
when this transition occurred. Disabling the adaptive policy keeps otherwise
eligible lookahead enabled for controlled performance measurements.

## Capability limits

Bonus-preserving promotion requires both:

- a backend whose cloned draft state is an exact, independently discardable
  assistant frontier and remains valid across a fully accepted target block;
- a sampler whose draft logit processing and sampling are pure functions of
  raw logits, explicit history, immutable configuration, and the supplied
  position-addressed PRNG state.

External Gemma assistants and the default/history-derived samplers satisfy
these contracts. Embedded Qwen does not, because commit advances target-owned
MTP state. Mirostat V2 does not, because its processing depends on adaptive
target-committed state. Both continue through the same scheduler without
same-request optimistic lookahead.

## Statistics

`optimistic_draft_tokens` counts every token computed on a lookahead branch.
On a match, the first token increments `consumed_optimistic_tokens`; it does not
increment `reused_optimistic_tokens` or `draft_tokens`. Only retained tokens
promoted into the canonical proposal block count as reused proposals and
ordinary draft tokens. A mismatch or terminal bonus counts every remaining
branch token as discarded. `optimistic_target_bonus_tokens` counts target
bonuses emitted while a branch exists; match and mismatch counters classify
non-terminal comparison outcomes. `adaptive_lookahead_disabled` reports
whether deterministic reuse-versus-discard accounting stopped future branch
creation. `optimistic_draft_time` records wall time spent producing optional
branches. `verification_in_flight_time` records submission-to-resolution wall
time, including scheduler work deliberately placed inside that interval. These
timings are diagnostic rather than correctness or adaptive-policy inputs.
