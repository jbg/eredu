# Speculative decoding and multi-token prediction

`safemlx-lm` supports lossless multi-token prediction (MTP) with either an
external assistant model or prediction heads embedded in the target
checkpoint. The target model always verifies proposals; accepted output has the
same target distribution as the corresponding non-speculative path.

Prepared chat, native tools, sampling penalties, stochastic acceptance,
cancellation, and semantic events use the same generation pipeline with or
without MTP.

## Assistants

An external assistant is loaded separately and validated against the target's
tokenizer, vocabulary, and model-specific interface. Gemma assistant artifacts
are supported through this path.

Embedded MTP is available for registered checkpoints with executable prediction
weights, including supported DeepSeek, Inkling, Nemotron-H, Qwen3-Next, and
Qwen3.5 variants. The usable proposal depth is capped by the checkpoint's
validated capability.

Applications should query `mtp_capability` or run model inspection instead of
assuming support from a family name.

## Stream placement

`MtpExecutionStreams` distinguishes three placements:

| Placement | Behavior |
| --- | --- |
| one target stream | target and assistant are ordered on one stream; no same-request lookahead overlap |
| two streams on one device | dependencies use completion events; arrays keep the same physical storage |
| streams on different devices | dependency boundaries synchronize before required arrays are copied |

Two streams on one GPU are an experiment, not an automatic optimization. The
target and assistant can contend for the same compute and memory bandwidth.
Compare the placement with lookahead disabled before adopting it.

## Verification and cache state

The target cache contains tokens already used as target inputs. After a fully
accepted proposal block, the target may sample one bonus token from the next
distribution. That bonus is emitted but remains the leading uncached input for
the next target verification.

A draft block owns its ordered proposal tokens, the processed distribution used
for each token, and the exact assistant frontier. Rejection rolls back the
target's speculative suffix and discards the corresponding assistant branch,
sampler branch, semantic events, and statistics changes. Chunked target caches
may retain allocation capacity while reducing their logical length.

Target and draft random streams are independent. Draft substreams are addressed
by logical output position, so scheduler interleaving or discarded optional
work does not advance the randomness assigned to canonical output.

## Optimistic lookahead

When the target and external assistant use distinct eligible streams, the
scheduler can draft one continuation while target verification is in flight.
This optional branch is promoted only after full acceptance of its parent block
and exact prefix agreement.

If the target bonus equals the branch's first token, that token is consumed as
the bonus and the remaining token/distribution pairs can be reused. The branch
is then extended from its saved assistant frontier to restore the normal
proposal capacity. If the bonus differs, the entire optional branch is
discarded. A terminal bonus also discards the continuation.

Promotion requires:

- an assistant state that can be cloned and discarded independently; and
- draft sampling that is a pure function of logits, explicit history,
  immutable settings, and the supplied position-addressed PRNG state.

External Gemma assistants and the default history-derived samplers satisfy
these rules. Embedded predictors whose commit advances target-owned state, and
adaptive samplers such as Mirostat V2, still use MTP but do not use same-request
optimistic lookahead.

`MtpSchedulerOptions::with_lookahead(false)` disables only the optional branch;
the same drafting, verification, acceptance, cache commit, callback, and
statistics pipeline remains in use. This is the reference setting for output
equivalence and performance comparisons.

By default, per-request lookahead can disable itself after enough resolved
branches when no proposals are reused or discarded proposals outnumber reused
proposals. This adaptive decision changes optional work only, not output.

## Constraints and tools

Each request owns its constraint sampler, tokenizer decoder, protocol parser,
stop matcher, callback, caches, and random roots. Draft and target logits are
masked at their exact logical histories. Drafting stops before a proposal would
cross a grammar-complete boundary.

Target resolution advances a transaction-local copy of constraint and semantic
state. Events become visible only when the matching target cache boundary
commits. Rejection, mismatch, cancellation, or terminal output drops every
remaining branch event.

## Diagnostics

MTP reports proposal and acceptance counts plus optional work that was drafted,
consumed as a target bonus, reused, or discarded. It also reports bonus
matches/mismatches, adaptive disablement, and time spent drafting or waiting for
verification.

These timings are diagnostic. In particular, time inside the in-flight
verification interval can include host scheduling of assistant work and should
not be interpreted as target kernel time.
