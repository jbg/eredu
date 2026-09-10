# Bounded capture during ordinary generation

`LoadedModel::capture_discovery()` returns the loaded session's retained catalog,
selected support, bounded transformations, and exact prepared-source identity.
MLX applications use the same generic model API. Cold
`inspect_architecture()` remains independent of native execution.

1. Prepare a chat using the loaded model's ordinary `prepare_chat` API.
2. Build a `CapturePlan` with exact catalog paths, unique selection IDs, phase and
   prediction schedules, transforms, and explicit per-step/cumulative limits.
3. Call `prepare_observed_chat`. It owns prompt token alignment, resolved settings,
   the chat's ordinary EOS/semantic policy, and immutable capture admission.
4. Call `generate_observed_chat` and consume ordered `ObservedGenerationRecord`s.
   The callback receives prepared alignment, committed token IDs and captures,
   ordinary semantic text events, and completion. JSON records carry run/session,
   exact artifact-source and plan identities.
5. Cancel the shared `GenerationCancellationToken` or return `ControlFlow::Break`.

The implementation composes `ControlledTextGeneration` and the existing
committed-token driver. It does not resample tokens or create a second EOS,
tokenizer, or text-decoding loop. `CapturePlan::none()` means none; an empty
selection list also means none. The older `ObservationRequest` retains its
capture-all interpretation of empty selectors for compatibility.

Use `intervention_discovery` and `prepare_intervened_chat` to admit prospective
activation/routing controls together with capture, then call the same
`generate_observed_chat`. Intervention outcomes and before/after evidence share
capture budgets and native completion ownership. Shared session validation and
observer forwarding serve capture-only and combined runs. Additional original
routing-decision resources use backend estimates at admission and runtime, while
evidence transforms keep their existing capture charges. See
[intervention plans](interventions.md) for the estimator and backend integration contracts.

Run the complete facade example, with an artifact containing tokenizer and chat
metadata:

```sh
cargo run -p eredu --no-default-features --features mlx --example observed_generate -- /path/to/artifact "Explain gravity" 8
```

The example discovers a supported activation, captures a summary and a 16-value
preview, uses ordinary seeded sampling, emits JSON records, and cancels after
eight generated tokens. It selects CPU explicitly; no native type appears in
the application.

## Bounds and outcomes

`CaptureLimits` provides independent per-step and cumulative limits:

| Quantity | Meaning |
| --- | --- |
| `captures` | Reserved value transformations; diagnostics alone do not count |
| `retained_bytes` | Conservative logical capture storage, including source/view allowances, contiguous backing and native transform temporaries |
| `host_bytes` | Conservative host buffers and materialization, including intermediate reduction scalars and record metadata |
| `encoded_bytes` | Conservative UTF-8 JSON size for capture records, including escaping and numeric expansion |

Reservations happen before native evaluation, retention or materialization.
Unknown dimensions remain unknown at admission and are checked against the actual
tensor before transformation. Overflow rejects the request. Reservations are not
refunded after work starts; cumulative retention is a sum of reservations, not
simultaneous allocator residency. Per-step retention likewise sums conservative
transformation costs even though transformations execute serially.
Admission conservatively assumes phase-enabled selections can coincide, and uses
the largest selected request shape when estimating cumulative costs.

`Fail` rejects the value before the prohibited work. `Skip` emits a structured
budget reason. Metadata for missing and skipped records also requires a budget;
when even the diagnostic envelope cannot fit, the operation fails rather than
emitting unaccounted data. Previews explicitly report truncation. Missing,
unsupported, skipped, truncated and failed work are never measured zero.

`TraceLimits` separately checks exact serialized JSON size for the entire facade
record stream: prompt alignment, text, provenance, captures, and terminal events.
Serialization uses a counting sink, without allocating an encoded copy. A record
that exceeds these bounds is not delivered, generation is cancelled, and a typed
error is returned. The caller must handle the returned error even if a terminal
record cannot fit within its transport budget.
The measured encoding is compact JSON; application framing, newlines, pretty
printing, or additional encoding require a separate transport allowance.

These limits do not bound the model, KV cache, accelerator allocator overhead,
native private reduction workspace, or a caller's retained history. MLX rejects
`physical_native_bytes`; logical accounting must not be presented as a hard
physical allocator ceiling. Native capture uses borrowed source values inside
the observer call, evaluates dependencies before slicing, and returns only host
records. It retains no views or lazy capture graphs across later model blocks.

## Numeric and position semantics

Transforms currently include axis-aware slices, bounded row-major previews,
explicit full tensors, finite-only summaries, fixed-edge histograms, and
`TopCandidates` from `model.logits`. Candidate scores come from the last logits
row and are explicitly labeled `RawLogitsBeforeSampling`: they precede token
filters, penalties, temperature and sampler processing and are not probabilities.
MLX ranks natively and transfers only the requested IDs/scores. Candidate capture
requires finite logits and positive count no larger than the vocabulary; it
consumes no RNG draws. Processed-score and normalized-probability summaries are
not currently provided.

Each candidate also carries `allowed`, its membership in the **effective token
domain at that decision, before any forced-token override**. This is the exact
intersection of tokenizer validity and the active grammar/semantic constraint,
including forbidden-tool suppression. It does not incorporate penalties,
temperature, top-k/p, Mirostat, or logit interventions into membership. Scores and
ranking remain raw: a forbidden token can still be the highest-scoring candidate.
`Original` and `Effective` scores refer to opposite sides of the logits
intervention hook; both use the same decision domain.

`domain: Some(CandidateDomain)` reports `allowed_tokens`, `vocabulary` (the actual
logits width, including tokenizer holes and padding), and `constrained`.
`constrained` is true only when a semantic restriction excludes at least one
otherwise tokenizer-valid ID within that output width. Tokenizer filtering alone
and a forced choice do not set it. Membership is not a probability adjustment,
nor a promise that subsequent sampler processing will select that token.

MLX ordinary and controlled text capture borrow the sampler's exact decision
filters. One-row `prepare_speculative_capture` also carries this domain for both
target and draft predictions, using their respective histories before forcing;
draft membership does not imply target acceptance. No extra grammar query, RNG
draw, or vocabulary-sized host transfer is needed. Only the requested candidate
IDs/scores, one boolean per candidate, and a small summary are exported; their
storage and encoding are included in capture reservations.

When a controller/sampler cannot expose an exact pre-override domain and its
tokenizer baseline, `domain` is `None` and unknown membership is `allowed: true`.
This includes legacy/custom owners using the default observation contract and
owners unable to provide an exact enumerable domain. Treat that combination as
**unknown**, never as proof of permission. Older JSON records deserialize with
these same defaults; the additive fields retain `CAPTURE_SCHEMA_VERSION = 1`.
Render known disallowed candidates as present-but-forbidden: their raw scores
remain visible, but the constraint domain never permitted sampling them.

Slices use exact catalog axis names and positive strides with half-open ranges.
The `Slice` transform requires at least one explicit slice; unsliced full capture
requires `FullTensor`. All modes remain budgeted.

MLX summary/histogram inputs convert to F32. Reductions run natively in chunks of
at most 1024 values; summary aggregation uses F64 on the host. Counts are integer.
Classification and histogram comparisons occur after conversion; F64 extremes
can become infinite, and integer statistics can round even though raw IDs stay exact.
Summaries report element, finite, total non-finite, NaN, positive-infinity and negative-infinity
counts. Min/max/mean/RMS exclude non-finite values and are absent for empty or
all-nonfinite inputs. Scaling before sums/squares avoids F32 intermediate overflow
for finite extremes; ordinary floating reduction error still applies. Raw/sliced
integer IDs preserve signedness and all 64 bits. JSON consumers must parse 64-bit
integers exactly rather than round them through JavaScript Number.

Histograms require finite strictly increasing edges, at most 128 bins. Intervals
are `[lo, hi)`, with the last upper edge included. Underflow and overflow count
finite values only; non-finite values have a separate count. Raw floating capture
JSON represents non-finite values as `"nan"`, `"+inf"`, and `"-inf"`; it does not
silently replace them with null or zero.

Prediction zero comes from prefill and covers input positions `[0, prompt_len)`.
Prediction `n > 0` comes from decode input `[prompt_len+n-1, prompt_len+n)`.
Each capture retains its catalog node/path and intervention position. Block and
logit captures precede intervention at that exact point; read-only routing
captures occur after dispatch. Current observed text records are committed and
owned by rank zero; partitioned and media capture require further
support and are rejected by admission or the available text-only entry point.

## Delivery and lifecycle

Delivery is synchronous, with at most one internal step of capture records. The
next step cannot begin until the previous step has been taken. A slow callback
holds up generation rather than growing an internal queue. Dropping a prepared
request submits no work. Returning `Break` requests cancellation at a committed
token boundary and stops further callbacks; the returned result provides the
terminal outcome. Cancelling the shared token leaves callback delivery open for
the current token's semantic events and completion. The generator's existing drop path resolves retained completions;
native errors continue to use retained recovery owners and may poison the session.
Consumer panic uses ordinary Rust unwinding and the same generator drop path.
Successful cancellation preserves the ordinary model state at the last completed
forward boundary; the run's sampler, decoder, and capture ledger are dropped.
It does not produce a resumable snapshot. Use the ordinary session reset before
an independent prompt (`LoadedModel::reset` for MLX).
Captured host data from a failed attempt can appear in a `CaptureFailure` event,
followed by `Failed`; it is never attributed to a committed token. Reuse follows
the existing backend state-preservation/reset contract, including proven rollback.

`step_seconds` includes forward execution, capture, sampling, token read and exact
completion. `capture_seconds` measures time in capture transformations and capture
record accounting. Completion's elapsed time includes consumer callbacks. These
are wall-clock categories, not independent accelerator profiler measurements.
Capture time can include evaluation of the source's lazy dependencies, so these
timings overlap and must not be added as independent costs.

LM Inspector can replace its tokenizer/EOS/sampling loop with preparation and one
`generate_observed_chat` call. Render semantic text events directly; associate
captures with the token event's prediction index and retained catalog point.
Replace post-copy preview/statistics with the corresponding admitted native
transform. Forward only already size-checked records, preserve structured missing
and skip outcomes, and return `Break` when the application transport disconnects.
