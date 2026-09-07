# Bounded capture during ordinary generation

`LoadedModel::capture_discovery()` returns the loaded session's retained catalog,
selected support, bounded transformations, and exact prepared-source identity.
The local facade exposes the same method on `LocalModel`. Cold
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
owned by rank zero; partitioned, speculative and media capture require further
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
an independent prompt (`LocalModel::reset` for the local facade).
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
