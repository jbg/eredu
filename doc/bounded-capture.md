# Bounded capture during ordinary generation

`LoadedModel::capture_discovery()` returns the loaded session's retained catalog,
selected support, bounded transformations, and exact prepared-source identity.
MLX applications use the same generic model API. Cold
`inspect_architecture()` remains independent of native execution.

Plans may select conditionally supported sites, such as media-only prefill
boundaries. Admission preserves the condition reported by discovery; it does not
assert that the request satisfies it. An absent scheduled site produces a typed
`Missing` outcome with no payload or value transformation. Unsupported and
unverified sites remain inadmissible.

1. Prepare a chat using the loaded model's ordinary `prepare_chat` API.
2. Build a `CapturePlan` with exact catalog paths, unique selection IDs, phase and
   prediction schedules, transforms, and explicit per-step/cumulative limits.
3. Attach the borrowed plan to `PreparedChatRequest::capture`. Startup admits it
   after the paid prompt producer supplies exact token/media geometry.
4. Call `start_controlled_chat(request, trace_limits, control, callback)` and
   advance or run the returned session. Ordered `ControlledGenerationRecord`s
   carry prepared alignment, committed token IDs/captures, semantic events and
   completion, with run/session, artifact-source and plan identities.
5. Cancel through the shared control handle or return `ControlFlow::Break`.

The implementation composes `ControlledTextGeneration` and the existing
committed-token driver. It does not resample tokens or create a second EOS,
tokenizer, or text-decoding loop. `CapturePlan::none()` means none; an empty
selection list also means none. The older `ObservationRequest` retains its
capture-all interpretation of empty selectors for compatibility.

Use `intervention_discovery` and the request's borrowed `intervention` declaration
to admit prospective activation/routing controls together with capture. Both use
the same controlled startup and ordinary committed driver. Intervention outcomes and before/after evidence share
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

For prepared multimodal input, obtain `InputTokenCount` through
`ModelCapabilityBackend::count_prepared_input` and use `model_positions` as the
capture request's prompt length. Composite adapters count the architecture's
admitted decoder positions, including media expansion, and retain media
workspace accounting. Submission validates the actual assembled batch and
sequence against the admitted capture shape before forwarding the model.

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

These logical limits do not bound the model, KV cache, accelerator allocator
overhead or a caller's retained history. Capture allocations also consume the
enclosing request's per-domain memory allowances through `MemoryLedger`.
Logical accounting does not establish a physical allocator or process ceiling.
Native capture uses borrowed source values inside
the observer call, evaluates dependencies before slicing, and returns only host
records. It retains no views or lazy capture graphs across later model blocks.

## Numeric and position semantics

`CaptureTransform::RoutedUnits` is specific to declared
`ObservationValueType::RoutedUnits` points. It accepts slices of the original
`token`, `route` and per-expert `component` axes. Its payload contains selected
unit rows with checkpoint-global expert IDs, original route slots and actual
coefficients. Native sorting and provider chunking do not change those identities;
an unselected expert has no row. A duplicate expert in two route slots retains both
rows. No expert-dense activation tensor is created, and no missing row means zero.

The partition receipt mechanism separately retains expert/unit ownership and an
authoritative source peer. It assembles selected TP columns by original
`(source_peer, token, slot)`, rejecting coefficient/expert disagreement, overlap
and incomplete coverage. Native receive-order chunks remain on each contribution
as `RoutedUnitCaptureProvenance`; the assembled payload's `source_token_ranges`
is empty. Delivery reservations cover route storage and provenance copies before
producer work. This mechanism uses ordinary bounded partition exchange and does
not itself admit a native forward or publish a committed step. Loaded MLX now
composes these receipts with the shared partition observer when retained placement,
sparse provider hooks and exact native invocation-group transport are available.

`CaptureBackend` provides separate partitioned sparse estimation and collection
methods. The request describes expected global coordinates and ownership; the
borrowed source supplies actual provider columns and original exchange tags.
MLX gathers only selected units, translates native receive rows back to the
admitted publication peer, and retains native chunk ranges separately from logical
token positions. Reservations include all potentially received route metadata and
the full invocation's chunk history. These methods do not bypass live session
admission or authorize publication.

The live session exposes `prepare_partition_routed_capture`,
`begin_partition_routed_capture`, `observe_partition_routed_units` and
`finish_partition_routed_capture` for composition with the ordinary partition
transaction. Preparation charges all ranks before fragment storage or native
collection. Begin pins actual receive rows and source precision; chunks must
cover that exact extent, not merely fit a worst-case reservation. An idle owner
explicitly reports zero rows and completes; missing instrumentation is rejected.
Malformed chunks and failed or repeated completion cannot become committed
records. Ordinary all-rank receipt delivery still stages provisional records,
and only the shared transaction's final commit publishes them.

`RoutedUnitObserver::begin_invocation` and `finish_invocation` bracket each
actual local prepared provider, including idle EP owners. The runtime scope
suppresses duplicate callbacks from nested bank adapters and always calls finish
after a local error, before reverse exchange or a downstream tensor reduction.
Begin borrows the original provider input, source origins and available unit map;
an EP outer scope can precede the bank map, which remains present on each actual
chunk. Native forwarding preserves these borrows and the original error source.
The shared `PartitionCaptureObserver` consumes these callbacks using
`PartitionCaptureLayout::routed_capture_placement`. Sparse placement identifies
distinct producers and every source member, including idle owners and replicas.
It reserves worst-case received input rows before source work, then checks actual
row extent, read width, dtype, scalar map and source origins. Geometry rejection
is agreed before native submission. Both original and effective selections finish
their prepaid votes before provider return; missing chunks, provider errors and
failed preparation cannot publish records or refund credits.

Execution also agrees provider success independently of these observer scopes,
including when no provider-unit capture is selected. TP, EP and shared pipeline
wave participants finish the applicable votes before reverse exchange or another
model collective. These execution votes create no observation records and do not
replace source completion or the final capture transaction.

This observer uses the existing source-completion, receipt and final transaction
protocols. Retained architecture layouts supply sparse bank and invocation placement.
Explicit stage groups join every tensor/expert replica at a fixed pipeline stage;
MLX uses bounded neighbor status reduction for connected Ring membership. Loaded discovery
checks `ObservationHookSite::RoutedUnits` separately from other internal layer hooks
and exposes only points whose selected native group can finish independently.
Disconnected groups still require coordinated relay support.
The verified Ring configurations include TP2, EP2, TP2/EP2, TP2/PP2, host TP2 and
disk PP2 for packed Qwen MoE. Full and strided captures, prefill, two cached decodes,
snapshot replay, sibling isolation and skip-on-limit use the ordinary public
capture contract. Sparse interventions share this observer's admitted session
and commit protocol. Original captures precede all operations, effective captures
follow them, and aborted forwards publish neither successful captures nor edits.
See [interventions](interventions.md) for operation ordering and affected counts.

The ordinary run reserves a complete selected-unit invocation before evaluating or copying
its first native chunk. One selection still produces at most one record per step.
Original source-token ranges and unique selected token/route receipts must cover
the invocation before the record becomes captured. Native errors, malformed
receipts, missing chunks and underestimated encoding preserve failed outcomes and
consume existing reservations. Delivery remains under the ordinary forward commit
boundary. `routed_unit_tensors` is a separate loaded collector fact; the ordinary
collector does not establish distributed sparse delivery.

The original and `.effective` unit paths retain their pre- and post-intervention
timing. Both are before down projection, optional projection input quantization,
and routing-coefficient multiplication. Reconstruct a routed write by multiplying
each selected unit by its loaded expert write column and coefficient, adding the
expert output bias once per route. Shared experts and other residual terms remain
separate. Quantized projections need their actual input-transform evidence; raw
selected units alone do not prove an exact additive decomposition.

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
owned by the ordinary output path. Loaded partition component capture assembles
global records on every participating rank with explicit producer evidence.
Media capture still requires further integration and is rejected by the available
text-only entry point.

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

LM Inspector can use prepared-chat startup and the controlled session's `run`
method for tokenizer, EOS, sampling and semantic delivery. Render semantic text events directly; associate
captures with the token event's prediction index and retained catalog point.
Replace post-copy preview/statistics with the corresponding admitted native
transform. Forward only already size-checked records, preserve structured missing
and skip outcomes, and return `Break` when the application transport disconnects.

Shared dense decoder component paths include normalized inputs, activated FFN
units and aggregated attention channels. Their `.effective` siblings are separate
read-only `AfterIntervention` observations. Original paths keep their existing
pre-intervention meaning. See [component analysis](component-analysis.md) for
geometry, support and reconstruction conventions.

Nemotron-H target partitions emit normalized inputs, dense/shared/routed ReLU² units,
attention channels, readout values and complete Mamba/sparse contributions through
these same contracts. Native SafeTensors/GGUF coverage spans TP/EP/PP and all three
residencies; see [the Nemotron validation](component-validation.md).
Shared-expert scalar observations use `model.layers.N.shared.feed_forward.units`;
their write is a constituent of the complete sparse write. Its declaration joins
the actual up/down parameters and the enclosing normalization. See
[the shared-expert extension](component-validation.md)
for original/effective timing and reconstruction without counting the write twice.

Shared routed Qwen and GPT-OSS decoders also expose their attention channels,
normalized FFN inputs and complete `feed_forward.contribution` values. The latter
include the expert reduction and have separate effective companions. Whole-write
metadata identifies the normalized input; it does not declare individual expert
neurons. Loaded TP, pipeline and combined capture use the actual provider hooks
and retained invocation ownership, including host-layerwise and streamed units.


`CaptureTransform::TokenScores { token_ids }` admits up to 64 unique IDs at
unsliced `model.logits` for batch one. It evaluates the final sequence row over
the complete model vocabulary, returning raw scores, strongest different-token
competitors, competition ranks (`1 + count(strictly greater)`) and stable log
probabilities. Tokenizer/constraint membership is separate metadata; it does not
filter this distribution. A vocabulary slice or truncated candidate list cannot
be used as a probability denominator. MLX uses native reductions and scalar host
reads, with chunked F32 exponent/sum work and F64 compensated aggregation. Target
log probabilities subtract the maximum before the log mass, preserving accuracy
under large common score offsets. Non-finite raw scores fail explicitly. These
operations use the ordinary capture reservation and failure lifecycle.

Distributed receipts also validate stable log probabilities against the target
and its strongest alternative. Those two values identify the global maximum;
their shifted probability mass gives a lower bound on the denominator, while
the vocabulary size gives an upper bound. This check remains meaningful when
adding log mass to a very large maximum rounds away in F64. It allows bounded
F32 reduction error and performs no additional allocations. Receipt validation
checks consistency; numerical comparisons still establish whether a producer
evaluated the actual model distribution.

Actual projection inputs can require a diagnostic transformation, as with the
F32 reconstruction of dynamically quantized FP8 inputs on GPU. Architecture
hooks expose these as read-only points (`*.write_input` and
`readout.projection_input`). Backend mechanisms offer an exact-shape prototype,
a conservative creation allowance and a deferred factory. The runtime uses the
ordinary capture reservation and transformation path: absent, skipped and
rejected selections never invoke the factory. Matching selections share its
result within that observation call; generated handles are released before
returning to inference. Each selection conservatively reserves creation storage,
and generation failures keep their charges and the enclosing native completion
owner. Original and `.effective` activation meanings are unchanged.


`CaptureRecord.source_dtype` carries actual native scalar precision separately
from the host payload's representation. BF16/F16 captures may export F32 numbers;
that conversion does not change the recorded source dtype. The backend reads this
metadata without evaluation or copying. Unknown values and older serialized
records use `None`. A deferred observation can retain its mechanism's declared
precision before generation. When the mechanism provides no type fact it uses
`None`; the shape prototype must never supply the actual precision. After a reserved
factory produces a value, runtime records that value's dtype even if its subsequent
transformation fails. Ordinary skipped transformations can retain known dtype
metadata without copying their values. Intervention evidence uses this same path.

Partitioned scalar-axis maps describe global component identities, including
which invocations a pipeline stage actually executes. They do not change
`CaptureRecord` into a global distributed record by themselves. Loaded component
capture uses bounded rank agreement, exact producer coverage and shared committed
delivery to produce that result. See [component analysis](component-analysis.md)
for the current native coverage and remaining ownership/transform gaps.

`CaptureSlicePartition` describes a bounded set of local native slices together
with their exact destinations in a global selected tensor. The runtime partition
capture mechanism retains the original admitted selection and uses ordinary
capture reservations before each transform. Raw fragment assembly reserves its
host buffers and rejects incomplete coverage, overlapping destinations, stale
identity and incompatible precision. Permuted component storage does not change
global result order. Fragment assembly itself supplies no transport or public session admission;
the receipt exchange below owns bounded host delivery and its rank decision.

Partition summaries and histograms use `assemble_reduced_fragments`. Core stride
intersection validates exact coverage without storage proportional to component
width. Counts and bins remain exact; finite mean/RMS aggregation preserves the
native F32-conversion convention and combines partial statistics in F64.
Different reduction orders can round differently. Deferred partition inputs use
`capture_generated_fragment`, sharing the ordinary reservation-before-factory
adapter and retaining actual generated-source precision. Public distributed
session admission connects these mechanisms to retained execution ownership and
the shared forward lifecycle described below.

Partition vocabulary reductions require exactly one complete, ordered final-logits
producer. Receipt admission rejects partial or permuted vocabulary geometry before
native capture or factory work. `TokenScores` and `TopCandidates` use ordinary
native full-vocabulary reductions; delivery moves only their bounded host payloads,
retaining source geometry and producer evidence. Receipt checks validate requested
IDs, counts, finite scores, rank/competitor consistency and optional domain bounds.
These structural checks complement numerical reference tests. They do not infer
global probabilities from shard-local reductions or top-k candidate lists.

Partition previews preserve the prefix of the globally selected tensor, including
indexed component maps and multiple sequence rows. Each fragment exports at most
`max_elements` values; global assembly allocates only the requested prefix and
rejects inconsistent local truncation or payload shapes. The payload is flat,
while `selected_shape` retains the complete selection. `Truncated` reports global
available/emitted counts. Producer budgets still account for each fragment's
bounded export, so a global prefix limit does not imply a single-producer cost.

Versioned core producer receipts can carry these bounded fragments between host
owners. Runtime `PartitionCaptureReceiptPlan` validates expected producers and
binds their exact geometry to execution/run/overlay identity and a forward epoch.
Receipt collection rejects wrong senders, stale context, duplicate ordinals and
missing acknowledgments, including for empty selections. Parser storage is
reserved before decoding; raw integers and non-finite floating values use the
existing exact capture wire representation. `ReceivedPartitionCapture` retains
context, producer acknowledgments and geometry identity with the assembled value.
`PartitionCaptureExchange` reserves both control rounds and the maximum allowed
payload gather before producer work. It checks rank readiness, exact receipt
identity and lengths, transfers padded portable words under the selected bounded
completion policy, then agrees local decode/assembly outcomes before returning a
value. A decoder budget failure can still vote because the control credits are
already charged. No reservation is refunded. Fully agreed rejection preserves the
typed local cause or names a rejecting peer; the enclosing forward driver decides
skip, rollback or termination. Transport failures and unagreed protocol mismatches
fence the shared communication owner. Exact native completion retains or safely
quarantines its buffers before any host resolution.

Admission of this move-only exchange submits no work and must itself be agreed
by the enclosing session before any forward starts. Every rank participates at a
common post-forward boundary, including inactive pipeline stages; a component
hook reached by only one stage cannot call a world-wide exchange. The native
adapter supplies rank, selected completion policy and logical transport storage
estimates. These records and transport credits do not authorize producer tensor
work. The live session and observer integration described below grants that
producer authority; loaded public composition is still required before support changes.

Architecture composition can obtain all component producers from the retained
selection through `ComponentPartitionLayouts::capture_producers`. Its remote-rank projections reuse
the same ownership rules as local admission. Identical invocation replicas select
one capture producer; distinct empty-overlap shards retain their acknowledgments.
The supplied maps still pass runtime receipt coverage checks and must be bound to
the session's live ledger, run and forward before native work.

The shared runtime provides transactional observer callbacks for local
admission, coordinated preparation, delivery after exact completion, and final
commit/discard notification. Local admission never submits a collective. Uniform
participation is checked before coordinated callbacks, including on inactive
pipeline ranks. A completed delivery is provisional until ordinary state commit
succeeds; delivery rejection uses the same rollback checkpoint as model failure.
These callbacks are the integration point for the bounded receipt mechanism.
`PartitionCaptureObserver` composes producer quotas and common identity into
these callbacks. Loaded public distributed generation binds this observer
through shared preparation and intervention admission.

Input preparation precedes observer admission. MLX forwards failed token
extraction, composite admission, capture prompt alignment, parameter-version
checks and decode framing into the shared `InputPreparation` agreement instead
of returning locally. Rejection discards the observer step before allocating its
capture envelopes, preserves installed model state, and consumes the forward
epoch. Retrying requires native settlement and a fresh epoch. Initial run and
prompt construction participate in the shared public readiness protocol.

Ordinary `CaptureObserver` also uses this lifecycle. Its shared owner refuses to
drain a pending transaction and rejects a failed attempt as a snapshot boundary,
even if the attempted step had no failed individual capture. Recorded local
observations and intervention application outcomes retain their existing meanings;
they are evidence of the actual boundary, not proof that generation committed.
Restore preserves consumed transaction epochs and cumulative resource charges.

Native text generation creates `CaptureObserver::for_step`, which performs no
reservation at construction. The shared local preparation callback starts the
step using the actual forward phase. It consumes the epoch before reserving
envelopes, so failed reservation cannot be retried under the same epoch. Pending
failed admission also blocks restore when no record envelope was created.
Admission rejection before mutable model-state access carries explicit runtime
evidence for native recovery; safe reuse still requires native settlement.

`CaptureSession::configure_partition_capture` binds an unstarted run to retained
artifact, execution, coordinated run, overlay and world identities. These labels
must come from loaded composition; arbitrary caller labels prove no provenance.
`prepare_partition_capture` grants move-only owner/epoch-bound work after globally
reserving producer, transport, decoder, assembly and public-evidence costs. Every
rank prepays the same global cost, while each handle can spend only its local
allowance. Restore and dropped handles refund neither budgets nor epochs.
Direct `begin_step` cannot replace an undrained transaction, including failed
preparation that created no record envelope. Partition-bound checkpoints also
reject discovery for a different artifact.

For loaded distributed composition, `PartitionCaptureIdentity::for_session`
accepts the `CommunicationSessionIdentity` retained by native setup. All ranks
derive it from the same complete versioned manifest/nonce transcript. Capture
then uses `capture:<setup>:<first-forward-epoch>` as its common run identity;
each record separately retains its actual forward epoch. The first attempted
preparation binds the run even if admission fails before allocating records.
Restore preserves that binding and the consumed epoch. Independent facade trace
labels are not a substitute for this shared identity, and host labels confer no
execution authority. Native composition must retain the matching model/transport
owners and provide the actual artifact, execution and overlay identities.

`PartitionCaptureObserver` performs this preparation in the shared local callback.
One fixed common coordination frame agrees the complete selection/layout/quota
set, including empty schedules, before source factories or transforms can run.
Selected hooks use separate prepaid status credits on their exact invocation
group. Replicas vote without exporting duplicate values; inactive pipeline ranks
do not inspect source tensors or enter that vote. A capture failure cannot spend
the status allowance or let successful peers enter the next model collective.
Missing hooks reject through the common receipt exchange instead of returning
early while other ranks communicate.

Completed global records remain staged until the shared final commit. Committed
`CapturedStep.outcome` distinguishes a committed model forward from an aborted
one. `Untracked` means a low-level caller or older serialized record provided no
transaction evidence; it is not a success claim. Individual records describe
their actual capture operations: an earlier observation may have completed before
a later provider failure. Consumers must check the batch outcome before treating
those measurements as evidence for a committed prediction. Aborted work retains
its charged usage, draining it does not make it a snapshot boundary, and restoring
an earlier boundary does not refund work. Native completion remains a separate
requirement. A committed forward can still be followed by a sampling failure.

`CapturedStep.partitions` includes the actual forward epoch, identities, producer
acknowledgments, exact local/destination regions and fragment charges. Aborted
delivery publishes no assembled value or partition evidence. Ordinary local
capture cannot bypass partition-bound source authority. The runtime observer
integrates previews, raw slices, full tensors, summaries, histograms and complete-source
vocabulary reductions. Loaded MLX
generation selects it for component points with retained ownership and an
independent hook transport. Partition interventions consume the same retained
ownership and global component coordinates. Each rank uses the same global plan and
participates in ordinary shared forward boundaries. Discovery remains local and
never starts a collective. Cold validation checks ordinary known shape costs;
exact all-rank producer/transport/assembly charges are reserved in shared forward
preparation, before model work. Receipt allowances are derived per selection from
its native encoded-size bounds and capped by the admitted byte ceiling.
Partition plans also admit `CaptureLimitPolicy::Skip`. The runtime reserves the
mandatory common vote before optional selections, then seals successful claims,
skipped indices and reasons, and cumulative accounting into one common digest.
Ranks that disagree reject before model work. A skipped selection runs no source
factory or hook and contributes no payload or partition evidence; its typed record
becomes visible only after final commit. Partial preparation credits remain
consumed, including a producer allowance followed by an exhausted hook allowance.
Record envelopes and coordination cannot be skipped: exhausting those mandatory
costs fails the step. Exceeding an already prepaid child allowance returns a
`PrepaidBound` error with the original typed quota cause; it cannot become an
application-budget skip. Geometry, unsupported mechanisms and native execution
failures retain their ordinary errors. Child capture re-admission uses the same policy.

Controlled forks preserve the loaded setup seed with a fresh capture owner and
first epoch. Existing bindings are checked against the actual loaded model and
parameter version on each observed forward. Empty deferred raw sources use
explicit `GeneratedCaptureSource` precision;
the prototype's dtype is never substituted. Ordinary capture builds empty raw,
preview, summary and histogram payloads after reservation without calling the
source factory. Partition delivery reuses that host constructor for its currently
supported raw, summary and histogram transforms. Unknown empty raw precision remains a typed rejection. Nonempty
sources are checked against their declaration before transformation; native FP8
supplies F32 through `eredu-nn::GeneratedTensorSource`. Dtype-dependent native
estimators use `CaptureBackend::estimate_generated` to price the actual source.

Ordinary native observation adapters preserve structured capture policy and native
transformation failures through the public backend failure's source chain. The
bounded diagnostic string in a failed record is a presentation field, not the
original error object. A generated source or collector failure does not refund
reserved credits. Reusing a model requires the existing state-restoration evidence
and native settlement; recovering a typed cause does not weaken either condition.


Intermediate component-read observations (such as normalized MLA Q/KV latents)
use the same bounded partition projection and receipt machinery as scalar units.
Their final axis describes the selected projection rows, rather than attention
channels. Original and effective paths have the same invocation owner; replicas
participate in interventions while one eligible producer publishes each complete
replicated value. A latent capture describes current input positions. It does not
retroactively observe earlier rows already stored in the cache.

All-dense V3 target partitions publish attention channels and gated FFN units
from their actual TP slices. Complete attention and FFN write observations follow
the ordinary projection reductions; enabling capture adds no diagnostic reduction.
Readout and replicated latent publishers follow retained placement. Native Ring
validation covers TP, PP and combined TP/PP across all three ordinary weight
residencies for F32 SafeTensors/GGUF and load-time affine 4-bit/group-32. Mixed V3 TP shared writes use the admitted additive receipts below; the native
F32 SafeTensors/GGUF and load-time affine matrix passes all 63 TP/PP/EP and
ordinary-residency cases.


Additive TP write receipts use partition schema 3 and an explicit
`PartitionCaptureCombination::SumF64ToF32`. Each producer supplies the complete
selected floating term. Runtime reserves raw export and assembly even for summaries
and histograms, sums raw F32 terms with compensated F64 arithmetic in canonical rank
order, rounds to F32, then transforms the complete selection. Preview exports only
its common prefix. Missing/failed terms are never zero. All invocation sources still
settle before common commit; rollback publishes no payload and restore refunds no
work. Evidence identifies host assembly and term precision, without claiming a
separate complete native tensor. Mixed V3 TP hooks pass neutral conformance and
native F32 SafeTensors/GGUF and load-time affine 4-bit/group-32 acceptance;
see [the component guide](component-validation.md).

Mixed V3 independently cached expert variants also pass all 63 CPU Ring cases
across F32 SafeTensors/GGUF and load-time affine 4-bit/group-32. The same capture
transforms, source completion and controlled replay apply with retained generated
companion ownership. An idle EP cache owner can receive no selected routes;
validation requires actual cache activity on every routed PP stage rather than
inventing local work for every owner.

V3 prediction component scopes are separate from primary target components. Their
fusion and head projection-input captures describe actual multiplication inputs;
all scope points require prediction execution. Ordinary loaded capture reports
reject them without an explicitly instrumented prediction call path. Typed
extension observers provide the internal hook seam, but bounded public
speculative admission, phase geometry and record delivery remain integration work.
Existing one-row speculative logits plans retain their existing meaning.


A separate internal speculative observer seam brackets real forward phases with
their physical sequence width. The runtime error bridge preserves the observer's
generated-source reservation callback and original failure; it never invokes a
skipped factory itself. Ending an invocation does not refund reservations or turn
tentative observations into committed token records. This seam has no public
`CapturePlan` admission or record-drain implementation yet. Existing speculative
one-row `model.logits` capture remains unchanged, and loaded prediction component
support remains disabled until that integration is complete.


Independently shaped forwards can use `CapturePlan::admit_invocations` with explicit
`CaptureInvocationBounds`, followed by `CaptureSession::begin_invocation`. The
actual `CaptureInvocationShape` is retained in the drained step. A multi-row cached
forward can share a prediction coordinate with a one-row proposal; neither changes
the meaning of ordinary one-row decode admission. Exact context is required only
for selected context-dependent axes and is never synthesized from prediction.

Invocation preflight checks one conservative maximum step. Prediction limits are
schedule coordinates, not counts of repeated forwards: the cumulative ledger admits
each actual repetition before capture work, including geometry and skipped-scope
metadata. Restore preserves expenditure. Scope masks narrow already admitted entries
and mark non-invoked points explicitly. Generated and sparse capture use the same
reservation, failure and record paths as ordinary capture. Partition invocation
admission also preserves these bounds and applicability masks. Receipt/evidence
schema 3 carries the exact optional invocation in `PartitionCaptureContext`; all
ranks agree on it before native transformations. Source geometry, fragments and
assembly use the same admitted shape. A context-only disagreement fails even when
the selected point has no context axis. Ordinary plans require absent explicit
geometry; independently admitted plans require it. Metadata reservations account
for every rank, and restoration preserves spent resources.

Bind the partition identity before `begin_invocation`, so its metadata reservation
covers every rank; late binding is rejected. `PartitionCaptureObserver` accepts
that already active invocation and the retained architecture layout supplies its
producers, hook members and exact source shapes.
This neutral protocol has live multi-rank conformance coverage. Public speculative
admission requires selected prediction producers and retained native transport;
loaded discovery reports which family/format paths supply those bindings.

Internal speculative forwards can use `SpeculativeCaptureObserver` with explicit
invocation admission and architecture-projected per-point applicability. It uses
the existing `CaptureSession`, including generated-value reservation, routing
receipts and original/effective intervention evidence. Its fixed-size scheduler
origin records a SHA-256 digest of the assumed generated prefix; it does not copy
the prefix or equate physical sequence rows with generated predictions. Capture
and intervention points outside the actual phase/depth receive `NotInvoked` and
inactive outcomes. Every queued record reserves its envelope before native work;
queue delivery and restore do not refund cumulative use or rewind invocation IDs.
The current adapter represents one request with batch one and no inferred context
axis. Context-dependent selections require an execution-supplied geometry adapter.

`ControlledSpeculativeStep.activations` carries these records separately from the
existing one-row sampler `captures`, under the same step run/restore identity and
trace accounting. Successful internal forwards remain tentative; interpret token
acceptance through the verification disposition. Runtime preserves a typed
`CaptureError` across native error propagation. Explicit CPU/Metal phase binding passes native collector conformance. Public
installation uses `prepare_speculative_activations` and controlled options; ordinary
loaded capture support keeps its original prediction gate.


Use `LoadedModel::speculative_activation_discovery` to obtain the separate loaded
internal report, then `prepare_speculative_activations` to admit its capture/edit
authority. Supply it as `ControlledSpeculativeOptions.activations`. The request's
maximum predictions must fit the admitted schedule domain; actual invocation widths
are checked independently before work. Every record carries the admitted identity.

After `step` fails, `take_activation_evidence` can deliver already reserved aborted
records while the session remains failed. It also drains evidence produced by
cancellation. Each returned envelope spends the same trace allowance as steps;
transport exhaustion consumes the rejected record and never refunds capture work.
Successful step records already include their internal evidence.

`generate_observed_prepared_chat_speculative` and
`generate_observed_managed_plain_text_speculative` drive
the controlled loop continuously with the same options and immediate per-step
delivery. Returning `ControlFlow::Break` cancels and settles the scheduler. The
scoped controlled form additionally exposes failure/cancellation evidence draining.


Internal observers can lend a single observer through `with_activation_observer`
for the whole admitted forward. `with_speculative_activation` keeps that borrow
inside its existing begin/complete/finish scope. Producer work remains live across
callbacks and is dropped before completion or failure finalization; no extra
allocation is needed for the default local adapter. Live five-rank conformance
covers this lifetime with actual partition captures, edits and evidence. This
shared lifetime contract does not by itself authorize public distributed internal
collection or add transactions to auxiliary prediction operations.

Auxiliary prediction observation spans a complete phase, including all depths
executed by prediction prefill. The shared runtime prepares and coordinates its
observer before mutation, agrees operation checkpoints and execution, settles
architecture-enumerated retained state and actual proposal outputs, and delivers
records only through the ordinary completion/final-commit protocol. Completion
failures preserve their original native source. Uncertain commit outcomes fence
subsequent mutation and produce no successful capture publication. This lifecycle
does not by itself supply partition producers or transport; distributed internal
admission requires those selected resources; the loaded V3 path binds and verifies
them as described below.

`PartitionCaptureBackendProvider` composes the ordinary speculative collector with
a retained layout and transport. One borrowed partition observer owns its prepaid
work for the entire phase, including preparation, callbacks and final transaction
commit. The collector binds partition identity before reserving invocation records.
Its existing scopes, failed-record delivery, restore and cumulative accounting
remain authoritative. Five-rank conformance covers two prediction scopes, local
causal edits, replicas, an empty shard, aborted publication and replay. This
provider is reusable infrastructure; public partitioned internal activation
admission requires selected prediction layouts and native factory binding. The loaded
V3 path below supplies both.

Prediction producer discovery reads the layout retained by actual module
materialization. It combines target placement with independent prediction scopes,
including tensor-sharded components, resident routed units, replicated fusion
values and separate prediction heads. The native execution context supplies the
retained transport and loaded identity to the existing speculative provider.
Admission revalidates the plan against that loaded report before reserving work.
Native V3 F32 CPU Ring acceptance passes all 21 TP/PP/EP and residency cases,
including compact channel and routed-unit deletion with original/effective evidence.
Prepared resident prediction providers retain their exact scalar columns and complete
source/failure votes before the following reduction. Serial capture retains local
metadata. V4 and Qwen prediction acceptance uses the same provider; the
component-analysis guide records the tested family/format/residency matrix.


The F32 distributed V3 matrix also exercises captures under coordinated parameter
overlays. Partition evidence retains the active edit identity; complete controlled
snapshots and sibling exchange reproduce the same captured values while copy and
capture budgets remain cumulative. Failed publication/removal preserves the prior
version, and successful removal restores baseline captures and generated tokens.

Speculative activation schema 2 binds context-cache inputs and fused proposal
values to separate invocation scopes. `PredictionContext` selects prediction
prefill/replay only; `FusedProposal` selects fused proposal execution only.
Sequential depth selections do not participate in a fused proposal. A point
outside its declared phase remains `NotInvoked`, never measured zero. Both scopes
use the existing shared capture ledger, exact physical tensor axes, transaction
epochs and final delivery; replay does not reset consumed allowances.

Prepared Qwen prediction captures include raw hidden/embedding inputs, both sides
of fusion normalization, fused residuals, attention channels, dense/shared/routed
FFN units and the actual prediction readout. Scope identities distinguish depths
while parameter identities preserve shared fusion and target embedding/head
ownership. Native TP/PP capture compares these values with resident execution;
failed routed capture closes its provider scope before later collective work.

### Qwen causal prefill row evidence

Dense Qwen2/Qwen3 and the shared dense-or-MoE Qwen factory declare causal equivalence for the common decoder's existing outer row hooks. Body hooks occur before physical readout selection and may preserve an originally admitted LastPosition contract. Coupled readout input and vocabulary-score selections require physical Sequence demand selected before the original quote; a declaration never upgrades an accepted LastPosition request. Exact capture source/catalog, immutable path owner, original geometry and canonical chunk bindings remain required.

Neutral numerical coverage compares all 18 declared hooks of a two-layer model against full-sequence execution, with a two-token cached prefix, uneven 2/1/2 prefill chunks and three further cached decodes. It includes Qwen2 mixed full/sliding attention across its window, dense Qwen3, both dense alternatives through the routed factory, normalized and unnormalized Qwen3-MoE selection, and tied/untied output heads. Each captured hook must be present, finite and nonzero with every requested row; cached state is compared as well. Separate source/discovery binding coverage preserves body-versus-readout demand rejection. These fixtures do not establish native physical capture admission, partition capture capacity or released-checkpoint numerical validation; existing source/funding/native gates remain unchanged.

### GPT-OSS and K2 causal prefill row evidence

GPT-OSS and K2 dense/routed factories use the existing shared outer row declarations. Nine nonzero scalar cases cover learned attention sinks and biased SelectedSoftmax experts, grouped/QK normalization, dense/shared/routed SwiGLU, routed activated values, attention gates, partial RoPE and full/sliding windows. All 18 two-layer declarations, including actual effective callbacks, must match a full five-row Sequence forward against uneven 2/1/2 chunks after a cached prefix, followed by three cached decodes. Tests compare all actual KV state fields and full vocabulary outputs. Observe-only collectors do not manufacture missing effective aliases.

Actual prepared-source/path binding tests cover every declaration for FullTensor and Preview(0). Body hooks may preserve LastPosition; readout-input and vocabulary-score hooks require physical Sequence selected before original admission. Equal independent capture owners, undeclared internals and downreadout reject. These are neutral equation/discovery fixtures, not released-checkpoint or native quantized/distributed validation. Native funding, original-source health, complete opening inventory and control/publication custody remain separate activation requirements.


### Selected-policy inspection contention

Cold retained-value inspection of the actual selected MLX policy returns `false`
when its mutex is held or poisoned. It cannot wait for callback reentry, and an
incomplete child retains the existing visited-prefix semantics. The result is
inventory evidence only: it neither performs native completion nor proves outer
manager/source coverage, collector capacity, source health or capture admission.
The change adds no fields or retained control allocation, so existing measured
control representations are unchanged. Callback-owned collectors and aliases keep
their existing original pricing and lifetime obligations.


### Hybrid text causal row evidence

Qwen hybrid text uses its actual target-unit paths and the same common readout
declarations as the ordinary decoder. Body hooks precede physical readout
selection; readout-input and vocabulary selections still require an original
Sequence request. Preview(0) retains this requirement. The declaration supplies
no native source pin, row writer, retained-owner capacity, budget grant or gate.

Neutral tests require all 18 real callbacks from two hybrid layers, comparing
full five-row execution with uneven 2/1/2 chunks after a two-token prefix and
three further decodes. They compare each convolution/recurrent slot, KV arrays,
positions, reset count and exact fixed-state keys, including intermediate chunk
frontiers. Nonzero fixtures cover dense and routed/shared Qwen3-Next/Qwen3.5,
expanded recurrent heads and history, tied heads and fixed rotary alternatives.
Actual prepared-source discovery binds every declared Full/Preview(0) selection
to the same shared path owner and original physical readout; internal mixer
selections remain undeclared. This is not released-checkpoint or native managed
capture validation.
Finite S includes each successful pin row's retained stamp Arc payload/counters and actual core/attempt controls, so escaped rows keep their original once-held custody. Capture H measures the boxed source slot, optional opening owner, parcel and quarantine-element construction overlap. Ordinary absent funding scopes store only a pointer; both real channel constructors allocate their priced slot before the accounting loan. Busy, comparison unwind or failed origin validation preserves the caller's pending group. No later hold, byte grant or refund is introduced. A group proves only its registered origins, not that an external inventory is complete.


### Original path seal for scheduled Sequence capture

Scheduled ordinary capture distinguishes a declaration-bound selection from
an accepted physical capture contract. The cold text-control producer seals the
actual immutable path identity together with original C, geometry and plan. The
accepted owner can lend a reference-only view; the shared gateway authenticates
that original association and the current prepared execution token before any
source factory or one-use prefill claim. A declaration alone remains diagnostic.

Physical Sequence rows enter the existing original finite frame/fragment bank.
All selected rows survive the per-chunk public-score release in run_final; only
the last public prediction is returned through the existing final index. Failure
or cooperative cancellation preserves spent quota and existing frame/error
ownership. This removes the private Sequence gateway rejection while preserving
public eligibility gates and the independent complete-opening, quiescence,
active-spend and outer/error-retirement prerequisites.


### Independently admitted compiled sources

A loaded compiled decoder source can retain its own original cold construction allowance while successive original R banks lease the same immutable program. Checked planning precedes every compiler reserve; partial failures retain their actual buffers and cause under that allowance. Idle source ownership permits ordinary host preparation while continuing to reduce available capacity. Exact domain/N/skip/mode checks precede the one-use atomic request claim, and bank attachment checks the accepted reservation's domain before consuming staging. Replays and foreign preflight failures add no source lease or error allocation.

The existing destination/kernel and original provider/cursor lifecycle are shared with the unique-source path. Freeze retires the decoder lease/destinations before returning the same token buffer; surviving loaded-source aliases and token-result aliases retain their distinct original tails. Source-only compilation does not retroactively fund HF, tokenizer snapshots, template/input graphs or caller containers. Shared stop sources and facade payloads retain their own admitted owners. Public managed activation uses the shared prepared-chat driver; its tested scope is described in [public conformance](prepared-chat-conformance.md).

The loaded decoder source native regression covers two separately admitted requests continuing one cache across resident, host-layerwise and disk-streamed execution. It compares each request across those realizations and against HF text decoding, and checks retained positions 6 then 12. After the first request, reset rejects with `ReservedWorkActive` even after completion and preserves position 6. Reset requires its own original admitted memory path for replacement state and old/new overlap; this source-sharing integration does not supply that path.


### Original TopCandidates under bounded prefill

The original ordinary/controlled collector supports the existing unsliced batch-one `model.logits` TopCandidates contract: one descending K-vector from the final prediction row, with raw scores and actual source/domain attribution. Candidate-only requests use LastPosition and StateOnly earlier chunks. Independently selected raw tensors retain their required Sequence spans; only the final physical span creates candidates. The original finite H bank owns the candidate Vec once, and the result moves into the existing frame without another payload copy. The frame remains provisional until whole-prefill completion.

Exact source/coordinates, logical cumulative limits, native buffer trace, original host construction and actual native completion remain distinct checks. CPU ordinary numerical behavior keeps its current sorter, but its stable-sort scratch is still unavailable for original physical-bound admission. Arbitrary unadmitted Sequence callbacks, independent invocation-window candidates, spatial candidate fusion and complete speculative original funding remain required follow-ons. No generic feature-axis or time-axis TopK, Preview/TokenScores/evidence expansion, or public managed activation is claimed.

### Original tensor control allocation custody

Original Preview/Slice/FullTensor host destinations include the final shared tensor Arc and concrete custody Box in their pre-accepted construction P. Every alias uses the same closed consuming drop path, so both allocations and the actual shape/data storage retire before final original custody. Payload-capacity inspection and wire/value APIs retain their prior meanings. Failed/partial and zero-length Preview destinations keep the same original hold until retirement.

This closes a host tensor allocation tail; it does not implement global Preview across independent speculative windows, fund an original speculative capture schedule, or complete native workspace/graph/Record/task/encoder accounting. Ordinary and controlled global Preview row assembly remains the existing source-bound fixed-buffer mechanism.


### Global Preview from independent captured-prefill windows

The existing logical companion also carries Preview. For original selected shape S and maximum M, it allocates only min(M, product(S)) output elements and maps physical selected prefixes into their global row-major destinations. Leading dimensions and global stride origins are preserved; physical records keep their own shape, values, outcome and attribution. No full prompt is concatenated for the host aggregate. The existing native worker may still process a whole selected physical rectangle, so the host prefix extent is not a native-workspace bound.

Controls, logical shape and numeric extent are charged before the first transform. Floating/Bool allocate immediately; Integer reserves exactly 8*N and creates one I64 or U64 buffer after the first actual source is validated, without another grant. Every required window is observed even for an empty or already-filled prefix. The complete companion's terminal Captured/Truncated wire growth is checked before infallible final publication. Failure or cancellation drops its provisional values without refunding successful physical work.

The original ordinary/control Preview path retains its existing source-bound Q/H target, frame and escaped-tensor retirement. Online speculative Preview here remains ordinary ledger-accounted evidence. Its original source-bound operation/host/native population is an explicit missing admission integration, not a new managed capability. Generated sources, prefix-valued assistant tensors, arbitrary callback splitting, intervention evidence and window×TP/PP composition remain unfinished.


### Ordinary captured prepared-media spans

The ordinary V2 attachment supports explicit architecture decoder-row FullTensor/Slice selections. It reserves one logical result and envelope plus the scheduled physical fragment estimates before selected extraction, uses the existing fixed slice/scatter algebra, validates zero/no-overlap hooks and precision, and retains partial results through failure/cancellation without refund. Final p0 delivery follows outer prefill indexing and settlement, not an intermediate native completion.

These are ordinary logical quotas and existing host/native lifetime owners. They are not an original source bank, complete native allocation bound, or managed activation. Source and checkpoint controls are reported separately from payload diagnostics. Shared source/tensor aliases preserve final-allocation custody; exact saved checkpoint markers contain no owning predecessor payload or allowance. Original prepared-media admission and global/encoder/distributed capture remain required follow-ons.


### Ordinary prepared-media global transform lifecycle

The collector extension binds Summary/Histogram/Preview fragments to actual Unit3 source geometry and reserves their complete logical destination plus every canonical native physical allowance before the first transform. Terminal candidates reserve the real last-row operation once. Fixed shape descriptors avoid allocation during cold estimation; slice/index buffers and host results are constructed only after the existing ledger comparison. Required metadata fails closed, value Skip stays permanent for the logical selection, and consumed usage is not refunded.

The final record uses the existing p0 transaction and cannot commit before the shared whole-prefill finalizer. Checked Summary/Histogram state and global Preview prefix algebra are reused from text/speculative capture; existing physical envelopes remain unchanged. Exact ordinary row/geometry/tensor-control sizes are accounted, while original managed media admission, CPU stable-sort original scratch and complete native allocator/scheduler/encoder/backing bounds remain unfinished. This fixture scope makes no end-to-end public managed-admission claim.

### Ordinary prepared-media final frame control

The complete ordinary frame, including record metadata, diagnostic Strings, Histogram and Candidates buffers, uses a preallocated SharedCapturedStep control under the actual HostPreparationAuthority. Its exact existing Arc/Box layout is a mandatory host metadata charge before records, including empty or skipped frames; Skip cannot bypass it and no drain-time refill exists. Shared tensor payloads keep their independent all-alias custody. A raw CapturedStep alone carries no host owner. Logical transform allowances remain distinct from complete native bounds, and diagnostic wire errors do not recreate original funding authority.
