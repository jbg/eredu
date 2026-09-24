# Completed-token execution control

The ordinary text facade supports completed-token stepping, pause/resume, canonical
token forcing, temperature changes, explicit reseeding, reusable native snapshots,
restoration and isolated serial branches. Complete snapshot support requires known
costs for every state owner. Currently this includes the forbidden-tool constraint
mode; active and automatic llguidance grammar modes explicitly reject snapshots.
See the [LM Inspector integration guide](lm-inspector-execution-control.md) and
the runnable [controlled generation example](../eredu/examples/controlled_generate.rs).

Sparse expert-unit capture uses the same capture owner in ordinary and controlled
text execution. Plan `CaptureTransform::RoutedUnits` from run creation using the
loaded sparse observation points. Records preserve original token rows and route
slots independently of prediction indices, and become captured only after complete
chunk receipts. Snapshots retain the immutable capture plan; restore and fork do
not refund accumulated capture, copy or transport allowances. Uncaptured historical
unit values require exact token-ID replay. Sparse unit edits use the same admitted
intervention plans, advancement and snapshot machinery. Their global expert/unit
identities are matched against each current forward, including changed routing.
Partial chunk receipts cannot complete a scheduled edit; restore never refunds
edit or route-copy credits. Loaded partitioned capture uses committed distributed
receipts for the supported native invocation groups described below. Partitioned
sparse edits use the same owner, operation ordering and committed outcome delivery.
Selected pipeline-stage status groups include every tensor/expert replica and use
bounded member-only agreement. Prepared component invocations select connected
stage or whole-session groups; storage replicas do not acquire ordinary callback
ownership. Arbitrary disconnected low-level Ring memberships remain explicitly
unsupported for independent status: their world-participation wave would require
inactive pipeline peers to enter a separate coordinated protocol.

## Prompt prefill

Ordinary and controlled generation share prompt chunking in the core driver.
`TextGenerationConfig::with_prefill_chunk_policy` and
`PreparedChatGenerationSettings::prefill` select `PrefillChunkPolicy::Bounded`
(512 tokens by default) or `Unchunked`. Each nonfinal chunk completes before the
next begins, advances the model cache without sampling, and retains the original
prompt identity. Only the final chunk predicts a token. The shared cancellation
token is checked between completed chunks; cancellation can therefore leave a
partial prompt cache, with no committed output. Reset the model before starting
a different request. Chunk boundaries are internal to the first controlled step,
so snapshots and pause/resume remain at completed-token boundaries.

The MLX adapter enables chunking for ordinary replicated plain-text execution
through the shared dense decoder architecture (including grouped-query and
sliding attention). Other architecture drivers retain a complete pass until
their incremental state semantics have conformance coverage.
`LoadedModel::prefill_chunking_support()` reports the executable's support or a
fallback reason. Prepared media/structured inputs, nonempty capture/intervention
plans, prediction extensions and distributed sessions retain a complete prefill
pass: their existing coordinates, transaction or publication contracts describe
one whole prompt. Speculative execution also retains its existing prefill path.
These are current implementation gaps, not architectural impossibilities. Memory
estimates for these paths must use the complete prompt. Dense shared decoder
readout projects only the last hidden position for an unobserved ordinary pass;
observed and speculative full-output contracts remain unchanged.

The focused native regression runs outside the sandbox:

```sh
cargo test -p eredu-backend-mlx --features metal --test chunked_prefill --locked -- --test-threads=1
```

Its nonzero two-layer Llama/Mistral fixtures cover CPU and Metal, grouped-query
and four-position sliding attention, resident/host-layerwise/disk-streamed weights,
and affine four-bit loading. Prompts of 1, 3 and 9 tokens use chunk sizes 1, 2, 4
and 32. Prefill and three cached decode steps agree with full observed logits
within `2e-4` maximum absolute error; four-token greedy output matches between
ordinary unchunked and controlled chunked execution. The observed prefill retains
all vocabulary rows. Portable tests additionally cover cancellation after a
completed prefix, with no token sampling or commitment.

## Timing

`ControlledGenerationSession::timing()` and every `ControlledGenerationRecord`
return the shared `GenerationTiming`. TTFT includes active request preparation and
the first completed token advance, measured before capture delivery or semantic
callbacks. Time between calls, including pauses and snapshot/inspection work, is
excluded. Inference-time capture transformations remain part of active execution.
EOS, structural and buffered tokens count even if no text is visible; cancellation
before commitment returns `None`. Restoring a snapshot never rewinds run timing.
A new ordinary branch measures its first new commitment separately.

## Controlled speculative generation

`with_controlled_chat_speculative` and `with_controlled_text_speculative` accept
the existing speculative request plus `ControlledSpeculativeOptions` and a worker
closure receiving `&mut dyn ControlledSpeculativeSession`. The backend lends its
prepared execution resources for that scope. No prefill or token sampling occurs
until `step`; between calls the worker can wait for Inspector commands. Returning
from the closure cancels and settles unfinished work through the existing scheduler.
The same request settings, grammar, token validity, stopping rules and semantic
callback apply as in uninterrupted speculation.

A step performs one scheduler action: prefill, draft a block, submit verification,
draft optimistically, poll, or resolve verification. A verification commits an
entire accepted prefix and replacement/bonus atomically. It cannot be exposed as
independently restorable single-token steps while the native cache is ahead.
`ControlledSpeculativeStep` reports new proposals separately from committed IDs,
per-proposal accepted/rejected/discarded dispositions, and optimistic prefix,
reuse, bonus consumption and discard counts. Capture records identify the target
or drafter and absolute generated-token prediction position. A UI can animate a
block's individual proposals without treating tentative tokens as conversation
history. `sequence` is monotone; `epoch` increments after restoration.

The returned output and session expose active TTFT, including facade/backend
preparation and target prefill but excluding time in the Inspector command loop.
Draft proposals do not establish TTFT. Timing and trace usage do not rewind.
`trace_limits` bound step records through the same JSON counter as ordinary
controlled generation. The request's semantic callback retains its existing
synchronous publication contract and is a separate consumer channel.

Supply snapshot limits before entering the scope. `snapshot_support()` explains
availability; `snapshot()`, `restore()` and `release_snapshot()` manage opaque
session-local handles. Snapshots preserve cache, assistant seed/context, sequence,
sampler, RNG, grammar/decoder and adaptive statistics. Restoration neither runs a
forward pass nor replays tokens. Handles cannot cross runs. Failed and cancelled
runs cannot be revived; normally completed runs can restore an earlier snapshot.
Copies are reserved before allocation, and cumulative copy/trace/capture costs
are never refunded. Snapshots currently require a canonical boundary after prefill
with no retained proposal block or in-flight verification.

Ordinary MLX compressed-latent state uses independent latent and rotary storage
for snapshots. Resident copies retain compact logical arrays, and continuation
allowances include cache allocation chunks. Paged copies also isolate sealed
blocks, catalog ownership and mutable tails. These copies use the same snapshot
and branch machinery as ordinary KV and fixed recurrent state; copying a cache
does not itself establish support for an embedded prediction executor.

MLX supports reusable external Gemma 4 and Muse Glimmer/DFlash checkpoints when
the target storage, sampler and semantic state provide complete isolated-copy
estimates. Complete embedded V3 and V4 sequential/DSpark snapshots also copy the
target, canonical prediction lane, independent prediction seed and capture tensor.
Native CPU/Metal acceptance covers resident, host-layerwise and disk-streamed
weights. V4 pooling copies preserve local keys, pending/pooled windows and overlap
buffers; each copied paged state owns an independent catalog in the same bounded
pool. Other state profiles require complete native copy bounds before admission.
Ordinary active llguidance constraints remain unsupported for snapshots because
their storage cost is unknown. These limitations do not prevent controlled
speculative stepping or proposal observation.

For captures, call `prepare_speculative_capture(settings, plan)` and pass its
admitted plan in `ControlledSpeculativeOptions::capture`. Each observation is one
vocabulary row, including prefill; ordinary whole-prompt admission is rejected.
The MLX implementation supports `model.logits` for target prefill, reached verifier
rows, and every sampled draft proposal, including rejected work. It reuses the
ordinary bounded transformations (including top candidates, summaries and slices)
and one shared cumulative ledger. Internal layer observations use the separate
`prepare_speculative_activations` admission and `activations` option. Loaded support
is verified separately for each selected execution; V3 internal component capture
is verified across resident/host/disk and distributed TP/PP/EP execution. The
component-analysis guide records complete family and encoding coverage.
At a canonical boundary, `fork(&snapshot)` creates an inactive branch slot and
`exchange(&branch)` swaps its logical run with the active run. `run_id()` and the
step record's `run_id` identify the active run; `branch_info()` describes the
inactive prefix. Forks may originate from any snapshot in the same scope, but
`restore()` accepts only snapshots of the active logical run. `release_branch()`
releases inactive retention. Returning from the scope returns the active run's
output and releases all inactive branches. Cancellation ends the whole scope.

Fork creation shares the immutable saved payload. Exchange uses the existing
exact snapshot/restore mechanisms, without forward execution, loading weights or
replaying semantic events. Unlike ordinary zero-copy exchange, speculative exchange
copies the outgoing and incoming states. The shared `SnapshotBudget` reserves both
copies before installation, including actual state growth. Branch slots consume
`max_branches`; a source snapshot retained by a branch still consumes its snapshot
reservation until its last reference is released. Unknown costs and budget failures
reject an exchange before installation. Native restoration failures fence the scope.
Timing, delivery sequence/epoch, trace and capture budgets belong to the whole
scope and never rewind or multiply when branches are created. TTFT continues to
describe the first token of the common request, including the inherited prefix.

`force_next_token(id)` uses the ordinary `TokenChoiceController` to validate the
canonical tokenizer domain and active grammar before restricting just that absolute
prediction. Tentative draft histories do not consume the choice; target commitment
consumes it once through the existing acceptance, sampling and semantic path. Step
records identify a consumed `forced_token`. `clear_forced_token()` removes a pending
choice. `override_sampling(SamplingOverride)` shares ordinary temperature/RNG
validation; temperature changes retain both speculative streams, explicit reseeding
replaces both, and adaptive sampling rejects zero temperature. Snapshots preserve
pending choices, temperature, RNG, adaptation, and immutable edit plans. Edits and
sampling changes reject retained proposals, in-flight verification and terminal runs.

Use `speculative_intervention_discovery()` and
`prepare_speculative_intervention(&capture, role, plan)` to admit tensor edits, then
`session.intervene(plans)` to replace future plans. Supply at most one plan per
`Target`/`Draft` role; an empty list removes future edits. Plans can also be installed
before the first `step()`. Pass the capture admission in controlled options even
for evidence-only edits. MLX supports the existing activation operations at
`model.logits`: zero, scale, mask, replace, add and logit masking. It observes raw
values, applies the shared intervention engine, then invokes ordinary constrained
sampling. Evidence is attributed by model role and absolute prediction position,
including tentative/rejected work, and charges the same cumulative ledger across
replays and sibling branches. Admission is rechecked against the exact loaded
source/session; unsupported layer/routing hooks are absent from discovery rather
than pretending to support speculative attribution.

Initial speculative snapshots still need a portable copy contract for backend-owned
prefill input. Embedded prediction forks remain unavailable wherever their complete
snapshot costs or isolation contracts are unavailable.

## Feature development rule

New inference functionality must support controlled sessions through the shared
driver. Add controlled/uninterrupted behavioral coverage. Exceptions need an
explicit capability or typed rejection with a concrete completion, state,
attribution or accounting reason recorded here and in the Inspector guide.

## State owners and boundary

| State | Existing owner | Required snapshot treatment |
| --- | --- | --- |
| Model KV, ring/sliding offsets, fixed recurrent and convolution state | Architecture-declared runtime state, realized by backend mechanisms | Independent native state copy plus all typed metadata |
| Prepared source, execution selection, native authority | Loaded backend session | Retain exact compatibility and completion ownership; never reload or replay |
| Pending prompt or last committed decode input | Core `TextGenerationMachine` | Preserve phase and pending input explicitly |
| Native RNG, standard/Mirostat policy and penalty history | Backend text-generation state and neutral sampler | Preserve RNG key and adaptive counters; overrides must be explicit |
| Canonical generated IDs, output limit, terminal precedence | Facade `CommittedGenerationCursor` / core `GenerationSequence` | Copy complete sequence and terminal outcome |
| Grammar/token constraints | Facade `ConstraintController` | Independent grammar fork |
| Incremental tokenizer, pending UTF-8, stops, tool parser and delivery | Facade `CommittedTokenPipeline` | Exact copy without token replay, output or finalization |
| Capture/admitted interventions and schedule position | Runtime `CaptureSession` / `InterventionRun` | Shared opaque checkpoint with drained evidence and child re-admission |
| Cumulative capture, transport and copying consumption | Run accounting owners | Never rewind or refund |

Prediction zero executes prompt prefill. After committing prediction N, the last
committed token can still be the pending input for prediction N+1: it must not be
inserted into the native cache twice. The full snapshot boundary must establish
native completion, finish intervention checks, resolve routing controls, deliver
the current bounded record batch and preserve that pending input. Pause does not
flush partial UTF-8, stop lookbehind or incomplete protocol state.

## Implemented shared mechanisms

`GenerationLifecycle` validates Prepared/Paused/Running/Completed/Cancelled/Failed
transitions. Running includes completion and delivery. It rejects a checkpoint
while running, cancelled or failed. Restoring a normally completed checkpoint
preserves its terminal state; restoring an earlier valid checkpoint into a normally
completed run can resume that earlier continuation. Restore increments a monotone
epoch for consumer reconciliation. Lifecycle state alone proves no native completion.

`GenerationControlHandle` carries a thread-safe pause request and the existing
permanent `GenerationCancellationToken`. A worker may retain thread-affine native
objects. Only an explicit resume acknowledges a pending pause; acknowledging pause
does not clear cancellation. Existing callback `ControlFlow::Break` retains its
cancellation meaning.

`ReplicatedTextSnapshotMechanisms` defines side-effect-free native estimates and
independently writable state copying. Shared replicated-session control slots retain
typed state plus prepared-input identity and commit metadata. Exact executable
identity and geometry are checked before mutation. Slots can be copied or exchanged
for serial branches while retaining the same executable and completion owner. The
native implementation still must prove isolated copies and completion, and the
facade still must compose all the other generation state listed above.

The MLX state-copy primitive now independently copies device KV arrays and fixed
recurrent/convolution tensors, materializing strided views contiguously first.
It preserves sliding windows, backing capacity, absolute offsets and typed layout.
Its estimates include native logical data, possible contiguous-copy storage and
host layout/role and input-identity metadata. Paged and compressed state copies are
explicitly absent from this primitive. The loaded MLX adapter implements the
`NativeTextStateBackend` contract with existing submission/recovery ownership and
checks exact executable identity before any state change. Copy completion failures
fence the shared engine when safe recovery is not established. An incompatible
slot fails before mutation without poisoning an otherwise healthy run. Slot exchange
is a host move under idle authority and opens no new native scope.

`TextSnapshotBackend` adds exact sampler/RNG and pending-input copying. Shared
`TextContinuationSnapshot` composes model state, sampling, pending input, explicit
independent constraint copies and capture/intervention checkpoints. It reserves
known component estimates before copying. Restore validates same-run identity and
stages all copies before atomic native exchange and infallible host installation.
Children get fresh identities and re-admitted shared record owners. Their retained
leases move with the logical child when it becomes active, even if the slot holding
its parent is dropped. Runtime combines backend model/sampler/pending-input growth
facts; the facade adds decoder, parser, constraint, history and child output growth.
The complete allowance is reserved before a runnable child is published.

The facade pairs this with complete semantic/cursor/lifecycle state and explicit
output checkpoints. Snapshot capability remains conditional on all component costs
being known; active/automatic llguidance grammar estimates are still unavailable.

A native CPU fixture restores and interleaves independently copied dense model
states with exact logit agreement. It distinguishes alternative input tokens and
checks counters for no extra forward execution, state reset, artifact reopening or
weight materialization during copy/exchange. Separate native storage fixtures cover
KV sliding windows/capacity and recurrent/convolution arrays.
`eredu-evaluation::execution_control::continuation_conformance` now runs against
host and loaded native dense, LFM2 convolution/attention, and Qwen3.5 MoE
recurrent/convolution/attention fixtures. It checks reusable initial and decode
snapshots, sibling interleaving, fresh admissions, controller state and cumulative
accounting across ordinary, captured, intervened and combined modes. The native
fixtures include nonzero temperature, standard and Mirostat sampling, penalties,
changing token constraints and exact bounded evidence. The MoE plan also forces
experts at an actual discovered routing point with prospective decode scheduling.
Probes exclude hidden
forward execution, reload and reset during restore/fork. Host failures exercise
unknown native/controller estimates and failures in host and native copying or
exchange. Native completion tests also exercise runtime-lock contention: a
successful wait retires observation/sampling ownership before returning, while
polling and errors preserve nonblocking recovery.

`CaptureSession::checkpoint` requires successful, drained portable records. It
rejects outstanding routing, missing/failed interventions, failed capture outcomes,
and partial admission failures. Its opaque checkpoint retains immutable admitted
plans, prepared-source provenance, absolute next-prediction position, and inherited
usage. It contains no native estimator or native handle. A same-run restore checks
opaque run identity and restores only schedule state: it publishes no old records
and leaves the cumulative ledger unchanged.

Drained capture batches expose `CapturedStep.outcome`. An aborted model forward
can contain valid observations collected before its failure, but cannot provide
evidence for a committed prediction or create a snapshot boundary. Its reservations
remain charged after draining and restore. `Untracked` identifies low-level or
legacy batches without transaction evidence. Completion and state reuse still
follow the existing native ownership protocol, independently of this outcome.

Routed provider success votes belong to execution and run even without capture.
Pipeline waves agree execution before preparing a boundary transfer. A completed
negative vote distinguishes a compute rejection from a failed transfer; actual
submission, completion and transport failures keep their existing terminal fences.
The failed iterator remains terminal. Synchronization and state-control admission
still govern reset or reuse; the vote alone does not establish native completion.

Shared-session capture also retains a nonrewindable transaction epoch. Pending
transactions cannot drain records or create snapshots. Exact completion stages
delivery before the ordinary final commit; a rejected or uncertain commit keeps
the attempt outside successful capture snapshot boundaries. Restore does not
refund an attempted epoch or cumulative capture work. The same callbacks run
through ordinary and controlled execution, including the native observation
adapters.

Step envelopes are admitted inside the shared observer preparation phase using
the actual forward's phase. Even an admission failure that creates no envelope
retains its attempted epoch and blocks restore until final discard. A completed
discard permits schedule restoration while preserving that epoch and cumulative
usage. Rejection before model-state access is distinct from successful rollback;
neither alone proves native settlement or permits reuse while work remains owned.

Ordinary MLX observation failures retain typed `CaptureError` and
`ObservationError` causes, or the original native exception, through the public
`BackendFailure` source chain. Diagnostic record text remains bounded separately.
CPU/Metal public tests across resident, host-layerwise and disk-streamed execution
verify that snapshot replay does not refund capture allowance and that both ordinary
and controlled native score-capture failures publish no token. Native recovery
tests establish settlement before retry; error provenance alone is not a reuse
guarantee. See [the observation error evidence](component-validation.md#observation-error-causes).

Cold request, prompt, sampler, instrumentation and initial-delivery results also
participate in shared preparation agreement. Failed ranks retain their local
cause; peers receive a typed stage/rank rejection before forward. Initial callback
cancellation propagates to every participant. The session-owned coordinator uses
bounded native transport and keeps attempt/copy charges monotone across reset,
restore and branching. It is not part of a restorable model snapshot. See
[cold text-run preparation](component-validation.md#cold-text-run-preparation) for
API migration, accounting and current verification scope.

Native Ring tests across TP, PP and combined TP/PP verify unchanged cached state
after agreed prompt, sampler and capture-installation rejection, followed by a
corrected run. An unagreed protocol failure permanently fences communication;
synchronization checks that fence before waiting on native queues, and reset
cannot restore the fenced session. A failure does not release resources still
retained by native completion.

`CaptureCheckpoint::fork` re-admits observations against the child's loaded catalog
and support. Interventions are re-admitted against child discovery with a fresh
facade session ID and the child's estimator. Request geometry and known costs are
checked for the remaining absolute schedule; frequency origins never move. Explicit
child cumulative limits include the usage consumed at the checkpoint. Later parent
and child work is charged independently. Removing all future interventions retains
the inherited ledger. Replacements apply only when future predictions execute; the
original admission remains available for lineage/provenance.

`SnapshotBudget` reserves known logical storage and copying costs before native
work. It bounds simultaneously retained snapshots, child states, total retained
bytes, and cumulative copy allowances. Unknown estimates, arithmetic overflow and
limits fail before reservation. The last handle releases retention; failed admitted
copy attempts still consume copying allowance. Restore uses a provisional retention
reservation. These are logical bounds, not a physical allocator/private-workspace
ceiling. Capture and transport budgets remain independent.

The ordinary one-shot driver now delegates to `CommittedGenerationCursor::step`.
This reuses the existing token source, tokenizer, constraints, EOS precedence and
semantic pipeline. The pipeline and built-in protocol parsers support exact state
forks, including buffered bytes, JSON/Python/tool-call state, stop matching, pending
delivery and terminal state. An unimplemented custom parser fork fails explicitly.

The core ordinary text machine also exposes `TextGenerationDriver` and detached
`TextGenerationContinuation` state. Existing asynchronous and controlled iterators
use the same machine. The detached state retains the pending prompt/token,
controller, sampler, remaining allowance and exact completions. A driver rejects
another driver's continuation before any work, requires completion and one-step
record draining between advances, and fences failures even after cleanup. This
low-level mechanism still requires facade composition with native state exchange,
semantic delivery and lifecycle policy; it alone is not a controllable session.

## Current public stepping workflow

Prepare with `prepare_observed_chat` (use `CapturePlan::none()` for ordinary runs)
or `prepare_intervened_chat`, then call `LoadedModel::start_controlled_chat` with
the prepared request, stop sequences, `GenerationControlHandle` and a synchronous
record callback. The returned `ControlledGenerationSession` exclusively borrows
the model and provides `step`, `pause`, `run`, `resume` and `cancel`.

Each `step` commits at most one prediction and settles native completion before
delivering its captures and semantic events. `pause` consumes no token, random
draw or buffered Unicode. `run` observes sticky remote pause requests at these
boundaries; `resume` explicitly acknowledges pause. The thread-safe handle can
cross threads while the native session stays on its owning worker. Callback
`Break` remains permanent cancellation and closes delivery. Explicit cancellation
discards unfinished semantic buffers; it never flushes them as normal completion.
Consumer panics fence the session before unwinding. Normally completed, cancelled
and failed sessions cannot step again.

`ControlledGenerationRecord` wraps the existing attributed generation record with
a monotone per-run output sequence and restore epoch. Compact-JSON limits
include the wrapper and lifecycle records. The same transport counter implements
ordinary observed delivery; pause/resume never resets it. The facade tests cover
all four record modes, pauses between UTF-8 pieces and before EOS, stop sequences
spanning steps, cross-thread pause, cancellation and consumer/transport failure.
`enable_snapshots(SnapshotLimits)` configures a non-resettable budget after checking
native, grammar and semantic support. `snapshot(callback)` returns an opaque reusable
`ControlledGenerationSnapshot`; `restore(&snapshot, callback)` restores the source
run. Creation supports the initial, paused and normally completed boundary. Completed
snapshots remain terminal; an earlier snapshot can resume a normally completed run.
Cancellation and failure cannot be revived. Native and semantic copies are admitted
and staged before installation, including fallible semantic preparation during restore.
The snapshot is tied to its original loaded driver and logical run; serialized
`GenerationSnapshotMetadata` cannot recreate it or establish compatibility.

`SnapshotCreated` carries metadata and its `GenerationOutputCheckpoint`. On restore,
`Restored` names the source prefix to retain in the consumer's event journal. The
next sequence remains monotone and the epoch increments. Earlier token/text events
are not emitted again. A consumer keeps the chosen prefix and replaces the later
continuation. Callback and transport failures retain the ordinary cancellation/fencing
policy. Capture, transport and copying usage never rewind. Snapshot drop releases
retention only. The facade conformance covers this in all four record modes, including
partial Unicode and terminal snapshots.

`fork(&snapshot, GenerationBranchOptions, callback)` creates an inactive isolated
child. `exchange(&mut branch, callback)` moves all native and facade state between
the active run and that slot; the slot then holds the formerly active run. The
exchange copies no tensors and makes no prediction or RNG draw. Cancellation,
transport consumption and semantic output follow each logical run. A cancelled
child may be exchanged out so a healthy parent can continue. Failed or unresolved
native state remains fenced; it cannot be made resumable by switching slots.

Options require explicit child transport limits and capture limits when the source
retains a capture owner. Sampling changes preserve inherited state unless `reseed`
is explicit. An optional intervention request replaces future operations through
shared re-admission. The absolute generation limit remains the snapshot's original
limit; this API does not extend it. The child may itself create snapshots or fork
descendants under the original tree's shared snapshot budget. Same-run restore
requires the snapshot's logical run; fork permits another run in that same exclusive
driver tree. A new generation on the same loaded model is a different driver and
cannot import the old tree's snapshots.

`BranchStarted` is sequence zero of the child stream. It includes parent snapshot
metadata, inherited capture consumption, future choices and explicit transport
limits, plus the immutable canonical prompt, generated-token prefix and exact
delivered semantic events. A
consumer can initialize text, reasoning and partial tool calls from that prefix
without replaying tokens or retrieving the parent's journal. Subsequent events use
the fresh child's run/session IDs and re-admitted plan IDs. Prediction/input
positions remain absolute; subtract the parent snapshot's next prediction for a
branch-local decision index. Existing capture values remain before-intervention
observations; intervention evidence retains its existing before/after meaning.

`LoadedModel<B>::start_controlled_chat` returns `ControlledGenerationSession<B>`;
snapshots and branches use `ControlledGenerationSnapshot<B>` and
`ControlledGenerationBranch<B>`. These generic types are also the MLX application API.

`force_next_token(id)` validates canonical vocabulary membership and the current
grammar, then restricts the ordinary sampler to that one candidate. It commits
through the same history, penalty, decoder and termination path. Conflicts fail
before advancement; `clear_forced_token()` removes a staged choice. A forced token
record includes `forced: true`; ordinary records retain their previous JSON shape.
At nonzero temperature the decision consumes one ordinary RNG draw. Greedy sampling
consumes none. Mirostat observes probability one and updates its adaptive state once.
A pending choice is part of the shared continuation snapshot. Replacing a token
already committed requires a snapshot from before that decision.

On backends implementing `TextSamplingControlBackend`, `override_sampling` accepts
`SamplingOverride { temperature, reseed }`. Temperature changes preserve RNG,
adaptive counters and penalty history. A stochastic run can become greedy and
later resume its retained RNG stream. A run created without an RNG needs an explicit
seed to become stochastic. Mirostat rejects zero temperature. `reseed` resets only
randomness, never adaptive state or history. Strategy changes and penalty/filter
overrides are not exposed in this initial policy. `SamplingChanged` records carry
the request, before/after facts and the first absolute prediction affected.
Native preparation uses the existing completion/recovery owner and executes no
model forward. Invalid requests leave the source unchanged; native failures fence
the public session. Callback failure retains ordinary cancellation semantics.

The shared forced-choice and sampling-override conformance scenarios run with the
host and all three native CPU fixtures in every record mode. They restore pending
choices and explicitly reseeded children, preserve the parent, and verify no replay.
Native sampler tests separately verify exact key progression and adaptive updates.

Built-in semantic parsers, stop/UTF-8 buffers, decoder history and the committed
cursor now have field-audited logical storage estimates. Immutable tokenizer data
is shared by `Arc`. Unknown custom parser/decoder costs fail explicitly. These
estimates count live logical values and owned bytes, excluding allocator capacity
and node overhead; `semantic_snapshot_bytes()` reports that component only. The
forbidden-tool constraint mode also has a known independent-copy estimate. Active
and automatic grammar modes remain unknown: llguidance 1.8.0 exposes independent
deep copying through [`Matcher::deep_clone`](https://github.com/guidance-ai/llguidance/blob/v1.8.0/parser/src/matcher.rs),
but no complete live matcher storage estimate or conservative continuation-growth
bound. `Matcher::last_step_stats` reports per-step work, while the lower-level
[`RegexVec::num_bytes`](https://github.com/guidance-ai/llguidance/blob/v1.8.0/parser/src/earley/regexvec.rs)
estimates regex tables only. Neither accounts for the full parser state, including
history buffers and caches, so neither satisfies snapshot admission. The dependency
uses the compatible `1.8.0` requirement without an exact pin; upgrading it has not
removed this limitation. Declaring tools with `ToolChoice::None`
selects the supported forbidden-tool owner on a profile with a tool surface;
ordinary no-tools requests can still use an active semantic grammar and therefore
must check support. Adding tools purely to change this capability also changes the
rendered prompt; the example makes that choice explicitly.

Branch allowances use checked conservative logical bounds. Native cache capacity
rounding, initially absent fixed state, sampler history and pending inputs are
included. Tokenizer decoder chains are priced for expansion and a maximum complete
decode per future decision; parser bounds include normalized text and structured
state. This can over-reserve substantially for long continuations. Child semantic
history is additionally bounded from its explicit trace budget. Allocator capacity,
driver-private workspaces and physical residency are not claimed as hard ceilings.
The parent baseline state and shared immutable weights/tokenizer remain owned by
the loaded run; additional snapshots, children and provisional restores are charged.

## Verification and remaining support limits

Portable suites cover lifecycle, independent state, constraints, sampling changes,
capture/intervention admission, failed host/native copies, limits, sibling switching,
pending forced choices, partial Unicode, stop lookbehind and terminal restoration.
Native CPU continuation coverage includes dense, heterogeneous LFM2 and recurrent
Qwen3.5 MoE with genuine routing interventions. Forward/reload probes establish
that copying/restoring/forking does not replay the prompt or reload weights.

The released LFM2-8B-A1B component probe also passes native Metal parity between
ordinary and controlled execution for baseline, deletion, keep-only, coordinated
parameter overlays and restoration: seventy controlled predictions match their
ordinary counterparts exactly. Each trial covers post-prefill restore, a sibling
from the initial boundary and parent replay after exchanging the sibling. Its
fork budget includes the trace-derived semantic storage reservation. Independent
Transformers numerical parity is recorded separately in
`doc/component-analysis.md`; lifecycle parity alone does not establish it.

The complete local facade has a native Qwen2 fixture for sampled partial-text
restore/fork after removing its loaded weight/tokenizer files. A second native test
executes the complete observed/intervened example and verifies unchanged restore
and fork before producing the modified child. These original CPU tests use
`--no-default-features --features mlx`. Later native Metal acceptance and released
reference comparisons are recorded in the component-analysis guide. A Metal-enabled
MLX build may require GPU access during initialization even when a test requests
a CPU stream; the CPU-only build remains the verified headless route.

Active/automatic llguidance state still needs a complete upstream storage estimate
and a conservative continuation-growth bound to broaden snapshot support; those
configurations fail before copying today.

Ordinary partitioned state copies require bounded all-rank preparation agreement.
Callers must advance the same operation and branch on every participant. Native
state-copy tests and six public distributed facade cases cover this contract,
including greedy and seeded sampled restore/fork with resident/host/disk weights. Selected prediction state, media and realtime control, persistent
snapshot files, cross-process/cross-backend restoration and layer-level pausing
remain separate capability decisions.

## Multiple routing banks in one layer

Routing control and observation dispatch include the architecture-declared bank
identity as well as the logical layer. Attention-value and feed-forward routing
therefore have independent paths, global expert cardinalities and effective
decisions. The observed provider uses the same bank identity before selection,
when attributing evidence, and when dispatching acquired expert work. Provider
boxing and bank lookup retain the original completion and accounting owners;
they do not create a second generation driver or a rewindable budget ledger.

K2 Horizon's replicated native tests cover both routing banks, device and paged
KV snapshots, independent interventions, and exact restored continuations.
Facade tests cover controlled/uninterrupted sampling with resident/host/disk
weights, partial-text restore/fork, captures and modified branches. Paged snapshot
copies retain independent namespaces within the same finite pool; estimates
include sealed blocks and mutable tails. See [K2 validation](k2-horizon-validation.md)
for commands and speculative validation results. Distributed
copies complete before preparation agreement, and exchanges leave every installed
state unchanged on preparation failure. Commit epochs remain monotonic across
branch exchange, manual rollback and prompt-cache restore. Snapshot copies and
branches continue to consume the existing nonrefundable copy and observation
ledgers.


K2 can use an independently executable external draft with the same tokenizer.
The existing external-drafting plan prepares both ordinary decoders before
execution. Proposal caches are private; snapshots preserve canonical target and
draft caches plus the immutable proposal seed. Rejected verification replays only
the accepted target prefix and advances the draft to that same frontier. Both
uninterrupted and controlled requests use the shared speculative driver, including
logit capture/intervention, sampling, termination and nonrefundable copy budgets.
K2 has no published embedded prediction head, so embedded drafting is rejected
by ordinary architecture admission.


Component observation and masks use the same controlled forward and intervention
re-admission machinery as ordinary block controls. A prepared-boundary snapshot
precedes prefill and supports testing the first prediction; a snapshot after a
forward cannot recover its unrecorded component activations. Select prompt tensor
rows separately from prediction indices, and do not force the token being scored.
See [component analysis](component-analysis.md) for actual family, encoding and
execution coverage.

For exact historical replay, `prepare_observed_token_ids` and
`prepare_intervened_token_ids` admit the saved IDs without rendering or tokenizing
them. Supply the original chat output/termination contract and start at an empty
model state; the normal controlled driver handles prefill, sampling and capture.
Unknown tokenizer IDs and empty prefixes fail before native prompt construction.

Parameter overlays use a completed, idle model boundary. Activation and removal
clear incompatible native state, invalidate native snapshots and change the facade
session identity, so prepare new requests after either operation. Ordinary and
controlled inference then use the same edited operators. Sampling snapshots retain
the parameter epoch; a stale low-level generation state is rejected before its
next native submission. Capture records carry the active `parameter_overlay_id`.
Persistent prompt-cache operations currently reject active overlays. These are
explicit compatibility rules; an old KV/recurrent state is never interpreted as
if it had been computed with the new weights.

The same lifecycle applies with independently cached expert banks. Parameter
queries borrow and release bank members sequentially, so they can change weight
residency without changing KV or recurrent state. Overlay publication replaces
the handles used by bank providers in both ordinary and controlled execution.
Their compact-memory accounting includes floating replacements of packed weights;
removal restores the original immutable cache sources. The component-analysis guide
records the tested cache budgets, formats and parallel configurations.

Dense and independently blocked fused FP8 FFNs retain exact component indices
when a tensor rank owns a short final intermediate block. Controlled captures,
masks and parameter edits use the same retained chunk placement as uninterrupted
execution. Fused gate/up branches retain separate scale-block origins, including
when each local branch has only three rows. Coordinated edits can affect both
branches of one primary parameter while preserving atomic publication, controlled
replay and restoration. The 259-unit fixtures check the final three units across
resident, host-layerwise and disk-streamed TP/EP/PP execution, including
independent-bank caching; the component-analysis guide records matrix acceptance.

Split attention FP8 tails likewise retain whole grouped-query groups across
Q/K/V/O projections, separate value banks and per-head normalization. Layers may
have different local head counts when their encodings differ; each layer's cache
geometry follows its retained placement. Controlled masks and overlays keep global
channel coordinates, including the last partial head. Parameter-query phases add
new allowances to consumed cumulative work; resetting state or evicting a bank
never refunds earlier parameter charges. The attention-tail acceptance matrix and
FP8 source-format scope are recorded in the component-analysis guide.

Distributed parameter transactions prepare a fresh state slot on every rank,
including ranks without edited weights. The outer parameter coordinator owns
publication and rollback agreement. Local state exchange preserves the original
cache and prompt identity without copying cache tensors or entering another
collective. A completed rejection restores them and keeps existing snapshots
valid. Only confirmed success invalidates snapshots and advances the parameter
version; restore never rewinds communication epochs or refunds budgets. Native
initialization is reserved from the selected state's complete bound, including
absent fixed components. Removal retains the active handles for rollback and
charges its known state/metadata work without reloading parameter sources.

Partition capture now uses the same transactional observer callbacks as ordinary
and controlled forwards. Its work handles belong to the actual capture owner and
monotone forward epoch; restored or foreign handles fail before source access.
Common pre-forward coordination agrees selected layouts, identities and global
reservations. Active invocation groups separately agree hook success before model
collectives. Receipt delivery stages global records after exact completion, and
only the existing final commit publishes them. Restore does not reset producer,
transport, status, host-assembly or encoded-evidence consumption.

Native communication setup now retains a common instance identity agreed from
the complete manifest/nonce transcript. Runtime capture can bind it to the first
attempted forward epoch without doing native work during cold configuration.
Failed admission and restore preserve that run identity; later records retain
their distinct actual forward epochs. A new capture owner uses the model's later
epoch, and a new model setup has a fresh instance identity. These descriptive
labels do not replace move-only submission authority or retained native owners.

Loaded component capture now uses this adapter in ordinary and controlled
generation. A capture checkpoint includes the retained setup seed; a re-admitted
child receives a fresh owner and binds its own first forward epoch. Parent replay
keeps the parent's run identity. Loaded preparation rejects a seed from another
setup, execution or parameter version before source work.

The native partition capture matrix includes shared Qwen MoE, GPT-OSS and
Nemotron-H provider execution. Attention channels, normalized FFN inputs and complete expert writes
follow the same controlled advancement, parent replay, sibling isolation and
nonrefundable ledger rules as dense capture. Public distributed activation edits
use the same admitted observer and final commit. Nemotron's mixed convolution,
recurrent and KV state also participates in coordinated parameter edits and replay;
see [its validation matrix](component-validation.md#nemotron-h-partition-component-execution).
Its always-executed shared ReLU² units use the same original/effective capture,
position-specific masks and prepared-boundary replay as ordinary dense units.
Shared input and output edits stay inside the shared branch of the sparse sum;
they do not define a second controlled execution driver.

The shared input-result phase coordinates token extraction, composite
admission, parameter-version checks, capture prompt alignment and decode framing
before observer admission or state mutation. A rejected step preserves prior
state and consumes its epoch; native settlement remains mandatory before retry.
Cold text-run preparation uses the retained preparation coordinator described
above. Distributed applications must
prepare compatible requests and advance every rank through the shared protocol.
Discovery combines actual hook coverage, retained global ownership and native
subgroup transport. Remaining specialized drivers and disconnected subgroup
mechanisms retain explicit implementation gaps.
Partition capture admits fail-on-limit and skip-on-limit plans, including child
re-admission. Before source work, every rank compares the successful selection set,
skipped indices/reasons and all consumed accounting. Disagreement rejects the
forward; an agreed skip is published only on final commit and never runs its
source factory. Restore cannot recover consumed preparation or coordination
credits. Mandatory envelopes and common coordination still require a budget.

Deferred capture precision is part of the shared observation contract for both
ordinary and controlled execution. Empty selections produce the same typed host
records without invoking a deferred source factory. Runtime validates nonempty
source declarations before transform and does not refund reserved creation costs;
partitioned empty producers must agree their source precision before publication.


Partition activation interventions compose with the existing controlled text
continuation, snapshot and fork machinery. Re-admission selects future global
operations and retains the inherited native context; it does not recompute earlier
KV or recurrent state. Every rank prepares the same global intent, while each
local admitted plan remains bound to its own loaded session. Child capture owners
use fresh run identities and forward epochs, and restore retains consumed capture,
projection, evidence and transport credits.

The shared partition observer validates and applies each active local projection,
including nonexporting replicas, then stages global outcomes and per-operation
before/after evidence until the ordinary forward commits. Inactive pipeline ranks
join only common preparation and completed outcome delivery. Direct ordinary
editing on a partition-bound capture owner is rejected: it cannot bypass the
prepaid operation authority or peer agreement. Missing invocation receipts fail
the attempt instead of becoming successful zero-valued edits.

Sparse partition capture now has ownership-aware receipt validation and bounded
assembly in the existing exchange protocol. This does not add a second completion
or inference owner: received host rows remain evidence pending the shared forward's
completion and commit. The live session now prepays sparse producers, pins each
actual invocation's receive extent, accumulates exact chunks and requires explicit
provider completion, including zero-row owners. Completed sparse fragments follow
the same staged delivery and final commit as dense captures. Shared local provider
callbacks now cover preparation, execution and failure before EP reverse exchange,
including idle owners, with nested residency adapters reporting one invocation.
The prepared gated/ReLU² EP tests verify rollback and no reverse exchange after
an agreed local failure. The shared partition observer now prepares routed source
bounds and uses its prepaid source/final votes through these callbacks, including
original/effective records, idle owners and nonexporting replicas. Abort and failed
work preserve cumulative credits and prevent publication. Retained architecture placement now feeds these callbacks through actual prepared
gated and ReLU² executors. The neutral tests cover prefill and two cached decodes,
idle received owners, all three residencies, TP/EP/PP ownership and source/collector
failure rollback. Loaded MLX now admits sparse capture through this same observer
when its exact native invocation group and provider hooks are available. The
packed-Qwen Ring matrix verifies full/strided original and effective values against
an ordinary native reference through prefill, two cached decodes, snapshot replay,
sibling isolation and skipped child capture. Restore preserves cumulative credits
and fresh branch identity. The shared operation authority now also carries sparse
edits through the provider scope. Operation-only plans activate these callbacks
without requiring sparse captures. Re-admitted future child masks start from the
same prepared boundary, receive fresh identity, preserve parent and sibling state,
and consume cumulative credits on every replay. Partial chunks or peer failure
prevent a committed edit outcome. Explicit tensor/expert stage groups now use
bounded native neighbor agreement; inactive pipeline stages do not enter those
observation callbacks. Disconnected native groups still need coordinated relays.

The released sparse score consumer also runs effective head/gain queries and
all-layer signed projections before controlled trials. Its combined LFM2-8B-A1B
Metal run passes seventy ordinary/controlled, cached-restore and sibling/parent
comparisons across baseline, deletion, keep-only, overlay and restoration. Parameter
query work remains cumulative across these phases. Fixed baseline target/alternative
IDs are scored along each trial's own unforced trajectory; branches within a trial
start at equivalent prepared boundaries. See the released sparse score results in
[component-analysis.md](component-validation.md#released-sparse-selected-token-and-token-difference-reconstruction).


Mixed dense/routed V3 pipeline component capture now follows the same controlled
partition driver. The effective input embedding travels in the typed boundary;
normalized Q/KV latents retain their invocation ownership, and effective current
KV values enter the compressed cache. Native two-process Ring tests cover
resident, host-layerwise and disk-streamed captures/masks with snapshot replay
and sibling isolation. All-dense V3 additionally passes native TP, PP and combined
TP/PP in all three residencies, including atomic multi-weight overlays, rejected
publication rollback and restoration, for F32 SafeTensors/GGUF and SafeTensors
load-time affine 4-bit/group-32. Corrected token-indexed query rotation is
shared by controlled and ordinary execution; MLA equation revision 2 rejects older
V3 cache fingerprints. Mixed V3 TP/PP/EP block instrumentation passes native F32
SafeTensors/GGUF and load-time affine 4-bit/group-32 acceptance in all three
ordinary residencies. Prediction internals use the shared scoped collector and
have separate encoded placement matrices; see the
[component coverage](component-analysis.md#execution-coverage).


The neutral additive-write capture path shares ordinary partition source completion,
hook votes and final commit with controlled sessions. It exports all admitted raw
terms before complete-value transformation, and publishes nothing on rollback.
Restoring a checkpoint retains assembly, export and transport charges. Sum-term
intervention projection binds the offset owner before native work; inactive Add
peers still participate in source validation and completion. Live four-rank neutral
coverage includes an idle pipeline rank, generated sources, failure/abort, and restore
without refunds. Mixed V3 TP execution hooks pass native F32 SafeTensors/GGUF and
load-time affine acceptance, including additive-only summary/histogram capture.

V3 prediction component primitives now share fusion/head execution with ordinary
inference. Their `prediction.readout.residual` intervention occurs before head
normalization and affects the returned prediction hidden state and draft logits.
The older `mtp.{depth}.output` remains after the head and retains that timing.
Neutral ordinary/provider/TP traversal tests verify selected-depth cache advancement
and fresh sibling replay. Public phase/frontier admission and delivery are verified
through `prepare_speculative_activations`, including the native distributed
matrices recorded in the component-analysis guide.

Mixed V3 independently cached F32 SafeTensors/GGUF and load-time affine execution
passes all 63 CPU Ring cases across the three ordinary weight residencies and all
TP/PP/EP combinations. Queries, coordinated edits, snapshots, siblings and replay
use the same selected bank identities. Idle EP owners remain valid participants;
every routed PP stage must demonstrate actual cache activity.

V3 component discovery now separates prediction depths and their score heads from
primary target attribution. Typed prediction operations can accept internal
observers during extension prefill, proposal and retained-input replay. Their
caller still owns prediction-lane restoration after failed work; restoring the
ordinary target state alone does not restore the separately owned prediction lane.
Public speculative component capture composes explicit phase attribution, bounded
delivery and cumulative budgets with the shared controlled driver. Loaded support
reflects the actual selected hooks; target-only support cannot authorize a separate
prediction scope.


Internal embedded observers now receive explicit target-prefill, prediction-prefill,
proposal-depth, verification and replay phases from the shared strategy. Successful
invocation completion stages tentative evidence; it does not make a proposal or
verification row committed output. Physical sequence width is supplied separately
from output prediction ordinals. Failures use existing target/prediction cache
recovery and cannot refund observer authority. The neutral tests cover prefill and
both replay delivery failures; native completion resources retain their original
owner. Public controlled capture admission and delivery compose these hooks with
cumulative reservations and exact frontier attribution.


The capture owner now supports explicitly admitted physical invocation geometry
independently of prediction coordinates. Snapshots retain immutable bounds and
schedules; restore clears delivered geometry and requires the next invocation to
supply its own exact shape. It preserves all spent capture/intervention resources.
Child re-admission carries bounds and inherited usage into the fresh session. These
contracts supply physical geometry to the shared controlled driver, which adds
speculative request/frontier attribution and public internal-capture delivery.

Speculative internal capture now has scheduler-owned request/prefix origins and
independent physical invocation geometry. Controlled step delivery has a separate
`activations` array, charged to ordinary trace transport, while `captures` retains
its sampler-row meaning. Invocation IDs and cumulative capture allowances are not
rewound by collector restore. Origin scopes are cleared after each operation,
including failure and unwind; optimistic prefix digests are never substituted for
a committed frontier. Public admission/installation now uses the separate loaded activation authority.
The neutral partition observer now accepts the same independently admitted
geometry and scope masks. Its receipts bind exact invocation axes and monotone
forward epochs; failed final commit publishes no payload, and replay does not
refund geometry, source, edit, receipt or assembly costs. Live multi-rank tests
cover actual edits, original/effective evidence and restoration with changing
physical widths at one prediction coordinate. Public distributed admission uses
the selected prediction layout and retained native transport; family coverage is
reported by loaded speculative discovery. The speculative provider retains the ordinary partition observer across each phase, including sparse
callbacks and final commit; its existing restore owner preserves spent budgets.


`ControlledSpeculativeOptions.activations` installs one immutable combined
capture/intervention authority before the first forward. Its collector and spent
budgets belong to the borrowed execution scope. Returning the executor restores
previous observer ownership; a later generation call receives no inherited edits
or observation allowance. Internal records remain distinct from sampler rows.

`take_activation_evidence` is available after failed actions and cancellation for
already charged host records, with run/epoch attribution and the shared transport
budget. It does not make a failed session resumable. Capture selections and
allowances remain fixed for the scope; future internal interventions can change
through `readmit_activation_interventions` using plans prepared before borrowing
the controller. This validates actual loaded discovery and requires a drained
canonical boundary. It does not recompute earlier target or prediction state.
Embedded sequential V3 snapshots copy the complete
target cache, canonical prediction state, separate prediction seed and capture
tensor. Reusable restore and serial sibling exchange preserve their independent
state while the collector keeps one cumulative budget and invocation sequence.
CPU and Metal acceptance covers resident, host-layerwise and disk-streamed weights.
Snapshots also retain exact internal intervention authority. Restore and serial
branch exchange prepare authority first and commit it only after all native copies
succeed. Removing edits does not remove the collector or reset its budget. Native
CPU/Metal acceptance covers adding and removing masks in a child, replaying that
child, stale-authority rejection and preservation of the parent and another sibling.
V4 sequential and DSpark now use complete pooling-state copies through the same
driver. Native CPU/Metal tests compare exact sampler captures and seven generated
tokens after a 255-token prefix, repeated restore and interleaved siblings across
resident/host/disk weights. Separate V4 component and distributed snapshot matrices
are recorded in the component-analysis guide. Capability checks require complete
native copy bounds for the selected state profile.

Continuous observed speculative generation is implemented by advancing this same
controlled session. The callback receives each `ControlledSpeculativeStep` and may
cancel with `ControlFlow::Break`; no second inference or sampling engine is used.

Observed prediction prefill, proposal and replay now share the ordinary session's
preparation, checkpoint, completion and commit protocol. Multiple prefill depths
belong to one invocation. Preparation failures perform no prediction work;
execution, native completion or delivery failures discard the invocation and
restore target state when the outcome is determinate. An indeterminate final
commit retains the existing fence against further mutation. Prediction-lane
rollback remains with the existing architecture/scheduler owner. Neither rollback
nor saved-state restoration refunds capture allowances or rewinds forward epochs.
Native auxiliary completion includes all retained prediction state, proposal
outputs and deferred token assertions, and preserves original failure sources.
This shared driver serves continuous and controlled execution, including selected
distributed collectors. Actual family and format coverage is recorded in the
component-analysis guide.


Loaded V3 internal activation plans now use the retained partition collector through
the same speculative controller. Independent prediction scopes carry their own TP
columns and non-tensor replicas; target input/output ownership remains unchanged.
All 21 F32 CPU Ring parallel/residency cases reproduce local phase records and causal
channel/routed-unit edits over prefill and cached generation. Prepared resident
providers finish their source/failure protocol before model reductions. Shared
embedding storage is present on every prediction replica, preventing different
pipeline ranks from choosing different speculative decisions. This does not change
snapshot budget semantics. All 21 F32 cases now also verify coordinated target and
prediction parameter edits, late peer rollback, active-overlay snapshots, repeated
restore, isolated sibling exchange and restoration after removal. Snapshot copying
and capture accounting remain cumulative. Additional encoding and family acceptance is recorded in
`doc/component-analysis.md` and its implementation checklist.


Internal speculative activation plans now use schema 2. The admitted scope binds
phase applicability: `PredictionContext` runs only during prediction prefill and
replay, `FusedProposal` only during fused proposal execution, and sequential
`Prediction { depth }` only during its own proposal depth or prediction
prefill/replay. Changing this binding changes admitted identity and requires a
new plan. Physical proposal rows remain distinct from prediction depth and
committed token positions. Neutral conformance verifies causal phase-selective
edits, original/effective evidence and replay without refunded usage. Complete
DSpark context prefill reports every accepted target row, including a one-token
prefix; sequential prediction reports the shifted hidden/next-token pairs instead.
The architecture extension supplies this physical row count to the shared driver.
DSpark public CPU/Metal capture, masks, controlled parity, snapshots and sibling
replay pass across resident, host-layerwise and disk-streamed execution. Broader
parameter, equation and distributed acceptance is recorded separately in the
component-analysis guide.

Paged sliding attention must preserve the sealed blocks needed by an outstanding
verification checkpoint. Advancing a speculative suffix can move the visible
window beyond those blocks; rewinding only the offset and mutable tail is
insufficient. Native checkpoint guards retain that catalog history until release,
without refunding charges or pinning all blocks to device memory. Ordinary
retention limits still apply, and incomplete visible history fails explicitly.

Prepared composite target sessions use the same native control-state adapter as
text targets. Snapshots retain the complete neutral runtime metadata, prepared
prompt identity and native hybrid state; bounded estimates, all-rank preparation
agreement, copy completion, overlay epochs and restore validation remain in the
shared session driver. Media inputs do not create a second generation engine.
Selected speculative execution continues to use its dedicated complete-lane
control contract rather than an ordinary target-only state snapshot.

Qwen3-Next and Qwen3.5 prepared prediction use these same observed controlled
operations for prefill, proposal and accepted-token replay. Their prediction lanes
remain separate from the target cache. Native text TP/PP tests cover causal unit
and channel masks, shared fusion input evidence, coordinated parameter overlays,
restore and sibling isolation. Resident routed banks finish their observation
scope before a tensor reduction. Every prediction replica binds its declared
shared embedding/head storage; that storage does not change ordinary target
input/output ownership. See the component-analysis guide for the residency and
conditional-media acceptance matrix.

## Caller-owned and speculative startup preparation

`LoadedModel::finish_text_preparation(stage, local_result, map_backend)` carries a
fallible caller-owned result through the loaded session's bounded readiness
agreement. The cancellable variant accepts `Result<Option<T>, E>`; `Ok(None)`
represents cancellation, and a peer failure takes precedence. Each participant
must enter the same stages in the same order even if local preparation fails.
For example, retain the result of `prepare_chat_multimodal_input`, agree it at
`TextPreparationStage::Prompt`, then construct `PreparedBackendInput`. A failure
returned before the agreement cannot be recovered retrospectively. These helpers
preserve local error types and expose peer/transport failures through the supplied
neutral error mapping. Usage remains cumulative across retries and restores.

Speculative ordinary and controlled entry points share batch host/prompt
preparation. Every host lane is validated before any native prompt is created;
readiness is agreed once per batch and stage. Controlled capture validation also
participates. Embedded native sampler/cache setup and initial lane delivery use
the same core result policy through retained transport while native authority is
held. An error preserves its original source, but does not establish completion
or permit reuse of a fenced broad native operation. Local artifact inspection,
immutable admission and media preparation remain local unless the caller invokes
an explicit agreement helper.

Ordinary and controlled token delivery agrees local cancellation and bounded
record failures before another native prediction. Controlled lifecycle and other
boundary records also agree delivery; a callback cancellation there is retained
for the next advancement, without changing the completed prediction. A completed
ordinary record cannot rewrite the existing termination reason, but a local
record-budget error is returned on its owner and as a typed peer rejection on
other ranks. Distributed participants must invoke collective control operations
in the same order. Local immutable admission and discovery do not add agreements.

Shared speculative scheduler creation also agrees its local policy and completion
support before any request is submitted. Controlled visitor setup agrees lane
geometry and collector installation first. These stages use the executor's
retained transport, including through architecture erasure; callers using the
neutral runtime directly receive the same ordering as facade consumers.

Speculative advancement coordinates per-request cancellation and exact completion
facts before selecting an action. A ready rank waits for peers; cancelling one
batch lane does not cancel unrelated lanes. Prefill and verified-token publication
agree their result and cancellation before installing a terminal lifecycle.
Controlled `cancel` drains the same action and bounded-record driver as `step`,
including an already submitted verification. It does not bypass record accounting.
A local record error or a controlling closure's returned error is agreed before
peers advance again. Already agreed failed steps do not submit another exit vote.
These agreements coordinate progress and failure; native completion ownership
remains with the retained completion resources. Distributed callers still invoke
snapshot, exchange and prospective edit operations in the same order.

### Prepared-request memory forecasts

`LoadedModel::forecast_prepared_generation` forecasts the same prepared request
used by ordinary or controlled startup. `forecast_observed_generation` accepts
`PreparedObservedGeneration`, verifies its session identity and uses its resolved
settings and capture/intervention contract. Trace-only preparations retain ordinary
chunking and final-row projection when the executor supports them. Instrumented
prefill uses a full pass and all-row logits. Admitted per-step/cumulative capture
limits bound logical transform storage, intervention execution/evidence, and
retained host records; immutable intervention payloads are included separately.
The host envelope allows one record history and one compact JSON trace. MLX uses
per-step native storage because transformations complete synchronously; other
backends default to the cumulative retained limit. These are logical estimates
with separate allocator/graph allowances, not physical process guarantees.
Independent autoregressive speculative forecasts include target/draft transaction
resources where both ordinary workspaces are covered. Embedded startup composes
ordinary prediction modules, state, retained target features and transaction copies;
it is finite only when the selected mechanisms are covered. Feature-conditioned
external assistants retain unknown components. Forecasts do not consume preparation, state,
callbacks, capture budgets or submission authority. Fresh-request forecasts require
fresh/reset state; speculative startup describes a fresh isolated lane. Active
sessions use the continuation operation below.
See [generation memory](generation-memory.md).


## Forecasting further tokens

After an ordinary controlled step, call
`session.forecast_remaining_generation(additional_tokens, &options)` to observe
cache position, retained state and a decode-only memory envelope. The exclusive
session owner supplies a quiescent boundary; forecasts consume no predictions,
random draws, snapshots, captures or transport budget. Restore and branch exchange
change which installed state is observed. Snapshot/branch reservations remain
conservative retained upper costs, not resident-memory credits. Unknown native or
host state retains an unknown upper end. The requested horizon is hypothetical
and never extends the run's configured token limit.

Ordinary sessions reject initial, terminal and unsettled boundaries. The object-safe
`ControlledSpeculativeSession` exposes the same method name for settled external
autoregressive and embedded-prediction lanes. It returns a `SpeculativeContinuationForecast` after prefill
or canonical verification/commit with no retained proposals, optimistic branch or
pending verification. It never polls or settles work to make a boundary eligible.
Before prefill, after termination/cancellation, on failed sessions, and during an
uncommitted transaction it returns `UnsupportedContinuation` without changing state.
Feature-conditioned external assistants remain unsupported. Embedded forecasts
require installed prediction state and retained-feature observations; missing
workspace mechanisms preserve an unknown upper end. Instrumentation
or custom sampler/semantic growth without a complete bound preserves known native
facts but leaves retention unbounded.

Speculative outlooks observe the target frontier and either the independent draft
frontier or the installed prediction-layer frontiers, native state capacity,
retained target features and assistant seed, sampling/RNG/semantic storage, and
live user snapshot/branch reservations. Shared embedded parameters remain in the
target residency report, including their cached conversions. Completed loading and prefill are excluded. Further drafting,
verification and commit keep the configured proposal/lookahead ceilings. Request a
new outlook after restore, branch exchange, advancement, or a horizon change.
For complete accounting and raw iterator settlement, see
[mid-session memory forecasts](generation-memory.md#mid-session-continuation-forecasts).

Resident parameter conversion retention is enforced by the shared native parameter
owners used by ordinary and controlled execution. The initial managed 256 MiB
budget covers retained plus reserved conversion payload across all permanent units
and embedded prediction owners. A denied reservation uses the unchanged temporary
cast path. Ordinary reset and controlled state advancement do not reset this budget
or release admitted conversions; parameter replacement revokes obsolete claims.
Forecasts and residency queries observe existing claims without populating caches.

Host-layerwise, disk-streamed and explicit device-ceiling executions retain no
optional conversions. Multi-rank native TP/PP/EP/DP construction reports typed
unsupported retention eligibility and effective disabled policy, since the local
controller cannot coordinate reservations across processes. This rejects retention
only; ordinary and controlled inference remain available through temporary casts.
Explicit settled-boundary trimming and its controlled-session operation contract
remain later retention-plan work.


### Conversion retention observations

Parameter-conversion retention is selected at load time through the neutral
execution plan or normalized load request. Eligible executions use a managed
256 MiB payload ceiling when no override is supplied; disabled, bounded and
explicit unlimited requests are supported. It is independent of allocator caching
and covers retained payload plus reservations, not total memory or temporary casts.
Controlled and uninterrupted inference use the same immutable budget and owners;
ordinary reset preserves admitted conversions.

`ControlledGenerationSession::parameter_conversion_retention` observes the same
budget as the loaded model. Controlled speculative reporting labels target and
external-drafter scopes; embedded prediction remains in the target group. Queries
never settle pending submissions, allocate execution resources, populate caches,
change tokens or RNG, or consume capture, snapshot, transport or copy budgets.
Missing backend observations remain unsupported or unavailable rather than zero.
There is no live limit setter or trimming operation in this phase.
