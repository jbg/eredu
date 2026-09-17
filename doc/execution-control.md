# Completed-token execution control

The ordinary text facade supports completed-token stepping, pause/resume, canonical
token forcing, temperature changes, explicit reseeding, reusable native snapshots,
restoration and isolated serial branches. Complete snapshot support requires known
costs for every state owner. Currently this includes the forbidden-tool constraint
mode; active and automatic llguidance grammar modes explicitly reject snapshots.
That public snapshot/restore/fork capability was already absent in ordinary
execution at tracked HEAD c513e17c; it is not a new managed-parity restriction or
an architectural limitation. These functional capabilities and logical copy/capture budgets do not imply a
finite managed-memory bound for every operation. See the maintained
[current managed-admission matrix](bounded-inference.md#current-integration-status)
for supported paths and request-specific workspace bounds.

`LoadedModel::start_managed_plain_text` exposes move-only controlled advancement
of the same driver used by `generate_managed_plain_text`. Both authenticate a
compiled tokenizer source, honor the same settings and cancellation, lend
`GenerationPlainTextEvent` fragments, and return retained `GenerationPlainTextOutput`
without copying its funded text. The public run/manual parity, pre-start cancellation,
source mismatch and output-retirement checks pass with neutral mock backends.
Native requests still reject missing ownership/arena bounds. Managed plain-text
snapshot, restore and fork now pass with selected dense and mixed recurrent/KV
resident fixtures, both before prefill and after the first committed token.
Fresh destinations require independent admission; saved state stays immutable,
and copy use remains cumulative across cancellation, refusal, restore and fork.
These cases do not establish managed chat, grammar, media, capture, speculative
or distributed snapshot support. See the detailed fixture scope and recorded
commands in [the public native validation record](validation/bounded-public-resident-chat-2026-09-16.json).

Selected managed external ordered-assistant generation now matches ordinary
and controlled execution, both on the target stream and with an explicit
same-device draft stream. The split fixture uses nonzero-temperature sampling and
uneven prefill chunks. The selected controlled split fixture also passes
snapshot/restore/fork/exchange replay and cumulative limits. Selected native
cross-device consumers now pass in both CPU/GPU directions, including saved-state
replay at their recorded scopes. Reproducible results are in
[the native integration record](validation/bounded-media-and-submission-2026-09-16.json),
under indexed_assistant_and_selected_external_placement and
publication_cpu_host_replay_integrated.

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

Controlled speculative capture carries an immutable shared frame through the
existing activation envelope. Draining or dropping a temporary output buffer
does not retire a frame still held by the caller; its capture account outlives
the final alias. Ordinary serialized capture records retain their existing wire
format. Invocation identity includes the admitted request, phase, absolute
prefill span, logical prediction and monotone occurrence; restoring mutable
model state must not refund cumulative capture usage.

Managed Embedded activation setup validates the retained architecture catalog,
selected hooks, artifact, execution, overlay and session without building a
replacement discovery object. The selected Qwen fixture now passes activation
capture, reductions, internal edits, snapshot/restore/fork and cumulative spending.
Qwen, Nemotron, Inkling, V3, V4 and DSpark pass their selected complete Embedded
state replay. V4 and DSpark also retire all 180 accounts per finite 12 GiB fixture.
The selected residency, controller and saved-state combinations are recorded in the
[current integration record](bounded-inference.md#current-integration-status).
Every request still requires its exact retained source and known workspace;
unsupported source/callback bounds receive typed refusal before execution.

The shared speculative sequence now distinguishes ordinary and retained storage.
Retained copies use the existing paid provider and preserve all sequence state;
terminal output moves its immutable token owner without an unowned vector copy.
The neutral copy/termination/partial-iteration fixture passes. Public managed
speculation now combines these owners with admitted native sampling,
request/statistics and controlled snapshot/view storage. Selected ordinary,
managed and controlled speculative comparisons and saved-state replay pass at
the scopes recorded in the current integration record; requests with unknown
required sources still refuse before that work.

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
history buffers and caches, so neither satisfies snapshot admission. The workspace
now selects checksum-pinned local llguidance 1.8.0 and derivre 0.3.12 sources.
Their local table changes reject new lexer states before insertion at the state
limit, including construction and optional warming, and provide a fixed-capacity
hash-cons primitive. Invalid limits and construction failures retain typed
library causes through matcher copies and later error returns. They do not bound all
derivative/parser allocations or independent copies, and do not change the
ordinary or controlled grammar admission gate. Deep copying now avoids a redundant
temporary parser-state copy, while preserving history, caches, remaining token
allowance and errors; a complete independent-copy byte bound is still required.
Declaring tools with `ToolChoice::None`
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

## Bounded prefill contract migration

The following prerequisite and validation notes record successive implementation
milestones. Statements that a later adapter is still missing describe that
milestone. Current managed support is summarized in the
[managed-admission matrix](bounded-inference.md#current-integration-status)
and the current fresh-resume section below; generic whole-facade funded restore
remains distinct from low-level native saved-source resume.


Runtime `PrefillDriver` supplies explicit `step` advancement and an uninterrupted
`run` wrapper over the same implementation. Completion gates chunk advancement;
cancellation prevents later submissions, and polling errors preserve unresolved
native ownership. Runtime reservations bind selected execution and geometry,
charge concurrent requests in one shared domain, and preserve consumed start
authority across cloned retention. A snapshot must not clone that authority into
another runnable prefill or refund retained bytes and cumulative budgets.
The scheduler now serves native ordinary token-only prefill with a default limit
of 512 positions. Facade configuration, controlled chunk advancement,
media and speculative integration remain in progress. Existing
controlled-session capabilities are not evidence of bounded-prefill support.
See [the remaining family/path and validation gates](bounded-inference.md).

Cold equation workspace tracing now composes with this same admission policy.
Metadata handles retain backing-allocation identity across clones and state
views. A completed span may start a fresh transient ledger only after its retained
roots have been assigned to persistent residency. `with_equation_workspace`
does not price snapshots, observers, speculative retention or sampling implicitly;
their separate bounds remain required. The new decoder conformance quotes cover
both prefill and multiple cached decodes, with unknown native primitives rejecting
strict admission at every candidate chunk size. These checks do not yet establish
production native budget support for controlled sessions.

Cold equation execution can now use the existing parallel layered traversal via
the metadata submission adapter. Forked inspection lanes share their allocation
charges; metadata completion neither refunds those charges nor supplies missing
native bounds. Exact rank ownership and uneven vocabulary widths are retained,
and projection observers run only after ownership validation. The conformance
trace exercises state-only prefixes, final-position generation and full-sequence
readout through the same traversal, with an existing prefix and three cached
decodes. This is coverage of inspection and shared readout geometry; native
collective costs and controlled production reservation wiring remain outstanding.

The same cold trace now retains routed selection controls and their optional
original decision, grouped-unit interventions, and hyper-head coefficient hooks.
It validates a routing control before recording selector work and validates a
unit replacement before the down projection. A failing observer leaves preceding
charges intact and prevents downstream operations. Multi-stream residual state
retains all four collapse outputs and rejects cross-trace mixing. These are
neutral inspection contracts; they neither execute numerical interventions nor
establish controlled native memory-budget support while required mechanism facts
and full-request composition are still missing.

Managed-host workspace is now explicit alongside tensor-buffer workspace in a
cold equation trace. Both domains must be bounded before ordinary or controlled
execution can reserve the trace's managed total. A tensor alias does not imply
zero host staging, and metadata completion does not refund host charges. A
diagnostic admission obtained without strict estimation cannot bypass this check
at the reservation pool. Current Metal facts cover tensor buffers; host staging
remains unknown, so they do not yet authorize native request budgets. Driver and
JIT bookkeeping, allocator caches and unrelated process memory remain outside
the managed-memory guarantee.

The exact-concatenation metadata state now traverses the same layered hybrid
equations with attention, convolution and recurrent components. A failed append
preserves its previous frontier and retained roots; preceding successful trace
operations stay charged. Segment reset and cheap metadata clones do not refund
the ledger, and a clone is not a bound for native isolated snapshot copies.
Sliding views retain the complete appended backing allocation in the persistent
root report. Blockwise attention likewise retains every recorded stage through
the containing span, including online normalization and value accumulators.
Its logical reconstruction-byte telemetry cannot authorize a request. These
inspection checks apply to shared execution contracts; they do not complete
controlled native budget wiring or price other selected cache mechanisms.

Resident compressed-state tracing now retains actual capacity increments and
the copy strategy across append, checkpoint restore and isolated snapshots.
Cheap checkpoint handles preserve allocation identities. Restoring state traces
the selected backing/logical copies, while isolated snapshots trace compact
materialization and later capacity regrowth. Previous branch charges are never
refunded by either operation. The shared resident metadata state covers all six
typed access profiles, including heterogeneous fixed/compressed models; this
does not itself price native snapshot transfer buffers, paged state or a complete
controlled request. Missing tensor or host facts remain unknown.

The native compressed-cache comparison also exposed a restore defect in the
underlying deep-copy wrapper: a strided logical view with spare capacity copied
padding into later batches. Deep copy now compacts non-row-contiguous input on
a reused default CPU stream before copying its logical data. This applies to
ordinary and controlled snapshot/fork/restore callers through the same native
primitive. Its possible
compaction remains part of the selected copy workspace, not an implicit free
operation or a budget refund. Selected Metal tensor facts now include possible
compaction and independent destination storage, as well as full-capacity update
copies and update conversion/reshape buffers. These copy operations have a
separate derived zero host-payload bound because their CPU compaction and copy
both use the shared allocator. This does not yet authorize controlled snapshot
or request admission: enclosing resource lifetimes still require reservations.

The selected Metal facts now include common tensor and activation equations in
addition to the recurrent scan. Their charges preserve possible copies when a
strided integer selection removes an axis, and include reduced-precision casts
and internal buffers through completion. These facts feed the shared trace for
either execution mode; they do not independently authorize a request while other
required native operations remain unpriced.

Convolution now has geometry-selected native tensor bounds, including actual
unfolding, split-K and Winograd buffers. Its disjoint host-payload bound is zero
by source derivation because those payloads all use the shared Metal allocator;
other operations' missing host facts remain unknown. Ordinary and controlled
execution use the same corrected transposed-output crop and F32 Winograd
computation with restored output dtype. Cropped backing buffers and dtype
conversion temporaries remain charged through completion. These operator facts
do not complete preparation, snapshot or full-request reservation wiring.

Audited dense, basic, indexing, padding, normalization, reduction and gated-delta
operators also have explicit host-payload facts. Constant and edge padding charge
native output and copy lifetimes without duplicating their unified-memory
payload in the host domain. Rotary construction prices its
Rust frequency vectors separately from native copies. Input-score attention
with long key rows prices the host mask for one completed key tile; it does not
retain a prompt-sized host mask. These facts apply equally to ordinary and
controlled equation traces. Unknown grouped, routing or transport mechanisms
still prevent complete managed quotes, and production reservation wiring is
unfinished.

Reduction and normalization facts now use the same completed-span accounting:
partial accumulators, casts and all normalization intermediates remain charged
until completion, including grouped/gated and learned-offset policies. Their
operator-level native validation does not establish production request-budget
support or price snapshots and other separately retained resources.

Packed projection and lookup facts use the same operator contracts in ordinary
and controlled execution. They account for quantized split-K reductions, direct
GGML kernels, selected-row decoding, and FP8 activation/scale transformation.
Selecting one hidden position before readout changes these shapes before the
quote is calculated. FP8 E8M0 host lookup storage has its own explicit charge.
These facts do not authorize an unpriced intervention factory, snapshot,
transport or request-preparation step.

Whole-request equation inspection uses the same prefill scheduler as explicit
and uninterrupted advancement. It covers every prompt chunk and conservatively
one decode invocation per reserved output position, even when the first token
is sampled from prefill. Decode pricing includes its score row after state-only
prompt processing. Completed spans contribute a peak, while simultaneous tensor
and host storage are summed within each span. Unknown coverage and untraced
resource bounds cannot disappear during aggregation. Context and exact request
identity reject before equation inspection; a successful complete quote still
needs the ordinary shared-pool reservation and native lifetime retention.

Dense-product accounting follows the same rule for ordinary and controlled
equations. Matmul, fused-bias tensor linear operations, constructed dense
projections and tied dense readout charge their possible casts, contiguous copies,
split-K accumulators and output buffers through completion. Selecting the final
hidden position changes the projection geometry before it is quoted. Remaining
grouped, attention, collective and separately retained-resource gaps still
prevent these operator facts from authorizing a complete production request.

Gather and dense embedding quotes include the domain-check arrays transferred to
the native completion owner, their reduction scratch, safe indices and sentinel
masks. Validation failure still settles those resources before release. These
charges apply equally to controlled and uninterrupted submissions. Floating
softmax similarly keeps its complete selected kernel/equation workspace charged
through completion; it does not assume that temporary donation will occur.

Attention quotes apply the same completed-span accounting to ordinary and
controlled equations. Fused eligibility, masks, sinks, grouped heads, score caps
and sliding chunks select the same native allocations in both paths. Two-pass
fused scratch uses a process-retained setting shared by cold inspection and the
native kernel; environment edits cannot invalidate an issued quote. These are
operator facts. Large input-score rows use the same blockwise accumulator for
ordinary and controlled callers. Each step now establishes both result readiness
and successful native ownership retirement before subsequent scratch allocation;
failed or unobservable work remains retained by existing recovery. The memory
quote therefore bounds a completed block plus retained state/query outputs,
including final casting and concatenation. Other unpriced mechanisms, retained
resources and production reservation wiring still prevent complete request-budget
coverage.

Rotary application has the same workspace facts in controlled and uninterrupted
equations, including first-use frequency construction, per-batch native outputs,
explicit rounded products and caller-supplied position embeddings. Internal
frequency state is conservatively included even when already evaluated; it is
not inferred to disappear on snapshot or restoration. Full request accounting
must still retain the actual operator state and other persistent resources.

Ordinary prefill and decode now carry final-position output demand through their
shared session strategy, so controlled and uninterrupted callers reaching that
strategy use the same projection selection. Direct scoring and prediction-target
operations retain sequence scores. Observers default to sequence readout, with
parallel agreement ensuring all ranks use compatible shapes. This is readout
parity; chunk-level controlled advancement and working-memory
admission remain pending production integration as described above.

The completed-span session operation now handles a state-only prefix without
score observation or publication and forces exact state completion before
returning. It uses the ordinary transaction and rollback engine, including
result-bearing input admission across ranks. The reserved session adapter connects
it to the shared prefill driver. Direct and composite native token-only adapters
now select this path collectively; media and sequence observers retain their
whole-request protocol while semantic span and attribution support is completed.


Reserved text-session conformance now exercises `SessionPrefill` through both
`PrefillDriver::step` and `run`, with identical state and three subsequent cached
decodes. It covers uneven spans, cached prefixes, resident/host-layerwise/disk
weights, cancellation and typed input-preparation failure. Cancellation is agreed
across active ranks before another span can submit; repeated terminal advancement
performs no collective. These are shared-driver checks, not evidence that facade
controlled sessions already expose chunk-level advancement. Native guards retain
request charges across unobservable completion independently of scheduler lifetime.

Reserved prefill now attaches its charge to the retained state before preparing
the first span. Dropping a completed/cancelled driver cannot refund that charge
while its cache or a checkpoint survives. Shared checkpoint restoration merges
current and imported charges before native copying; an older snapshot or a failed
restore cannot discard newer accounting. Native state snapshots/forks inherit
the charge, and ordinary control slots exchange it with their state. Full state
replacement releases only that owner's reference. Copying retention does not
reserve the additional physical storage of an independent branch; shared copy
pricing and production preflight remain required.

Reserved session state now includes its exact request and logical decoder
frontier. Ordinary and prepared-input execution agree batch, span length, phase,
readout demand and state position before checkpointing or model work. Decode
cannot exceed the reserved output allowance. The completed span must reach the
expected position, including a logical position on stateless partitions.
Restoring a checkpoint restores this frontier so replay uses the same quoted
position range; it never refunds backing charges or existing cumulative capture,
transport and copy budgets. Restoring a snapshot without an admission preserves
an installed admission and therefore cannot silently resume unbudgeted execution.

The common transaction retains a native reservation guard through publication
and requires completion for reserved calls. Settlement failure fences the session
and leaves unresolved ownership with native recovery. Sequence execution now
uses this same outer lifecycle. Auxiliary proposal equations and retained capture
resources still require their separate complete quotes; decoder geometry alone
does not prove their bound. These mechanisms are shared by controlled and ordinary
callers, but production facade preflight and chunk-level controlled advancement
are still outstanding.

Equation quotes now include the overlap between opening state and replacement
state through completion. Unchanged state counts at its full backing capacity;
displaced opening buffers remain transient until that span completes. A narrow
logical view cannot refund its larger backing allocation, and an unseeded trace
or unknown existing capacity cannot authorize inference. This accounting is
shared by ordinary and controlled request inspection.

The native resident compressed-cache adapter projects live, restored and compact
snapshot state directly into metadata. It includes independently copied capacity
buffers as well as logical views and retains the actual growth increment.
Projection itself performs no native evaluation or prefix replay. Unknown native
storage remains unknown. These facts support subsequent continuation quotes;
they do not reserve the independent copy operation, its additional branch
storage, or shared paged resources. Those still require common-pool admission.

Ordinary and hybrid resident-state projections now preserve actual absolute
frontiers, key-only storage, empty one-token-window history, fixed-role absence
and cross-layer aliases. A restored or forked state's live buffers supply the
inventory; prefix length alone is insufficient. The portable import validates
architecture geometry without evaluating or replaying native state, and lazy
fixed replacements leave strict estimates unknown. These projections do not
copy the source's submission authority or refund its retained reservation.
Exact dtype validation precedes conservative floating workspace conversion, so
restored BF16 state cannot silently satisfy a role that declares exact F32.


Loaded MLX executables now retain the exact architecture/source blueprint and
selected Metal allocation facts for equation inspection. A quote projects current
resident state without copying authority, advances only a metadata clone through
all prompt/decode spans, and checks its opening and closing frontiers. Shared
runtime inspection rejects active, indeterminate and fenced transactions. This
preserves controlled state and does not implicitly wait for lazy buffers to
become known. The replicated equation quote still needs enclosing preparation,
sampling, materialization and copy bounds plus shared-pool admission before it
can authorize a budgeted ordinary or controlled request.

Explicit no-budget requests and enforced requests share the same scheduler via
`InferenceRequest`; no-budget execution does not advertise memory coverage.
Cloning or converting authority cannot restart a prefill, and a session adapter
rejects replacement by a different request even when geometry matches.

Persistent driver state no longer carries an executor lifetime. Controlled
composition can create a short session adapter for each advance while retaining
the same request and outstanding completion; the family conformance fixture now
exercises that arrangement through cached decode. This does not yet expose
chunk-level advancement through the public controlled facade.

Ordinary and controlled commitment now pass their live cancellation token into
the same core generation machine and native prefill driver. A request cancelled
between completed text spans returns no output, performs no sampling or grammar
commit, and publishes the ordinary cancelled finish reason. No first-token time
is recorded without a token. Backend failures take precedence over simultaneous
cancellation; they are not converted into successful termination. MLX explicitly
settles its operation and token validation before returning the no-output outcome,
so successful cancellation retains only the completed prefix and permits safe
session reuse. A peer cancellation also sets each participant's local token.
The token remains outside snapshots and cannot be rewound by state restoration.
Each public controlled step still advances at most one committed prediction;
it may process multiple completed prefill spans internally. Media, sequence
observation and speculative prefill still require their remaining span integration.

The shared ordinary source gateway also accepts an already admitted request.
Its execution identity, geometry and cached frontier are checked before semantic
source construction, and the one-use start is consumed before that factory runs.
Rejected or unavailable ingress cannot silently resume through the unbudgeted
fallback. Retained state owns the same request after caller handles are dropped;
cached decode remains bounded by its original output allowance. MLX input clones
and borrowed views retain that request identity. This connects the existing
scheduler and completion retention to supplied admission for both invocation
styles, but public ordinary/controlled preparation still needs complete quoting
and reservation before native input allocation.

Ordinary iterators, borrowed controlled sessions and detached continuations now
share core prompt/sampler preparation. Both facade entry points pass owned host
token storage to that driver. Cold admission receives exact host capacity and
read-only controller/configuration inputs; the loaded session agrees its result
before creating native prompt or sampling payloads. Local causes survive the
agreement and peer rejection prevents later preparation and model work.

The same preparation owner stays with the machine after startup, through pending
completion and payload destruction. Detached children retain the original owner
after their driver or parent is dropped. This does not authorize another run or
pay for independently copied native state; snapshot composition must reserve
those resources separately. MLX retains supplied request authority during detached
prompt/sampler recovery, including unresolved native errors.

Public native startup still uses explicit unbudgeted policy unless an internal
request was already supplied. Complete native quoting, dynamic constraint/capture
storage, controlled delivery's retained prompt copies and speculative/media
preparation must be connected before this becomes full budget admission. The
speculative batch's all-lane readiness protocol remains intact and is not replaced
by per-lane ordinary startup votes.

Prepared source storage now has a shared physical-owner inventory. A restricted
target or prediction view retains the full source payload, and ordinary/controlled
branch aliases do not multiply that source's capacity. The inventory retains
owners while admission composes bounds; it is not itself a budget reservation or
a copy allowance. Original in-memory tensors and packed quantization overlays
both count, including spare capacity. GGUF reader buffers have a cold fixed
ceiling. Dynamic state copies, native payloads and returned output handles still
need their separately owned charges; source accounting does not complete public
native budget enforcement.

Native residency snapshots now retain physical array and immutable host-transfer
owners alongside exact source storage. Aliases do not multiply capacity, and an
inventory continues owning its payloads after a manager evicts its logical
entries. It is not a ledger pin or a new execution allowance. Cold inspection
does not poll or retire asynchronous work: any in-flight/failed transfer makes
the snapshot's bound unknown, as do lazy/unrecognized custom arrays. A snapshot stays unknown
even if some other operation subsequently establishes completion; admission must
obtain fresh evidence at its quiescent boundary. Full controlled admission still
requires state, copied resources, output retention and shared-pool composition.
The synchronous residency-transfer boundary now finalizes published array handles
before making their generation ready, so subsequent cold inspection can observe
the settled allocation without touching its completion event.

Cold target-parameter inspection now uses the shared session's resolved-boundary
guard and an immutable parameter traversal. It cannot publish replacements,
acquire streamed units or resolve pending completion. Current resident values and
stored reload overrides are visible after an ordinary parameter transaction;
retained inventories keep their prior physical owners across subsequent edits.
Inspection at an unresolved/fenced or active transaction fails before visiting
values. Separately retained banks, prediction state and copied/output resources
remain additional owners for complete budget composition.

The same session guard now protects a combined read of native mechanisms and
state. Target inventory joins actual weight residency, current parameters,
addressable-bank pools, native state arrays and every retained paging manager.
Cache blocks on host/device and buffered shards retain their physical owners;
duplicate handles and logical bank scopes cannot discount or multiply that
storage. Pending worker transitions remain unknown and are never completed by
inspection. This supplies target-side accounting evidence for ordinary and
controlled admission, while prediction/observer/output ownership and shared
request reservations remain separate requirements.

Retained-work evidence spans the cache workers' completion publication and final
local payload destruction. Completing native state arrays happens in execution,
not inspection. Native ownership facts now certify MLX host-transfer array aliases
with the same identity as their host buffer. Controlled admission can count that
shared allocation once; arbitrary custom backing and unfinished arrays still
remain unknown. Retaining or copying an inventory remains separate from granting
request authority or reserving future workspace.

Cold prediction-resource inspection also uses the resolved-session guard for both
replicated and composite execution. The architecture borrows actual modules and
immutable state prototypes, while MLX includes inactive replacement tensors and
their retained sources/managers. Parameter loans, loading, state cloning and
completion polling are not part of inspection. An unevaluated replacement makes
the bound unknown without evaluating it; an earlier inventory continues owning
its original backing after restoration or executable destruction. These owners
can be merged with the target inventory without double-counting shared buffers.
Active speculative lanes, observers, escaped outputs and shared request-budget
enforcement still require their enclosing ownership/reservation contracts.

An explicitly admitted text request now claims preparation once before native
prompt/sampler construction. Both uninterrupted and controlled startup use that
claim. Cloned preparation handles retain its charge but cannot repeat either
construction stage; a changed startup sampling configuration is rejected before
sampler work. A failed stage never restores unused startup authority. Prefill
requires the prepared request identity and successful prompt binding plus sampler
readiness, so an earlier retention clone cannot bypass a failed preparation.

Sampling snapshots carry their request charges, restoration preserves old and
imported charges, and pending-prompt copies preserve chunk policy and request
identity. A copied pending prompt does not mint another prefill start: an
independently runnable branch requires its own admitted authority. Sampled token
handles and their copies keep the shared charge after native completion,
generation teardown and session destruction. Releasing the session lease does
not release token-owned budget. Native copy/reseed recovery independently retains
these owners on failure. These contracts are prerequisites for complete budgeted
branch admission; they do not price all copies or supply full public request
budget enforcement by themselves.

### Preparation and shared-storage charge lifetime

Preparation validates the exact finite output allowance before claiming prompt
or sampler construction. Startup sampling configuration is retained with that
claim. Clones cannot initialize another sampler or bypass incomplete preparation
through direct prefill. Sampler restoration preserves newer charge owners;
pending prompt copies preserve request identity and chunk policy. Returned token
handles retain charges after the generation driver and session are dropped.

Shared physical-storage registrations compete with inference reservations for
one pool's remaining capacity. Copying a registration shares its original charge;
separate inventories sharing certified physical owners also count those owners
once. A registration remains charged during deferred native cleanup, and neither
restoration nor registration failure rewinds the pool's historical peak. Copies
that allocate new storage still need a separate complete copy allowance and
admission. Independent runnable prefill branches still need their own request
authority. Public budget composition for those paths remains required.

### Pooling and composite text workspace inspection

Pooling metadata preserves local keys and their persistence sentinel, incomplete
windows, pooled history and overlapping windows across uneven prefill and decode.
Its checkpoint shares metadata allocation identities; restoring requires matching
context, batch and exact declared geometry. This does not grant an independent
native-copy allowance or refund any cumulative budget.

Routed and composite text inspection advances a clone of the live state through
the same shared prefill/decode quotation driver. Composite spans use ordinary
prepared-input admission, and native state inspection remains fenced while a
session has unresolved work. These quote routes do not constitute budgeted native
execution or cover media preparation, active prediction lanes and independent
snapshot/copy allocations by themselves.

Pooling prompt-cache validation reads overlap views in logical order: its native
numeric diagnostic snapshots first copy evaluated strided storage into a compact
array. Reading shape-sized contiguous slices from overlap backing can otherwise
make a correct save/restore appear to change values. The production prompt-cache
serialization is unchanged. Empty pooled sequences are valid rotary inputs;
workspace inspection keeps their scalar and frequency work instead of rejecting
the zero sequence extent.

Cached Muse continuation preserves the absolute rotary position while its causal
mask uses the retained key suffix. Implicit masks are constructed per layer in
ordinary and partition execution. The numerical regression crosses a sliding
window with uneven multi-position appends followed by three cached decodes;
native inspection also resumes actual cached state with multi-position input.

Pooled attention and pooled-position workspace now include the selected native
gathers, score products, masks, normalization, reductions and partition buffers.
Holding a top-k result retains the complete sorted-index allocation; a small
logical selection does not release its backing charge. These operation bounds
feed the shared quotation path for ordinary and controlled sessions. They do
not by themselves complete request admission, prediction-lane accounting or
independent snapshot/copy allowances.

Relative-profile attention and multi-axis rotary now supply selected native
workspace facts to the same ordinary/controlled equation quote driver. Rotary
outputs retain their full backing allowances; Rust frequency payloads are priced
separately from native copies. Signed coordinate offsets saturate, and the largest
valid relative-query endpoint is accepted by the shared contiguous/paged kernel.
The resident Inkling text fixture has a finite initial and cached equation quote.
This does not supply complete controlled-request admission, media preparation,
prediction-lane accounting or independent snapshot/copy allowances.

Multi-stream equation bounds retain the collapsed value, pre/post coefficients
and stream-mixing matrix across each sublayer. Final-head coefficient construction
and its weighted sum remain separately priced around the ordinary observer
callback. Sinkhorn workspace follows the selected iteration count with checked
arithmetic. Empty spans preserve that callback and output dtype while bypassing
invalid native empty reductions. These are shared operator facts; complete
controlled admission, observer retention, prediction lanes and independent
snapshot/copy allowances still need their enclosing resource accounting.

Same-operation output aliases preserve one complete storage root across logical
views and retain possible input owners. Dropping another output does not release
the surviving view's charge. The native joint router uses this for selected and
always-on coefficients, which share one normalized buffer, and retains the full
partition-index backing behind selected IDs. These declarations feed the shared
ordinary/controlled quote path; they do not grant independent-copy authority or
complete routed-request admission by themselves.

Top-k router estimates cover supplied IDs and every routing intervention through
the authoritative neutral intervention driver. Original and effective decisions
retain their own complete index backing and any shared score/coefficient roots.
The estimate includes the successful predicate path, strided row replacement,
all score-stage adjustments, and a possible CPU tie partition. This same cold
operation fact is available to ordinary and controlled equation traces; observer
delivery, retained captures, independent copies and complete request admission
still require their enclosing resource accounting.

Packed expert operator bounds include ordinary and TP output construction,
including the replicated down-bias correction, and the separate activated-unit
observer boundary. Empty grouped output reduction preserves coefficient dtype
promotion and avoids the invalid native empty reduction. The current metadata
trace uses the selected native callback schedule, including large gated
invocations delivered in uneven token chunks. Each original/intervention/effective
callback preserves its source offset and total invocation size, and failure stops
later deliveries. Completed-span accounting retains allocations already made;
chunking or an observer failure does not refund them. A missing schedule raises
`WorkspaceGroupedObservationScheduleUnavailable` before callbacks. Independent
expert providers still require their member and residency schedule. These
operation facts alone do not authorize independent copy/capture budgets or
complete ordinary/controlled admission.

Retained-state host reads preserve logical row order across capacity-backed cache
prefixes, broadcasts and reversed views. Serialization writes one host byte
result directly, and value comparison does not acquire another tensor or host
payload. A borrowed contiguous slice rejects an incompatible layout instead of
reading a shape-sized region from a smaller or differently strided backing.
Independent snapshot cloning still performs its separately priced native copy.
This corrects the copy mechanisms; enclosing snapshot/capture admission must
still reserve each retained result and retain its authority through completion.

Centroid-selected prediction readout computes masked vocabulary scores from each
position's selected minimum. Batch neighbors and chunk boundaries cannot change
that floor. Ordinary and controlled traces share its workspace descriptor, which
prices selected weight replicas and all lazy children through completion. Empty
positions bypass native selection and reduction. This is an operation contract;
it does not by itself admit a speculative round or its retained captures and
branch state.

A runtime request may also retain a ceiling on its shared managed-memory domain.
The smallest live ceiling governs both new reservations and registered storage,
including requests that do not supply another ceiling. Controlled continuations,
snapshots and native recovery retain the same request owner and cannot release
that ceiling by restoring state or failing a completion poll. Only final safe
retirement releases it. An explicit successor capacity may raise a completed
funding account's ceiling under the owner-delegation contract below; it does not
release storage or allow further submissions from the old run.
Ordinary and explicitly advanced prefill use the same
capacity-aware chunk planner. Public policy and native-domain installation still
require the complete enclosing admission integration.

Ordinary and controlled text preparation now receive the same public
`TextInferencePolicy`. Both support its prefill maximum through the shared MLX
chunk driver. The managed capacity and chunk policy are included in controlled
configuration identity. A strict capacity requires finite output allowance;
facade resolution supplies the ordinary 256-token fallback, while a direct
unbounded core request rejects before native or controller work. MLX currently
rejects public managed capacities with `WorkingMemoryError::UnknownBound` until
all domain owners participate in accounting.

Prepared speculative generation currently rejects non-default inference policies
with typed `InferencePolicyUnavailable` in coordinated host preparation. This
covers ordinary and controlled runs and a policy appearing on any batch lane.
Direct native speculative entry also rejects during coordinated admission. The
reason is incomplete speculative preparation, branch/capture accounting and
shared-domain integration, not an inherent limitation of speculation. This
exception must be removed when those paths consume the shared admission driver.

Cold resident quotes include the checkpoint before each transaction and the
possible rollback copies after a late failure. Sharing checkpoints add no
independent payload; compressed checkpoints include all capacity and logical
copies. The successful metadata frontier remains unchanged by rollback pricing.
The runtime unquoted-owner barrier also survives cloned owners and excludes live
reservations even when their charge is zero. Completing a native operation alone
cannot release that barrier while surviving storage remains unaccounted for.

The native model owner now joins the shared memory domain before loading and
survives prepared-model transfer, session retirement and unresolved loading work.
Ordinary and controlled submissions preserve that owner after releasing their
session lease, so retained opaque token handles cannot clear the unquoted barrier.
Completed output arrays also attach it to physical backing before publication;
converting logits to a tensor or array and retaining a view preserves ownership.
Final native release queues Rust destruction for an unlocked host boundary.
This is conservative lifetime participation, not a byte quote or an independent
copy grant. Complete quoted snapshot/preparation admission, observer escapes
and rank-coordinated budget admission still need their enclosing integration.

Load-time storage publication keeps one charge per native backing rather than
sharing a whole-inventory charge with every surviving array. Retirement of one
allocation can release its charge while another escaped alias remains accounted
for. Native generations and pinned Arc identity tokens prevent deferred cleanup
from mistaking a new allocation for a retired one. Key destruction precedes byte
release and runs outside the pool lock. These charges cover existing storage;
the model's unquoted barrier remains until complete state, preparation and copy
admission is established. Snapshots still need a successful quoted-copy route and
a checked transition from request charges to retained-state charges.


Ordinary prompt/sampler preparation and independent snapshot copies now acquire
separate unquoted domain ownership before allocation. A reservation in that domain
rejects new independent allocating work before state mutation; reserved preparation
continues through its already retained request authority. Explicitly unbudgeted
preparation still acquires an owner. Reserved prompt backing retains its existing
charge before binding and through any escaped array aliases. Prompt backing, copied
pending input, greedy host history and RNG copies retain ownership after their
source runtime or preparation handle is dropped. Recovery retains new copy owners
until native work settles. Native state exchange transfers owners with installed
and displaced state; sampling restore preserves both old and incoming authorities.
Neither operation refunds request or cumulative control budgets. Empty pending
input and a temperature-only override do not acquire a new owner. These changes
close ordinary lifetime gaps, while successful shared-domain quoted copies,
complete request promotion and speculative-copy integration remain unfinished.

Speculative seed, target/draft RNG and positional RNG snapshots now retain native
memory authority independently of sampler lifetime. Reseeding acquires before key
creation; every derived key and RNG mutation preserves its source and execution
domain authority through native recovery. Cloning immutable key handles does not
reserve twice or refund cumulative control budgets. Prepared greedy samplers also
retain the enclosing preparation owner for host history.

Embedded target-state and external/autoregressive target-cache copies acquire a
fresh owner before native copying. The state itself retains the neutral lease,
so an empty snapshot remains protected after its source runtime disappears.
Restore/import retain current and incoming ownership before fallible work; final
payload retirement precedes the last lease release. Copied tensors attach owners
to completed backing before publication. Reserved domains reject these independent
operations before mutation. Prediction-extension layer snapshots receive a distinct
snapshot context carrying the domain as well as native execution resources. Empty
and populated sequential/pooling copies retain independent owners through cache
checkpoint, clone, clear and restore. Restore merges incoming and installed guards
before fallible mutation; an escaped copied array retains its owner independently
of the cache wrapper. Prediction model-state copies reuse the target-state path.
This preserves existing controlled snapshot semantics without refunding request or
cumulative control budgets. Successful quoted snapshot admission remains unfinished.
Strict managed budget requests continue to reject unknown coverage.

Speculative distributions retain their native storage and domain authority through
proposal, verification and residual sampling. Processing, capture and adaptive
commit callbacks enter the same recovery/admission mechanism before native work or
policy mutation. Sampler snapshots and forks share allocation-authority history,
including domains added by a later callback on another clone; they still clone
ordinary mutable sampling state according to the existing sampler contract.
This preserves ownership for shared custom policy/capture payloads without
refunding cumulative control budgets. Verification no-work cases do not acquire
additional authority; actual event/copy handoff admits all inputs before mutation.
Complete shared-domain accounting for delivered host capture records remains
separate from their cumulative capture quotas.

Ordinary and controlled session operations now check allocation authority for the
backend context as well as retained model storage. A live reservation rejects an
unrelated domain entry before reset, token creation, forward execution or output
observation; rejection leaves the native state and session lease unchanged.
Successful unquoted entry retains authority with surviving session payloads.
Native owners hold immutable leases, so inventory publication cannot revoke a
snapshot, output or other descendant's existing authority.

Fresh session construction can now publish a complete idle storage inventory
after the initial reset settles. Only certified nonstate storage with an empty
decoder inventory qualifies; incomplete native backing or enclosing ownership
keeps the loading barrier. Publication releases only the constructor's local
loading handles after every charge is installed. It does not change snapshot,
restore or fork authority, and does not release a nonempty decoder's owner.
Subsequent ordinary and controlled work still acquires operation authority before
native submission. Strict request admission remains separate from this transition.

Direct model output ownership also preserves retained request reservations without
depending on sampling. The completed backing receives only neutral charge handles
and native allocation authority, avoiding a cycle through native/session roots.
Raw clones and views preserve the charge until the backing retires. Idle storage
inspection separates actual decoder backing from nonstate resources and requires
explicit enclosing coverage; it does not perform state copying or complete public
strict admission. Ordinary observer callbacks now receive tensors with deferred
descriptor/backing authority, including prototypes, generated values, routing
components and returned interventions. Attachment does not evaluate tensors or
invoke skipped factories, preserving callback admission and controlled execution
order. Independently retained prediction observers and delivered host payloads
still need their enclosing accounting before strict admission can be enabled.

The current request span checks do not price changes made after preparation.
Capture/intervention installation has its own cumulative ledger, and temperature
override has sampling-semantic validation; neither is shared-memory admission.
Reseeding and independent copies already require fresh native authority and reject
a live reserved domain. Those checks must remain until the selected request quote
explicitly covers replacement overlap and subsequent work. A future strict
ordinary/controlled request must retain its admitted sampling, instrumentation and
output-retention contract and validate these transitions before mutation. Raw
low-level outputs likewise need independently priced lifetime or a bounded release
protocol; attaching the same charge to every escaped output only preserves that
charge's lifetime.

Ordinary and controlled advancement now share explicit step-permit entry before
controller decisions and native input creation. Controlled advancement retains
the permit through token observation, exact completion and controller commitment;
ordinary delivery keeps its asynchronous behavior. Prediction readiness and
decision failures use the existing bounded peer agreement, with a separate
controlled commitment vote.
Permits are never copied into snapshots. Core-issued attempt identity survives
restoration; unrestricted mutable policy access changes the revision identity,
and a fork receives a distinct run identity. Instrumentation setup changes policy
identity before its mutable hook, including when setup later fails. These
identities now bind MLX's private cold quote to the original core run. A quoted
permit claims the runtime attempt before callbacks; other entries still acquire
unquoted authority.

Original run binding now occurs inside admission before prompt/sampler creation
and before callers can access mutable policy. The runtime preparation owner
serializes this binding with preparation and prediction claims. Its logical run
permit consumes an ordinal/output slot before callbacks, and an unfinished drop
fences the shared run without refunds. Decode requires the exact preceding
completed receipt, so a clone or restored old token cannot substitute for the
next input. A provisional receipt becomes usable only after permit finalization;
its identity holds no native payload or charge. Native completion still owns
unresolved resources. MLX now consumes these claims only with its immutable
complete native quote, validates the actual controller decision and every host
state frontier, and attaches provisional receipts to successful native tokens.
The shared native execution driver retains reservation ownership before input or
model work. Ordinary and controlled paths share this driver and differ only in
when the core finalizes the permit.

Resident and selected host-layerwise token-ID admission use the selected family/native workspace,
complete idle publication, actual prompt capacity and controller declaration.
Missing facts still reject before preparation. Direct admitted prompt preparation
checks length and allocation capacity against the original evidence before
creating a native array. Only that worker marks the prompt as originating from
the quote; rebuilding input or changing its policy invalidates the marker. The
actual decode receipt is checked again when the issued permit is submitted,
preventing a token swap between issuance and claim. Strict capture/intervention configuration rejects before
installation: its storage and mutation overlap are not yet included in this
ordinary quote. Mutable policy exposure invalidates the original run revision;
it cannot silently reuse the old quote. Direct backend sampling-override hooks
also reject before changing a quoted sampler, including temperature-only changes.
Independent snapshots and copies still
need fresh authority and cannot refund the original run's spent allowance.
Quoted device-resident decoder state can admit a subsequent token-ID request using its
current cached position. Admission records the exact opening state revision;
advancing another continuation makes that evidence stale before controller
callbacks. The first native submission atomically fences the predecessor's
future run permission while preserving its charges. Both ordinary and controlled
execution use this path. Subsequent decode requires a token stamped with the
current producing revision as well as the immediately preceding completed receipt.
Restoration and exchange cannot revive an earlier revision or spent attempt.

Funded native requests now publish surviving physical allocations through the
existing shared registry. Old request metadata no longer retains all expired
workspace. The run owner closes after controller, pending input, completion and
sampler payloads have retired; unresolved native scopes independently preserve
their coverage. Successful completion alone does not retire a still-owned native
completion's roots. Unknown or failed publication leaves funding uncertified.
Preparation uses the same payload-before-funding order through readiness
rejection and unwind. Completion settles submitted RNG descriptors before cold
publication, without native host readback.

Strict native controller admission requires both a complete size witness and a
complete `TextControllerStorage` lifetime declaration. Wholly run-owned filters
retain their existing contract. Shared tokenizer masks use a neutral immutable
owner with independent physical accounting, so loaded models, controller clones
and snapshots keep their charge after the generation run retires. The facade's
text controller now declares these shared owners explicitly. Ordinary and
controlled preparation publish the charge before prompt/sampler work and check
the original inventory before each decision. Shared provenance also survives
forced choices and is checked again before native submission.

Loaded-model construction is fallible and publishes shared tokenizer storage
before returning, even when that model never starts a run. Its deferred host
factory requires unquoted domain authority, so another live admitted request
rejects construction before allocating the mask. Later inference reuses the
existing charge instead of attaching another copy. When the complete controller
source inventory is already registered, runtime now pins it and removes its exact
fixed contribution from the new reservation. Full diagnostics remain available in
`WorkingMemoryReservation::admission()`. Cached continuation combines this host
credit with the independently proved decoder-state credit; both pins transfer into
the same run and work scopes. The ordinary and controlled paths share that planner.
Unknown or incomplete registration receives no source credit and retains the full
allowance. Actual owner attachment and identity/provenance checks still precede
prompt preparation and prediction. Final masks and forced-choice overlap receive
no source discount.

Controlled mask quotes include the original pre-override filter alongside the
final mask. Generic `All` controllers now declare optional filtering through
metadata, permitting unfiltered decisions and domain-sized forced masks in the
same admitted run. Nested choice wrappers bound both domain lengths and temporary
mask overlap without querying the controller or constructing a witness mask.
The shared sampler quote includes each possible mask allocation and unfiltered
input alias, including retained outputs across mixed steps. The executable score
backing capacity comes from architecture inspection rather than its logical shape.

A choice staged before admission can be consumed and followed by unforced steps
under the same policy. This does not authorize changing an admitted controller:
`controller_mut` still revises/fences the retained policy, and dynamic choice
updates still need repricing and authority integration. Grammar parser storage
remains incompletely bounded. Snapshot ownership of the shared mask does not
establish complete quoted snapshot/restore/fork admission for the remaining state.

Cached successors with resident or selected host-layerwise weights reserve an
identity-bound residual quote for device-resident decoder state. Existing
cache roots are pinned in the same registry and excluded individually throughout
the equation trace; later replacement buffers remain incremental even while old
raw aliases survive. Full state diagnostics are retained. The reservation transfers
its existing-storage pin into run/scopes before preparation, and stale opening
revision checks still precede controller or native work. Ordinary and controlled
execution consume the same quote and planner. Quoted reset and snapshot/copy
admission, complete instrumented/media/disk/distributed and remaining host-policy quoting, and independently
escaped completion/root accounting remain implementation work.

Sampling quotes retain all potentially delivered token allocations through the
output allowance, including physical allocation capacity and aliases. Successful
token completion can release temporary logits and old random keys while yielded
tokens remain alive. A separately retained completion handle or raw score output
requires additional lifetime accounting and is not covered by this token-only
retention policy.

Cold inventory distinguishes editable checkpoint parameters from other retained
numerical values. Neural modules and architecture static owners now explicitly
report whether read-only traversal covers every numerical helper, including
rotary frequencies and cached execution representations. Incomplete traversal
preserves known values without authorizing a complete budget. Inspection remains
behind the existing resolved-session boundary and performs no native evaluation,
snapshot, state mutation or refund of retained authority.

Selected host-layerwise weights now use the same managed token-ID admission,
chunk planner, step permits and generation driver as resident weights in both
ordinary and controlled sessions. Decoder state must remain device-resident.
Admission requires complete idle publication, exact completion, a finite output
allowance and complete facts for every ready immutable MetalShared host transfer.
Missing facts or changed source, policy, parameter or state identity reject before
prediction. After reservation, the private quote pins the exact ready host stores
so eviction cannot substitute an unquoted disk load during the run.

Construction and per-name transfer windows join the existing equation and
sampling quote. Registered decoder credit composes with this path; prospective
parameter destinations receive no such credit. Existing controller provenance,
input-marker, opening-revision and predecessor-receipt checks still apply.
Both generation modes use the same cold candidate search even when the first
chunk lacks a numerical workspace bound. The first diagnostic candidate binds
the output width and controller contract; a later change rejects before
reservation. Only a structurally validated incomplete-workspace result permits
retry. Source-readiness and native failures stop immediately with their original
cause. Rejected candidates do not create prompt/sampler state or acquire host
residency pins; these remain after successful reservation.
Successful completion publishes surviving physical storage before releasing
workspace; failed work retains coverage through the existing recovery owner.
This extension does not authorize dynamic controls, capture/interventions,
independent quoted snapshot/reset/restore/fork/copy, media, speculation, disk or
parallel execution. Those remaining paths are implementation work, not model
limitations. Independently escaped raw logits/completions also retain their
separate lifetime-accounting requirement.

Disk equation inspection now uses the same neutral layerwise traversal as host
inspection. Reusable direct-read plans retain exact file admission, and cold
manager snapshots expose canonical alias dependencies and actual warm capacities
without retaining device arrays or source-wrapper payloads. This is preparation
for the foreground direct disk route. Its private receipt activates once per
existing `SessionOperation`, before input conversion, and covers every prefill
chunk plus final output selection. Decode and sampling use the same operation
owner. Recovery tickets retain the route through unfinished work; settled old
token owners do not prevent the next operation from activating it.

The selected loader must have zero host-prefetch budget, complete retained direct
read plans and device-resident decoder state. The manager enforces exact declared
windows and canonical-owner capacities before acquisition. The policy completes
and evicts each preceding window before refilling, including group boundaries
and shortened tails. Ordinary and controlled runs use the same quote, candidate
search, permits and execution path. Background prefetch, ordinary conversion and
the other unfinished managed paths above remain implementation work.

For successive managed text requests on one session, an explicitly larger
`managed_memory_capacity_bytes` can raise completed predecessor limits. The
private session must hold the original funding owner's one-time delegation; a
request clone, snapshot, output token or receipt cannot supply that authority.
Both ordinary and controlled admission use the shared atomic candidate planner.
The old run must be closed and every native funding scope certified before its
ceiling can change. Active or quarantined funding rejects an increase; unchanged
limits still participate in ordinary admission.

Old token allocations, raw aliases and historical request owners remain charged
after succession. The adopted ceiling stays with each old funding account until
its final retirement, including when the new request retires first. Fixed pool
capacity and unrelated request limits continue to apply. Rejected candidates
leave accounting and limits unchanged. Successful admission commits the new
policy even if subsequent preparation fails. This changes neither predecessor
fencing nor snapshot, output, transport or copy allowances.

Immutable non-mask controller sources now share the same accounting custody as
shared tokenizer masks. Borrowed byte slices cannot outlive their owning handle;
pre-existing aliases preserve attachments until the final allocation retires.
Ordinary and controlled semantic setup pack vocabulary offsets and bytes through
one deferred constructor guarded by the selected backend's managed domain.
Another active reservation rejects before that factory runs, with the original
neutral backend error preserved.

Managed decisions with byte-source declarations include a complete post-callback
source witness. Submission compares source kind, identity and capacity with the
admitted inventory, so replacing an owner during a callback cannot pass solely
because its size is unchanged. Forced-token overrides retain that witness and
existing observation provenance. This is source validation, not permission for
unbounded allocation inside a callback.

Grammar parser growth, compiler/trie storage, semantic output state and independent
snapshot/fork/restore copies remain separately unproved. Their managed admission
stays unknown until those owners and operations have complete bounds. Shared
vocabulary custody does not change cumulative control, transport or copy budgets.

Runtime prepared chats now retain registered immutable reconstruction recipes
instead of compiled grammar matchers. Eager validation and syntax fallback still
happen during preparation; the recipe keeps the original schemas used to validate
completed calls. Prepared-chat clones share this source and its accounting
custody. Explicit text mode leaves it encoded, so an unused semantic matcher no
longer survives solely because the prepared chat remains alive.

Ordinary, controlled and speculative semantic setup use the same host-preparation
hook before tokenizer reconstruction, grammar copying, protocol parser creation
or schema compilation. A live managed reservation rejects that unquoted work;
the original backend cause survives facade mapping. Actual mutable parser owners
and shared validators retain the authority through internal copies and source
destruction. This is exclusion while complete bounds are unavailable, not a
finite parser budget. It does not turn the logical snapshot copy ledger into
physical allocation permission, refund cumulative budgets, or complete host
capture/restore/fork destination admission. Those remain separate requirements.

Snapshot, restore and branch creation now acquire destination host authority
before facade metadata and semantic copies. The shared continuation driver also
acquires independently before owned discovery, controller/capture copying and
logical copy reservation. Rejection leaves copy usage and the source continuation
unchanged. It retains its typed `HostPreparation` cause and leaves the facade
lifecycle usable; an actual native copy failure continues to follow native
settlement and recovery rules. Once copying begins, failures still consume the
existing cumulative logical allowance.

Authority follows the copied continuation through installation and branch
exchange. Shared capture/checkpoint/partition owners retain attachments until
their final aliases retire, including handles created before attachment. Source
and destination custody coexist during restoration and partial installation.
The independent facade and runtime acquisitions intentionally prevent retained
source custody from being used as fresh destination admission.

This closes an early-copy ordering gap. It does not replace the logical ledger
with a physical estimate or enable finite MLX snapshots: quoted sampler copies
were already rejected, and are now rejected before host work. Complete funded
copies, independently exported application values and unbounded parser state
remain implementation work. Cumulative capture, observation, transport and copy
budgets are never refunded by restoration.

Standard and Mirostat sampler snapshots now bind their source controls and full
boxed history through one checked copy plan. Clearing history keeps its retained
capacity, so the copy estimate continues to include those unused slots. Copying
preserves the RNG key without drawing or reseeding; later source and copied
samplers remain independent. The native array program used for pending inputs,
RNG and concatenating KV state has a matching workspace trace that retains
source/destination overlap and every requested destination, including copies
from aliased source slots.

These component plans provide no fresh execution grant or complete finite copy
allowance. They keep existing host/native admission, recovery and logical ledger
semantics. Full controller, metadata, decoder, delivery and state-copy coverage
must be established before a finite snapshot or runnable branch can be admitted.

A funded sampler component now carries the original sampling stage's host custody.
Its sample permit consumes one attempt before native filtering or sampling;
unused permits do not refund the original allowance. Host custody is a closed
scope separate from native PRNG recovery and retains a protected canonical
history-growth allowance inside the existing account.

Independent sampler-component copies use fresh destination accounts under the
same pool and its tightest live ceiling. The frozen copy retains its own history
and controls through source destruction and can fund another copy. It exposes
no mutation or continuation grant. This component facility does not enable a
whole controlled snapshot: complete RNG, pending-input, cache, decoder, capture
and delivery funding is still required, and the existing whole-copy rejection
remains.

The native saved-array component uses an independent copy account for the current
RNG key and optional pending scalar. It validates the completed receipt without
issuing a new text step, and keeps the same session lease through inspection and
copying. Preparing or inspecting its sealed copy plan does not draw randomness,
advance the frontier or consume the logical snapshot ledger.

The destination is published only after native completion is established. An
ordinary settled copy error leaves the source unchanged and usable; destination
funding stays conservative on failure. A successful saved component can outlive
its source, but contains no execution grant or ready-to-submit pending token.
It does not enable complete controlled capture, restore or fork until all
remaining state, controller, decoder, capture and delivery payloads are covered.

Read-only snapshot capture, branch copying and restoration staging preserve the
source run's policy identity. Their quiescent copy borrow exposes no mutable
generation state. Successful restoration still revises policy when it installs
the prepared state, and issued prediction attempts remain monotone. A rejected
copy before installation does not itself revoke the original continuation.

Sampling/input snapshots now have a distinct immutable saved type. Complete
unquoted capture uses one aggregate copy; restore and fork stage one independently
owned sampler/input pair before installation. Copying that saved data is a
separate operation from preparing an executable continuation.

A bounded MLX saved component admits canonical history, RNG and an optional
pending scalar together before copying either host or native payload. The host
hold and native completion scope share one account while retaining independent
lifetimes. An immutable saved component remains a valid copy source after its
original live run advances or is dropped; duplication checks its own account and
physical storage rather than replaying an old prediction receipt.

Bounded prefill-input copying still requires a closed prompt/cache-identity host
plan. Funded saved state also cannot yet become a runnable restored or forked
continuation: the resumed initial decode needs fresh execution admission and a
new binding to the installed state revision. These paths return the typed unknown
bound before copying or installing unsupported state. This is remaining
implementation work, not an architectural limitation. Whole finite snapshot
capture continues to require its native-state and facade-host aggregate.

Live layout metadata uses shared immutable ownership rather than allocating a
fresh layout for every checkpoint. Its exact retained capacity and per-domain
attachment follow all aliases, including those made before registration. Native
idle publication includes that owner before releasing loading authority. Input
identity descriptors have measurable immutable storage and a closed text-token
construction plan included in prompt admission. Session prefill and control
checkpoints preserve the actual owner, including its attachments, through commit,
rollback and copying. Borrowed legacy descriptors remain compatibility inputs;
neither a raw layout borrow nor a copied logical estimate authenticates source
funding. The full bounded snapshot aggregate and fresh resume admission remain
separate unfinished work.

Prepared resident-cache copy components preserve their actual storage semantics:
padded key/value buffers stay padded, compressed logical views become compact
independent buffers, and pooling retains each optional partial/overlap slot.
Aliased inputs remain separate requested destinations. Hybrid fixed-role tables
preserve absent components and scalar frontiers while making their retained
slot extent measurable. The recovery worker retains produced intermediates and
results before the next fallible native operation. Its collector is not a list
of final publication roots and does not itself establish completion or funding.

These native copy components do not relax the early whole-snapshot guard. The
mutable live host records need source accounting, destination host copies need
protected capacity in the joint account, and paged catalogs need independent
namespace and transfer plans. A saved numerical copy also needs an immutable
owner before a later resume can acquire fresh runnable authority.

The resident pooling-attention copy composes local key/value storage with zero,
one or two pooling streams in a closed source-bound program. Fixed-role copying
uses the same isolated-array worker and recovery collector. Existing paged
snapshot behavior stays on its manager-copy path; resident component preparation
rejects paged storage before native work.

Fixed mutable layer and role containers now retain source charges through their
actual storage owner. Independent clone/snapshot tables receive new identities;
in-place checkpoint restore preserves the existing outer table. Existing
same-extent role-table and DeviceState clone_from reuse preserves their prior
identity, while whole native model-state clone_from retains its replacement
semantics. Shared immutable layouts continue to share their original owner.

A fixed-table token can outlive destroyed elements and retain an old charge, but
retirement rejects any new attachment. Table/element and final charge destruction
happen outside the lifecycle locks. Native publication attaches each actual
source charge before releasing its original authority. Complete destination
construction, paired decoder/sampler admission and fresh resumed-run permission
remain separate requirements for bounded controlled snapshots.

Source publication remains conservative when other retained model components
are unknown. The DeepSeek V4 rotary-helper fixture retains its loading authority
while exact host tables can be inventoried and registered separately under that
same authority. This partial source coverage is not complete copy or run admission.

Shared ordinary and controlled continuation snapshots retain decoder, sampler
and pending input as one opaque saved pair. Restore and fork invoke one paired
resume preparation, which stages all components before installation. A late
copy failure leaves the active and saved states unchanged; cumulative logical
copy charges remain consumed. Read-only sampling diagnostics borrow the paired
owner and do not expose a writable decoder slot.

The MLX paired implementation preserves its established unquoted workers under
their actual allocation owners. Its bounded resident KV route uses one aggregate
decoder/sampler/native account and immutable paired output. Other native state
representations, pending prefill payloads and fresh funded resume remain
unfinished; their typed rejections are implementation gaps, not architectural
limitations. Saved copy custody cannot grant resumed-run authority or waive
the whole-host snapshot guard.

A prepared saved copy is bound to the session that issued its submission lease.
Passing it to another session rejects before admission or copying. Saved data
can still be copied within its pool after fresh preparation on the destination
session. Neither source provenance nor shared pool membership substitutes for
that exact destination lease.

Fixed destination-slot construction uses an actual source borrow and a closed
vacant-cell initializer, followed by sequential fill and exact completion. A
partial finish retains its builder, and an extra fill returns the incoming
value unchanged. This primitive does not itself fund the table, authorize nested
resources, or certify native completion.

Joint component admission protects sampler history and decoder-slot initialization
independently from native publication. Partial decoder construction and incomplete
finish errors retain both installed payload and its host scope. Completed slots
expose borrowed values and a new copy plan, with no mutable table export or run
grant. The decoder initialization envelope remains held until that owner retires.
The actual native copy driver must still supply complete source inventory,
retain source data through recovery and publish destinations before certification.


The resident KV component worker now retains the exact live decoder or actual
saved decoder through native recovery and commits both host holds with its
numerical copy account. It preserves cache controls and shared input identity,
publishes only destination arrays after completion, and exposes immutable saved
data. Copying those saved values does not replay an earlier live frontier. Other
native state representations remain explicit unfinished copy paths; fresh funded
resume still requires its own inference admission. The enclosing controlled
snapshot host guard and cumulative budget rules remain in force.


A funded sampler may now prepare preserved host state for a fresh sampling stage.
The stage must match the new configuration, output allowance and exact destination
account. Its owner starts a new attempt count while retaining history capacity,
penalty controls and Mirostat state; dropped attempts still consume the new
allowance. Existing source authority is not reset or reused. The new host hold
covers the original destination copy and all permitted history growth overlaps,
and remains unavailable to native storage adoption until payload retirement.

This host prerequisite does not install a decoder, restore native RNG or pending
input, construct a new controller, or create step authority. Funded native resume
therefore still rejects before installation. Resident Pooling tables can now use
the same sealed decoder/sampler copy account as KV tables, with actual optional
cache slots and shared layout preserved. Stateless absence and authentic complete
model admission remain separate unfinished work.


The resident copy adapter now preserves actual stateless Pooling absence. Saved
absence has no fabricated table, global index or mutable state, while the shared
copy engine still retains sampler/native and complete-source custody. An admitted
absence can only be completed from the actual source instance it borrowed.

Dense decoder initialization now has a closed one-buffer primitive, including
partial completion ownership. This remains separate from live-state installation:
a new request must fund it, and its protected host coverage must transfer to the
actual table's registered charge without a gap. Neither the new primitive nor
stateless copying changes the funded native resume rejection.


Cold retained-array inspection now uses separate metadata and inspection-clone
operations that perform no evaluation, completion polling, housekeeping or owner
reclaim. A busy runtime or failed clone leaves the inventory unchanged and keeps
the original typed error. A backing observed as unknown remains unknown even if
another alias completes later. Callers must keep the source settled and unchanged
while binding the snapshot to its retained handle.

This closes an inspection-side effect without releasing a source pin or granting
new work. The existing exact-completion and quarantine rules still govern copies;
funded native resume remains rejected until fresh request authority and decoder
installation are implemented. Lazy model-helper publication is separate loading
work and remains gated by its actual unknown inventory.


A fresh prompt stage can now consume an actual decoder source to construct one
fixed dense table under its already admitted request. No source request, receipt,
quota or copy account is promoted into execution authority. Incomplete fill and
typed publication failure retain the entire destination and consumed claim.
Successful table publication returns only the original prompt's finish token;
finishing marks construction complete, and the existing owned-input binding is
still required before prefill is ready.

The table's retained charge is attached before its construction hold retires.
Escaped metadata conservatively keeps that charge after payload retirement.
Host completion never certifies the separate native scope or clears quarantine.
These primitives do not yet install a native decoder or enable funded generic
restore/fork; the fresh shared inference driver remains unfinished.


A completed resident KV dense copy can now produce fresh native state without
inheriting an old request or transaction flag. Neutral prepared-state binding
preserves that state and its existing custody, leaving publication to the ordinary
control exchange. Origin validation is available before construction and repeated
at binding, so a parameter change between those steps invalidates the candidate.

Saved native pairs preserve their original executable origin even when copied
through another compatible session or after the original model retires. Matching
shape and numeric epoch alone are insufficient for resume. The funded resume hook
now checks this provenance and still rejects for missing fresh request/controller/
input construction. No new controlled or uninterrupted resume loop is enabled.


Public MLX loading now settles its already-retained numerical module helpers
before initial storage publication, under existing loading exclusion and native
recovery. Cold inspection still never evaluates a helper. A low-level executable
that has deliberately skipped this loading finalization may therefore still have
an unknown inventory; completing ordinary public loading does not justify
upgrading that separate executable's evidence.

This loading change supplies no saved-state installation, restored controller,
new request or inference step. Complete initial publication may retire its local
loading exclusion only when the full existing inventory is known and the normal
empty-decoder condition holds. Native errors retain their causes, and unresolved
work retains its loading authority and numerical roots through existing recovery.


`TextGenerationDriver::resume_saved`, borrowed controlled generation and ordinary
generation now share an optional backend resume extension and the same prediction
machine. A fresh context starts at local attempt zero; historical preparation and
receipts are never replayed. Zero output agrees no-work, and cancellation before
installation returns no continuation. Installation happens inside the Sampling
result; peer rejection, cancellation or unwind there must retain recovery or
fence the exact session. Successful readiness alone permits finish-only
extraction. No active native submission lease may cross these agreements. The
neutral extension does not yet enable MLX funded resume or generic snapshot fork.


Fresh dense KV quoting now preserves source aliases while modeling each copied
slot as independent destination storage. Multiple future cache updates can start
from those projected outputs without assuming their capacity equals the source
view's backing. This is cold evidence only: actual saved origin, complete source
custody, native completion and fresh installation still need separate checks.

The populated sampling quote is a prerequisite for fresh resume admission. It does not copy a saved component, reuse an earlier preparation, validate a new configuration against the saved static policy, or authorize a resumed step. Source custody, copying decoder/pending/random state, complete host and native bounds, fresh request admission and installation through preparation readiness remain independent requirements. Ordinary and controlled consumers can use the same prepared architecture quote; this prerequisite alone does not add an MLX resume implementation.

Saved MLX copies preserve cache frontier F and absolute prediction ordinal A independently. Ordinary request quotes use origin zero; receipt and step checks use the private checked local ordinal, while capture and prediction diagnostics keep absolute A. Nonzero-origin arithmetic is tested internally, but no native fresh-resume route or funded generic restore is enabled by this prerequisite. A future resume constructor must bind the actual saved origin, fresh request and post-install revision before any step is usable.

The native opening-seal prerequisite preserves ordinary revision/predecessor checks and rejects unsealed or resealed runnable evidence explicitly. Pending metadata may support future prompt preparation, but it issues no step or receipt. Installation, failure custody through readiness, and final fresh-run binding remain the future closed resume adapter's responsibility; generic funded restore/fork remains unchanged. Tests of changed empty retention identities are owner-level conformance, not a native resumed-generation claim.


Fresh-resume preparation now has closed pending-token host construction and final typed dense-state binding mechanisms. The pending token becomes a one-part incremental input with no replacement cache identity, no decoded host token vector and no history append. Its original committed prefix remains with the saved decoder control state. Cloning this input shares the actual protected container. Consuming the original prompt completion prevents a second construction claim; closed or quarantined funding rejects before numerical entry or host fill.

Final dense-state binding preserves the exact supplied saved prefix and fresh empty inference retention. It does not compare against a different currently installed prompt, inherit old requests or exchange live state. Saved-source quotation checks finite future allowance, absolute prediction ordinal and decoder frontier independently, and preserves typed idle-boundary errors. These mechanisms still require source-bound admission, native construction, installation and failure custody through shared preparation readiness before ordinary or controlled native resume is enabled.


The fixed constructor host fact is consumed by the same selected equation quote and admission used by uninterrupted and controlled text generation. It does not add a second sampling or execution driver. The proposed V4 regression matrix uses real resident, host-window and direct-disk loading, a five-position prompt followed by three cached decodes, exact capacity, and rejection one byte below that capacity before callbacks or source work. Results remain subject to central native validation. Snapshot/fork/restore and fresh-run grant requirements are unchanged.


## Current fresh native resume (2026-09-14)

Fresh native saved resume uses the existing ordinary/controlled text preparation and generation machine through TextResumeBackend. A public immutable MlxSavedTextComponents provides data only; a positive finite request must acquire fresh source-bound authority. Zero output and initial cancellation remain inert. The saved pending token is already committed to history and becomes one Prefill position; it is not appended again. Saved absolute prediction A and decoder frontier F remain distinct from the new run-local attempt zero, and new outputs receive only new request receipts.

Prompt construction is provisional and native work settles before readiness. Installation rechecks exact session, original target revision, parameter/source origin and health, final controller and core context, then uses the existing checked exchange. The failure guard stays armed until Sampling Ready and borrowed finish-only extraction. A rejection/cancellation before exchange preserves the old branch; failure or unwind after exchange fences the exact affected session. New state has fresh empty inference retention and the original shared prefix. Historical source grants never authorize the new run.

The validated native construction program covers resident dense KV and grouped Hybrid ordinary, key-only, fixed/recurrent and compressed MLA state with finite allowance and exact completion. Table-backed Pooling also passed combined validation. Absent-table and paged state and remaining capture/media/speculation/distributed compositions retain explicit incomplete gates; they are unfinished integrations, not inherent model limitations. This fresh driver does not enable generic funded snapshot restore/fork of unpriced facade/controller/parser state.


The fresh saved-source driver also accepts the closed KV-only Hybrid representation used by routed and composite executables. Typed binding consumes its actual published Hybrid table with fresh inference retention, preserving the shared original prefix, global layer coordinates, attention state and fixed offsets. The original prompt claim, source admission, sampler copy, checked exchange and readiness guard remain shared. The grouped extension below handles recurrent/fixed and compressed Hybrid state. Paged Hybrid resume remains unfinished.


Compressed-copy projection now models the actual compact destination: two logical arrays are independently copied and each becomes its destination store and view, preserving the saved offset and capacity step. All four possible source roots remain in the complete source inventory. Grouped MLA saved resume consumes this projection; the projection itself supplies no snapshot execution authority or native completion.


Fresh grouped decoder construction preserves one prompt-preparation claim across
all outer and child tables. A child can finish and publish its exact live table,
but cannot produce or finish a prompt completion. The outer coordinator releases
the original finish-only remainder only after every child publication and exact
outer completion. Existing pending-input construction, native settlement, final
publication, owned-input binding and shared readiness still apply. Failed group
construction consumes the original claim without enabling a retry; partial and
published payloads keep their independent custody through recovery. Immutable
saved group results retain no runnable request or previous preparation grant.

The shared fresh-resume adapter now has a grouped Hybrid construction path for actual nested role tables, including an absent optional value that still occupies a role slot. Its one original prompt claim, saved F/A coordinates, populated sampler and PRNG key, checked installation and readiness guard remain shared. The prior KV-only fast path remains valid for actually empty role tables. Paged-manager and facade semantic-state restoration remain separate work.


Table-backed resident Pooling now uses the shared saved-resume driver, one original
prompt completion, protected sampler copy and typed published-state binding.
Local caches and primary/index pooling streams keep their partial-window state.
Actual absent-table fresh resume and paged managers remain unfinished integration;
they are not model-family limitations.

Ordinary and controlled resume use the same sealed incremental admission before
host/native preparation. Existing physical source charges remain held. The reduced
reservation preserves saved frontier and absolute coordinates, fresh local
receipts, immutable source identity, target exchange, opening readiness and exact
target failure fencing. It does not enable whole-facade snapshot restoration.


A closed shared TensorObservation host destination is available as a prerequisite
for bounded capture. It preserves standalone DTO serialization and, through a
borrowed adapter, the existing capture wire policy. Cheap aliases retain actual
shape/data custody; incomplete builders can be recovered without reallocating or
releasing that custody. Finishing scalar fill proves only a complete host payload,
never native completion or permission to advance a controlled session.

Managed ordinary/controlled capture and intervention ingress remains unchanged and
explicitly gated. Integrating this host leaf still requires exact source/native
transformation coverage, record ownership, quota attribution and the shared request
preparation/readiness path. The current floating raw-transform and rank limits
belong to this constructor, not model-family capability declarations.


The no-decoder Prompt constructor preserves actual rank-local absence. It does not
yet enable absent-state saved resume: selected local absence differs from an
ordinary layout containing NoState layers. Existing realized-state/control and
saved-quote layout checks remain until selected-state and applicable distributed
agreement integration is complete. Prompt cancellation cannot reissue construction
or certify native work; unresolved cleanup retains source pins in quarantine.


The runtime capture-tensor host owner now has a closed constructor using an original parent reservation, shared by future ordinary and controlled capture work. It protects only the exact host destination and introduces no capture invocation, quota reset, policy revision, preparation stage or native completion assertion. Original managed instrumentation admission and the native transform/readout/record-delivery path are still required; current managed capture/intervention rejections remain unchanged.


If multiple native scopes fail, their complete registered source inventories all remain pinned, including additional prompt/transfer roots and zero-byte identities. Successful sibling completion releases only its own pins and never clears another scope's quarantine. Native payload recovery and source accounting remain separate obligations.


The shared fresh native saved-source path also composes ready host-layerwise and foreground direct-disk weights with Device decoder storage. Ordinary and controlled preparation retain the same exact transfer/read policy as ordinary finite generation. Copied saved F/A, original prompt prefix, populated sampler/key and fresh receipt ordinals are unchanged; the existing installation guard and preparation agreement remain responsible for cancellation and failure. A completed output may retain its physical charge after its operation's disk restriction retires. Unknown/unsupported source schedules, paging, prediction/communication and rank-local absent-state semantic selection remain separate unfinished integration. The nonzero host/disk parity, exact/one-byte-short and cancellation/retirement tests passed centralized validation; the bounded-inference record contains commands and results.


Ordinary and controlled `next_cancellable` now agree cancellation before each prediction as well as at safe prefill spans. Saved-source runs reuse the same check. A cancelled call terminates the source without a new token or controller callback, after settling prior native submissions. Local completion or admission failures remain errors. All ranks participate in the existing Prediction agreement; no extra agreement or refundable attempt is introduced.


Managed capture/intervention admission remains gated. Its new F32 Metal selection leaf and source-bound host destination are shared mechanisms for future ordinary/controlled integration, not a second observation driver. They preserve admitted output/source geometry, whole source backing, parent-account host protection and caller-owned native recovery. Successful host finish provides no readiness, quota renewal, invocation authentication or native completion certification. The caller must settle and retire/publish actual roots before certifying the exact borrowed scope; errors after native entry retain its full source pin bundle through quarantine.


CapturePayload raw and shared tensors use the same kind:tensor wire encoding, including nonfinite tensor values, and compare by value across ownership representations. Deserialization yields raw caller-owned DTOs. SharedCapturedStep serializes like CapturedStep and exposes borrowed records; it has no owning export or deserializer. Existing ordinary/controlled capture delivery contracts remain unchanged until complete frame admission is integrated.


The capture mechanism prerequisite supports actual F16/BF16 through native AsType(F32), preserving the existing native exceptional-value conversion rather than substituting Rust half widening. Full, Slice and Preview use the same selected program; zero final half output finishes an empty protected host owner after selection settlement without a hidden cast or F32 read. F64 remains a typed rejection for this selected Metal route, whose GPU primitive implementation rejects F64. This extension does not remove the managed ordinary/controlled capture or intervention gate, supply invocation attribution, or claim that host finish certifies native completion.


The fixed frame constructor reports the enclosing ledger step/cumulative usage and transaction outcome without creating quota, execution or success authority. It preserves extra invocation/failed-attempt charges, rejects further mutation after parent closure/quarantine and keeps payload plus custody on finish error. Existing ordinary/controlled capture delivery interfaces and gates are unchanged pending complete original-admission and shared-driver integration.


Capture checkpoint forks preserve the original cached opening origin and local observation schedule while extending only the child prediction limit; they do not fold cache into prompt length, rebase observations or refund usage. Nonempty intervention plans currently lack an ordinary cached-origin contract and reject nonzero-origin coupling before work. Zero-origin and explicit-invocation paths remain unchanged; origin-aware intervention integration is unfinished rather than architecturally inapplicable.


SharedCapturePlan preserves the exact semantic admission through cheap aliases and later accounting attachment. Changed or independently cloned admissions have separate physical ownership; a caller's raw clone receives no custody. The owner supplies no invocation, quota, snapshot or fresh-run permission and does not change ordinary/controlled instrumentation gates.


Capture checkpoint validation now borrows its original admission rather than creating a temporary re-admitted plan. Current selected capabilities and exact declarations are checked before copying checkpoint state. Failed validation does not change schedule, delivery or cumulative consumption. This removes the temporary validation copy; checkpoint payload copying and controlled-session source ownership still require their existing preparation/admission contracts.

Options-aware ordinary, controlled and detached starts share Admission → Prompt → Sampling → Instrumentation and retain the exact shared capture source through failure cleanup and machine lifetime. The Instrumentation vote also occurs for explicit options with no capture. Legacy starts and late legacy capture/intervention configuration keep their existing order; saved resume is unchanged. This additive core contract does not yet enable managed capture in the facade or a concrete backend.

Shared capture-plan source publication retains physical accounting across earlier aliases and inventory/request retirement without altering semantic plan identities, checkpoint policy or capture budgets. A plan's source registration grants no capture hook, native completion, resumed invocation or frame-delivery permission. Managed capture gates remain unchanged by this prerequisite.

The cumulative capture host foundation uses request-local ordinary prefill/decode coordinates and nonrefunding one-use claims. It can preserve metadata frames at p0 and schedule gaps and keeps protected output aliases alive independently of the request's metadata. It does not install capture, alter controller policy, clone/replay preparation, create readiness or permit a native source read. Ordinary and controlled managed capture still require shared original-admission/installation/delivery integration; their existing typed gates remain. Snapshot/fork capture credit and claim-state inheritance are not changed by this unit.


Scheduled capture tensor failures can leave a lexical observer as owned typed errors without copying their payload or releasing original host custody. Conversion consumes the partial owner and cannot restore the spent claim. It releases an exclusive native-scope borrow without certifying that scope or a sibling. This is a capture-composition prerequisite and does not enable managed observed sessions.

Scheduled native tensor transfers spend the original one-use coordinate claim even when later validation or native work fails. Same-shaped independent admitted plans cannot substitute for the retained source plan. Completed read-only tensor aliases keep cumulative host custody; a failed host finish retains its protected partial payload and exact scope borrow until explicit retirement. Host finish is separate from native settlement, publication, logical quota and frame delivery. Managed capture gates remain unchanged.

Shared capture source ownership now survives portable sessions, partition work/receipts and checkpoint source aliases without re-admitting or copying that source. Existing checkpoint destinations remain real independent copies, and child fork admission remains independent of the parent's source registration and run identity. Restore retains the existing nonrefunding consumption policy. Scheduled host-frame claim installation, managed delivery, native readiness and capture/intervention gates are unchanged; this source-lifetime unit alone does not enable managed controlled capture.


Precommit capture sealing validates account health, logical usage and existing wire encoding before constructing the immutable shared owner; its one-use outcome finalizer cannot allocate or certify native work. Pending abort delivery keeps original payload/custody and spent claims, including after unwind, parent closure or quarantine. The observed equation entry uses ordinary prefill/decode coordinates for both controlled and uninterrupted drivers, preserves full-readout consumers and rejects partial-prefill capture with a typed attribution error. Actual native resident, host-layerwise and foreground disk sessions preserve shared path identity across both execution modes. These components still need the shared funded CaptureSession transaction/drain and production native hook installation before managed observed generation is enabled.


Ordinary, controlled and detached text drivers share fallible retained capture delivery and completion settlement. Failed completion polls/waits preserve the exact handles, and failed drains preserve undelivered records. Legacy raw drains park shared records without copying; None cannot establish a drained boundary. Pending records block another prediction, with detached drivers rejecting before consuming input and borrowed iterators retaining their existing termination semantics. The funded lexical observer uses the original transaction policy: fallible sealing precedes agreement, commit only publishes a sealed frame, and abort/unwind detaches original custody without allocation or refund. Empty transactions still require a terminal drain; a successful retry cannot clear an earlier failed continuation.


Selected projection-input metadata and native execution share one fixed reconstruction sequence, with compact input roots and intermediate lifetimes represented around the observation callback. The generated factory remains lazy for unselected observations. This mechanism preserves ordinary and controlled shared traversal semantics; original funded observer installation and retained partial-factory recovery still require integration before managed capture is enabled.


The native funded capture bridge reuses the existing session/observer transaction
and completion paths for both ordinary and controlled composition. It retrieves
only the work installed by the original text entry, and its lexical observer drops
before operation finalization. A typed failure or unwind retains an aborted frame
and original native recovery roots without refund or certification. Delivery still
requires independent exact native completion. This private prerequisite does not
open managed gates or claim prefill, generated, partition or intervention support
before their original quote and installation paths are integrated.

Decode Full/Slice/Preview diagnostic traversal accounts one logical prefill metadata row across prompt chunks and uses request-local predictions with the original cached prefix. It does not allocate a frame per chunk or reset cumulative quotas between spans. Readout demand is available before observer prepare and shared with the funded collector. Selected whole-prompt tensor attribution and generated-source consumption remain explicitly unfinished; selected unsupported paths retain typed errors while skipped factories stay lazy. Ordinary and controlled physical capture installation is not enabled by this diagnostic prerequisite.


The private original captured-admission route retains the same request geometry, source identity and prepared traversal used by the shared text preparation. It protects cumulative host capture storage before Prompt and preserves a pending source-health witness through Prompt/Sampling for a one-use installation check. Semantic validation borrows retained facts without rebuilding discovery or resolving checkpoint identity. Failed validation leaves the pending bank intact. This prepares the common ordinary/controlled integration; it does not enable managed instrumentation or establish snapshot, restore or fresh-resume capture parity. Original-context installation, native execution and completion-first shared delivery still need to be connected and verified.


The MLX core-options route now shares original captured admission, installation, native execution and completion-first delivery between ordinary and controlled text sessions. Installation accepts only the original ready, unstarted core context and actual reservation; it issues no permission and leaves the Instrumentation vote to core. Decode Full/Slice/Preview capture uses the cumulative original bank; p0 contains schedule metadata when prefill selection is disabled. A pending shared frame blocks the next prediction, including when the legacy raw drain has parked it. Empty plans still have a terminal drain after every prediction. Escaped immutable frames retain their original scheduled host hold; after healthy run/native closure, unused transient headroom retires while that hold, live registrations and the original capacity ceiling remain. Selected-prefill attribution, captured snapshots/restores/fresh resume, intervention, the facade host envelope and speculative/distributed capture remain unfinished. Their typed rejections are implementation gaps, not inherent family limitations.


Chunk-level state commitment is distinct from whole-prefill capture delivery. A captured p0 frame must stay provisional through the final cancellation check, score indexing and reservation settlement. Cancellation or failure after earlier chunk commits aborts the frame and preserves its spent claims; it does not imply whole-prefill state rollback. Ordinary and controlled runs use the same neutral outer guard.


Ordinary post-token grammar/termination queries preserve the initial bound policy across later cached decodes. Explicit controller_mut still revises policy and invalidates a finite request's prior binding. The query does not drain pending captures, make a continuation quiescent, or grant additional output allowance. This semantic bridge is independent of unfinished complete facade managed-host/capture admission. A true query reports the controller condition without stopping core advancement or consuming pending input; the facade remains responsible for applying termination policy.


Ordinary native completion waits now make a complete owner-filtered retirement pass after the existing bounded progress observation. A settled current-thread owner cannot be indefinitely skipped by alternating 64-record rotations around foreign-thread owners. Polling, state commitment, errors, lease release and quarantine retain their existing rules; unresolved and foreign owners remain retained.


### Shared facade capture records

The ordinary and controlled token sources use the same completion-first `CapturedStepDelivery` path. Shared token and failure events retain their original frame and its protected custody. Both paths deliver settled terminal frames with no token, and preserve an original execution error if best-effort delivery also fails. Callback cancellation cannot clear cancellation or replace that error. Legacy raw-drain APIs remain available; a shared frame requested through them remains parked until the retained delivery API is used.

The closed `controller_is_complete` query remains unchanged: it delegates the controller's existing mutable semantic query without revising policy identity. It does not automatically terminate the core machine. The facade's existing committed-generation cursor applies its termination result.

No funded facade snapshot, intervention, partition, source admission or envelope accounting support is established by this delivery bridge. Existing rejection boundaries remain in force.

Controlled speculative prediction records use the same legacy/shared capture
carrier and preserve a producer's retained row buffer through the existing
`Session::advance` publication. Escaped record clones retain the original frame
custody without changing role, prediction position, cumulative usage or tentative
status. Their JSON representation is unchanged; decoded records are ordinary
caller-owned data. The raw original invocation producer now joins the existing
ProcessLogits phase with a paid shared frame/tensor bank, actual selected native
trace and nested completion. Its physical ordinal never substitutes for logical
role/position/disposition. A failed native call can deliver its already sealed
host evidence through the prepaid queue without settling or releasing native
Recovery. Observation failure stays Aborted; successful capture before a later
policy error stays Untracked, matching the ordinary capture callback. The first
execution error wins if best-effort evidence sealing also fails. The current
same-request continuation shares live
capture spending, as ordinary sampler snapshots share their collector; no fresh
request can import saved usage yet. The existing managed controlled entry now
constructs a fresh original C source from the caller's admitted declaration and
installs a paid shared record queue before advancement. Its continuous observed
wrapper uses the same continuous-step helper as ordinary speculation. Source
copying preserves admission identity, schedules and limits; it does not adopt
caller buffers. Full/Slice/Preview, TokenScores and TopCandidates on the selected
F32 sampler-input profile use the same numerical occurrence and funded delivery.
Scores preserve requested ID order, full-vocabulary normalization and exact
pre-force domain flags; candidates preserve the shared sort/readout order.
Snapshot/restore/fork preserve the same live source, queue and cumulative ledger.
Other transforms and internal activation producers still require their own
complete original mechanisms.


The new neutral capture source channel is an ownership prerequisite shared by ordinary and controlled execution. It does not change observation scheduling, selected prefill support, advancement, frame delivery, or the native gate. A source/parent rejection preserves the channel; only a later canonical successful chunk integration may remove it after exact native settlement. Aborted logical frames and spent host/ledger claims remain nonrefundable even when a previously successful chunk can eventually retire its own physical capture sources.

Persistent host prefill capture targets are an additive prerequisite shared by ordinary and controlled capture banks. No production selected-prefill gate is removed. Zero initialization does not establish captured success: every contributing fragment must finish exactly once and every canonical chunk must pass host coverage checks. Partial targets move into the existing aborted-delivery owner on cancellation/failure, retaining original H without refund or implicit native completion. The future observer integration must preserve this same transaction and delivery engine.

The prefill fragment axis/program prerequisite changes neither ordinary nor controlled generation. It supplies shared geometry and the existing MLX selected-operation sequence only; it does not create a capture claim, install a collector, complete a chunk, or publish partial evidence. Selected full-prompt capture through bounded chunks remains unfinished until original host targets, canonical source retirement, readout and semantic attribution are joined in the shared driver; controlled execution must consume that same integration.

The shared capture bank now has a private short-fragment transfer prerequisite for ordinary and controlled prefill. Source joins roll back only before constructor success; later failure retains the complete original-scope source pins and the partial host target. Typed errors remain intact, and host finish never certifies or retires native work. The canonical stamped public wrapper and actual selected-prefill execution/readout integration remain unfinished; no managed capture gate changes in this unit.

The additive MLX retirement result does not change ordinary or controlled execution, cancellation, saved-state behavior, or delivery. It creates no chunk-success ticket, certification, new scope, retry engine, or budget refund. Future capture cleanup must consume canonical exact chunk/scope completion and route retirement rejection through the existing all-rank readiness protocol while retaining original error/recovery owners. Those integration obligations remain separate from this native mechanism.

Ordinary and explicitly advanced prefill use the same canonical source-retention callbacks. A successful chunk’s ticket is delivered at post-chunk cancellation readiness before another chunk or output escapes; final host capture stays provisional until the existing outer finish. Registration/ticket Drop preserves unresolved source pins, and retirement rejection aborts the logical frame without refunding host or logical capture claims. No selected split-prefill gate is changed, and no success ticket is produced if the model, exact commit check or a required completion guard fails.

Ordinary and explicitly advanced prefill share the token-aware boundary: cancellation requested by the final retirement callback is agreed before output delivery and causes the provisional capture frame to abort without refund. Legacy executors retain their boolean callback through the additive default; cancellation after the completed vote retains existing boundary semantics.

Actual-source fragment preparation and native selection now share the existing capture worker, with a canonical stamp required for public short-target construction. The focused Metal cases cover source/selection behavior only. This does not activate assembled p0 capture, establish architecture causal row equivalence, alter prefill readout, or claim uninterrupted/controlled selected-prefill parity. Those paths still require original admission, the native recovery carrier and real multi-chunk completion/cancellation tests.

Nonblocking scope entry requires both nonblocking runtime admission and nonblocking native bookkeeping publication. The native registry prerequisite removes only the guarded first initialization. It does not solve failed-rank communication, certify scopes, or authorize an unretained failure vote; terminal communicator fencing and failure abandonment remain separate required work.

Nonblocking try-begin is a prerequisite only. Busy establishes that no scope or work was entered; success creates an empty scope, not admission or earlier completion. An owner-preserving begin error cannot authorize a failure collective. Existing blocking callers remain unchanged; terminal communicator fencing, failure abandonment and distributed caller integration are still required before activation.

The native capture carrier has actual SessionPrefill lifecycle coverage using an original admitted bank, exact native scope and runtime-issued chunk registration/success ticket. The tests use one full Sequence chunk and a pre-settled external observation through an existing whole-tensor claim. They cover exact/-1 admission, cancellation before bootstrap, original typed failure, retirement Busy, final publication rejection, escaped frame/native aliases and unlocked payload retirement. They do not establish selected split-prefill capture or facade parity. Fragment ingress remains private without a production callback until shared p0 progression, architecture row/readout contracts and the original full physical span quote are connected. No managed gate or reduced-peak assumption changes.

Selected p0 progression now has a shared, allocation-free logical mechanism and a private original-H host adapter. A missing expected zero/nonzero hook prevents chunk advancement; a duplicate hook, failed attempt or abandoned attempt cannot restart the row or refund quota. Legacy/decode one-shot selection and public host-target construction keep their prior behavior. Full logical block-FP8 creation is charged once per accepted selection, while Preview(0) still requires each nonempty physical factory. Actual source validation, per-hook factory sharing, physical tracing and original-step fragment dispatch remain separate integration work. The geometry-only Sequence declaration does not prove causal equivalence, and selected readout still needs an explicit physical-readout/returned-output contract.

A canonical native communicator can now be monotonically fenced against new submissions without a replacement vote, native buffer, wait, or completion claim. Repeated cached initialization, aliases and split descendants observe the same fence. Marking does not cancel already-entered transport, settle a request, issue a prefill ticket, or permit capture delivery. Selected strategy binding and shared ordinary/controlled guard-failure propagation remain separate integration work; only native primitive/source tests are included in this stage.

The same retained causal/readout companion is intended for ordinary and controlled preparation. Each candidate binds the actual physical InferenceGeometry output before quote/admission: body-only may keep LastPosition/StateOnly, while coupled readout or vocabulary rows require Sequence. No accepted request is upgraded. A canonical ordinary-text causal declaration excludes supplied masks, media/proposal invocation and activation intervention semantics. Generated projection input preserves its selected factory and quota even for Preview(0). This foundation leaves the managed split-prefill gate and physical delivery driver unchanged.

Prefill observers that request opening state receive the current local state before each source preparation, including uneven chunks after a cached prefix. The existing preparation guard, original request, upcoming epoch and input agreement remain authoritative. Initial cancellation skips the callback; callback failure prevents source preparation and retains the original typed error. A partial state traversal is an error, never an empty or complete inventory. Existing observers keep the prior retention callback, including the new callback's default fallback. This adds no second controlled-inference driver and no native completion assertion.

Retained span diagnostics are additive to the shared ordinary/controlled quote planner. Existing unsealed constructors preserve their admission numbers and cannot return an accepted span association. Opt-in binds the same actual candidate to its successful reservation; equal geometry or equal totals do not bind an independently sealed candidate. The schedule includes every original prefill and conservative decode span. No controller callback, admission gate, execution path, termination rule, readiness vote or completion behavior changes. Future managed chunk integration must opt in for all relevant work, not treat this as an inherent capture-only feature.

ordinary and controlled execution continue to share canonical SessionPrefill ticket retirement. The private funding foundation never reopens an active account on Busy/error/unwind or an abandoned association. Only exact source-parcel handoff clears the exclusion atomically; independently retained storage remains physically charged until final owner retirement. Existing host holds may be consumed without changing unheld headroom. This does not enable selected split-prefill capture.

### Original-account span metadata ownership

Ordinary and controlled preparation can consume the same sealed accepted span quote into a compact owner under their existing original funding run. This adds no inference driver or independent controller policy. Original plan aliases preserve full host P through account retirement and error/unwind; a losing duplicate retains the typed original quote. No native work is certified by host-only retirement. Actual ordinary/controlled installation and canonical pre-chunk opening-state activation remain gated until their complete original quote and native settlement integration is present.

### Prefill fragment transaction composition

The bound capture observer uses the same ordinary/controlled prefill transaction lifecycle. The first chunk claims p0 and resets its ledger once; intermediate commits retain the active frame, the final local completion seals it provisionally, and only outer `finish_prefill(true)` publishes a committed delivery after cancellation and retirement checks. Errors, missing or duplicate zero hooks, abandoned writers, and outer cancellation preserve spent quota and partial/complete payload custody in an aborted frame. Decode and unbound observer behavior are unchanged.

The new route requires the actual immutable causal/readout companion and fixed original geometry. It provides neither a second bank nor native execution authority. Controlled and uninterrupted activation still requires the same original numerical span proof, opening-inventory publication, exact carrier settlement, and completion-first drain. The selected-prefill native/public gate remains closed until that joined path is verified.

### Shared borrowed state-source callbacks

Ordinary and controlled execution use the same whole-state source contract and direct native layer callbacks. The callback borrows the currently owned arrays without cloning handles or changing a frontier, source identity, inference retention or capture ledger. Empty and stateless sources visit nothing. No new engine, snapshot behavior, native submission permission or capture gate is introduced.

The direct visit is distinct from a complete active opening inventory: shared paging managers, worker payloads and metadata retain their existing separate contracts. It does not make partitioned capture, partial-prefill capture or remaining-span admission supported. Those joins must preserve the original account and successful canonical completion before any reuse.

Ordinary and controlled callers that enter the shared SessionPrefill driver use the same opt-in mechanism opening-source boundary. The default false path performs no execution traversal and does not call the mechanism hook. The original preparation guard is already live, begin_prefill_chunk has succeeded, and the source has not prepared its next input. A mechanism failure retains its original ReplicatedTextSessionError::Mechanism variant and enters the existing input agreement/guard-settlement path. No extra vote, state mutation, completion claim or rollback of prior committed chunks is introduced. The callback borrows actual state and current execution rather than a pre-loop snapshot. Mechanism implementations remain responsible for resources retained on error or unwind; the existing guard-creation failure contract is unchanged. The neutral tests cover real uneven SessionPrefill progression, partial/incomplete visits, opt-out, and failure before later source/model work; they are not native collector or facade-admission conformance.

Native live opening-source collector foundation (activation remains unwired):
`MlxReplicatedTextMechanisms` can privately bind a request-scoped collector for
its canonical pre-prepare callback. It borrows the actual state and current
execution visitor supplied by the shared runtime. Four separate inventories
retain state/managers, host slot/layout tokens, weight-manager/checkpoint sources,
and execution arrays; duplicates retain their full physical backing and unknown
or partial coverage cannot become complete. No family/residency branch, model
reacquisition, publication, origin-health grant or new funding scope is introduced.

The private guard closes collection on drop without clearing pending inventory;
exact request/chunk/epoch checks protect the one-use successful handoff. Error and
unwind preserve completed domains and visited execution prefixes. Existing
Result-returning builders can discard their own partial temporaries on error;
those underlying owners remain with the original session/recovery, and this
foundation does not claim a complete replacement recovery inventory. Host-slot
tokens retain accounting custody, so actual state/table ownership remains required.

Fixed field/slot/guard/inventory construction controls have measured helpers.
They are not complete collector bounds: BTreeMap and Vec capacities, SourceStorage
reconstruction, actual state/manager traversal, temporary inventories, Rc allocator
overhead and original enclosing controls remain separate. These obligations,
original-account publication/health validation and retention alongside the actual
operation must be resolved before wiring the private binder. Prediction banks,
outer model/blueprint sources, controllers, copy/sampling joins and current work
roots are not authenticated by this four-domain collector. Existing manager locks
may block; this is not a new bounded nonblocking manager-inspection guarantee.
Managed gates, span debit exclusion, fragment activation, admission and installation
are unchanged. Seven source/ownership tests exercise the private collector; they
do not constitute canonical runtime-hook, managed budget or native transfer proof.

The additive bound-prefill executable quote route is shared cold preparation infrastructure, with no ordinary or controlled execution gate change. Its candidate/path validation and borrowed host plan do not grant a step, reserve another account, install a collector or authorize partial-prefill native work. Ordinary/control activation still requires the same original prepared source, physical geometry, funded span and completion/retirement contracts.

Original capture preparation now retains the same selected geometry and span plan for later ordinary/controlled use. Installing the private payload moves its bank, semantic companion, source-health witness and original plan together, once, after the existing run/request checks. Full sequence readout may be quoted for p0 hooks, including Preview(0), while body-only hooks retain last-position demand. This does not yet allow either driver to execute the selected-prefill route: current live-source publication, retained-control holds, remaining-span admission and native delivery must still be joined. Snapshot/copy gates remain unchanged.

Original text control custody is shared by the existing ordinary and controlled preparation/account contracts. The neutral accepted text receipt preserves original source, geometry, plan, reservation, and run identity; it does not rebind policy or create another engine/reservation. A compact control guard can outlive span record payloads and native certification, so physical owner retirement governs the P+Q hold. This increment does not enable capture activation or snapshot copying of a funded ledger. Native owner integration and completion/retirement behavior must be verified before either public flow claims the joined route.

Prompt, Sampling and inference use the same accepted native control custody in
ordinary and controlled captured sessions. Pending and installed capture owners,
the historical fixed quote after installation, and every original FundedWork
retain it through errors, quarantine, escaped aliases and native certification.
Certification never releases Q while the control object remains alive. Named
facts are sealed before the original reservation; no resumed or later operation
can replace them with another byte value. Existing unobserved and saved routes
retain their separate W accounting and cannot use this as a live-span proof.

### Native retained-leaf visitor parity

Ordinary and controlled native executions share the same audited leaf
retained-value adapters. Direct field visitation preserves lazy sources,
full backing aliases, and explicit incompleteness for opaque native storage.
Callback unwinding leaves the original module owning its fields; the visitor
does not create an intermediate owner or provide completion evidence.
This internal prerequisite adds no capture gate or selected-prefill support
claim and does not change frame, quota, or retirement policy.

Opening-source allocation inspection now reads fixed-size backing facts without copying tensor shapes. Contention returns typed unavailability, lazy backing remains unknown, and neither queued owners nor housekeeping callbacks are advanced by inspection. These facts do not themselves authorize chunk submission or establish completion.

Cache worker inspection can now return immediately when its retained-work registry is busy. Prepared and active tasks remain evidence of retained payloads until actual worker-resource release. The query grants no cancellation, completion or retirement authority and does not prevent new work after the observation.

### Shared checkpoint-source inventory contract

Ordinary and controlled consumers can use the same borrowed checkpoint
physical-storage visitor. Authorization views cannot discount hidden physical
payloads, and an incomplete child does not stop a composite from reporting
later known owners. Source errors retain their original type and leave owners
already retained by the caller intact. No frame, quota, native completion,
remaining-span marker or capture gate is activated by these source facts.

### Borrowed manager inspection and controlled execution

The private borrowed native storage visitors do not advance ordinary or
controlled sessions, settle work, retire resources or refund funding. Manager
mutex contention returns typed Busy; incomplete workers/transfers remain
incomplete and original callback errors preserve the retained caller prefix.
Existing public owned-inventory methods and execution gates remain unchanged.
Any future opening collector must couple these same physical facts to the
original request, price its slots and controls before admission, preserve source
health and pending recovery owners, and validate later manager mutation. The
visitor alone supplies none of those permissions for either shared driver.

### Source-bound retained-owner count diagnostics

The optional retained-owner slot diagnostics are shared by ordinary and controlled
preparation. Borrowed native state descriptors preserve the exact state/table
instance; immutable weight descriptors preserve the actual manager and source
roles. Active layerwise windows or unknown custom owners stay unknown even when
a current override count is known. No mutation, settlement, storage allocation,
refund or replacement source authority is granted by a scalar count.

Snapshots, restored managers and later topology replacement still require exact
original-source binding and growth/entry validation before an opening collector
can be activated. This change introduces count facts only; bounded publication,
control custody and the complete physical opening inventory remain separate work.

### Shared terminal-failure behavior during prefill

Ordinary and controlled inference use the same source, chunk, observation, and commit runner. A reservation-entry, phase-transport, successful-settlement, or unwind failure fences further advancement through that same session; it preserves the first phase and original cause rather than granting retry authority. Source preparation rejected through a completed negative agreement retains its existing pre-state-mutation classification. Both nested observation retention and outer chunk retention abandon unresolved work, and final provisional host records receive the existing abort notification exactly once. Previously committed chunks stay committed; a terminal failure does not fabricate an `Aborted` commit outcome or refund spent capture claims.

The explicit final-commit `Indeterminate { epoch, phase }` protocol is unchanged. Subsequent model advancement and reset check `ensure_commit_resolved` before new model work. Existing local prompt-cache save/load may preserve that exact unresolved outcome; they are explicit state-persistence operations, not permission to resume model execution. Distributed restore/load paths retain their existing resolved-commit checks. An indeterminate result does not itself become a new canonical terminal mark in this increment, but its error still abandons nested/outer retention rather than spinning in successful completion. Native multi-rank terminal-path validation and original communicator construction accounting remain explicit prerequisites, not architectural limits on controlled inference.

### Original pin-bank custody across controlled advancement

The bounded existing-only pin bank is extracted once from the consuming original
text-span owner. It consumes the same canonical SessionPrefill contexts and
stamped source scopes used by ordinary and controlled drivers; it introduces no
alternate advancement or completion engine. An empty group still retains its
original control custody, and multiple escaped groups or failed attempts retain
the complete finite S contribution until their last owner retires. Successful
pinning is neither native completion nor permission for the next chunk.

Only the existing canonical ticket transition can detach the capture segment;
Busy, errors and unwinding do not clear it or restore pin attempts. The closed
source publication transition below is separate from this existing-only bank;
bounded native opening collectors and their original installation remain separate
prerequisites. This mechanism enables no additional capture gate.

Retained-owner count preparation distinguishes absent architecture children from active loans: Option::None contributes zero only for a missing child in the actual immutable module topology. Resident missing-unit loans, bounded pending work, unknown modules and busy managers remain unknown. A later insertion or topology replacement requires revalidation before opening; these diagnostic changes do not activate a controlled inference path.

### Qwen causal declarations in the shared capture path

Qwen's dense and routed block factories now provide the same ordinary causal-row evidence to the shared path/selection machinery used by ordinary and controlled preparation. No second advancement or delivery driver is added. Readout-input and score capture still require original physical Sequence geometry; body-only declarations do not force it. A prefill/decode invocation's routed resource capacity remains independently bound to that invocation even though each row's expert equation is causal. Snapshot/restore budgets, interventions, partition capture admission, source custody and managed applicability gates are unchanged. The new neutral fixtures validate declared equations and exact selection binding, not end-to-end controlled native capture support.

Ordinary and controlled sessions share the repaired outer boundary observation order: original value, optional intervention, then the effective value that subsequent work consumes. An effective-input observation error stops before the unit changes state and uses the existing failure and loan cleanup path. Prepared traversal retains the effective names during cold preparation, so this repair adds no path construction inside bounded execution.

Controlled and uninterrupted observation use the same declared original/effective unit boundaries. Effective points are read-only evidence after the single existing intervention; they are not additional intervention targets. Architecture-owned Muse/V4 hooks remain single emissions with their actual axes, and repeated decoder passes retain their logical invocation identities.

Prepared ordinary and composite partitions now report traversal-owned outer
`.input.effective` and `.output.effective` values at the same shared unit boundary
as resident traversal. Ordinary and controlled session drivers consume that same
executor; effective reporting is read-only and observes the value actually sent
to the unit or next boundary. Architecture-owned hooks remain suppressed by the
existing ownership declaration. Observer errors follow existing agreement,
rollback and terminal classification; no extra completion vote or retry authority
is introduced. Managed partition capture remains subject to its existing gates.

### Original capture source readiness

Ordinary and controlled preparation consume the same accepted one-key C
publication transition before installing capture. A pending owner cannot become
an ordinary text control guard, accepted span view or pin bank until its exact
source registration and complete known S+C health join succeed. Earlier plan and
source aliases retain original custody through failures. No callback, completion
assertion, capture quota refund or alternate inference engine is added.

Typed source reuse validates the existing private source owner; opaque legacy
attachments have an explicit typed incompatibility result instead of being
assumed healthy. Both drivers use the same terminal host-only accounting:
healthy run/native closure releases unused transient headroom while original
host holds, live registrations and the original ceiling survive. Quarantine or
an active span prevents trimming. Successful capture-plan publication
is not native settlement, complete opening publication or permission to activate
remaining-span execution. Additional collector/installation joins remain required.

### Original custody for private opening collection

The fixed native opening capsule uses the same original accepted text-control
binding and reservation for either ordinary or controlled composition. It does
not add an inference driver, expose controller state, or enable a capture route.
The present API borrows the actual state/execution and thus cannot escape their
lexical owner or be rebound to a saved/replaced runtime. Persistent installation,
snapshot/fork rebinding and canonical capsule retirement remain later work.

After a source failure, native inspection Busy/unknown result, or panic, the
caller retains the capsule and every installed owner prefix. Reuse is rejected;
no completion, source publication, rollback or budget refund is inferred. A
failed pre-inspection buffer allocation returns the same sealed plan while its
original aggregate hold remains live.

The scope purpose is assigned only by existing closed constructors; it is not a
completion assertion or a user-selectable resource policy. Ordinary and controlled
aliases retain identical exact host/source custody, and no certify call releases
live host payloads or resets consumed capture, copy or transport quotas.

If accounting is poisoned, both drivers retain conservative original funding
through run, metadata and physical-owner cleanup. A host-only tail is trimmed
only after healthy completion; catching an accounting unwind does not authorize
reuse of that headroom.

### Shared quiescent inspection with execution

The ordinary and controlled paths can borrow the same retained session tuple
through `inspect_runtime_execution`. The existing inspector delegates to this
one check: control fences and indeterminate commits reject first, followed by an
active epoch, before invoking the inspection callback. Existing typed callback
errors and rejection precedence are preserved. A shared returned borrow excludes
concurrent mutable session advancement; no test-only mutable session access is
introduced to manufacture an active borrow. Neutral tests cover actual cached
steps, partition-local inspection, and real indeterminate/fenced callback
exclusion. This inspection does not establish native completion or activate
bounded capture through either generation driver.

### GPT-OSS and K2 row-contract scope

Ordinary causal GPT-OSS and K2 dense/routed rows now share the existing architecture declaration path used by capture preparation. This changes no controlled/ordinary driver, capture permit, callback schedule or capability gate. Full/uneven-prefill and three-decoderow comparisons check the same declared hooks and retained state; partition/routing intervention, explicit masks, media and tentative proposals are not admitted by this declaration. Managed selected-prefill activation still depends on the common canonical frame/chunk, original account and complete source/control proofs.


### Selected-policy retained visitor reentry

Ordinary and controlled inspection use the same selected-policy retained visitor.
Its existing mutex is acquired without waiting; a nested callback through the
same policy or a cloned policy handle sees incomplete evidence rather than
blocking the owning thread. Poison also remains incomplete. This changes no
execution/acquisition locking, transaction state, completion, budget or gate.
The actual selected enum contains thread-affine native recovery and is not
`Send`; tests therefore exercise real owning-thread contention and callback
reentry without manufacturing a cross-thread policy. CPU native fixtures cover
resident and bounded nonzero owners, aliases, poison, unwind prefixes and an
acquired resident unit while preserving the remaining available inventory.


### Qwen hybrid target row declaration parity

Ordinary and controlled preparation consume the same Qwen hybrid target-row
declarations; execution continues through the existing shared drivers and
unchanged recurrent/KV state. The declaration does not alter the current MTP
observation availability gate or enable conditional/media, proposal, masked or
intervened row assembly. Native bounded collection, original funding, complete
opening inventory and partition capture support remain independent work.

LFM2/LFM2-MoE causal row evidence is for ordinary intervention-free target text
invocations. Controlled and uninterrupted drivers consume the same retained
source/path declarations and original candidate geometry. Body observations
precede readout and retain all physical rows under StateOnly or LastPosition;
coupled readout input and vocabulary scores require an originally selected
Sequence geometry. Binding never upgrades an already accepted LastPosition
request. Preview(0) still binds the same hook and physical readout requirement;
generated-source creation accounting and existing capture/source/session gates
are unchanged. This declaration/width-one repair does not activate managed
partition capture or alter MTP/media availability.

### Nemotron-H target capture semantics

The ordinary Nemotron-H target path now supplies causal row/readout declarations for its real embedding, outer target unit and output hooks. The declaration does not enable a controlled capture route: original accepted source/path binding, physical readout demand, span funding and the existing native gates remain required. Appended MTP groups retain their distinct state and existing proposal/observation availability; no target declaration is inferred for those invocations. A width-one Mamba convolution uses zero carried convolution history while preserving its recurrent matrix, and cached mask construction selects the attention state of the actual invocation.

Neutral conformance compares a cached prefix followed by uneven chunks and repeated decode against full-row execution, including every declared real callback and complete fixed/KV state. It separately checks body StateOnly/LastPosition behavior and rejects binding coupled readout capture to LastPosition. Prepared resident, host-windowed and disk-streamed TP/PP cases use a Dense/MoE-only first pipeline stage, retain zero positions there, and compare downstream cached attention results and owned body callbacks after bounded rank work joins. These are source-staged tests pending central execution, not a native multi-rank or released-checkpoint validation claim.


### Retained opening snapshots are not session readiness

The private existing-only opening-pin adapter can run from the shared original
SessionPrefill callback, using its real request, scope and source stamp. Its
completed snapshot retains the earlier inspected physical owners and exact
path-source alias; intervening state mutation does not refresh those owners.
The snapshot exposes no current-runtime prepared-binding or readiness proof and
does not enable ordinary or controlled capture dispatch.

Tests retain the snapshot outside callback failure and panic, preserve the
original typed pin error, prohibit a second attempt, and keep original host
custody through terminal trim. Callback failure itself is not native completion:
test cleanup separately synchronizes and retires the submitted native guards.
The eventual native gateway must join current pairing, complete inventory,
publication and canonical retirement in the existing shared driver before
claiming controlled/uninterrupted parity for that route.
A canonical source parcel can retain one exact installed opening group in addition to capture transform sources. The legacy no-ticket retirement helper rejects such a group; matching ticket take removes it with the existing span marker atomically. Backends retire actual payloads before the parcel outside their locks. Failed retirement and uncertified scope teardown retain the installed group in the same original quarantine. Explicit provider certification and the separate P-only accounting primitive retain their existing contracts; joined P+Q text activation requires its matching opening group. This neutral custody mechanism does not enable controlled or ordinary native capture paths by itself.

Finite capture-source publication rows are spent from one original accepted bank. Canonical SessionPrefill context, current stamped slot, exact account and all retained origins are checked before issuance/commit. Issued attempts remain caller-owned and terminal across Busy, typed failure, provider panic or abandonment; there is no replacement row or later allocation grant. Whole-batch mutation is atomic and comparison-free inside Usage after its first mutation. Key cleanup occurs outside Usage before opaque per-allocation outputs can escape. Successful registration does not itself settle native work, attach native owners, certify completion, or authorize a remaining-span execution path.

### Muse-Glimmer causal ordinary text observations

The retained Muse ordinary-text companion distinguishes body rows from physical coupled-readout rows before original candidate admission. Body-only selection preserves StateOnly/LastPosition behavior; readout input and vocabulary-score selection require the same original physical Sequence demand. The prepared composite adapter forwards the declaration to shared controlled/uninterrupted mechanisms; it does not add a second driver or open the native split-capture gate. Nonzero neutral fixtures compare every real original/effective callback and complete cached KV across uneven chunks and repeated decode. Media input, caller masks, interventions and external DFlash proposal equations remain outside this ordinary row proof, with their existing support policy unchanged.

The prepared native backing-owner handoff is shared native storage machinery,
with no ordinary/controlled generation branch. It does not advance a session,
certify completion, open a capture gate or grant another reservation. Successful
attachment consumes a prepared owner once; Busy/errors preserve it, while earlier
successful attachments remain with their actual physical backings. Initialization
waits belong to cold preparation only; bounded handoff never invokes the Once.
Retirement queues a prepared Rust owner only after backing and native node release;
queued and pending Rust nodes are freed before their accounting payload drops. Closed
ordinary and controlled opening publication must use the same original accepted
capacity and recovery owner when this primitive is joined. The staged native
mechanism tests alone establish no end-to-end capture support.

### DeepSeek V3 ordinary prefill row evidence

V3 supplies its ordinary target row contract to the existing shared controlled and uninterrupted drivers. BeforeReadout selections preserve StateOnly and LastPosition execution; selected normalization/projection inputs and vocabulary scores require physical Sequence readout before candidate binding, including Preview(0). Five neutral tests cover all eighteen actual hooks in a two-layer fixture, a cached prefix of two positions, a five-position continuation split 2/1/2, and three cached decodes. They compare every compressed latent/rotary row and offset, retained block-scan mode and the complete state layout, and explicitly reject unexpected KV/recurrent/pooling state. No native split-capture gate or second execution driver is introduced.


DeepSeek-V4 target row declarations use the shared retained capture selection
for ordinary and controlled callers. Body hooks preserve all chunk rows before
StateOnly/LastPosition selection; complete stream-collapse and score capture
requires an originally admitted physical Sequence demand. MTP groups, DSpark
context/proposal execution and the existing observation/intervention capability
policy are unchanged. In particular DSpark's bidirectional proposal mask is not
an ordinary target causal-row claim. No native capture or funding gate is opened
by the declaration. Neutral conformance covers real original/effective/readout
callbacks, prefix and uneven chunks, complete persistent pools and three cached
decodes; native, distributed and released-checkpoint comparison remain separate
validation obligations and are not inferred from this source-only package.

### Kimi Linear causal target rows

The direct Kimi Linear target's ordinary row companion binds the original capture
source and prepared path owner for fixed KDA, compressed MLA and mixed state
profiles. Body observations run before readout selection and retain every input
row under `StateOnly` or `LastPosition`. Readout-input and vocabulary observations
require original physical `Sequence` demand; this does not upgrade an accepted
request. Tests compare a populated two-token prefix plus a five-token continuation
run whole or as 2/1/2 chunks, then three cached decodes, including all fixed
convolution/recurrent and compressed latent/positional state and scalar cursors.

Width-one KDA remains stateful through its recurrent matrix while requiring no
convolution history. This is the same pointwise convolution equation at kernel
width one, not a generic relaxation of missing-state errors. Existing Kimi
no-positional-MLA validation, MTP rejection, media/proposal/intervention policy,
snapshot/restore policy and native capture gates remain authoritative. Selected
replicated wrappers require their separately reviewed explicit causal opt-in;
this direct-family change does not claim their activation or a second controlled
inference engine. Numerical and source-binding cases are staged, not executed.


Selected fixed/compressed ordinary sessions preserve the family's explicit causal
row declarations across the existing resident, host-layerwise and disk-streamed
owners. The neutral selected-owner fixtures compare a cached two-token prefix,
a five-token continuation versus chunks 2/1/2, and three cached decodes, including
every declared real outer/effective/readout callback. Complete mutable state is
compared across runs at the prefix, completed continuation and each decode;
intermediate chunk snapshots check the actual frontier and nonzero state. The
first prefix uses prepared observation, preserving the original path token.
Body-only observation retains StateOnly/LastPosition behavior before readout.
These are ordinary session equation/source tests; they do not activate a native
finite-capture gateway or establish a new admission/resource proof. Kimi's selected adapters now forward their explicit causal fact, contingent on
the separately staged direct recurrence proof and selected tests passing.
Partition row binding, the remaining undeclared families, media, interventions
and speculative/proposal execution still require their own contracts.

### LFM2 width-one state and cached masks

A width-one convolution has no persistent token state: its exact NoState position
and fixed offset remain zero. Logical request advancement belongs to the shared
session driver; attention masks read actual attention cache positions. Selected
ordinary Stateless and KeyValue profiles therefore execute the same equation
without relaxing either profile's rejection of fixed-state operations. The local
mask helper bounds its search to the invocation segment in the supplied exact
StateLayout; stateless leading/trailing pipeline cuts cannot create an offset-zero
mask for another rank's attention.

The selected width-one tests retain real original/effective/readout callbacks,
Full/Preview source binding, StateOnly/LastPosition behavior and full versus 2/1/2
continuation after a two-token prefix, followed by three decodes. Complete state
is compared at prefix/final/decode as in the shared selected fixture; intermediate
snapshots check frontier/nonzero state. Existing direct LFM2/LFM2-MoE cases keep
their complete per-chunk state comparisons and now require exact zero NoState
frontiers. The prepared partition regression checks actual owned body rows,
returned predictions and each local NoState/attention position. Native capture,
intervention, MTP/media, funding and snapshot policies are unchanged.

### Shared completed-prefill inventory boundary

Controlled and uninterrupted prefill continue to use the same SessionPrefill driver. Opted-in mechanisms may inspect the actual current execution/state at both the guarded opening and the post-completion, pre-retirement boundary. The provided legacy callback behavior is unchanged. A requested prepared-token check runs before collection; the callback lends the session's actual token and grants neither persistent traversal authority nor native completion. The existing readiness/cancellation agreement handles its typed error. Panic unwinds through the existing terminal fence and abandons recovery; it does not mint another ticket or certify unfinished work.

The private native row installation is available only for an actually selected p0 hook. Empty, decode-only and skipped p0 schedules do not receive an orphan row bank or synthetic bootstrap. Original logical capture quota and the single full target remain in the existing shared capture policy. Final native owner retirement can precede a freshly observed cancellation; the provisional shared frame then aborts without refunding capture or inventing a second execution engine. A failed late handoff retains both the published prefix and original segment marker. The facade/native activation gate remains closed pending the remaining original-span and enclosing-owner obligations.

### Gemma4 ordinary prefill row evidence

The ordinary Gemma4 target uses the existing causal-prefill declaration contract for the same parameters, causal prefix and ordinary text input. The shared execution driver remains unchanged. StateOnly and LastPosition retain every body row; selected readout or vocabulary observations retain their existing Sequence requirement. `readout.linear` precedes final softcap and `model.logits` follows it. Media, custom masks, interventions and speculative assistants/proposals remain outside this declaration.

New neutral fixtures stage a two-token cached prefix, continuation five as 2/1/2 and three identical subsequent decodes. They compare complete persistent KV/cursor state at every frontier. The numeric cache's per-call attention-history scratch can have different width after a sliding-window chunk: tests check its exact invocation-derived extent and every value against its absolute-position reference slice, then require entire histories to agree after the same next decode. Ordinary optional PrefixEmbedding remains absent and is checked as such. No mutable field is silently omitted to force full/chunk equality. These tests await central execution and do not replace released-weight or native validation.

### Gemma4 bounded sliding-history correction

A multirow submission after a bounded sliding prefix can expose fewer keys than its absolute sequence endpoint. The previous automatic prefix-wide mask therefore failed to broadcast for both native contiguous history and the neutral numerical cache. Gemma4 now obtains mask-free sliding causality from the existing window-aware mechanism or the cache-owned scan. Explicit masks, full-attention masks, shared-cache ownership and caller rotary positions retain their separate contracts.

The correction preserves every existing direct, prepared-source, body/readout and partition numerical/state assertion. Four added tests cover nonzero publishers and shared consumers with W-minus-one persistence, cache-owned invocation history, cached prefix two plus 2/1/2 continuation and three decodes, explicit masks, uncached prefix/window causality with nonuniform explicit rotary embeddings, and actual prepared workspace descriptors. Global NumericCache retention is unchanged. These source-staged tests require central execution; they do not establish native or released-weight validation.


### Complete-unit partition observation ownership

The shared architecture placement compiler resolves directly bound tensor/hidden observations for actual target decoder invocation nodes. This repairs missing outer input/output ownership in prepared composite partitions while preserving the already emitted original/intervention/effective callback order and all state behavior. All TP/EP execution replicas retain their full hidden-axis mapping; shared physical passes retain distinct logical identities. Prediction/media applicability and capture admission are unchanged.

Focused neutral placement tests cover Gemma and Inkling composite declarations, Nanbeige shared passes, V4 stream axes and conflict/foreign-owner rejection. The existing Gemma causal TP/PP/residency test and queued Conditional Qwen TP/PP/EP tests keep their full callback, nonzero value and complete-state assertions. This source increment does not claim those execution checks have run; central validation is recorded separately.

Gemma4 partition observation comparisons use `atol=1e-4, rtol=1e-4` for F32 partial-sum rounding through the eight-layer residual path, with strict shape, dtype and finite-value checks. Direct full/chunk execution, selected public scores and complete state retain the shared `atol=2e-4` comparison. Two identical diagnostic runs covered 18 worlds, 84 rank sessions and 263,952 compared values: maximum observed activation error was 0.00074482 (normalized by one plus reference magnitude: 0.000064645), versus 0.000007235 for public scores and 0.000007272 for state. Every expected observation row and state field remains checked. Temporary measurement instrumentation was removed before final conformance validation.


Selected Kimi conformance uses the same shared prepared-session fixture: a
nonzero two-token prefix, five-token continuation whole or in chunks 2/1/2,
then three cached decodes. Full state comparisons cover every declared fixed
q/k/v convolution history and recurrent matrix, MLA latent and unrotated
positional channels, payload geometry and scalar cursors at
prefix/final/decode boundaries; intermediate snapshots assert actual frontier
and nonzero state. Width-one KDA has no fabricated convolution history.
StateOnly and LastPosition retain the existing body/readout policy. This is an
ordinary shared-session numerical proof, not native capture admission or new
controlled-session execution. Existing snapshot, partition, media, MTP and
intervention contracts are unchanged. Central execution remains pending.


### Private captured opening admission continuation fixture

The private opening-row successor exercises actual `TextGenerationDriver` continuations from original `admit_with_capture_opening_rows` through context binding, installation and four predictions. Read-only quiescent boundaries verify the same original bank/source/account; attempted bank reuse rejects. Empty, decode-only and prediction-zero-skipped plans follow the shared inactive predicate and allocate no row bank. Empty sources drain without frames or frame claims; other schedules retain one logical frame per prediction. This is a test-only route selection with unchanged production gates, pending centralized native execution.


### Installed funded prefill observer binding

Controlled private capture installation now lends the original selection together with its collector into the shared funded prefill observer path. Exact source/geometry validation precedes native work acquisition, and ordinary native owner, scope and stream validation remains shared with funded decode. The bound observer uses the already admitted physical output demand, so a pre-readout observation does not independently demand sequence logits. The same original bank, row schedule, quota and terminal ownership remain in force.

The existing private-entry regression exercises nonzero prefill capture followed by three decodes, all three weight routes, full versus one-position prefill, exact original quota, and empty/decode-only/skipped-prefill cases. This correction adds no synthetic observer events or test-only fact bypass. The ordinary admitted source gateway still accepts `LastPosition`; selections whose actual physical readout is `Sequence` continue to meet that existing typed rejection. Supporting that additional gateway is separate work, as are complete outer inventory and public eligibility. No execution result is claimed by this off-tree source package.


### Inkling ordinary row continuation

The staged Inkling tests compare a nonzero prefix of two tokens followed by a
five-token continuation whole or as 2/1/2, then three cached decodes. All nineteen
actual original/effective/readout callbacks compare. Every intermediate chunk
state is also compared with an independently advanced same-prefix reference;
all persistent KV values, offsets, windows and four convolution histories are
checked, including local head overrides and dense/routed/shared branches. Kernel
widths one, three and four exercise zero, short and initially padded histories.

Body hooks execute before readout selection and preserve StateOnly/LastPosition;
readout inputs, including `readout.scaled`, require physical Sequence. The
prepared composite and direct text owners compare actual callbacks and complete
state across the same seven invocations. Prepared TP2, PP2 and combined TP2/PP2
runs span resident, host-layerwise and disk-streamed policies, with full local
state snapshots after each call and observations matched to actual component
ownership. KV/key/value convolution shards follow the actual local head range;
hidden-width residual histories remain replicated. Rank results join before
numerical comparisons. All physical observation rows, including model.logits,
compare before public output selects its final prediction row. The separate
public-return assertion still checks that final row.

A separate two-depth MTP width-one regression checks the shared history-free
DecoderLayer, persistent full/sliding KV and repeated cyclic depth use. It does
not grant ordinary target row declarations to MTP. Media, intervention, proposal,
restore/fork and managed-capture admission policies are unchanged. The existing
cache-owned relative-attention fixture remains distinct: the generic numerical
state does not implement that override, and its unsupported reason is not
suppressed. These tests have not been executed in the source package; central
checks and existing released-checkpoint/native validation remain separate.

### Missing auxiliary convolution state roles

Missing declared convolution-history roles preserve their original typed `StateError::UnknownComponent` across numeric, workspace and MLX adapters. The same role lookup still rejects before the requested convolution can run; earlier operators in a layer may already have changed their own state, and this error-source correction does not promise layer rollback. Existing kernel-three/four Inkling tests retain all four independent missing-role assertions. Focused workspace/native bridge tests retain the exact role through error cloning and state-owner destruction. This is error attribution only, with no gate or execution-policy change.


Qwen hybrid kernel width one keeps the ordinary recurrent frontier across full
and chunked prefill, cached decode, and target/draft/target use. The bounded
correction's neutral fixtures cover actual direct, selected, parallel and
prepared conditional owners, including full-attention MTP draft state without
granting MTP observation hooks. No controlled driver is duplicated. Compilation
and numerical validation of these added fixtures remain pending in this staged
increment; the prepared conditional tests are state-regression evidence, not a
conditional causal-capture declaration or native gate activation.


The Qwen3-VL zero-section mRoPE regression uses nonzero actual checkpoint payloads
through the prepared composite input wrapper. Dense and routed text compare a
two-token prefix, a five-token continuation split 2/1/2, all persistent KV and
position-delta fields at each frontier, and three cached decodes. This validates
the existing rotary/state path, without a new controlled driver or conditional
capture declaration. These newly staged tests still require central execution.


Conditional Qwen ordinary-text row proofs use actual admitted composite inputs
and the existing prepared and partitioned session drivers. Nonzero dense/routed
fixtures compare a cached two-token prefix and five-token continuation against
2/1/2 chunks, each complete state frontier, and three subsequent decodes. Body
rows are checked under StateOnly and LastPosition readout. Prepared TP, PP,
TP+PP and routed TP+PP+EP fixtures use the actual component coordinate maps and
resident, host-layerwise and disk-streamed policies. This adds architecture
declarations and conformance tests; it introduces no separate controlled engine
and does not open the native capture gate. Newly staged numerical tests require
central validation; no released-checkpoint or native distributed result is claimed.


Ordinary final-output consumption now discards intermediate score owners at the
same settled chunk boundary used by explicit advancement. The shared driver still
executes required physical Sequence projections for observation consumers and
preserves final output indexing. Neutral tests track real score-owner drops before
next submission, compare nonzero recurrence and three decodes, and cover cancellation
and later submission failure. No other retained-input, native entry or managed
memory gate obligation is completed by this release fix.


### Retained committed-sequence storage prerequisite

The core committed-sequence algorithm supports one non-Clone retained mutable
owner in addition to its unchanged default Vec mode. Retained preparation is a
consuming local result with owner-preserving errors. It must eventually be
resolved inside the existing first readiness call before `next_token`; this
prerequisite adds no hook, vote, prediction or public managed route. Cancellation
can freeze an empty prefix without materializing token slots. Terminal freeze
moves an immutable owner into `GenerationTokenIds`; cloning or partially consuming
that result retains the original supplied owner.

Legacy constructors, Clone, logical snapshot estimates and Vec extraction remain
available only on the default sequence. Controlled/uninterrupted callers and the
public `GenerationOutput` field retain their current implementation in this unit.
The later facade migration must preserve the callback-cancellation override and
stop/grammar/EOS/max precedence. It must preflight managed snapshot/restore/fork
before host or native copying, using a separately admitted destination mechanism;
this unit supplies neither that copy mechanism nor completed controlled parity.
Sequence-only original preparation must also avoid adding an Instrumentation
vote while preserving the old capture and explicitly empty legacy-options phases.

### Record readiness and structural scope lifetime

MLX can observe whether entered records in an original submission scope and its
durable descendants still lack their existing terminal transition. This cold
query excludes only structural empty-child accounting. It does not seal or
finish a scope, progress a record, cancel communication, detach capture pins,
release backing owners, or make a subsequent mutation safe. A healthy quiescent
record observation can coexist with pending ordinary scope lifetime; a failed
or blocked quiescent observation remains failed or blocked. New unrelated
scopes cannot observe earlier work: callers must keep the actual original
ancestry/preparation handles, and protect against future submission separately.

The wrapper exposes no conversion from record observations to completion
authority. Controlled/uninterrupted drivers and their existing completion and
error recovery are unchanged; this foundation alone activates no inference path.

Ordinary, controlled and detached core text preparation now share an opt-in
borrowed sequence request. Extraction occurs after run binding inside the existing
Admission operation, before its readiness vote; existing None/empty/capture option
phases remain unchanged. The retained owner is taken once, and the legacy detached
copy/composition boundary rejects retained mode even after extraction until an
admitted destination-copy mechanism exists. This is an unfinished managed-copy
prerequisite, not model inapplicability. The facade must preflight before its own
earlier copy work and materialize token slots before its existing first finish_step
Delivery agreement, preserving delivery-error and cancellation precedence. No
second token engine, readiness vote, result-field migration or native gate is added.
New neutral lifecycle/order tests still require central execution.

## Moshi selected canonical-frame numerical fixture

`reference_numeric::moshi_frames` drives the existing `RealtimeSessionScheduler`
and `execute_realtime_frame` with nonzero selected source payloads. A two-frame
prefix, five-frame continuation and three further frames use one actual session.
Ordinary turns and caps 2/1/2 compare complete temporal/depth K/V, offsets, resets,
state layout, schedule, delayed coordinate payloads, sampler/RNG state and outputs
at every committed frontier. A turn cap bounds scheduler work; it does not pack
multiple frames into one model call. This serial transactional fixture permits
one branch per request, so each committed frame has physical width one.

Callback failures after temporal work or a depth slice never reach completion
creation and publish no third frame. Cancellation before the next frame invokes
no model callback. The scheduler removes terminal canonical state: the fixture
checks its actual state at Drop, the last successful full-generation snapshot,
exact completed-work count and absence of retry. It does not claim a live terminal
generation or cover pending native completion, snapshots, release/resume or fork.


### Retained sequence disposal

Ordinary and controlled retained sequences keep the same startup, materialization, commit and readiness lifecycle. Disposal now calls the actual provider's owned retirement operation from dormant, partial-error, prepared and immutable-result owners. Construction failure returns a closed owner that supports read-only inspection and consuming revalidation; it no longer exposes the raw Box. Preparation cause disposal remains ordered before its sequence, including unwind. Freeze still hands the original Box to the consuming provider once, which must preserve allocation custody on its own success and panic exits. Semantic fixtures cover concurrent final token aliases, partial owning iterators and callback failures; they establish no runtime funding or facade output guarantee. Legacy sequence/Vec behavior is unchanged.


### Span custody final-owner handoff

The original full and raw host-custody populations now retain their accounting payload through a consuming final-owner handoff instead of ordinary Arc field destruction. Concurrent aliases all use Arc::into_inner; no external Weak can retain either allocation afterward. Existing reservation Weak identities refer to a different allocation and preserve their original validation behavior. Original source health, pending publication, run closure and native quarantine checks are unchanged.

Neutral tests use actual admitted span accounts, detach all plan/run/reservation owners, and concurrently retire full or raw aliases while checking the original hold. Unwind tests distinguish a certified native scope from an abandoned one: host retirement cannot certify native completion or clear quarantine. Existing exact/minus-one, source-health and publication tests remain applicable. These lifecycle tests are not allocator instrumentation; allocation order follows the closed ownership graph and supported standard-library extraction semantics. No result provider, managed-copy support or public admission gate is enabled by this change.


### Original plan aliases at retirement

Earlier and escaped span-plan aliases retain the same immutable records and original once-held custody through concurrent final-owner disposal. A promoted-owner panic or failed post-attachment source validation keeps records readable until the last real plan owner retires. Actual accepted-account tests check the exact held P and preserve independent source quarantine. The change adds no native completion signal, retry, replacement quote or execution permission; only the plan's private owned retirement and its original control sizing change.

### Moshi frame continuation across scalar storage policies

Selected resident, host-layerwise and disk-streamed numerical sessions use the same canonical scheduler and [2,1,2] bounded-turn continuation fixture. New comparisons cover depth 2/3, batch 1/2, forced decisions with and without diagnostics, two callback failures and idle cancellation after two commits. Full state/history, sampler/random state and observed/output values remain equal for matching requests. Without diagnostics, both depth configurations skip their fully forced depth tail: the interpreter maps Existing user-audio targets to Force directives too. With diagnostics, every depth unit executes. Actual unit destruction and disk rereads are asserted. No packed frame equation, finite frame-admission or pending native-completion claim follows from these tests.


### Grouped pin final owners

Original grouped-pin aliases and canonical opening parcels share one concrete final-owner retirement path, including transfer into original scope quarantine. Existing stamp/source/account checks still precede the allocation-free installation move, and only the canonical ticket can detach an installed source parcel. Actual SessionPrefill fixtures retain the original P+Q+S/H and physical origins through concurrent typed or mixed typed/parcel teardown; existing failure, Busy, source-closure and quarantine tests keep their original paths. This adds no retry, replacement hold, scalar credit or native completion proof.


### Original token-result admission prerequisite

The optional runtime retained sequence is priced from the actual borrowed
maximum/EOS request before original acceptance, then extracted once after the
existing core run binding within Admission. Pending or abandoned extraction
blocks both new-prompt claims and owned-input binding, as well as sampling.
The mechanism reuses the existing prompt stage byte, adds no second engine or
agreement, and leaves legacy/capture-only preparation available. Sequence-only
preparation preserves Admission, Prompt and Sampling phase order without an
Instrumentation vote.

Explicit core preparation later materializes the already priced token slots
and revalidates original source health. A future facade caller must perform
that step before its existing first finish_step readiness agreement. This
prerequisite does not migrate public output fields, implement managed snapshot
copies, enable native capture/span entry, or add a native error conversion.
The extraction failure remains a by-value owner until the separately priced
closed-error successor is composed. Controlled copy restrictions and all
existing termination/cancellation precedence remain unchanged.

Neutral lifecycle cases cover real ordinary and controlled Admission, exact and
one-byte-short acceptance, new and owned prompt routes, optional capture,
foreign/abandoned claims, source quarantine, zero output, pointer-preserving
freeze, concurrent aliases and partial owning iterators. These tests do not
submit model work or establish complete facade/native memory accounting.


### Retained sequence rejection and core error retirement

A rejected original retained-sequence provider continues to own its payload in the typed core rejection. `BackendFailure` now retires its concrete source allocation before the rejection payload is destroyed; flattening a previously classified failure returns its original source owner. Genuine Admission fixtures exercise invalid returned policy and extraction failure, preserving the shared phase order and original cause through escape/unwind. A safe post-unboxing witness checks callback order; it is not an allocator deallocation probe or native completion evidence.

`BackendFailure::source_retention_peak_bytes<E>()` and the core-owned rejected-sequence diagnostic report one finite envelope's requested layout and named controls. They supply no grant, retry budget or readiness. A future runtime/native result bridge must include its exact concrete source/error populations in the original seal and retain first-step preparation under the existing readiness agreement. Repeated public wrapping, nested native errors, provider payload funding, controlled copying and enclosing facade storage remain separate.


### Capture sidecar final-owner retirement

The original C publication protocol retains its same reservation, identity, finite commit and witness validation. Its typed, erased source-attachment and erased witness aliases now share one closed concrete retirement path. Earlier source aliases keep the original C and host hold after quote/run owners disappear. Reused C still retains both actual original accounts without a replacement credit or alias chain. Source-health rejection never implies native completion.

Real-account tests cover mixed final-owner races, exact surviving C/host charges, raw/owned cross-use rejection, post-C source quarantine before witness completion and key-clone unwind. The narrow test-only post-publication fault is source-scoped, one-shot, reset on unwind and runs outside source/Usage/TLS loans. Existing exact-minus-one, healthy/raced reuse, counterfeit source, Busy/poison and comparison-unwind fixtures remain. Safe payload/drop probes and concurrent uniqueness supplement the pinned Arc contract; they do not observe allocator events or activate native span readiness.


### Moshi exact temporal window and failure assertions

The canonical-frame fixture derives temporal retention from the normalized visible-key window: temporal context counts preceding positions, so context 3 produces window 4; depth context already includes its visible keys and remains window 4. It asserts those distinct equations, actual cache windows, complete K/V contents and offsets/reset history at every committed frame. The ordinary/bounded and resident/host/disk comparisons remain exact.

Callback-failure coverage checks the same typed source before scheduler stringification, then retains the existing failed status, two committed frames, no completion for the failed branch, exact retired committed state and no retry. Cancellation and forced-tail semantics are unchanged. Physical checkpoint replacement is checked at the actual prepared-source acquisition boundary, not at the earlier metadata-only handoff.


### Accepted physical Sequence with ordinary final output

The original physically Sequence capture can enter the shared scheduled text
gateway only with its accepted span view and original pre-admission path seal.
Current prepared execution binding is checked before the one-use driver or
source factory. A truthful sequence observer requirement, raw bound selection,
equal-content foreign path source, unsealed source or different reservation
cannot substitute for that association. Missing sources still reject rather
than falling back to whole unbudgeted execution. Legacy unbudgeted selection,
LastPosition body capture and decode continue through their existing paths.

The same chunk transactions, completion guards, cooperative cancellation checks
and final-index guard remain in force. Each complete observed chunk contributes
its actual rows to one original prediction frame. Intermediate public Sequence
outputs retire after settlement and before the next chunk; final indexing
precedes successful outer-frame notification. A source-factory failure before
the observer guard creates no frame or terminal notification. Later failure,
cancellation or unwind aborts the original frame without undoing already
committed chunk state. No new agreement, copied chunk loop or readiness vote is
introduced.

Portable cases exercise a real two-token cached prefix, exact integer recurrence
and full state, all four actual outer original/effective hooks and actual logits,
whole versus uneven chunks and three cached decodes. They check original/-1
admission, missing/duplicate/foreign/stale associations, source and observer
errors, retirement and final-index errors, initial/intermediate/final cancellation
and identical unwind payload. The native private-entry fixture covers resident,
host and disk logits capture, full/uneven chunks, three decodes, exact original
headroom and escaped frame/source custody. These are staged source tests, not
execution results or complete native memory-activation evidence.


### Consuming original token-result bank

The genuine shared Admission path can consume one original sequence bank after run binding and before Prompt/Sampling. Replay or absence takes no owner; fixed Busy/identity/unavailable errors retain no original custody. A consumed foreign, fenced or failed construction spends that bank and retains its exact original reservation/source controls in the returned core error. Cancellation/unwind cannot restore it. Core prepare_storage remains a consuming operation: recovering its sole sequence destroys the prior cause before retry, so it cannot accumulate concurrent owning preparation errors. The shared facade first finish_step, result/copy ownership, native quote/work retirement and public capture activation remain separate composition obligations.


### Original native quote/work aliases

Historical native quote and work aliases now share private closed Rc owners.
Final owner retirement extracts the concrete payload after retiring its outer
allocation, preserving the same original control guard through the boundary.
Taking an installed capture bank, certifying completed work or unwinding a
caller does not release custody retained by another quote/work alias. Scope
failure, quarantine and native completion rules are unchanged.

New staged tests retain actual original quote/work aliases after bank/run/source
retirement, unwind other aliases with an identical panic payload, and require
original protected controls plus C and explicitly joined path storage until the
final remaining work owner. A real
canonical native source-retirement probe verifies that its temporary test alias
is cleared on normal exit and unwind without a Weak retaining the work block.
Existing controlled/uninterrupted, exact/-1, source-failure, copied and resumed
execution tests remain required. These source tests do not observe allocator
events or establish a new complete public memory contract. Ordinary and saved
routes without an original aggregate host guard retain their explicit custody
limitations; no late amount or completion substitute is introduced.


### Submission aliases preserve the original control account

The native submission owner carries the same original work guard through escaped token/completion aliases, synchronous completion, failure and recovery quarantine. Its closed Rc population destroys the allocation before the concrete payload, and retires funding after the other fields. Original finite M+2 work controls include one measured submission representation per slot; sampling observations reuse their producing owner. Copy composition includes the same fact through its existing single-account helper. No completion predicate, source inventory, public activation or budget eligibility changes.

Focused fixtures use an actual accepted capture account: a nonzero native token remains readable after sibling-owner unwind and retains the exact protected-controls plus published C and explicit path-source tail; a real live descendant keeps the original owner and submission lease quarantined until ordinary lifetime retirement. Existing exact/one-byte-short and output/residency/recovery tests remain intact. These are source-staged tests pending central execution, not allocator-event measurements or complete native-inventory evidence.


### Capture carrier ownership on failure and unwind

The same original carrier now has a closed concrete Box owner. Native Busy, a late transform error, cancellation and canonical settled-source retirement keep their existing meanings. Releasing the Box allocation does not certify an aborted source segment or refund its original scope. A new fixture exercises both late transfer failure and retirement Busy, preserves the exact typed cause or Busy error and the corresponding failed/captured record, then unwinds earlier owners while a work alias retains the intact carrier and full original charge. Existing nonzero canonical retirement, source-alias, exact/one-byte-short and cancellation tests remain unchanged.

The closure is allocation ordering only. Existing native retirement guards still enclose applicable payload Drop; the canonical ordinary-host path still drops payloads after its loans end. Accepted-record quiescence is not consulted. Source staging and formatting/replay checks do not claim native execution or allocator-event measurement.


### Shared retained cursor readiness prerequisite

The private retained committed cursor uses the same advancement and termination
algorithm as ordinary generation and controlled one-step advancement. Core's
one dormant token preparation attempt produces a local result before the
existing first `finish_step`; it adds no readiness vote or inference engine.
A failed preparation carries its exact source and sequence owner through that
agreement. A source readiness failure or peer cancellation submits no token;
a pre-cancelled or zero-output request allocates no token slots. Cancellation
keeps the existing final disposition, and zero output otherwise publishes
`MaxTokens` through the same pipeline/settlement calls. Token-slot preparation
is the scope of this allocation statement; parser and event work is separate.

Neutral tests compare ordinary loop-style stepping and explicit controlled
steps for max-token, EOS, grammar and stop precedence, including callback
cancellation. They retain all committed IDs/events and readiness counts, check
the same token pointer after terminal freeze, and keep its owner through a
partial consuming iterator. Tests also cover original typed local failure,
source first-readiness failure, initial delivery-error precedence and provider
unwind without retry. These are source-staged tests pending central execution.

Loaded/controlled public constructors remain legacy; retained Clone/snapshot
and managed copying remain unavailable. A future facade integration must bind
the actual borrowed request in original Admission and receive the runtime's
one admitted bank/provider, then fund complete facade/result/error/copy
representations before use. This cursor-only prerequisite does not complete
those owners, enable native grants, or add a second original quote.


### Fixed original consumer contribution

The optional original sequence consumer descriptor is admitted and verified in
the same shared Admission operation as EOS/max and the original storage bank.
An older sequence provider does not silently opt in. Missing or foreign returned
associations reject while retaining the provider, before Prompt/Sampling.
Sequence-only requests still add no Instrumentation phase.

The private typed consumer wrapper advances the existing commitment cursor and
prepares dormant token slots immediately before that cursor's existing first
`finish_step`. It introduces no readiness vote. Initial cancellation and zero
output still skip token allocation; peer disposition and delivery-error
precedence remain in the shared algorithm. The earlier explicit Delivery phases
in loaded and controlled startup remain separate and unchanged. A source or
pipeline failure consumes the cursor into its owner-preserving error; a local
preparation failure keeps its concrete owning error and retires the empty cursor
first. Terminal extraction preserves the token pointer. Legacy Clone/snapshot
behavior remains confined to legacy storage.

Public loaded/controlled routing and the Vec-based public output field are not
changed by this prerequisite. Fixed controls do not establish bounds for dynamic
request/parser/tokenizer/events or copied results; these remain required before
public managed activation. Neutral tests cover original three-driver optional
capture admission, exact width-one minus-one rejection, returned association
checks, owner retirement, cancellation/readiness and result aliases. They do not
claim a newly executed native consumer integration or full facade payload bound.


### Retained ordinary terminal result prerequisite

The private retained consumer can now freeze an originally selected ordinary output containing the same committed token owner, finish reason and timing. Tuple admission cannot select this output mode after acceptance. Nonterminal extraction returns the unchanged cursor; preparation and other failures preserve their existing opposite owner-drop orders. Retained output Clone is unavailable, while legacy Vec output/cursor cloning and snapshots retain their existing APIs.

Neutral staged conformance covers loop-style and one-step controlled advancement through the same algorithm, initial/peer cancellation and zero output, first-readiness failure, terminal pointer identity and partial owning iteration. Original runtime fixtures exercise all three sequence constructors, optional capture, exact/minus-one and the explicit registered-source full-to-raw transition. These are prerequisites only: no public loaded/controlled path, extra readiness vote, speculative output mode or copied token result is enabled.


### Decoder configuration lifetime across loaded and controlled use

Fresh facade decoders and controlled vocabulary now retain the loaded HF tokenizer's immutable snapshot. Their mutable token/prefix state is unchanged, and decoder forks can complete pending Unicode independently after the loaded model and other forks are dropped. Mutating a neutral chat tokenizer configuration creates a separate inner when snapshots exist; template cache/variable edits alone do not change the token configuration or vocabulary fingerprint.

Staged portable behavior checks cover old/new configuration snapshots after mutation, template rendering/fingerprint stability, loaded-owner drop, incomplete Unicode forks and both special-token modes. This adds no generation engine, snapshot budget, readiness vote, public managed admission or tokenizer-storage completeness claim.


### Fixed-destination decoder prerequisite

The eredu-text plain-join/ByteLevel stream counts total successful token steps independently of compacted lookbehind. Exact destination mismatches and exhausted limits reject before prefix repair or append. HF InvalidPrefix retains its appended ID and actual expected/candidate frontier; fixed diagnostics do not own escaping text. Output borrows the candidate, so another step or finish cannot overwrite a live chunk. Finish is only the existing residual-length check; cancellation does not introduce a flush.

This unit adds no facade driver or preparation vote. Loaded and controlled routes keep their current decoder behavior. Using the new mechanism requires the same shared cursor, original source/layout/request association, one-use dormant extraction, existing earlier Delivery dispositions and first finish_step, plus real parser/event/error/copy ownership. Those integrations remain pending; no lower-level buffer proof advertises complete managed inference.


### Ordered fixed decoder pipelines preserve the existing stream frontier

The fixed-destination text kernel accepts the actual singleton ByteLevel wrapper and the six ordered fallback/Replace/Fuse pipelines used by current GGUF constructors, including the optional single-leading-space Strip. It keeps the existing shared HF-compatible step algorithm: successful calls have a separate finite ceiling from compacted lookbehind, capacity rejection precedes mutation, and a genuine InvalidPrefix retains the appended ID, prefix/index and full candidate. A valid ASCII byte followed by an invalid byte and an ordinary character exercises this failure from fresh state.

Finish remains the existing residual byte-length check and emits no flush; cancellation remains the caller's decision to skip work. Full-frontier comparisons cover the ordered pipelines, partial Unicode, unknown/special/empty boundaries and exact source lookup. The actual facade is not redirected to this kernel yet: controlled and uninterrupted source/admission/event integration must continue through the shared driver, with original ownership and finite parser/event bounds established first.

### Private plain projection over the shared commitment driver

An originally admitted plain decoder/stop source can lend text and at most one
terminal descriptor through the same private cursor used for repeated ordinary
stepping and controlled advancement. The existing first readiness, source
failure, grammar/commit and final delivery agreements remain unchanged. Matched
stops become terminal before callbacks, including callback cancellation; ordinary
cancellation discards lookbehind, and successful decoder finish precedes any
pending stop-text flush. Borrowed descriptors expire before another token or
terminal provider extraction.

This is a private storage/consumer contract. Public owned SemanticEvent and
controlled-record delivery/copy remain unactivated until their separate original
owners and physical capacities are admitted. Existing legacy snapshot behavior
continues; the mutable original provider remains non-Clone and has no implicitly
funded copy or structural/tool-parser destination.


### Explicit submitted-work tracking ceiling

Ordinary and controlled text admission forward the same optional
`submission_tracking_capacity_bytes` through their shared original quote. It
requires total managed capacity and a finite output allowance, but does not
predict graph fit. The native provider seals one actual fixed arena plus its
constructor/retirement controls into original Q before reserve, including the
ordinary route without capture or a retained sequence. Optional capture, sequence
and plain decoder/stop sources keep their own existing identity and once-only
bank semantics.

Prompt/Sampling Busy retries preserve their same retained input, Work, recovery
node and arena. All five genuine prediction roles inherit this one arena; token
clones retain their existing one-shot observation, and no token copy receives a
replacement role. Exhaustion reports a fixed typed source while existing native
records stay retained until actual completion permits destruction. Freed arena
blocks may be reused inside the original ceiling; no allocation or observation
can grow/refill it. Saved/copy admission explicitly rejects a selected ceiling
before work until its separate original authority is wired.

Staged tests cover actual nonzero ordinary/controlled generation and complete
selected state across resident, host and disk routes, optional C/R and the
existing resident opening-row fixture; exact/minus-one original admission;
same-owner Busy; actual Eval/Wait storage; partial capture failure; native nested
inheritance and post-deallocation custody. They are source-only until central
validation. Registry/graph/task/error allocation closure and default selection
remain unfinished; no new completion proof or managed gateway is enabled.

### Shared native descriptor retirement

Ordinary and controlled execution continue using the same MLX array ownership
path. Final descriptor release drains known input/sibling descendants before
releasing the parent's remaining authority, without an allocating traversal.
Replacement and exception cleanup preserve complete sibling groups. This adds
no advancement engine, cancellation disposition, completion certificate,
copy-budget refill or public managed gateway. Primitive-held/external graphs
and a complete original graph-storage bound remain separate obligations.

### Realtime decision demand and required observation

Moshi/PersonaPlex continue through the same canonical frame coordinator for
ordinary and bounded advancement. Demand affects vocabulary projection, not the
frame schedule, decision order, sampling state, or publication/rollback engine.
A sampled decision and a retained diagnostic always require real scores. Forced,
unobserved decisions may omit scores; a wholly forced non-diagnostic depth tail
may retain its existing body-skip optimization. Required observations instead
execute that tail, including its frame-local state updates. Explicit
opportunistic tracing sees only values actually produced and does not force it.

An initialization-only absolute-delay transition has no model call. Required
observation rejects that already-prepared transition before model/completion;
input preparation has occurred and its owners retire under the existing failure
path. No initialization callback is manufactured. Splitting the single schedule
advance from materialization for an earlier rejection remains separate work.

The numerical fixtures cover nonzero Moshi and private reduced PersonaPlex
resident/host/disk frames and low-level wider inputs. Released PersonaPlex
checkpoint parity and native TP numerical parity remain validation gaps.
Ordinary PP frame transport and reusable frame snapshot/fork/persistence were
already unimplemented; the existing release/resume path is distinct. Synchronized
PCM/codec lifecycle would require a new application contract: the existing
public realtime frame API accepts encoded tokens and does not own standalone
codec state. The
published PersonaPlex admission profile is unchanged. Draft/speculative and media
capabilities are not enabled by this demand mechanism.


### Realtime transformed-source binding validation

Resident and bounded realtime loading retain the same original validated task
plan through native materialization. A packed replacement under an original
dense key is validated as an authoritative selected output; untransformed keys
retain exact original-source validation. This fixes transformed loading before
either ordinary or controlled execution is constructed and leaves the shared
frame driver and selected state equations unchanged.

Focused cold cases cover affine companion precision, MXFP4 encoding, malformed
or unpublished outputs, unchanged-key provenance drift, and generic source-backed
companion group geometry. The latter tests the neutral post-transform contract;
it does not enable native packed-to-packed transcoding. The existing native Moshi
conformance suite remains the actual dense-to-packed construction and generation
regression. These new cases are staged for central execution; their source review
does not establish execution or released-checkpoint validation.

### Original TokenIds startup and controlled parity

The original-input adapters for ordinary, controlled and detached generation
share the existing Admission → Prompt → Sampling flow. Input/sequence construction
occurs inside genuine Admission after run binding; first consumer preparation
stays at the existing Delivery/first `finish_step` boundary. There is no additional
readiness vote or frame loop. Pairing errors, including zero-output requests,
reject before backend admission; controlled peer cancellation uses the existing
shared cleanup. Caller IDs can be mutated after successful synchronous startup
because only the admitted immutable destination is retained.

The private facade original-plain composition uses the same token-source adapter
and retained cursor for looping and explicit advancement. It does not activate
public managed decoding or provide resume/fork/copy destination admission. Native
repeat-request tests use cached continuation, preserving original cache/request
custody; an original-funded reset path remains a separate required integration.

### Enforced graph metadata capacity in original text execution

The optional `graph_metadata_capacity_bytes` policy is a physical ceiling for the declared native graph metadata component, not a graph-fit estimate or a replacement for total managed capacity. It requires original managed admission. The same original arena follows Prompt and preparation Sampling, then ModelExecution, prediction Sampling, SamplingEvent, ModelValidation and TokenScalar. Ordinary and controlled text use the shared drivers and genuine claim. Busy retains the same pending input, Work, Recovery node and arena; neither retry nor token observation refills it.

Allocation refusal is a typed `GraphMetadataFailure` source. It does not certify completion: already submitted work and its original scopes retain custody through the existing completion/recovery/quarantine path. Native storage returns before its constructor-owned Send capsule reaches the existing unlocked retirement queue. Untouched prepared-work cleanup keeps its previously established no-observation rule.

Source fixtures cover a 32-session ordinary/controlled/capture/sequence/opening/residency matrix, exact and one-byte-short original admission, genuine Busy retry, minimum-capacity refusal, and typed native-to-neutral projection. Native and safe-wrapper fixtures cover partial construction, cross-arena/legacy buffer transitions, empty-container lifetime, sibling rollback, deep/reentrant retirement, primitive Weak roots, submitted prefixes and failed C/callback vector conversion. They are execution requirements, not an execution record. Saved/copy authority, worker-generated metadata and default capacity policy remain required follow-ons; no public gate is widened.


### PersonaPlex initialization and scalar controlled parity

Private PersonaPlex numerical coverage uses the same canonical coordinator and scheduler for uninterrupted and bounded 2/1/2 advancement. An initialization transition precedes two model-frame prefix calls, five continuations and three more calls. Required observation on initialization is rejected after real input materialization but before model execution/completion. The fixture checks exact typed rejection, zero committed mutation/sampling/projections/completion, real host input retirement and no resubmission. An opportunistic initialization can transition through the existing consuming executor API to required final-row observation/intervention for later calls. Later callback failure retains the exact injected source before scheduler display erasure; cancellation/failure preserve the complete committed prefix and retire leases.

Diagnostics or required observation execute all sixteen depth bodies. Non-diagnostic execution may omit only the actual fully forced suffix: one warmup depth decision, eight later generated decisions, or zero depth bodies when all generated targets are forced. Full-state comparisons use execution-equivalent demand variants; the intended depth cursor differences against diagnostic execution are asserted explicitly. Public frame snapshot/restore/fork/persistence remains a pre-existing implementation gap, not an inherent PersonaPlex limitation. The existing release/resume path retains complete token-session state. Standalone PCM codec state belongs to external application composition and is outside the token-frame account.

### Explicit preparation for an admitted reset

Ordinary `ModelRuntime::reset` still synchronizes then resets, and explicit
`synchronize` retains its existing contract. `reset_admitted` does not implicitly
wait: typed Busy preserves installed state and is not successful completion.
Unsupported provider/producer coverage remains a distinct rejection. This avoids
performing unreserved synchronization before the original reset claim.

Opt-in backends can provide `prepare_reset_ordinary`, whose consuming result
exclusively borrows the actual runtime until reset or abandonment. Both new
backend hooks default to Unsupported. Dropping a readiness owner may only release
its borrow; it must not reset, synchronize or progress work. Changed capabilities
are rejected after preparation and again before claim consumption. Failure also
consumes readiness, so a later attempt must prepare again. This owner exposes no
source, session, callback or clone route and works independently of generation's
step scheduler.

No loaded, controlled or distributed native route is activated by this core
contract. Their actual producer exclusion, completion, account, source publication
and retirement obligations remain implementation work. Ordinary external
settlement can set up a private test, but is not evidence of an originally funded
wait. A strict session retaining earlier accounts may therefore still reject
ordinary preparation until a complete original settlement path is implemented.


### Resident-KV reset publication and pending consumers

The private native reset publication path clears the actual selected KV state and committed prompt identity while preserving the newly funded revision. Its source, revision and numerical contents remain unchanged on pre-install failure; the error retains any constructed destination. Displaced state retires through a prepared ordinary node after all session loans end. Native fixtures cover resident/host/disk histories warmed by ordinary and controlled generation, followed by host reset publication and a second reset from the installed original table.

Those fixtures perform ordinary loading/synchronization externally and consume genuine core claims through a scoped test hook. Production native activation remains rejected until the separate readiness entry covers actual producers. Subsequent funded generation currently rejects original reset tables at publication-source preparation; a fixed pre-admitted original-table consumer is required before claiming uninterrupted or controlled post-reset generation parity. This is an implementation gap, not an inherent KV-model restriction.


The fixed original resident-KV table consumer uses shared native quote, model
Work and publication paths for ordinary and controlled generation, including
resident, host-layerwise and disk-streamed weights. Coverage compares emitted
IDs and every retained KV shape/value and checks successful Work publication
and retirement. Public admitted native reset remains unsupported until its
separate producer-readiness contract is implemented; the reset fixtures perform
ordinary external settlement before their genuine private core claim.

### Cancellable independent-draft initial prefill

Ordinary speculative runs and controlled registration share a cancellation
outcome before their first sampled token. Independent autoregressive target and
draft prefill use the same selected-source scheduler as ordinary text: target
`LastPosition`, draft `StateOnly`, with full `Sequence` verification unchanged.
An agreed cancellation after a completed span prevents the next source/model
submission, restores the joint lane checkpoint and produces no first token or
sampler update. A failed completion remains an error under existing retention and
fencing; it is never converted to cancellation or successful settlement.

The existing run/step and snapshot/fork mechanisms remain shared. Snapshot replay
does not replenish capture or copy budgets. Cancellation of a legacy executor is
observed at its completed prefill boundary; only migrated selected sources gain
per-span responsiveness. Explicit chunk policy remains rejected for embedded and
assistant modes until their complete captures and shifted seeding are integrated.
Media encoder-once ingress and complete original speculative funding remain
required follow-ons; strict unknown-memory rejection is unchanged.

Completed target work remains in speculative telemetry after cancellation or
rollback. The selected-source progress result reads the shared driver completion
frontier; cancellation during draft prefill retains the completed target count.
This scalar measurement grants no memory or native-completion authority.

The facade checks explicit speculative chunk support through a cold neutral backend
query for the selected target/draft pair. Its default rejects; the native
independent-draft adapter opts in using the same predicate as execution admission.
The batch host vote still rejects unsupported chunks and all unfinished memory
or tracking policies before prompt preparation. Ordinary and controlled routes
share this check.

### Reentrant native publication cleanup

Publication can reclaim a previously retired native allocation owner at an ordinary native entry. Such cleanup cannot recursively publish or certify the active Work. The actual scope is retained without an open cell borrow and restored on success, failure or unwind; existing completion evidence remains required. Failure after model publication but during decoder attachment keeps the earlier physical charges and the uncertified remaining envelope. Ordinary and controlled execution use this same native Work finalizer.


### Chunked materialized embedded prediction (2026-09-14)

Ordinary and controlled registration share cancellable captured prefill for materialized sequential embedded predictors and DSpark. Target spans complete and publish their declared capture before a separate auxiliary transaction begins. Sequential seeds pair hidden row i with token i+1; a single last hidden row crosses a span boundary. DSpark consumes aligned target-layer captures, retaining the existing concatenation across selected layers within each span. No full-prompt capture or second production chunk loop is introduced. The installed target frontier authenticates each capture; the auxiliary frontier must equal the completed seed prefix. All lane values plus the carried row complete before the shared chunk finalizer. Cancellation cannot skip local/peer agreement or unresolved work, and completed target spans remain in telemetry even when registration rolls back without sampling.

Bounded collectors accept an explicit invocation window for FullTensor/Slice records and activation edits without evidence. Global slices, strides and edit payload coordinates use the existing funded fragment projection; non-overlapping windows skip native transformation. Invocation records keep physical shape and explicit span provenance, and cumulative limits are not refunded by snapshot restoration. Arbitrary whole-tensor callbacks remain incompatible unless they explicitly opt into span semantics. Global reductions, top-k, whole-result Preview, intervention evidence, sparse routed collectors, and combined partition/window projection still require their aggregation integrations and return typed Unsupported for this path. They are unfinished integrations, not architectural limitations. The same window contract is reusable by ordinary Sequence observers, whose current fallback is unchanged here.

External assistants, captured/speculative media, distributed native execution validation, and original managed speculative memory remain separate required work. The facade preserves strict rejection of unknown managed memory/control policy. Assistant chunk rejection remains until its distinct prompt-state seed consumer joins this lifecycle. New CPU fixtures exercise six materialized variants across three residency routes, numerical scores over at least three rounds, controlled snapshot replay, physical span records and cancellation. Central validation records distinguish executed coverage from remaining gaps.


### Ordinary selected media prefill advances

The new Qwen VL/conditional-Qwen retained-media source runs through the same shared step/run driver and selected transaction used by ordinary prefill. Encoder work occurs once in the first decoder transaction; later steps retain the original compact media roots and consume architecture-aligned decoder spans. Numerical conformance includes state, requested rows, cancellation boundaries and cached decode after complete checkpoint rollback. This is direct shared-driver conformance, not activation of full facade controlled-media sessions or managed copies.

Ordinary unfunded cancellation leaves the actual committed prefix available to existing cached decode, with VL's prefix delta derived from the original axes. It does not restart the terminal prefill driver or shorten a funded request. A live unfinished source cannot be cloned or restored onto a different revision; shared resume/fork authority and captured-media observations remain follow-on contracts. Existing required Sequence/prepared/transactional observers receive a typed rejection before model work on this new route.

## Muse shared media advancement

Muse-Glimmer now supplies its own compact normalized raster-media source to the shared selected media lifecycle. It preserves image/video placeholder IDs, text-embedding normalization and per-layer cache-derived masks, completing all future projected rows under the first span. Received encoder boundaries rebuild placement metadata without repeating patch/learned projection. The nonzero local and selected TP/PP/combined cases cover run/step, resident/host/disk, full/last local rows, mixed image/video 2/2/1 and ordinary cancellation-prefix cached decode. Central validation records describe executed coverage.

This is an ordinary/unfunded mechanism, with no native/public retained-media activation or complete peak claim. Gemma4/Inkling still require a neutral observed-inactive dependency state and their own per-layer/shared-attention or zero-unit audio semantics. Native original source/completion, funded continuation and remaining-family integration remain required follow-ons.


## External assistant prefill spans

The ordinary and controlled speculative drivers share selected local Gemma4/DFlash prefill. Actual target spans commit through the existing guard, and architecture receivers settle capture-derived seed roots before the next span. Cancellation after a completed span retains its target-work count but returns no seed, logits, proposal or emitted token. Existing checkpoints, lane exchange and controlled snapshot state remain authoritative; whole-sequence verification is preserved.

Prompt-row observers must explicitly support physical spans. Gemma K/V values additionally require complete-context support, announced at the actual installed target frontier; they are not relabeled as prompt fragments. Unsupported transformations are rejected before the corresponding target work. The public native chunk predicate also checks the selected executable; partitioned multi-tensor capture publication and speculative media capture remain required integrations.

Tests compare nonzero Gemma ordinary/control/full-prefill results and repeated snapshots across all three weight-residency routes. An observed six-token/chunk-two case preserves two final physical score rows while the common guarded tail selects one sampling row. DFlash tests retain tap order and lazy initial encoding; released-geometry protocol coverage is explicitly shape-only.

## Optional media through shared run and step

The selected retained-media cut now distinguishes Unseen, observed Inactive, and Produced dependency outcomes. Existing layered traversal and partition scheduling establish inactivity; later spans import it without fabricating a tensor. Missing active dependencies still reject. Inkling's active zero-unit dMel group executes its existing completion equation. Default hooks preserve other executors.

Gemma4 and Inkling provide architecture-owned retained-source/span hooks through the existing selected media visitor. Gemma4 keeps independent vision/audio projection owners, original placeholders, per-layer inputs, cache positions and shared-KV transport; each input projection runs only at its existing root begin. Received encoder continuations use original metadata, without dummy shape tensors. Inkling preserves exact hMLP folds, dMel offsets/normalization/valid frames and sconv state. Full future media roots settle under the first span; original semantic vectors move into the source while encoder preprocessing temporaries remain under first-span completion.

Five nonzero numerical cases cover: compact optional combinations across resident/host/disk and TP/PP/combined, ordinary versus shared full/last rows, run/step, cancellation at two actual boundaries, complete local state and three cached decodes. The exact large Inkling hMLP receives focused resident/host/disk and bounded PP coverage with 512 MiB fixture budgets. Its remaining raw-image topology/residency cross-products remain explicit unexecuted validation obligations, not unsupported behavior.

Central validation records report executed coverage. This is ordinary/unfunded selected-driver support. Public/native retained-media activation, full original source/control/graph/completion admission, captured-media attribution and funded continuation remain Unit C work. New source and context allocations are not caller-owned or free; no native peak or whole managed execution claim follows from these changes.


### Logical reductions during captured embedded prefill

Ordinary observed and controlled speculative execution share the same span lifecycle for Summary and fixed-edge Histogram. Every physical target/seed envelope keeps its own coordinates; only the last real envelope carries a logical-prefill companion. Separate target and selected shifted/unshifted seed lanes become Complete only after full coverage, agreed reduction validation, final score indexing/reservation settlement and target exchange. Cancellation/failure keeps spent quota and physical evidence, and removes provisional global payloads. Zero-row seed is explicitly NotInvoked. Open groups cannot be snapshotted/readmitted/restored; completed restore never rewinds invocation ids or capture usage.

This component does not authorize arbitrary whole-tensor observers, complete-prefix assistant K/V, ordinary/funded Sequence collectors or parallel/global transform composition. Their precise semantic/ownership integrations remain unfinished. Native completion and unresolved failure custody stay with the existing driver; host aggregate completion alone is not a universal native retirement signal.


### Ordinary prepared-media advancement boundary

The low-level core `TextGeneration::from_prompt` and `ControlledTextGeneration::from_input(Prepared)` use the same selected native retained-media source when an ordinary chunk width is supplied. The latter advances the existing generation machine explicitly. Their staged native parity case compares committed IDs, actual whole-input cache identity and full mutable state. Ordinary cancellation may leave a committed prefix that supports the existing unfunded cached-decode behavior; the cancelled source is terminal and cannot be resumed or replenish original authority.

This does not complete public facade `ControlledGenerationSession` prepared-media startup. Its prepared-token IDs, Started/delivery position attribution, initial capture, snapshot payloads and source ownership require the coherent follow-on described in the control audit. Required prepared/sequence/transactional media observation remains rejected before the media plan factory/model execution. Pure-text capture on the same selected model retains its established path and is tested with actual nonzero capture records. Managed prepared-media admission remains closed until original source, transient, graph, event and copy accounting is complete; neither missing control attribution nor missing admission is an inherent family limitation.


### Terminal candidates share ordinary and controlled prefill

Original-funded TopCandidates uses the same collector and SessionPrefill/PrefillDriver lifecycle in uninterrupted and controlled generation. Earlier canonical spans do not spend a candidate extraction; the final span supplies the actual terminal row. A separately required Sequence selection keeps its full physical span, including a two-row final span. Final indexing/reservation settlement still precedes Committed delivery. Cancellation/failure never resets the once-only candidate slot or refunds spent logical quota.

The source package stages nonzero candidate/token/full-KV parity across resident, host-layerwise and disk routes, exact/-1 admission, initial cancellation, mixed Sequence capture, native precision/tie/sort branches and final shared-owner retirement. Central validation is recorded separately; these statements do not claim completed public managed capture, new native peer-failure injection or migration of arbitrary legacy/explicit-window observers.


### V2 controlled startup from actual prepared input

`LoadedModel::prepare_controlled_input` accepts an opaque backend prompt and existing chat output policy. `start_controlled_prepared_chat` and `start_controlled_prepared_text` start the same controlled driver used by the existing token-ID APIs. They do not retokenize the chat to replace the supplied prompt. `PreparedInputInstrumentation::Unobserved` installs no capture owner. `Capture`, including an empty plan, and `Intervention` currently return the typed `InstrumentationUnavailable` rejection; existing text observed/intervened entry points remain available. Captured media is an applicable, unfinished integration.

V2 Started, snapshot and branch records carry actual ordered source segments, semantic/prepared identity, the opening cached frontier, decoder extent and canonical IDs only for real token-ID segments. Projected/media segments carry `NotTokenized`. With opening frontier c and source extent n, prediction zero covers [c,c+n); prediction i > 0 covers [c+n+i-1,c+n+i). `complete_token_ids()` returns None for a mixed/projected source. V1 `prompt_token_ids()` keeps its original complete-token meaning. The V2 wire parser rejects unsupported envelope/nested versions; deserialized records remain diagnostics. Closed live records and metadata may be cloned and serialized, but their library-owned payload cannot be detached from ordinary custody.

Startup may prepare native input, token reads and sampling state, but performs no encoder/model prediction. Initial cancellation, Started callback break and zero output therefore commit no token. The existing pause/step/run, forced-choice, sampling, Unicode, termination and cumulative snapshot/branch semantics apply. An input prepared against a different loaded session, changed installed revision or changed frontier is rejected at consumption, including an equal-position restore. Snapshot/restore and branch exchange retain the same original source map; they do not refund trace or copy budgets.

Initial mixed-media pending-input snapshots remain unknown until the mandatory ordinary copy unit constructs a checked inventory of all ordered payload/metadata arrays and extents. The existing copy primitive will produce independent contiguous per-slot storage while preserving values, order, dtype/shape, metadata/extents, cache attribution and request policy; it does not preserve arbitrary shared-view backing identity. The authenticated pending extent retained by this unit is available to that follow-on, but `text_snapshot.rs` and `saved_sampling.rs` are unchanged here. Later boundaries are supported only when the existing complete native/controller/parser estimates succeed.

The initial V2 native numerical matrix covers Qwen VL/conditional-Qwen attribution in resident, host-layerwise and disk-streamed selections, plus actual Qwen VL media/cached-state equivalence and signed/projected source distinctions. The public portable facade matrix covers run/step Unicode parity, cancellation/zero/unwind, exact trace limits, foreign source, error/alias retirement and snapshot/restore/serial branch budgets. Native V2 TP/PP/combined frontiers, additional families, initial mixed-media copies and capture/intervention remain explicit validation/implementation obligations; absence of a new test is not architectural inapplicability. Managed/submission/graph policies and already-reserved opaque sources stay rejected without a new original source/control/record admission.


### Preview across captured prefill spans

Supported row-window captured prefill now reports a single global Preview in the historical `prefill_reductions` companion while preserving each physical envelope. The result uses the original selected row-major order and exact global Truncated counts. It remains provisional even when its entire prefix was filled early; cancellation, missing hooks, changed precision or later final indexing/exchange failure makes it unavailable. Skip remains explicit and consumed observation limits do not rewind on restore.

Ordinary and controlled speculative drivers share this collector and retain physical Sequence demand where required, including a multirow final span before sampling selection. Existing original ordinary/control Preview remains separately funded. This does not activate original managed speculative observation, arbitrary whole-tensor callbacks, generated/routed/media or prefix-valued assistant aggregation, or window plus TP/PP composition. Those require their own source and admission integration.

### Ordinary V2 prepared-media snapshots and continuation

The pending-input copy mechanism now preserves every ordered payload, metadata key/value, extent, cache identity and authenticated source attribution. Distinct logical array views receive independently copied storage; shared source backing identity is not preserved or deduplicated. The existing snapshot/restore/fork driver uses the authenticated decoder extent for future prefill growth, including Prepared snapshots, with cumulative copy and transport budgets retained across restores and branches. Raw or relabelled media without consumed attribution and attached original reservations reject before ordinary copy work. Installed capture, including an installed empty plan, remains a separate V2 capability until the captured-media lifecycle unit is composed; ordinary Unobserved is not an empty capture installation. Staged native acceptance covers initial/paused/completed snapshots, repeated restores and serial branches across resident/host/disk; central execution evidence remains required.


### Prepared media with ordinary decoder-row capture

PreparedInputInstrumentation::Capture now uses actual V2 semantic decoder coordinates for supported FullTensor/Slice hooks. Empty Capture remains distinct from Unobserved. The same pending-source installation, collector and physical media chunks serve ordinary core iteration, manual advancement and public prepared controlled startup. One logical p0 frame retains requested full/default shape and scatter order, and is delivered only after final output selection and completion. Cancellation publishes no sampled token from unfinished prefill.

The source owns a finite unbudgeted PrefillDriver request; the ordinary outer text preparation validates it without retaining a second request. Valid snapshot copies receive one fresh source request before restore/fork publication. Exact checkpoint/run lineage prevents equal-plan or lower-ledger substitution, while same-run restoration does not refund observation/copy usage. The shared hook leaves unrelated token-only original inputs unchanged.

This increment does not authorize original managed prepared-media inference, interventions, encoder/non-row/global transform capture, or temporal+spatial distributed aggregation. Those selections retain typed rejection pending their required construction, attribution and workspace mechanisms.


### Global captures on ordinary V2 prepared-media input

The ordinary captured-media path supports Summary, Histogram, Preview and terminal TopCandidates through its existing shared driver. Global transforms cover the full admitted decoder-row selection; Preview retains global selected flatten order and exact Truncated counts, including zero output. Body selections retain minimal readout, while selected readout stages require Sequence before request preparation. Candidate-only plans retain StateOnly earlier spans and LastPosition at the end, preserving raw-score order and the actual sampling-decision domain. Aggregate payloads remain provisional until final indexing and settlement succeed.

Controlled startup, iterator/manual execution and snapshot restore/fork share Unit3's exact source/request/checkpoint binding. Failed/cancelled attempts emit no successful aggregate, and consumed capture/copy limits do not rewind. Staged acceptance adds five local families across resident/host-layerwise/disk selections, actual partial-prefix failure and public restores/two serial branches; this source change alone reports no execution result. Original managed media/speculative input, distributed media capture, interventions, generated/encoder/prefix-valued hooks and remaining global transformations require further source, transport and completion integration.

### Prepared-media frame ownership through controlled delivery

Ordinary prepared-media capture drains as SharedCapturedStep, including empty, skipped and failed frames and decode after the whole-prefill frame. Raw `take_captured_step` probes return None while retaining the same delivery; callers use `take_captured_delivery` and borrowed capture views. Pending delivery blocks another prediction. Frame aliases retain their actual ordinary host authority after session/source teardown; snapshot and fork never refund consumed control or value quotas. Public events retain the existing token/capture_failure wire form. This component does not establish universal custody for standalone raw errors or activate original managed media capture.


### Compiled prepared-media semantics and ordinary control

Original host/semantic sources can enter the existing low-level `TextGenerationInput::Prepared` path after genuine ordinary native upload. The iterator and manual `ControlledTextGeneration` use the same constructor and driver. New source fixtures compare actual compiled-source iterator/manual results and complete local state. The compiled certificate binds the real selected source and current session; it is not a serialized execution grant.

Ordinary copied pending media clears the optional compiled packet and executes its independent per-slot copies through ordinary inspection/admission. Cache identity, exact source-derived attribution, capture attachment, request and cumulative snapshot/copy policy remain unchanged. Original managed copy and request admission remain unfinished. Ordinary cancellation can retain a committed media prefix and use the existing cached-decode behavior; it does not establish funded cancelled-request resumption or refill a budget. Captured prepared startup retains its existing separate validation and accounting obligations.


### Standalone failures from ordinary prepared capture

A completed, skipped or aborted Shared capture frame and a standalone operation error can escape independently. Installed ordinary prepared capture keeps the actual host exclusion through either owner's final retirement, including errors before frame creation and after p0 drain. Cancelled force/clear/sampling-state validation and duplicate snapshot configuration follow V2's existing retained public error route. Restore/fork reuse existing source/copy authentication and nonrefunding budgets.

The error token does not certify native settlement or unchanged model state: existing recovery and the actual original failure determine those facts. It supplies no finite error capacity and does not activate original managed prepared-media or unsupported distributed/intervention capture. Explicit caller formatting/copies and unrelated raw error constructors remain caller-owned.

### Original native media source and shared ordinary advancement

An original B1 native source can be consumed with its genuine A semantic certificate through the same ordinary low-level `TextGenerationInput::Prepared` iterator and manual driver. The closed source does not expose owning native arrays or a serialized execution grant. Source/selection/current-session checks precede ordinary lowering, and completed native leaves are published only after the actual synchronous copies finish.

The two-family iterator/manual fixture uses the real selected driver. Existing ordinary cancellation, cached decode, snapshot/copy attribution and cumulative control policy remain unchanged. Pending copies create independent contiguous slots and clear the optional compiled packet; they do not redirect execution to old B arrays. Public managed prepared-media startup and original copy/capture accounting remain closed until B2/B3/C/D/E are composed. This unit does not claim captured-media applicability beyond existing ordinary behavior.

### Original B2 input ownership and current control boundary

The complete original prepared-input owner can feed the existing shared prepared-media driver under ordinary model execution. The same source route is exercised through iterator and manual core advancement, using the same semantic packet and completion paths. No second inference loop is introduced.

The new original part/cache witness is included in `has_original_input_custody`, so ordinary V2 capture and snapshot-copy paths do not silently adopt it. Controlled original-media startup, source-authenticated snapshot copying, original capture, distributed preparation and managed request admission still require their concrete source/control/completion contributions. B2 does not relax those typed rejections. Binding consumes one unbound source; a rejected attempt retains A/B and cannot be retried from a body or cache clone.

The private original plain-string session advances the same committed cursor manually or to completion. C-domain authentication remains mandatory before and after controller callbacks; cancellation before startup, after S/E, and between predictions preserves original owners without introducing another engine. The terminal text carrier shares R with committed IDs and has no mutable/raw extraction. There is no managed plain-string snapshot/resume/fork or public/default activation in this increment; those operations still need their actual state/copy/event ownership. Native coverage is specified for ordinary/controlled/detached core use, two cached requests and full numerical K/V across resident/host/disk; source-only staging does not imply those gates executed.


### Shared prefill role ownership

C1 retains one shared ordinary/controlled prefill driver and names before/after boundaries independently of pending polls or observers. Original text control roles belong to the actual first active request; their prepared native owners remain through cancellation/failure and independent retirement. Ordinary prepared-media consumers retain the complete B3 first interval and future roots through the same finalizer. Component acceptance does not activate original managed media, original observation/copy, speculative auxiliary role populations or distributed completion controls; those remain required integrations with exact source and resource contributions.

### C2 completion ownership in shared prefill

Uninterrupted and controlled/manual generation retain the same prefill driver and
transaction finalizer. Original completion uses the actual bound current Scope;
a retained sealed Scope may later be observed without starting new work. Scoped
NeedsFundedProgress, Unobservable or Busy is a prompt refusal, not an infinite
wait or permission to perform ordinary global cleanup. Pending accepted frontiers
may wait, while Recovery alone determines when retained roots can retire.

The actual native failure and original custody survive escaped error aliases,
failed-frame/transaction unwinding, scheduler first-error storage and final task
capture retirement. Native diagnostics are copied through fixed scratch rather
than allocating another diagnostic String. Legacy constructors and ordinary
paths remain available. Full event/task/receipt demand and initial native error
producer bounds, original-media admission and distributed completion remain
explicitly unfinished; this source unit does not advertise their activation.


## Original prepared startup and terminal identity

OriginalPrepared enters the same core startup and generation machinery in looping and manual modes. It requires the genuine original sequence request and a complete concrete producer; default rejection and the native precise missing-contribution result are explicit unfinished integration. The nonzero neutral fixture exercises identical state/history/media/position and outputs, uneven 2/2/1 prefill and two cached decodes. Cancellation retains the actual committed prefix and future compact media through the existing owner; it grants no resumed or replenished original request.

Run and continuation identities share the checked never-reused core run ID. Failed/cancelled startup returns no continuation; restore preserves its run and monotone policy/attempt, while fork/resume use a new run. Exhaustion rejects before backend/controller preparation, advancement and preparation-agreement hooks; infallible mutation cannot clear it. Driver ownership remains independently checked by its existing setup Arc. Reserved completed-step receipts can escape retirement as scalar evidence, but a new/foreign/replaced preparation cannot authenticate them.

This low-level D1 work does not activate public prepared-input V2 startup under managed capacities, captured-media startup, original pending-copy/snapshot or funded resume/fork. Those operations still need their exact attribution, record/source ownership, cumulative budget and completion contributions in the same original admission. Ordinary supported behavior and existing V1/V2 wire contracts remain unchanged.


### Original prepared-input inspection progress

Low-level ordinary and manually advanced core startup share D2's authenticated original-input preflight. Both derive the same source/canonical/decoder facts and reject `MissingInspectionStorage` before prediction while the full native producer is unfinished. An installed empty capture still rejects at its own earlier boundary. The complete neutral D1 producer retains actual nonzero run/manual, exact-Q and Q-minus-one coverage through the same shared admission/driver; this does not activate public managed V2/capture/copy/fork startup.

Borrowed inspection cannot outlive its actual session. Fixed outer errors preserve the real fence/commit phase and leave callback errors unboxed; ordinary error compatibility remains. Native capture, pending-input copy, snapshots/funded continuation and cumulative budgets retain their prior ownership gates. Original text-only/projected-only media source admission and the other-family/parallel matrices are unfinished integration work, not architectural exclusions.


## Fixed reports and controlled parity

D3 keeps the existing core generation and prefill drivers. Its closed neutral producer uses the same accepted request for ordinary iteration and manual advancement, retaining exact flat graph/source/scratch custody through results and errors. The tests require nonzero 2/2/1 prefill plus cached predictions, exact-Q/Q-minus-one behavior, source substitution rejection and alias retirement. The ordinary B3 encoder/decoder diagnostic trace also uses the common report reducer.

This does not enable managed native prepared-media startup or V2/capture/snapshot/copy/funded continuation. Those paths still require complete native inspection and completion facts and the existing E ownership/attribution work. Ordinary no-encoder and controlled behavior is preserved; an unimplemented original no-encoder admission producer is not an architectural limitation.


### Request lock contention and terminal fencing

Request preparation and run-state synchronization preserves blocking completion under contention: an unfinished step waits for the same exclusive state and fences its run before Drop returns. It never treats temporary contention as a skipped fence or a refunded ordinal. Supersession still locks both actual preparation owners in pointer order; reserved scalar receipts and ordinary preparation aliases retain their existing identity and lifetime semantics.

The private runtime implementation spins on an inline exclusive try-lock instead of creating native waiters. This trades CPU and fairness under contention for bounded control storage; no finite waiting-time guarantee is claimed. Poison handling remains terminal for the affected authority. Unsupported original lock storage remains a typed unknown bound before admission. This correction does not activate managed prepared-media, V2, capture, snapshot or copy paths.

## Shared original native controls across prefill roles

Ordinary/manual/controlled sessions retain the same shared driver and exact before/after span coordinates. Every original role, including source/span preparation and final score indexing, must configure its own genuine carrier/quotas before native producers; missing or spent authority refuses without ordinary fallback. Empty nested children can replace only an inherited parent carrier with their own preallocated owner, preserving parent state/cause and no-refund semantics.

The component preserves ordinary Graph-only diagnostics and existing whole-sequence verification. Original complete native media, speculative auxiliary and distributed paths still require the listed source/workspace/command/exception producer contributions; these are implementation gaps rather than inherent model limitations. Positive component tests do not widen Complete capabilities.

## Shared original evaluated-array route (source-only)

SourcePreparation, SpanOuter and FinalIndex now share the same original scoped synchronous evaluation in uninterrupted/manual/controlled drivers. The adapter preserves original carrier identity and actual native causes through fixed refusals, owned-array failure and escaped errors. It does not replace the inference loop or release failed-prefix roots on a polling error. Ordinary behavior is preserved.

Timed/cross-stream completion and the later inspector String/Vec construction remain precisely named unfinished mechanisms; they are not model limitations. New tests cover actual core-request role progression across all three residencies plus retained Busy failure after bank/runtime retirement, while existing numerical, snapshot and exact-Q tests remain required. No managed capability is widened; compilation and execution are pending root validation.

Ordinary and controlled external Gemma4/Muse default prefill share the same
completion-gated driver and demand selection. Span-compatible consumers use the
runtime default; callbacks without span/context semantics keep one whole default
invocation and their declared Sequence demand. Explicit incompatible split
requests still reject before target work. Cancellation is checked at shared
boundaries and cannot refund already completed target work. Direct verification
continues to consume every evaluated score row.

Ordinary and controlled materialized embedded default prefill share the existing
captured-span driver, lane exchange, shifted seeding and completion boundaries.
Callbacks without span semantics retain one whole default invocation and their
declared Sequence demand, with their existing invocation callbacks. Explicit
incompatible split requests reject before target work. Cancellation preserves
completed target counts and cannot publish a partial initial seed as success.

### Original prediction diagnostic replay

Repeated original SamplingEvent failures preserve the source snapshot observed
at each point: an earlier Busy without a native cause stays distinct after the
carrier publishes its single immutable native cause. Repeated equal observations
retain an existing source. A contradictory/unclassified result is a typed sticky
terminal error retaining the first offending cause; it cannot become successful
output or authorize resource retirement. Original scalar reentry/interruption
and poisoned-health errors retain fixed sources, including old SamplingEvent
custody. Public controlled and uninterrupted error conversion shares the same
provider hook. Original producer-fit, allocator measurements and all remaining
prediction-role execution closure are still required integration work.

Controlled startup, prospective sampling changes, snapshots, restore and branch
operations preserve the selected provider's error ownership at public erasure.
Invalid policy and host/preparation failures retain their existing typed branches;
a backend polling/conversion error remains neither completion nor permission to
release retained resources. This conversion change adds no original admission,
new snapshot capability, or producer-fit guarantee.

### Speculative retained-error transfer

Controlled scheduler and reseeding failures can transfer an existing neutral source through an explicit executor/sampler hook, retaining native custody without a second source wrapper. Default refusals preserve ordinary classification and driver source chains. Failed reseeding leaves temperature and randomness unchanged; typed control, output and cancellation branches are not flattened. Native speculative error storage, capture's dual-owner transport, wider controller publication allocation, and original speculative completion remain separate required work. Six new neutral behavior cases accompany this source change; execution is pending the combined validation record.

### Dense speculative first-error custody

An opt-in paired capture transport preserves the first stored control error while a later execution signal independently retains its own source. Neither observer drop nor dropping one alias releases the other source's custody. Default ordinary mappers, typed admission, sparse transport and completion semantics remain unchanged. Original producer/source populations, typed original observer projection and native completion fit are required before activation; the paired helper supplies no admission authority. Nine source-level behavior cases accompany this change, pending combined execution.

The private original speculative completion reports Busy/funded/unobservable outcomes with retained immutable source custody, validates actual completed roots, and uses existing same-node retirement for success, error and cancellation. Paired capture transport can retain independent execution/controller aliases of those sources. This source component does not activate original speculative scheduling or close dense/sparse transforms, request occurrence banks, sampling/cache/snapshot owners or native producer fit. Existing ordinary and controlled speculative drivers remain the policy owners; new fixture/source coverage is not full original-execution support.

An original exact-root completion retains the same prepared recovery/payload
owners through root reserve, count refusal, native failure and retirement. A
nonnull partial native output is owned even when construction returns failure;
the safe adapter establishes that owner before error propagation. Empty and
evaluated roots still submit a real selected-stream frontier. Readiness alone
does not release original authority, and the same-node payload-before-cleanup
order remains required. The new storage facts do not activate speculative
requests or close per-request producer counts, source/capture storage, native
DAG/Record/platform fit, or the original mechanism/context projection.
### Descriptor observation and retained control

The low-level MLX descriptor companion borrows the exact source and an existing runtime guard through count/fill. Controlled and ordinary orchestration can share this mechanism without a second inference loop. Its loan grants neither submission nor completion authority, and a changed source/state refuses before destination writes. The ordinary owning metadata adapter remains available; shared outer-guard retirement and selected inspection storage must be closed before this primitive can support a complete original prepared-media request. No controlled feature is advertised solely from descriptor availability.

### Shared native metadata ownership

Controlled and uninterrupted existing-state inspection share the same native ExistingArrayProjection and fixed descriptor owner. The real runtime guard remains live through shape/layout consumption, and unknown backing is not promoted to known. Busy and unsupported-dtype errors release only this operation's own guard; foreign runtime ownership is preserved. The ordinary waiting policy changes only the private lock's blocking route, preserving deadline and immediate recovery behavior. This adds no inspector timeout or separate controlled inference driver.

### Private original chat request preparation

Original J/H preparation feeds the same original plain-string startup and shared cursor used by uninterrupted and controlled advancement. Both generation-prompt variants are measured and constructed before H publication; the exact selected borrowed rendering uses add_special_tokens=false as ordinary chat encoding does. H authenticates the same J/C account, source objects and concrete consumer layout. It remains owned through later S/E/I/R startup failures and retires after the actual E→I handoff; aliases retain their original charge until the final strong owner.

The source tests cover cancellation before file work and after H/S/E, EOS-domain rejection before S, exact-short admission, actual reserve/settlement errors, concurrent renders, repeated cached requests and returned shared text/ID lifetime. A full config/tokenizer oracle compares 48 private requests with ordinary rendering/encoding (44 completions and four existing typed empty-input rejections). Native Metal coverage reuses the ordinary/control/detached, three-residency full-state matrix. These are source validation obligations until a central record reports executions. Native CPU numerical chat validation remains pending.

The supported private request borrows string role/content pairs through synchronous H construction. General request ingestion, additional chat semantics and public/default whole-inference activation remain mandatory follow-ons. This limitation describes implementation coverage, not an inherent model or controlled-session restriction.


Ordinary MLX prefill coordinates access to the runtime before each one-shot
reservation entry. This prevents unrelated ordinary runtime contention from
turning an otherwise valid prefill boundary into a terminal Busy failure.
Coordination may wait for the current runtime owner; it adds no hard latency
guarantee. It retains no runtime loan across input preparation, numerical work,
transport, callbacks or completion. The existing shared cancellation boundary
samples the current token after entry and readiness, before its cancellation
vote and the next chunk. Ordinary and controlled execution use the same seam.
The underlying immediate entry APIs and installed original control banks keep
their nonblocking refusal and exact resource-retention semantics.

Public managed plain text now applies the caller's domain ceiling before stop
compilation and prompt encoding, for both uninterrupted and manually advanced
sessions. They share the same preparation account and existing generation driver.
Pre-start cancellation still performs no source preparation. A refused source or
later startup retains the ceiling through its owning failure; successful startup
hands enforcement to the admitted request without a gap. Focused portable checks
cover the early one-byte refusal, concurrent source accounting and ordinary/manual
parity. Complete native activation still requires the remaining built-in kernels.


The private original chat preparation now carries the same caller ceiling before
rendering and across the delay until session startup. Startup consumes that
preparation and rejects a changed capacity before stop compilation or encoding.
Both manual advancement and uninterrupted execution continue through the shared
plain-text cursor. Cancellation retires rendered sources before their ceiling;
a retained startup error keeps both until its real resources retire. Four focused
portable checks cover output parity, early render refusal, cancellation, changed
capacity/source identity, and failure lifetime. This does not add public managed
chat tools, grammar, snapshots or fork support; those integrations remain required.

`LoadedModel::start_managed_chat` and `generate_managed_chat` share the existing
consuming `ManagedPlainTextSession` cursor and borrowed text-event/output owners.
Use `compile_managed_plain_text_source` for the original tokenizer, then
`compile_managed_chat_source` for the selected tokenizer-config template. Startup
requires the caller's managed-memory ceiling before rendering and encoding;
missing complete backend fit still rejects. Pre-start cancellation produces no
session, and explicit advancement has the same sampling, stops and first-terminal
reason as uninterrupted generation. Tools, parallel tool calls, explicit
reasoning and managed chat snapshot/resume/fork still need their original
policy integration. Template contexts now borrow the actual tokenizer defaults
and caller replacements through both measured and admitted renders. Ordinary
precedence applies, including replacement of `messages` and
`add_generation_prompt`; template loop locals still take priority. No supplied
replacement is silently ignored. The selected scalar context supports string,
boolean, null and JSON-number values through the existing formatter, truth and
qualified expression operations. Message-list replacement uses the same exact
role/content profile. Structured-value formatting, general indexing/iteration,
and other unimplemented operations still return a typed render/source error;
this does not qualify arbitrary templates. The source compiler now handles text, message role/content access,
concatenation/inequality, short-circuit `and`, conditionals and nonnested message
loops with the actual local variable name. Decoded source, syntax and emitted
instructions are reserved under the original source allowance before compilation.
Filters, calls, macros, nested loops and expressions exceeding the current VM
operand capacity still return typed source-profile rejection. These are
implementation gaps, not inherent model restrictions;
managed chat snapshot/resume/fork remain outside this initial public connection.

The selected managed dense ready-host fixture now matches ordinary and controlled
public generation with a one-unit device window across two nonzero layers.
Five prompt tokens use 2/2/1 chunks; three cached decode steps produce four total
output tokens (`[8,38,26,1]`, `iMAb`). The shared request also covers pre-start
cancellation, a one-byte refusal and output retained past model/source teardown.
Transfer consumers and completed residency pins retire at their exact owners;
no separate controlled inference loop is introduced. This is a selected-path
milestone, not whole-family, disk, parallel or state-operation coverage. Commands
and scoped results are in `validation/bounded-public-resident-chat-2026-09-16.json`.

Ordinary snapshot capture and restore now consume borrowed complete host-copy plans through the shared runtime transaction. Snapshot-budget reservation precedes pipeline, cursor, delivery and metadata destination copies; restore also stages its prompt before native exchange. Failed copy attempts retain the existing cumulative-budget consumption. The managed capture path adds separately admitted host preparation and native-copy custody to this transaction. Fresh restore and fork now pass for selected resident dense/hybrid plain-text fixtures, as detailed in the later restore/fork milestone; broader state and residency combinations remain unfinished.


The managed plain session now has a snapshot capture gateway over the shared
completed generation machine and retained cursor copier. It reserves logical
snapshot usage before independent host/native destination admission and keeps
failed-attempt copy consumption monotone. Its snapshot is immutable saved data;
capture itself grants no detached driver identity. The shared restore/fork API performs fresh admission and destination construction; selected resident dense/hybrid plain-text activation is validated by the later four-case milestone.

MLX implements `original_saved_components_preparation_bytes` for a quiescent
decoder with no pending input, one decode token, or the complete prepared
plain-text token matrix. Its cold query composes the
actual finite source, projection, registration, copy-program, native execution,
publication and returned-owner constructors. Host preparation is admitted before
those constructors; the independent copy account admits the larger of the
portable numerical bound and the native physical copy envelope before copying.
The shared saved-copy driver retains both authorities through native completion,
publication and final saved owners. Original observation checkpoints still need
their own admitted destination producer. The public post-commit capture passed against a nonzero dense fixture: the saved
cursor contains token `[8]`, live generation continues with `[8,38,26,1]`, and the
saved owner outlives model/source teardown. Snapshot-count and one-byte refusals
preserve nonrefundable copy accounting. This qualifies the selected resident
post-commit capture path. The later four-case public restore/fork milestone below
also covers selected resident dense/hybrid pending-prefill and post-commit state;
the wider input/state matrix remains.


Original snapshot copies retain authenticated source tables and prepared-input
identities without acquiring unquoted allocation authority or charging their
existing storage again. The shared capture worker passes its accepted host
preparation lifetime to native source containers and recovery. Numerical copies
still require their own bounded admission. The joined managed capture path uses
these source owners together with finite native execution and destination
publication. Selected resident dense/hybrid plain-text restore/fork is now
validated as recorded below; the wider state and input matrix remains unfinished.

The public foreground disk regression now exercises three nonzero decoder units with a two-unit device window, zero host-cache capacity, zero lookahead, and no background queue. Managed chunks of 2/2/1 plus three cached decode steps agree with ordinary and controlled generation on four output tokens. The same managed case checks cancellation before work, one-byte capacity refusal, and output custody after the model retires. Device inventories include the immutable host sources retained by canonical device cells; ordinary calls on these managers consume the retained detached reads through the existing transfer path. This milestone does not cover background prefetch or complete managed snapshot support. Exact incremental results and build scope are recorded in `doc/validation/bounded-public-resident-chat-2026-09-16.json`.

The pending-prefill capture successor borrows the complete prepared plain-text token matrix and feeds it through the same isolated-copy/native/publication plan. It retains immutable pending-kind and geometry metadata, preserving the distinction between a prompt and a committed decode token. The facade delegates qualification to the backend instead of unconditionally rejecting pending prefill. The public prefill phase passed: its saved cursor remains empty at prediction zero, subsequent live generation produces `[8,38,26,1]`, and the saved owner survives model/source teardown. It also verifies count/one-byte refusal and nonrefundable copy accounting. Selected resident dense/hybrid prompt resume now passes as recorded below. The later prepared-media milestone also passes selected pending-prefill and postcommit restore/fork with resumed execution and retirement; broader media/state combinations remain unfinished.

The same public snapshot case now also passes with the selected Qwen hybrid recurrent-state fixture. Its saved prefix is `[63]`, and live continuation produces `[63,32,32,32]`; saved custody and logical-budget refusal checks match the dense case. This extends capture coverage only.

Managed restore and fork now share the core saved-generation driver and the runtime snapshot transaction. The immutable saved owner supplies the original cumulative snapshot budget; callers cannot replace that budget or increase the remaining output allowance. After independent destination admission, the logical copy reservation moves into the actual copied provider account. Generation and output aliases retain that account, so dropping a returned session cannot release the charge while its output remains alive. The focused owner-lifetime regression passes, including final output destruction on another thread. MLX public managed plain restore and fork now pass for selected dense and recurrent/full-attention hybrid resident fixtures, both before prefill and after the first committed token. The cases verify output parity, cancellation, one-byte rejection, immutable snapshots and cumulative copy usage. Final model destruction queues native semantic payloads; their logical leases remain valid until the existing explicit `MlxNeuralBackend::reclaim_retired_resources()` boundary retires completed owners. The tests preserve exact zero-retention assertions after that boundary. Broader state, residency, chat/controller, capture, media, speculative and distributed combinations remain separate required work.

### Speculative submission and rollback failure custody

The shared verification submission transaction now retains both typed backend
errors when submission and checkpoint restoration fail. The submission error
remains the primary source; the rollback error is retained alongside it without
allocating another diagnostic wrapper. Restoration still runs while the original
failure, checkpoint and draft block are owned. Successful restoration returns the
unchanged submission error. Failed restoration grants no completion, cache reuse,
publication or budget refund. Ordinary and controlled schedulers consume this
same transaction. This correction does not qualify original speculative request
funding, round temporaries, or the broader resolution/cancellation copy paths.


Managed resident execution with appended prediction groups must route each
initial/group boundary through the request's prepared neural completion bank,
even when drafting is disabled. The released Qwen3.5 checkpoint's retained
prediction group exposed an ordinary nested submission at this boundary; the
tiny hybrid fixture without prediction groups did not exercise it. The resident
policy now uses the same finite boundary producer as the bounded policy, and
both original policies consume their accepted single-stream choice in the
shared layered loop. Original errors retain their underlying cause and request
custody through neutral transport. This change does not activate request-wide
speculation or alter cumulative budgets. The pinned released managed run remains
the native validation target; this staged change has not itself been built or run.


### Source-explicit managed plain speculation

`ManagedPlainTextSpeculativeRequest` combines the existing borrowed
`ManagedPlainTextRequest`, explicit draft selection, proposal/scheduler options,
cancellation and committed-event callback. `generate_managed_plain_text_speculative`
and `with_controlled_managed_plain_text_speculative` consume the same original
host/prompt preparation and shared backend driver. Controlled mode supplies
`DriveControlledSpeculation`; stepping, tentative proposals, commit, rollback,
termination and cumulative snapshot/copy budgets remain in that driver. The
original tokenizer provides a borrowed vocabulary census instead of allocating a
legacy vocabulary map. Source identity and the request ceiling are authenticated
before native source work. This new consumer has no ordinary preparation fallback.

Original EOS, callback and semantic owners keep the authenticated preparation
through handoff. Returned events and failures retain their actual accounts. A
controlled failure keeps its exact typed cause; its paid outer transport uses a
fixed marker instead of constructing a second diagnostic String. The existing
ordinary error message path remains unchanged. Closed speculative records count
compact JSON with an infallible byte counter before the same transport comparison;
per-record or cumulative refusal changes no consumed-byte count and creates no
secondary io/serde error owner. This is not admission for user serializers.

The selected original native path currently requires one independent external
assistant with completed prompt and exact role/occurrence bounds. Wider batches,
embedded prediction mechanisms and structured semantic profiles remain unfinished
applicable integrations. The observed managed entry points use original capture
sources and the same sampler-input/claim driver; raw numerical parity and shared
snapshot spending are covered by the public ordinary/managed/controlled cases.
Static edit installation remains gated pending its actual native producer.
Snapshots use their existing known-policy and native-copy checks, retaining
cumulative refusals and source identity rather than creating a new request account.

The static-edit preparation component now copies immutable declarations into an
original source account and validates loaded discovery by borrowing. Evidence
`None` has attributed outcome destinations in the same numerical/frame claim row,
including paid aborted delivery. These records cannot report success before an
edit receipt, and a failed attempt cannot refund claims or metadata. This is not
yet public original edit execution: the shared native activation worker still
needs its exact input/region/mask/payload mechanisms, and Preview/Summary evidence
needs source-owned companions. Ordinary intervention execution and wire records
retain their existing behavior.

Original capture has a closed host-only checkpoint and fresh continuation-bank
producer. It preserves the admitted plan identity, absolute frontier, and saved
cumulative usage without copying declarations or resetting the source bank.
The native saved-components transaction and observed continuation quote now
construct that fresh destination for the existing raw model.logits profile.
Same-run restoration preserves the live cumulative ledger; an independently
admitted fork inherits saved usage. Joined native validation remains required.


Original native text capture quotation now consumes the same observed equations,
logical capture ledger and physical prefill-fragment policy as ordinary capture.
For the existing raw `model.logits` transforms, actual accepted transfers feed
native completion/publication requirements; skip decisions and cumulative limits
are not recomputed by another scheduler. Sequence readout is preserved when the
request requires it. This lower-level join depends on the corresponding native
prepared-completion/source-publication worker. It does not enable speculative
capture policy or other observation transforms. Captured snapshot/restore/fork
use the separately admitted saved ledger and collector described below.

The funded capture restoration producer preserves post-snapshot cumulative
spending through a closed shared source. Repeated same-run restorations cannot
reuse spent observation allowance. Explicit independent forks retain the
existing separate-budget rule: their newly admitted ledger inherits saved usage.
The native saved-capture route consumes this distinction through the core
original-resume intent and the shared Sampling installation transaction.

Original managed text snapshots can preserve the existing funded raw
`model.logits` capture source at a successfully drained boundary. The snapshot
stores the immutable frontier and shares its cumulative ledger through the
existing saved-component transaction; capture plans, callbacks and native
execution grants are not copied or reset. Logical copy allowance is consumed
before the new fixed host destination and native copy. Fresh observed restore
and fork install their own original capture bank and retain the actual source
on the shared generation machine. A saved pending token uses one physical
prefill at its absolute logical decode coordinate, with original source/role
checks before execution. Full-sequence demand, quota refusal and native
completion remain shared with live capture. Pending-prompt and post-commit
snapshots support a shorter output allowance through the sealed continuation
geometry. Pending prompts retain the original axes and fragment schedule; only
the remaining output interval changes. Joined native validation is pending.

A shortened original plain-text restore or fork applies its new remaining-token
allowance to the copied shared generation sequence as well as the backend driver.
Core owns the monotonic logical endpoint; the retained provider keeps its exact
original token capacity, decoder storage and custody. Readiness validates that
physical capacity, and subsequent copies preserve it. This prevents an extra
source request after the resumed backend has produced its admitted allowance,
without resizing storage, refunding budgets or reopening a terminal sequence.

Speculative managed Summary capture uses the same ProcessLogits phase as raw
rows and selected scores. Ordinary, managed and controlled entry points share
that phase, its cumulative capture ledger, delivery queue and failure evidence.
Summary reads the selected original sampler input before forcing or truncation;
restoration does not rewind its capture spending. A separate numerical claim
validates each invocation even when target/draft or replay coordinates coincide.

Original speculative static activation evidence reuses the shared before/apply/
after intervention order for Preview and Summary. Each companion selects the
operation's exact region and keeps its operation ID, before/after position and
logical coordinate. Later operations observe earlier effective outputs. Snapshot
restore/fork retains the same live capture ledger; it cannot refund evidence
charges. A value-limit skip keeps its attributed metadata, and an after-evidence
failure preserves any completed before payload in an aborted frame. Successful
host delivery still waits for the enclosing native numerical completion. The
source/destination producer is separate from installation preflight, whose
estimates must consume these same immutable companion declarations.


Original managed independent speculation installs admitted static model.logits
edits through `intervene` at the same controlled boundaries as ordinary runs.
The loaded target declaration authenticates both role wrappers; native source
geometry is checked separately for each role. Installation prepares both paid
immutable role sources before replacing the live set. Snapshot/restore/fork
retain these sources and the existing cumulative capture lineage. A failed
replacement leaves prior edits intact, and clearing edits does not reset capture
usage. Initial original installation supports `InterventionEvidence::None` with
paid attributed outcomes; source-owned Preview/Summary companion installation
remains an explicit unfinished producer, not a model limitation.

Static evidence installation preflight consumes the same paid companion source
that later supplies numerical claims. It compares before/after metadata and value
estimates against the actual enclosing capture limits, using the existing Fail/
Skip policy. A failed replacement preserves current interventions; snapshot/fork
continues to alias the installed immutable source without resetting capture usage.


### Source-funded static speculative interventions (2026-09-16)

The existing controlled speculative setup callback installs admitted target/draft
`model.logits` edits through `intervene`. Managed execution authenticates each plan
against the retained loaded declaration and target session, copies its immutable
source under the existing source budget, and validates both candidate role slots
before replacing the active pair. Both roles preserve ordinary admission semantics.
An explicitly supplied empty capture plan can retain the outcome ledger without
requesting raw logits. No additional generation driver is introduced.

Static Zero, Scale, Mask, Replace, Add and MaskLogits use the shared activation
worker at this declared vocabulary boundary. Component masks require a separate
point with a declared component axis. Before/after Preview or Summary evidence
uses each operation's exact selected region, is charged to the enclosing capture
ledger, and is delivered only after the numerical completion frontier. Ordinary
captures continue to observe the original logits. Failures retain completed
before evidence and already spent credits; restoration does not refund them.

The saved sampler retains immutable source aliases and their actual installation
funding. Duplicate-role rejection leaves current edits intact; clearing changes
the live plan, while restoration recovers the saved plan. Public nonzero fixtures
compare ordinary execution, continuously drained managed execution, and controlled
restore/fork/clear/recovery, including six composed actions and independent
per-operation Preview/Summary host oracles. Exact commands and artifact hashes are
recorded in `validation/bounded-media-and-submission-2026-09-16.json`. Broader internal
activation, prediction-assistant and distributed original integrations remain
separate unfinished work, rather than inherent model restrictions.


Embedded prediction has a source-authenticated prepared lane-state producer.
It preserves the architecture's selected strategy and actual prototype geometry,
including populated model-state copies, and retains the original host account
through destination and error retirement. This component alone does not enable
managed Embedded generation or controlled continuation: the shared embedded
occurrence, cache, observer and checkpoint adapters must consume it first.
Until that join is complete, the public managed Embedded route keeps its typed
preparation refusal; the supported independent-drafter route is unchanged.

The shared speculative executor copies optimistic draft state through an explicit
fallible `copy_draft_state` operation. The coordinator agrees copy failure before
any rank enters independent proposal work. Embedded state erasure forwards that
operation instead of requiring infallible `Clone`; prediction transactions use
the same fallible copy worker for checkpoint and advanceable replicas. Ordinary
implementations retain their existing clone behavior. This seam does not certify
original Embedded cache construction, tensor copies, or erasure allocations;
those producers remain required before public Embedded activation.


Managed chat template contexts additionally support borrowed JSON member and
array indexing through the existing renderer, signed indices, scalar equality,
`not`, and `is defined`/`is undefined`/`is none` conditions. Missing members retain
ordinary lenient behavior, including failure when a further lookup targets an
already undefined value. Tokenizer defaults and caller replacements keep the
same precedence in manually advanced and uninterrupted sessions. Structured
formatting/deep equality, JSON-container loops, string indexing and remaining
compiler operations still reject explicitly; this does not enable tool or
reasoning controllers. Escaping output/error owners contain no caller references.

Original embedded prediction startup now retains the actual prepared stream,
copy environment, exact executable origin and authenticated host preparation.
Its fallible saved-lane copy dispatches architecture-owned current state
membership to the existing registered compressed/pooling/resident copy workers;
it never recreates a prototype. Per-member copy custody outlives destination
arrays, and native resident variants move into their exact typed state without
an extra erasure allocation. The closed lane also retains architecture-declared
physical occurrence shape and prefill alignment separately from requested width.

These copies run at settled boundaries outside an active numerical scope. Target
copies can consume the existing completion-bound resident source witness. New
prediction role births still require their actual source publication before a
subsequent registered copy; identity alone supplies no source discount. Public
original Embedded execution still requires the shared cache/erasure constructor
and occurrence/native invocation joins; this copy producer does not open that
gate or change the independent External strategy.


Embedded occurrence accounting accepts the root Embedded schedule, exact claims
and paid continuations directly. Target/prediction startup and model roles use
the shared original account worker; full equation geometry and exact invocation
must match. Failed attempts stay spent, successful aliases retain one charge,
and restored continuations add paid slots while retaining every prior charge.
This does not enable public original Embedded execution by itself: source/state
binding, actual native invocation and completion still require their consumed
backend producers. Independent AR APIs and driver behavior remain unchanged.


The shared embedded cache can retain a source-bound prepared copy provider and an
independent completion witness for each target/prediction state. Saved state,
prediction forks, commit copies and restored replacements use actual current
payloads; a failed second copy drops its completed prefix without changing the
source. Prediction-prefill rollback consumes the existing checkpoint and does not
require fresh copy funding after the failed operation. Temporarily installed
target state still yields no control snapshot, matching the ordinary path.

The native producer must install the exact state witness after its own successful
completion and retain independent-copy custody on copied states. This neutral
carrier does not enable original Embedded public execution by itself and does not
reset occurrence, capture or copy spending. The existing original External route
and its controlled behavior are unchanged.


Managed chat rendering additionally supports built-in scalar/container type
tests and short-circuit `or` with the same ordinary operand and undefined-value
semantics. Manually advanced and uninterrupted requests consume the same
selected default/caller context, and retained output/error buffers keep their
original host custody after caller context retirement. This expression coverage
does not open the separate tool/reasoning semantic gates or certify arbitrary
templates, JSON-container iteration or structured formatting.


Original speculative controller sampling and its RNG snapshots now borrow one
common numerical-source owner. The owner retains the existing exact tagged
request and cumulative account across every shared program. Binding it without
an AR model occurrence is supported; this grants only access to the existing
numerical admission path, not a model role. Controller capture/intervention
sources keep their existing request identity and ledgers. Model logits still
require an actual completed model-role producer before numerical consumption;
this source factoring alone does not enable original Embedded generation.


A context-bearing Embedded cache factory now consumes the actual paid startup
lane and copies canonical target state at the shared target-preparation boundary.
Commit/fence checks, exact layout validation and rank agreement precede handoff;
failed preparation consumes its one-use startup slot without changing canonical
state. Per-state completed-source witnesses drive later snapshot/fork copies,
and copied registered values retain their own copy account. Neither a shared
mutable completion witness nor a replay budget reset is introduced.

The final erased cache retains its paid host owner after the typed payload.
Construction failures travel as retained BackendFailure without native diagnostic
formatting. Original Embedded generation remains pending its actual model-role,
completion and subsequent invocation/error-destination joins; this constructor
does not alter the already supported original External controlled path.


The selected native operation bank can retain an exact Embedded invocation role
through source construction, neural submission, failure and retirement. Each
Embedded prefill role is already one physical span; its one-use claim remains
spent on failure. AR and Embedded role identities remain distinct while sharing
the same bank workers and cumulative account mechanisms. This component does
not change snapshot budgets, optimistic capability or the public Embedded gate;
actual source-bound invocation/completion integration remains required.

Embedded executor error transport now preserves original typed causes and their
funding through the shared ordinary/controlled driver, including captured prefill
and observation failure resolution. Retained source extraction survives executor
erasure; error conversion does not declare native resources completed. The
context-bearing prediction invoker borrows its target session separately from the
request context, so provisional state recovery and source custody retain their
existing lifetimes. Public original Embedded enablement still requires the actual
source-bound phase, input and readout producers; this error migration adds no
capability declaration and leaves external-drafter behavior shared.


### Internal activation readmission under managed execution

The shared controlled driver always invokes the selected executor's activation
validator before replacing prospective edits. Ordinary execution supplies loaded
discovery; a managed original observer validates against its retained, paid
declaration source. Missing discovery still rejects through the default validator.
This preserves mandatory source/plan authentication without rebuilding a second
discovery report during managed advancement.

Internal activation outcomes describe the enclosing model observation transaction.
A completed transaction is Committed; its outer speculative phase and origin still
identify tentative proposal/verification work. Sampler-input capture keeps its
existing Untracked disposition. Logical aggregate records attribute physical
contributor charges; only the report-level additional charge represents new
aggregation storage/encoding spending. Restore and fork refund none of these costs.

Original managed paged Host generation uses the same request, append, scan and
commitment driver in uninterrupted and controlled modes. Its actual accepted
Host source component enters the same native role/account; source extraction
does not replenish quotas. Canonical Device occupancy may be lower than live
physical pool occupancy while completed roots and replaced backings remain
owned. Host saved-state copy/resume/reset remains gated pending its source/copy
producer; the already-qualified Device paged lifecycle is unchanged.

Managed live Disk paging is connected to the shared plain-text request and
controlled advancement drivers for the currently qualified full-attention paged
state. Snapshot, restore and fork share immutable exact writer-file sources in
independent managers, preserve mutable tails, and consume the existing copy and
source admission. Reset uses its independently admitted empty namespace. A cold
option match never substitutes for current source, queue, destination, completion,
capacity or pin validation; unknown physical sources and unqualified pending
storage remain typed refusals. Public staged coverage forces actual Disk writes
and reloads, compares ordinary/managed/controlled output, and checks saved
continuations plus final file and account retirement. Central native results are
still required; sliding/key-only paging and external persistent-file parser
admission remain separate implementation work.


Managed plain chat now admits explicit profile controls through the same
behavioral selector and request-validation workers as ordinary chat. It maps
Inkling thinking to its named effort, preserves Muse strength/required generation
prompt and IFM/Qwen effort rules, and retains exact tokenizer/template/control
custody through the original render and shared startup. Unsupported source VM
operations or insufficient funding remain typed failures; no profile is inferred
from a model name. Explicit thinking requires `allow_unparsed_reasoning` for raw
text output. Semantic parsing/tools retain their own source admission. Requests
with no explicit controls also use the ordinary behavioral profile recognizer;
its selected policy and source custody continue through the shared renderer and
startup.


The reverse external placement checks use an explicit CPU target and Metal
assistant in the same public generation and controlled-continuation fixture.
Ordinary, managed and controlled modes load the identical artifact pair and
selected CPU Float32Tiles mechanism. The continuation case includes saved-state
restore, fork/exchange, forced-token isolation and calibrated cumulative copy and
trace refusals through the existing shared driver. These ignored checks are
queued for retained target-source qualification; they are not a claim of reverse
placement support. The production binder remains closed until the full CPU
target, numerical, transfer and saved-copy consumers qualify.

### Managed prepared-chat controllers and tool completion

Source-explicit managed prepared-chat speculation uses the shared uninterrupted
and controlled drivers for the selected built-in Forbidden, Active and Auto
controllers. The retained tokenizer, grammar recipe and declared source supply
the existing constrained sampler, semantic decoder and commitment worker.
Controller/history/parser copies use independent paid destinations; immutable
events and escaped failures retain their actual source and payload payers.

The public native semantic fixtures compare ordinary, managed and controlled
output for Forbidden, Active and Auto. The Auto case samples a real complete tool
token, emits the function name and JSON argument fragments, invokes registered
full-schema validation, emits exactly one ToolCallEnd, and reaches Finished.
Its first token is terminal, so it correctly submits zero speculative verification
rounds. The Active and Forbidden cases also exercise verification rounds.
Component conformance covers split/atomic Auto activation, history and EOS,
independent controller copies, and failure/escaped-output custody. These are
completed controller paths; new combinations still require their actual sources.

Full-schema completion uses the pinned ordinary engine with a paid borrowed JSON
tree. Validated source coverage includes primitive and nested constants/enums,
property/required/additional-property rules, array items, string lengths,
primitive numeric bounds and integer multipleOf, recursive references and URI
storage, compiler-selected literal patterns, propertyNames, dependencies, and
small/large uniqueItems. Shared tests verify ordinary result parity, duplicate
key/order behavior and first-cause funding refusal with retained input custody.

Required workspace with no qualified source remains a typed refusal before the
unknown worker executes. Concrete built-in gaps include general regex/pattern-key
execution, fractional multipleOf, retained unevaluated evaluator graphs and some
object-valued contentSchema annotation storage. Draft-selected asserted format or
content helpers also need their own source. Arbitrary extension callbacks have
no inferred workspace bound; the facade's default compiler installs no such
callbacks or remote retrieval. These refusals describe the selected operation
and required workspace, rather than an entire model family.

The focused modern-draft format/content string annotation and
contains/minContains/maxContains checks pass at their recorded scopes.
Annotation-only schemas keep the ordinary draft semantics; their source does not
authorize format assertion or content decoding.

The public comparisons live in
[the prepared-chat semantic fixture](../eredu/tests/native_execution_control/managed_plain/embedded/semantic.rs);
the bounded schema cases live in
[the schema source fixture](../eredu/src/runtime/chat/tool_schema/authority_tests.rs).

### Explicit managed realtime frames

The MLX realtime context exposes `prepare_realtime_frame` with an explicit
shared-domain capacity and optional frame limit. It inspects the retained loaded
source, canonical state, frame inputs, sampling and actual execution policy before
constructing a branch. A returned branch owns its one-use preparation.
`submit_prepared_realtime_frame` consumes that preparation; completion and
publication use the ordinary realtime coordinator. `run_realtime_bounded`
composes those callbacks with the existing fair scheduler, including its
cancellation and terminal-work retention rules. The caller chooses capacity;
physical-memory observations do not supply an enforced limit.

The source program prices numerical work, frame inputs, copied prior state,
metadata, host output and native completion ownership. Host/Disk operation banks
and pure-TP communication sources feed the same frame traversal. Prepared
metadata funding remains retained on failure. This is an Eredu-managed resource
bound, not a whole-process or operating-system memory guarantee.

Resident managed realtime passed four-frame ordinary parity, typed tiny-budget
refusal before branching, unsubmitted-branch discard and release/resume in
milestone 223. Milestone 229 closes the complete Host check (0.59447 s) and pure-TP
check on both real Ring ranks (0.72220 s): all four frames match their ordinary
reference, and managed TP records the same actual model-collective count as
ordinary TP, separately from scheduler consensus. Those results use artifact
SHA-256 `48cea70f9d33f4585701331026d04dbf537630e858dceec62ae7f29ccc1ff9c8`.
Milestone 235 closes the Disk prefetch check (0.44317 s), with all four frames
matching ordinary numerics and state frontiers. Its witness measures positive
DiskToHost count/byte deltas after the pre-frame baseline, joined completed
background work, zero background failures and a peak queue depth of at most one.
The runtime SHA-256 is
`ad5ca3b5041d368e5ac2c24a66224e3375c57fff33cd3e6bfe1df1cc563775ca`; production
is unchanged from the 234 itinerary correction. The three neutral distributed
preparation/completion checks pass, including partial refusal, completion-gated
commit and retention of the first typed failure while a peer remains pending.
Separate paid full-pager preparation, rollback, canonical revision custody and
manager-report retirement checks have passed. All six feature configurations
resolve to passing results, including native CPU-only and default facade plus
image/audio. Portable behavior combines 124 original backend-conformance passes,
one corrected exact retry, 30 portable facade cases and one realtime case:
156 passes with one preexisting checkpoint-dependent ignored case. Initial failed
commands and their follow-ups remain recorded separately. Exact commands,
artifacts and results are retained in the
[native milestone record](validation/bounded-media-and-submission-2026-09-16.json).

Standalone PCM codec calls outside the public token-frame contract do not share
this account. Ordinary realtime pipeline and combined TP/PP were already
unimplemented in tracked HEAD c513e17c; the existing typed rejection remains.
Realtime snapshot/fork APIs were also absent; release/resume of the existing
session is a distinct supported operation. These are preexisting implementation
boundaries, not architectural limitations or extensions of decoder prompt-cache
persistence.


### PCM composition and the realtime request boundary

Realtime frame control accepts already encoded audio tokens through
`RealtimeInputFrame`. The managed prepare/submit methods and the ordinary
scheduler use the same token-frame state and publication driver. Existing
low-level Mimi PCM encode/decode calls are application composition outside that
request: neither `AudioTokenizer` nor `Mimi::encode_step`/`decode_step` accepts a
managed frame account or joins its session transaction. Cancelling, releasing or
restoring a token-frame session therefore does not claim to rewind an external
codec's convolution tails, overlap buffers or transformer caches.

`personaplex_full_path_bench` explicitly labels its ordinary external codec mode.
Its timing includes standalone encode/decode calls and an ordinary host token
bridge; it does not establish a working-memory bound for PCM-to-PCM processing.
Historical PCM-lifecycle notes describe that separate application integration,
not a missing operation inside the current encoded-token request. Media encoders
selected as part of an ordinary inference request remain subject to that
request's managed preparation, cancellation and completion contract.

Managed tensor-parallel realtime frames enter the same distributed schedule and
completion protocol as ordinary frames. Source preparation is a callback of that
driver, with topology-wide admission agreement before model submission; partial
admission cannot publish a branch or leave peers dispatching a refused frame.
Rank-local turns remain rejected for non-replicated topology. The frame account covers its admitted state-copy and numerical work. Every
managed scheduler exchange uses the retained world's independently admitted
initialized-word, Gather and completion producer in the same memory pool.
Protocol word/result directories and exact frame descriptors use the transport's
cumulative Host source; borrowed native output retains its own paid destination
through validation. The native wrapper preserves the first typed model or
transport failure across rank agreement and cleanup.
