# Completed-token execution control

The ordinary text facade supports completed-token stepping, pause/resume, canonical
token forcing, temperature changes, explicit reseeding, reusable native snapshots,
restoration and isolated serial branches. Complete snapshot support requires known
costs for every state owner. Currently this includes the forbidden-tool constraint
mode; active and automatic llguidance grammar modes explicitly reject snapshots.
See the [LM Inspector integration guide](lm-inspector-execution-control.md) and
the runnable [controlled generation example](../eredu/examples/controlled_generate.rs).

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

`LocalModel::start_controlled_chat` exposes the same behavior through opaque
`LocalControlledGenerationSession`, `LocalGenerationSnapshot` and
`LocalGenerationBranch` wrappers. Applications need no native backend type.

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

The complete local facade has a native Qwen2 fixture for sampled partial-text
restore/fork after removing its loaded weight/tokenizer files. A second native test
executes the complete observed/intervened example and verifies unchanged restore
and fork before producing the modified child. These CPU tests use
`--no-default-features --features mlx`. Metal coverage is opt-in and was attempted
in the development environment, which reported no accessible Metal device. A
Metal-enabled MLX build may require GPU access during initialization even when a
test requests a CPU stream; the CPU-only build is the verified headless route.

Active/automatic llguidance state still needs a complete upstream storage estimate
and a conservative continuation-growth bound to broaden snapshot support; those
configurations fail before copying today.

Partitioned, speculative, media and realtime execution control, persistent snapshot
files, cross-process/cross-backend restoration and layer-level pausing remain outside
the initial support target.
