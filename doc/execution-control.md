# Execution control

Tool-name normalization is prepared-request policy shared by JSON and typed chat.
Controlled semantic events expose original application names just like ordinary
and speculative execution. Grammar and schema validation retain wire aliases.
The immutable restoration map belongs to the semantic plan and is shared by
parser snapshots/forks; mutable partial-call state remains independently copied.
Host continuation bounds include restored-name expansion, so aliases cannot
understate delivery/history storage. The existing active/automatic llguidance
grammar snapshot rejection still applies when a complete mutable grammar storage
estimate is unavailable; normalization does not bypass that capability check.

Completed tool calls use the same `ToolCallCompleted` event in uninterrupted,
controlled and speculative generation. The event carries the call index, ID,
original application name and parsed, schema-validated object arguments.
Provisional start and argument deltas do not authorize tool execution. Speculative
proposals publish no completion until commitment; cancellation discards unfinished
calls. Snapshots preserve partial arguments and completed payloads, and delivery
checkpoints retain their existing replay rules. Storage estimates include parsed
JSON containers; tool-enabled history bounds allow for their expansion relative
to encoded trace bytes. Text-only history keeps its encoded-byte bound.

Typed chat preparation uses the ordinary shared drivers. A counted
`PreparedChatRequest` enters capture/control admission through
`prepare_observed_chat_input`, or `prepare_intervened_chat_input` for prospective
interventions, and then uses `start_controlled`.
The request retains the exact text IDs or native prepared media prompt;
controlled startup does not render or encode it again. Media alignment includes
architecture-declared placeholder IDs and must cover every counted decoder
position. MLX obtains these IDs through the architecture's existing semantic
token construction. An adapter without `prepared_input_token_ids` observation
rejects typed media capture/control admission with
`BackendFailureKind::Unsupported`; ordinary and speculative execution can still
consume its counted payload. Missing alignment is an adapter observation gap,
not a model-family limitation.

Prepared source-backed requests and ordinary requests use
`PreparedChatGenerationSettings.decoding_mode` for initial selection. Omission
selects AR for a causal target; self-speculation carries its options in the mode.
The target-only source enters the shared request table without an assistant and
supports the target's declared ordinary refinement policy. Controlled and
uninterrupted visitors borrow the same prepared target executor.

Live capture admission reads `CaptureDeclarations` without hashing checkpoint
contents. Session-provenance records carry the loaded session and admitted plan,
with absent artifact attribution. Explicit content-exact capture, intervention
admission and capture snapshots resolve source identity. Snapshot restore and
fork retain exact source checks, drained delivery and monotone budgets in either
observation mode; snapshots may perform the first content-resolution pass.
A child adding interventions receives content attribution from the newly admitted
intervention plan, while its execution and epoch ownership remain separate.


The ordinary text facade supports completed physical stepping, pause/resume, canonical
token forcing, temperature changes, explicit reseeding, reusable native snapshots,
restoration and isolated serial branches. Complete snapshot support requires known
costs for every state owner. Currently this includes the forbidden-tool constraint
mode; active and automatic llguidance grammar modes explicitly reject snapshots.
Context encoding and tentative refinement advance without querying the canonical
constraint controller or parsing semantic text. Selecting operations use the
current controller decision and commit one authoritative token. `Progress` records
carry phase, block, iteration and independent execution coordinates; token records
also declare the meaning of their scores. Causal conditional scores, refinement
position scores and constrained finalization scores remain distinct. Generation
records use schema version 4; execution-control metadata uses version 3.

Snapshots preserve the logical physical-step and committed-token coordinates,
alongside backend phase state, sampler, pending input and constraint state.
Restoration does not rewind output reconciliation epochs or refund capture,
transport or copy budgets.

For iterative generation, `step` completes one physical context, refinement or
finalization operation. A progress-only step commits no token and remains a pause,
snapshot and fork boundary. `draft_events` enables replaceable token-block updates;
their branch, epoch, block and revision identify the active tentative output.
Drafts clear before canonical delivery and on cancellation, failure, restoration
or branch exchange. Snapshot replay preserves numerical state while advancing
the output epoch, so consumers cannot mistake replayed drafts for older revisions.

A prepared observed request can retain a seeded `RefinementCanvas` with pinned
positions. Ordinary and controlled execution preserve those positions during
refinement and carry them through snapshot, restore and fork. Fixed positions use
one-hot feedback and do not dilute confidence at unresolved positions. Explicit
position score captures support reads without commitment: advance through context
and refinement, collect the scores, then cancel before finalization. See
[seeded and pinned canvases](refinement-canvases.md).

The `decoding_mode` setting selects `Autoregressive`, `Diffusion` or
`SelfSpeculation` with validated invocation options. Absence selects AR when the
retained realization supplies causal target sampling. An intrinsically iterative
architecture such as DiffusionGemma retains its trained position-score policy.
Diffusion and masked drafting carry their numerical `RefinementPolicy` in the mode.
Incompatible modes fail before context execution. The common runtime
refinement state retains masked selections, selected score rows, independent RNG,
the suffix transfer cursor and cumulative invocation count. Constraint-driven
suffix refinement preserves the committed prefix and fixed pins; snapshot clones
isolate the numerical cursor and random stream.

`request_decoding_mode` on a controlled session or its thread-safe
`GenerationControlHandle` validates the request against immutable realized-mode
facts. One pending transition is allowed; another returns `ModeChangePending`.
Request-local sampling requirements are checked at admission as well. Mirostat
rejects diffusion position scores with `RequiresCausalConditionalScores`, while
leaving active execution and the pending command unchanged.
Executions without a settled handoff reject changes before altering the run.
The shared physical driver finishes the current block or verification transaction,
appends any committed suffix once, preserves next-target logits and removes
tentative output. `ModeChanged` follows tentative-output clearing and carries the
effective generated-token position. Cancellation wins over an unacknowledged
handoff. Terminal runs reject commands.

`GenerationRequest.control_handle` supplies the same mode command
mailbox to uninterrupted advancement. Mode state, including pending commands and
the last acknowledgement, is copied with complete snapshots. Restore installs it
with the corresponding cache and sampler; a fork has an independent mailbox.
Capture, transport and copy usage remain cumulative. Native loaded-family support
is reported separately from the neutral command mechanism. Nemotron's native
physical driver supports AR/masked-diffusion handoff at completed block boundaries.
Committed block append retains only the final causal score row for resumed AR.
Refinement and target random streams remain independent, and snapshot accounting
includes retained next-token logits and spent refinement randomness.

Prepared executor lanes carry `GenerationRequestConfig.decoding_mode`, including
the same validated speculative options used by request preparation. The common
`GenerationRequestTable` runs causal lanes with zero proposals and no optimistic
drafting. Controlled executor sessions expose `request_decoding_mode`,
`decoding_state` and `bind_control_handle` after prefill. Supplying
`GenerationRequest.control_handle` or a batch lane's
handle binds uninterrupted commands before prefill. Failed submission closes a
bound mailbox; terminal output closes commands before its delivery callback.

AR and speculative executor handoffs settle through a zero-proposal transaction.
The final committed token enters causal cache once and its next-target row stays
retained for subsequent sampling. The acknowledgement precedes that sampling,
does not consume either random stream and clears promoted tentative proposals.
Durable snapshots retain the row, explicit pending committed token and complete
mode state. Each serial branch has an independent command owner; inactive
commands survive exchange, and branch release closes only its mailbox. Branch
creation charges host command metadata while sharing immutable saved tensors.
Per-request lookahead and completion deadlines stay within shared scheduler
ceilings. Continuation forecasts cover active and pending proposal allowances,
lookahead, adapter workspace and retained next-target rows. Numerical unmasking
support and complete diffusion commitment are distinct capability facts.

Causal sampling overrides and causal likelihood-ratio speculative verification reject
noncausal target scores with `RequiresCausalConditionalScores`. Constrained
finalization applies the evolving canonical constraints independently to each
position. Explicit reseeding remains a prospective change to the retained stream.

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

Realtime retained payloads participate in the same unpublished model-state branch.
Their exact identity includes the declared channel layout, positive batch,
session owner and history generation. A branch with changed geometry, cadence,
feedback or owner fails before model-state publication. Dense and categorical
payloads share this retention contract; cloning history does not publish output
or establish native completion. Moshi and PersonaPlex keep their existing
categorical frame advancement and completion-gated publication.

The shared realtime generation transaction accepts architecture-owned schedule
state, sampling controls and transition inputs. Schedule policy may retain
separate audio, model and tool progress; the transaction imposes no common frame
clock. Exact contract identity, sampler cardinality and schedule successor checks
precede model publication. Every commit and idle sampling replacement advances
a revision, invalidating older branches before they can replace canonical state.
The ordinary transition helper and explicit branch advancement use the same
completion boundary. Selected realtime sessions retain a neutral frame contract
and accept architecture-owned input, schedule and sampling types through the
same scheduler. The exact policy descriptor and sampler count are checked at
registration. Release/resume preserves the selected contract, independent clocks
and session incarnation; sampling realization borrows the requested controls and
replaces them only while the request is idle. Local and distributed advancement
use the same batch admission and completion rules.
Unpublished branches can draw uniforms and standard normals without categorical
decisions. These draws share the decision driver's retained stream and become
canonical only when the entire transition commits.

VoiceChat's agent sampler retains effective feedback after function/turn overrides.
Raw selection runs first, including any categorical RNG draw. The branch adopts
the decision driver before accepting effective feedback and drawing speech noise;
commit publishes their state together. Neutral conformance verifies rejection
without completion, deterministic replay and stale-fork rejection. Session
snapshots compose this sampling state with the other conversation state.

The channel handoff selects raw agent/function tokens before applying explicit
overrides. Tool speech can use acknowledgment text while effective agent feedback
remains PAD. A function-selection failure after an agent draw leaves branch RNG
and history unchanged; native completion handling still belongs to the enclosing
submission owner. Neutral tests compare committed effective feedback through the
next cached backbone call, including retained recurrent and attention state.

Optional `ReplicatedChannelDecisions` coordinates that same channel handoff with
one recorded request/turn/invocation identity. Admission agrees head geometry,
domain, perception evidence, sampling policy and effective history. The designated
owner alone samples and resolves overrides; followers validate its raw and
effective IDs and execute a forced decision without RNG. Phase failures preserve
the pre-call RNG and history, including failures after owner selection or local
publication decoding. The completed driver and prospective history are validated
before final agreement and adopted together afterward. Authority is consumed on
admission even when work fails; replay uses fresh authority with an unpublished
branch. This object owns neither native completion nor session snapshots and
does not authorize replaying delivered tool calls or audio.

Realtime materialized inputs retain channel-keyed ordinary values and selected
overrides with their exact layout and batch. Mixed host admission validates both
groups before copying either. Partial categorical forcing copies only selected
columns; unselected sentinel entries never become native inputs. The categorical
interpreter checks the retained ingress contract and complete input roles before
advancing its unpublished schedule or payload history. A materialization failure
does not establish native completion; the submission owner still handles any
work already accepted by the backend.

Completed realtime outputs retain typed channel identities and coordinates for
both current and delay-aligned values. Dense frames and categorical decisions
use the same exact-completion and prepublication observer boundary. Host
validation checks declared geometry, categorical domains and finite dense values;
an observation failure discards the scheduler branch and exposes no output.
Moshi's portable projection assembles its public codec frame after validation.
Typed component outputs use `submit_realtime_output` and the same prepublication
observer. A failure after submission attaches its exact quarantine completion to
the branch; a duplicate attachment rejects before invoking native submission.
The VoiceChat facade composes this publication boundary with session advancement,
variable-length transcription events and conversational snapshots.

## Prompt prefill

Ordinary and controlled generation share prompt chunking in the core driver.
`TextGenerationConfig::with_prefill_chunk_policy` and
`PreparedChatGenerationSettings::prefill` select `PrefillChunkPolicy::Bounded`
(512 tokens by default) or `Unchunked`. Each nonfinal chunk completes before the
next begins and advances the model cache without sampling. Only the final chunk
predicts a token. A controlled step can publish `Progress` and pause at a
completed nonfinal chunk; it consumes no output token or controller decision.
Snapshots and branches preserve pending input together with committed model state.
The uninterrupted driver repeats the same operation until token production.
Cancellation is checked between completed chunks and can leave a partial prompt
cache with no committed output. Reset the model before starting a different request.
`Unchunked` removes the caller-imposed chunk limit but still honors an architecture's
required invocation limit.

`LoadedModel::prefill_chunking_support()` reports the executable's support or a
fallback reason. MLX plain-text adapters support ordinary replicated shared decoder
architectures, including grouped-query and sliding attention, and dense/routed LFM2
and LFM2.5, plus target-only Qwen hybrid text and composite text input. Qwen carries
convolution history, FP32 recurrent state, attention caches and positions across prefixes.
Its selected embedded prediction path remains separately scheduled. Their plain-text splitting path retains a complete pass for capture and
distributed execution. LFM2 retains convolution history, attention caches and absolute
positions across passes; convolution-only schedules use the same driver.

Composite adapters with retained prefill requests support prepared media and absolute
capture spans through the shared cursor described under [Flash-Next](#flash-next).
Adapters without that cursor retain a complete pass for structured input. Nonempty
intervention plans reject partial prefill spans with a typed error. Speculative
execution uses its own shared prefill operation and does not expose between-chunk
control advancement. Memory estimates reflect the actual selected path.

The shared dense and heterogeneous decoder shells project only the last hidden
position for an unobserved ordinary pass. Observed and speculative full-output
contracts retain every required row. Nonfinal prefixes compute one unused vocabulary
row; they never sample or commit a token.

Ordinary prefill completes both output and retained mutable state before
publication, including when ordinary decode is asynchronous. Completing logits
alone is insufficient: a convolution-history slice can otherwise retain an
unevaluated graph. The neutral convolution compacts its retained history; native
backends bound its backing storage independently of prompt length. Completion
failure follows the existing rollback/terminal-failure protocol. Cancellation
stops further submission between completed chunks and does not preempt native
work already submitted. Chunk size controls work per submission, not a fixed
wall-clock cancellation deadline or the total attention-cache size.

Chunk size remains caller-selected, with the existing 512-token default and
`Unchunked` option. This policy imposes no allocator cap or device-size heuristic; architecture
geometry and history limits apply independently. See [LFM2 prefill validation](lfm2-prefill-validation.md) and
[Qwen3.5 prefill validation](qwen35-prefill-validation.md).

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

`PlannedModel::with_controlled` accepts the same `GenerationRequest` as
uninterrupted generation, `GenerationControlOptions`, and a worker closure.
`request.output_mode` selects semantic or literal text; the owner retains its
source and plan defaults. The returned output includes optional draft statistics.

The common controller supplies explicit `step` and `run`. For speculative
inspection, `control.speculative()` lends `&mut dyn ControlledSpeculativeSession`;
for ordinary snapshots and interventions, `control.ordinary()` lends the typed
ordinary controller. Speculative options retain their phase-aware capture and
snapshot budgets. Ordinary capture uses `GenerationControlOptions.capture`;
ordinary snapshot budgets are enabled on its typed controller. Opaque prepared
inputs use backend-reported original decoder IDs, including media placeholders.
Returning early cancels and settles unfinished work through the existing driver.
Requests cancelled before ordinary startup return a cancelled output without
opening a controller scope. Request cancellation and command-handle cancellation
both remain effective when supplied independently.

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
admitted plan in `GenerationControlOptions::speculative_capture`. Each observation is one
vocabulary row, including prefill; ordinary whole-prompt admission is rejected.
The MLX implementation supports `model.logits` for target prefill, reached verifier
rows, and every sampled draft proposal, including rejected work. It reuses the
ordinary bounded transformations (including top candidates, summaries and slices)
and one shared cumulative ledger. Internal layer observations use the separate
`prepare_speculative_activations` admission and `speculative_activations` option. Loaded support
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
| Pending prompt or last committed decode input | Core `GenerationRequestTable` completed-decision source | Preserve phase and pending input explicitly |
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

The MLX state-copy primitive independently copies device KV arrays and fixed
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
`eredu-evaluation::execution_control::continuation_conformance` runs against
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
remain charged after draining and restore. `Untracked` identifies low-level batches without transaction evidence. Completion and state reuse still
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
API contracts, accounting and verification scope.

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

The ordinary one-shot driver delegates to `CommittedGenerationCursor::step`.
This reuses the existing token source, tokenizer, constraints, EOS precedence and
semantic pipeline. The pipeline and built-in protocol parsers support exact state
forks, including buffered bytes, JSON/Python/tool-call state, stop matching, pending
delivery and terminal state. An unimplemented custom parser fork fails explicitly.

The common `GenerationRequestTable` also drives `TextGenerationDriver`, detached
`TextGenerationContinuation` state, and native-token and controlled iterators.
A native completed-decision source retains the pending prompt/token, opaque
sampler, non-cloneable controller and exact completions. The common request owns
canonical generated IDs, commitment allowance, cancellation and mode control.
Native token iterators yield the original handle after exact completion and
canonical identity validation. An invalid identity fails that iterator advancement
before commitment or delivery.
Snapshots copy the real canonical sequence with native state and pending input;
restore and fork preserve spent observation, transport and copy budgets. A driver rejects
another driver's continuation before any work, requires completion and one-step
record draining between advances, and fences failures even after cleanup. This
low-level mechanism still requires facade composition with native state exchange,
semantic delivery and lifecycle policy; it alone is not a controllable session.

## Public stepping workflow

Prepare with `prepare_observed_chat` (use `CapturePlan::none()` for ordinary runs)
or `prepare_intervened_chat`, then call `LoadedModel::start_controlled` with
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
the active run and that slot; the slot then holds the displaced run. The
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
cannot import another driver's snapshots.

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

`LoadedModel<B>::start_controlled` returns `ControlledGenerationSession<B>`;
snapshots and branches use `ControlledGenerationSnapshot<B>` and
`ControlledGenerationBranch<B>`. These generic types are also the MLX application API.

`force_next_token(id)` validates canonical vocabulary membership and the current
grammar, then restricts the ordinary sampler to that one candidate. It commits
through the same history, penalty, decoder and termination path. Conflicts fail
before advancement; `clear_forced_token()` removes a staged choice. A forced token
record includes `forced: true`; ordinary records omit this marker.
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
overrides are not exposed by this policy. `SamplingChanged` records carry
the request, before/after facts and the first absolute prediction affected.
Native preparation uses the existing completion/recovery owner and executes no
model forward. Invalid requests leave the source unchanged; native failures fence
the public session. Callback failure retains ordinary cancellation semantics.

The shared forced-choice and sampling-override conformance scenarios run with the
host and all three native CPU fixtures in every record mode. They restore pending
choices and explicitly reseeded children, preserve the parent, and verify no replay.
Native sampler tests separately verify exact key progression and adaptive updates.

Built-in semantic parsers, stop/UTF-8 buffers, decoder history and the committed
cursor have field-audited logical storage estimates. Immutable tokenizer data
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
uses the compatible `1.8.0` requirement without an exact pin. Declaring tools with `ToolChoice::None`
selects the supported forbidden-tool owner on a profile with a tool surface;
no-tools semantic requests use tokenizer validity without a grammar. Complete
snapshot and fork support still requires known native and semantic storage costs.
The example uses an ordinary no-tools chat request.

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

Partition capture uses the same transactional observer callbacks as ordinary
and controlled forwards. Its work handles belong to the actual capture owner and
monotone forward epoch; restored or foreign handles fail before source access.
Common pre-forward coordination agrees selected layouts, identities and global
reservations. Active invocation groups separately agree hook success before model
collectives. Receipt delivery stages global records after exact completion, and
only the existing final commit publishes them. Restore does not reset producer,
transport, status, host-assembly or encoded-evidence consumption.

Native communication setup retains a common instance identity agreed from
the complete manifest/nonce transcript. Runtime capture can bind it to the first
attempted forward epoch without doing native work during cold configuration.
Failed admission and restore preserve that run identity; later records retain
their distinct actual forward epochs. A new capture owner uses the model's later
epoch, and a new model setup has a fresh instance identity. These descriptive
labels do not replace move-only submission authority or retained native owners.

Loaded component capture uses this adapter in ordinary and controlled
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
subgroup transport. Unsupported specialized drivers and disconnected subgroup
mechanisms return explicit capability failures.
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

Sparse partition capture has ownership-aware receipt validation and bounded
assembly in the existing exchange protocol. This does not add a second completion
or inference owner: received host rows remain evidence pending the shared forward's
completion and commit. The live session prepays sparse producers, pins each
actual invocation's receive extent, accumulates exact chunks and requires explicit
provider completion, including zero-row owners. Completed sparse fragments follow
the same staged delivery and final commit as dense captures. Shared local provider
callbacks cover preparation, execution and failure before EP reverse exchange,
including idle owners, with nested residency adapters reporting one invocation.
The prepared gated/ReLU² EP tests verify rollback and no reverse exchange after
an agreed local failure. The shared partition observer prepares routed source
bounds and uses its prepaid source/final votes through these callbacks, including
original/effective records, idle owners and nonexporting replicas. Abort and failed
work preserve cumulative credits and prevent publication. Retained architecture placement feeds these callbacks through actual prepared
gated and ReLU² executors. The neutral tests cover prefill and two cached decodes,
idle received owners, all three residencies, TP/EP/PP ownership and source/collector
failure rollback. Loaded MLX admits sparse capture through this same observer
when its exact native invocation group and provider hooks are available. The
packed-Qwen Ring matrix verifies full/strided original and effective values against
an ordinary native reference through prefill, two cached decodes, snapshot replay,
sibling isolation and skipped child capture. Restore preserves cumulative credits
and fresh branch identity. The shared operation authority also carries sparse
edits through the provider scope. Operation-only plans activate these callbacks
without requiring sparse captures. Re-admitted future child masks start from the
same prepared boundary, receive fresh identity, preserve parent and sibling state,
and consume cumulative credits on every replay. Partial chunks or peer failure
prevent a committed edit outcome. Explicit tensor/expert stage groups use
bounded native neighbor agreement; inactive pipeline stages do not enter those
observation callbacks. Disconnected native groups lack the coordinated relays required for this protocol.

The released sparse score consumer also runs effective head/gain queries and
all-layer signed projections before controlled trials. Its combined LFM2-8B-A1B
Metal run passes seventy ordinary/controlled, cached-restore and sibling/parent
comparisons across baseline, deletion, keep-only, overlay and restoration. Parameter
query work remains cumulative across these phases. Fixed baseline target/alternative
IDs are scored along each trial's own unforced trajectory; branches within a trial
start at equivalent prepared boundaries. See the released sparse score results in
[component-analysis.md](component-validation.md#released-sparse-selected-token-and-token-difference-reconstruction).


Mixed dense/routed V3 pipeline component capture follows the same controlled
partition driver. The effective input embedding travels in the typed boundary;
normalized Q/KV latents retain their invocation ownership, and effective current
KV values enter the compressed cache. Native two-process Ring tests cover
resident, host-layerwise and disk-streamed captures/masks with snapshot replay
and sibling isolation. All-dense V3 additionally passes native TP, PP and combined
TP/PP in all three residencies, including atomic multi-weight overlays, rejected
publication rollback and restoration, for F32 SafeTensors/GGUF and SafeTensors
load-time affine 4-bit/group-32. Token-indexed query rotation is
shared by controlled and ordinary execution; MLA equation revision 2 requires a
matching V3 cache fingerprint. Mixed V3 TP/PP/EP block instrumentation passes native F32
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

V3 prediction component primitives share fusion/head execution with ordinary
inference. Their `prediction.readout.residual` intervention occurs before head
normalization and affects the returned prediction hidden state and draft logits.
`mtp.{depth}.output` occurs after the head.
Neutral ordinary/provider/TP traversal tests verify selected-depth cache advancement
and fresh sibling replay. Public phase/frontier admission and delivery are verified
through `prepare_speculative_activations`, including the native distributed
matrices recorded in the component-analysis guide.

Mixed V3 independently cached F32 SafeTensors/GGUF and load-time affine execution
passes all 63 CPU Ring cases across the three ordinary weight residencies and all
TP/PP/EP combinations. Queries, coordinated edits, snapshots, siblings and replay
use the same selected bank identities. Idle EP owners remain valid participants;
every routed PP stage must demonstrate actual cache activity.

V3 component discovery separates prediction depths and their score heads from
primary target attribution. Typed prediction operations can accept internal
observers during extension prefill, proposal and retained-input replay. Their
caller still owns prediction-lane restoration after failed work; restoring the
ordinary target state alone does not restore the separately owned prediction lane.
Public speculative component capture composes explicit phase attribution, bounded
delivery and cumulative budgets with the shared controlled driver. Loaded support
reflects the actual selected hooks; target-only support cannot authorize a separate
prediction scope.


Internal embedded observers receive explicit target-prefill, prediction-prefill,
proposal-depth, verification and replay phases from the shared strategy. Successful
invocation completion stages tentative evidence; it does not make a proposal or
verification row committed output. Physical sequence width is supplied separately
from output prediction ordinals. Failures use existing target/prediction cache
recovery and cannot refund observer authority. The neutral tests cover prefill and
both replay delivery failures; native completion resources retain their original
owner. Public controlled capture admission and delivery compose these hooks with
cumulative reservations and exact frontier attribution.


The capture owner supports explicitly admitted physical invocation geometry
independently of prediction coordinates. Snapshots retain immutable bounds and
schedules; restore clears delivered geometry and requires the next invocation to
supply its own exact shape. It preserves all spent capture/intervention resources.
Child re-admission carries bounds and inherited usage into the fresh session. These
contracts supply physical geometry to the shared controlled driver, which adds
speculative request/frontier attribution and public internal-capture delivery.

Speculative internal capture has scheduler-owned request/prefix origins and
independent physical invocation geometry. Controlled step delivery has a separate
`activations` array, charged to ordinary trace transport, while `captures` retains
its sampler-row meaning. Invocation IDs and cumulative capture allowances are not
rewound by collector restore. Origin scopes are cleared after each operation,
including failure and unwind; optimistic prefix digests are never substituted for
a committed frontier. Public admission/installation uses the separate loaded activation authority.
The neutral partition observer accepts the same independently admitted
geometry and scope masks. Its receipts bind exact invocation axes and monotone
forward epochs; failed final commit publishes no payload, and replay does not
refund geometry, source, edit, receipt or assembly costs. Live multi-rank tests
cover actual edits, original/effective evidence and restoration with changing
physical widths at one prediction coordinate. Public distributed admission uses
the selected prediction layout and retained native transport; family coverage is
reported by loaded speculative discovery. The speculative provider retains the ordinary partition observer across each phase, including sparse
callbacks and final commit; its existing restore owner preserves spent budgets.


`GenerationControlOptions.speculative_activations` installs one immutable combined
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
V4 sequential and DSpark use complete pooling-state copies through the same
driver. Native CPU/Metal tests compare exact sampler captures and seven generated
tokens after a 255-token prefix, repeated restore and interleaved siblings across
resident/host/disk weights. Separate V4 component and distributed snapshot matrices
are recorded in the component-analysis guide. Capability checks require complete
native copy bounds for the selected state profile.

Continuous observed speculative generation is implemented by advancing this same
controlled session. The callback receives each `ControlledSpeculativeStep` and may
cancel with `ControlFlow::Break`; no second inference or sampling engine is used.

Observed prediction prefill, proposal and replay share the ordinary session's
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


Loaded V3 internal activation plans use the retained partition collector through
the same speculative controller. Independent prediction scopes carry their own TP
columns and non-tensor replicas; target input/output ownership remains unchanged.
All 21 F32 CPU Ring parallel/residency cases reproduce local phase records and causal
channel/routed-unit edits over prefill and cached generation. Prepared resident
providers finish their source/failure protocol before model reductions. Shared
embedding storage is present on every prediction replica, preventing different
pipeline ranks from choosing different speculative decisions. This does not change
snapshot budget semantics. All 21 F32 cases also verify coordinated target and
prediction parameter edits, late peer rollback, active-overlay snapshots, repeated
restore, isolated sibling exchange and restoration after removal. Snapshot copying
and capture accounting remain cumulative. Additional encoding and family acceptance is recorded in
`doc/component-analysis.md`.


Internal speculative activation plans use schema 3. The admitted scope binds
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

`PlannedModel::forecast_generation` forecasts the same prepared request
used by ordinary or controlled startup. `forecast_observed_generation` accepts
`PreparedObservedGeneration`, verifies its session identity and uses its resolved
settings and capture/intervention contract. Trace-only preparations retain ordinary
chunking and final-row projection when the executor supports them. Instrumented
prefill forecasts conservatively use a full pass and all-row logits, including for
executables that process observed prompts in chunks. Admitted per-step/cumulative capture
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
owners used by ordinary and controlled execution. The managed 256 MiB
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
Explicit trimming uses the settled-boundary contract below.


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
There is no live limit setter. Read the effective policy and scope from these
observations instead of maintaining a second limit ledger. Group charges are
admission accounting: aliases within a group charge once, but separate groups
sharing backing each admit their own claim. Do not sum these claims as physical
residency or add retained payload to resident parameters a second time.

The [consumer workflow](generation-memory.md#controlling-parameter-conversion-retention)
and [facade example](../eredu/examples/conversion_retention.rs) show configuration,
live reports and fresh-request forecasting. A finite conversion allowance and
the independent allocator-cache allowance are not a combined memory limit;
request state/workspace, original weights, graph/driver storage and padding remain.

### Trimming retained parameter conversions

`LoadedModel::trim_parameter_conversions()` first uses ordinary synchronization to
establish healthy completion, then releases optional conversion claims without
resetting the request. Controlled ordinary and speculative sessions expose the
same operation through their shared execution drivers. They require a canonical
completed, drained boundary; pending native work or speculative transactions
produce typed `ParameterConversionTrimError::NotQuiescent` rather than being
settled, committed or rolled back implicitly. Unsupported backends return a typed
`Unsupported` error; native failures retain their source through `BackendFailure`.

Trimming preserves committed tokens, pending input, cache frontiers, sampling/RNG,
epochs and capture, transport, snapshot, fork and copy accounting. Existing
snapshots and branches remain valid. Embedded prediction participates in its
target's group; separately loaded external drafters retain separate result scopes.
The same backend claim-release operation serves ordinary and controlled runs.
For ordinary `PlannedModel` trimming, the target is trimmed before the external
drafter. An error from a later participant does not roll back earlier releases;
re-query participant usage after an error. No cross-participant atomic trim is
promised.

Repeated trimming is idempotent and empty retention stays empty. Eligible source
bindings remain registered, so later inference may repopulate within the original
load-selected cap. Parameter replacement still retires obsolete bindings and is
not a trim. Reports describe released claims and remaining admission usage, not
bytes returned to the OS. Other owners or native graphs can retain backing;
allocator-cache flushing is a separate operation. Ordinary reset continues to
preserve retained conversions.

Continuation forecasts refresh parameter-conversion allocation and retaining-binding
observations at each canonical boundary, for both target and external drafter.
They retain immutable selected geometry, never startup conversion credit. After
trim or invalidation, subsequent forecasts project the newly uncached casts while
preserving state frontiers and observation budgets. Retention policy and usage
are a scoped subset of the forecast's existing parameter/workspace accounting;
a finite ceiling does not bound temporary casts. If a backend reports outstanding
reservations without publication/backing attribution, the phase upper bound stays
unknown even at a zero-token horizon; forecasting does not settle those resources.

## Flash-Next

Qwen3.8-Flash-Next (`qwen4_exp`) uses the shared ordinary and controlled generation
and speculative drivers. SafeTensors and GGUF targets support resident,
host-layerwise and disk-streamed weights. Architecture defaults select a 512-token
target invocation limit and retain the declared context capacity; explicit load
policies override those defaults. Cumulative history capacity is independent of
per-invocation chunk size and includes subsequent decode tokens.

### State and request ownership

Flash-Next's typed state combines ordinary K/V, named index-summary and position
streams, and fixed recurrent, convolution, lexical-history, partial-block and
position components. Snapshots, restore, fork, reset and paged prompt-cache persistence
use the shared lifecycle mechanisms. Integer n-gram history remains exact. QSA partial
blocks retain positions, lengths, raw keys and first-token text/media rotary provenance.
The bound state identity includes source geometry, integer hash controls and selected
parameter transforms; same-shaped artifacts with different controls cannot reuse a
prompt cache. Restoring state never refunds observation, transport, copying or source
work.

Image/video requests retain explicit original token IDs through embedding replacement
and pipeline transport. Embedding-only inputs without those IDs fail at admission.
Media rotary positions and the signed position delta survive prefill chunks and cached
decode; the delta resides in the target's fixed state. The lexical reset token is
independent of facade-owned generation termination. Encoder and target units retain
separate layer-sized residency owners.

### Chunked media prefill and capture

Composite requests use an architecture-owned request type through the shared
`PrefillCursor`. `start_prefill` validates request and chunk geometry;
`advance_prefill` performs one ordinary transaction; `finish_prefill` repeats that
same operation. The complete single-sequence request fits within the selected history
capacity, while assembly and lexical lookup operate on the current chunk. Oversized
requests fail before native request construction or encoder execution.

The pending native prompt retains the cursor. A controlled `step` can finish one
nonfinal chunk and emit `Progress`. Cursor snapshots share immutable request
proofs and completed encoder output while copying mutable model state. Restoring a
completed prefix does not rerun its encoder. Complete-input prompt-cache identity
publishes only after the final chunk commits; partial progress clears stale complete
identity. Foreign cursors reject, as do stale cursors whose prefix differs from the
committed state.

Output and retained encoder projection complete before cursor publication. A failed
chunk rolls back through the shared driver; completed prefixes stay committed.
Completion and restoration errors propagate to the caller, and an unchanged cursor
alone does not establish successful rollback. Partitioned advancement agrees request
preparation, extent and each step's position before execution. Local preparation
failures rendezvous with peers through the same agreement protocol.

`prepare_observed_input` accepts a prepared backend prompt and its explicit original
decoder IDs, including media placeholders. Callers preserve that correspondence. It
feeds ordinary observed generation and controlled text/chat entry points. Capture
records retain an absolute `prefill_span`; sequence slices intersect and rebase for
each physical chunk. Final prediction scores are captured only on the final chunk.
Restore preserves cumulative capture and copy usage. Low-level token iterators that
install capture directly drain each `advance_prefill` result before requesting the
next token.

Nonempty intervention plans receive a typed rejection for partial prefill spans:
mask and replacement payloads require span projection that this path does not supply.
Direct low-level composite forwards with a full-request observer reject chunked
capture; shared generation supplies fresh collectors per cursor advancement.

### Embedded prediction

Default loading runs the target even when MTP weights are present. Explicit embedded
prediction installs separately selected prediction state and the shared speculative
executor. Target residual capture preserves every stream before collapse, and the
target owns shared embedding/output parameters. A GGUF target accepts a matching
SafeTensors prediction companion through `ExecutionPlan::with_prediction_source(path)`;
full and prediction-only companions retain source admission and provenance separately
from the target.

Text and image/video prediction prefill use causal chunks at the selected invocation
limit. The final residual of a chunk pairs with the first original token of the next
chunk before seeding the remaining shifted pairs. Whole-request token IDs, rotary
positions and encoded media remain available to the request cursor. The complete
prompt is one speculative prefill advancement; failure restores both target and
prediction state to the initial checkpoint. Ordinary facade `PrefillChunkPolicy`
does not control this operation, and speculative control has no between-chunk pause.
Activation scopes identify actual physical target/prediction rows while scheduler
provenance identifies the complete speculative operation.

### Distributed execution and tools

TP, PP, EP and combined execution use architecture-owned placement and shared
collectives. Recurrent, K/V, lexical and QSA state follows its declared tensor/stage
placement. Pipeline boundaries retain complete residual streams, original IDs,
visibility and rotary products. Table acquisition belongs to one declared tensor
owner and compact lookup results are shared with peers; failure agreement includes
expert participants and idle stages. Prediction retains separate local state and
replicates shared vocabulary parameters across its declared groups.

Distributed parameter and component coordinates use exact bound-rank placement.
Queries requiring unavailable cross-rank coordinate projection return a typed error;
local parameter queries and ordinary distributed inference remain supported. Capture
and intervention availability comes from loaded discovery. The
[family validation report](qwen4-exp-validation.md) records supported native and
portable coverage, tolerances and validation limits.

The pinned official chat template supports thinking controls, tagged tool arguments,
reasoning history and tool-result replay through the shared parser and termination
policy. Active/automatic tool grammar snapshots return the shared typed unsupported
capability because complete matcher storage estimates are unavailable. Rejection
preserves committed progress and output attribution and leaves generation usable.

### Media policy and resource observations

`MediaExecutionPolicy::new(processor)` declares input readiness without mandatory
encoder/request/native-allocation budgets. Optional coarse host-processing limits
reject oversized requests before expensive work. Token, history, chunk and geometry
limits, exact original IDs and completion-safe ownership remain required. A GGUF
projector can use architecture-default media intent. Cold memory ranges, storage
diagnostics and dominant weight/table/cache controls describe selected execution;
completely bounded/accounted inference is not a goal.

Prepared-input stream copies retain source, destination and host staging through
completion or terminal recovery. The staging ceiling applies per copy operation;
it neither refunds cumulative control budgets nor caps recovery retained by other
operations. Cache identity survives transfer. Storage surveys expose known backing
identities and allocator capacities without evaluating lazy tensors or retaining
native roots. Media surveys include request proofs, prepared ingress/rotary products
and a supplied encoder continuation. A survey is metadata, not a session checkpoint
or proof of completion.

Host parameter-cache and transfer observations preserve backing identity independently
of cache eviction. Read-only reporting performs no housekeeping, polling, evaluation
or resource acquisition. Detached native submission reports describe at most 64 nodes
and 256 allocation records on the calling thread; truncation, borrowed queues, active
reaping and uncovered owners remain explicit. Empty or partial observations do not
authorize reclamation or budget refunds. Native completion ownership governs release.

Realtime discovery and loading preserve the controlled session boundary.
`api::realtime::prepare_realtime_model` discovers the architecture and retains exact
component sources, pinned assets and source guards. `prepare_mlx_realtime` admits
`RealtimeLoadOptions::Speech` for PCM conversations or `Categorical` for
Moshi/PersonaPlex frames before native binding. A protocol mismatch fails before
native resources are created. The consuming `PreparedMlxRealtime::load` accepts
execution contexts, with no replacement sources or host policy, and returns the
selected bound execution. Speech loading registers the existing session and queues
prefill; neural work starts only when that session advances. The same session
provides PCM ingress, tool waits, committed output, snapshots, restore and forks.
The [VoiceChat validation contract](voicechat-validation.md) distinguishes artifact
admission, executable-session conformance and independent numerical parity.

The portable `VoiceChatToolChannel` accepts effective function tokens only after
their shared transition commits. Callers capture its `VoiceChatTurnIdentity`
before admitting work and pass that identity with the committed observation;
interruption invalidates earlier observations and outstanding result tickets.
Complete schema-validated semantic tool events appear together at EOTC. Payload
token, decoded-byte and call limits return typed errors without clearing retained
input or publishing partial calls. External applications execute the calls.
Submitting a matching result prepares bounded ASCII tokens without adding markers.
`response_policy` only peeks at the next forced function token; the cursor advances
when that exact token commits, so failed candidate execution consumes no response.
After supplied tokens are exhausted, unforced tool decoding retains the pending
ticket until the model commits EOTR. Committed observations within a turn must
arrive exactly once and in order; the turn stamp does not deduplicate callbacks.
Rejected results and mismatched feedback preserve the pending state. Delivered
events and external tool effects are not undone by interruption. The enclosing
session driver retires interrupted neural work, records timing and composes
these host policies with turn-taking and session snapshot/restore.

`VoiceChatToolWait` consumes explicit monotonic timestamps for tool waiting.
An initial selected reminder finishes even if the external result is ready;
unstarted generic reminders are skipped once a result arrives. At most two
distinct generic selections have one-second gaps by default. The fifteen-second
final timeout starts after the last selected phrase finishes playback, with
portable duration overrides. Native completion of EOS returns a playback receipt;
it does not start the timer. The application records the matching presentation
completion through `finish_playback`. Offline execution supplies the equivalent
recorded presentation frontier without sleeping.

Each reminder admission retains an identity until shared native work commits or
is safely discarded. Repeated advancement returns that same receipt. Re-admission
after discard has a fresh identity even at the same timestamp; duplicate or
interrupted commits cannot advance the phrase cursor. A result arriving during
admitted speech does not replace that work. Waiting for playback or a timer leaves
perception-only work available. Timeout submits a prepared response through the
existing function channel and rejects later external results; natural EOTR still
ends response decoding. Results submitted before timeout advancement win at the
same recorded timestamp. These host-policy transitions use the shared neural
execution path and participate in session snapshots.

`VoiceChatTurnTaking` separates first-prediction RNNT evidence from label-loop
termination. Perception-only work advances acoustic counters; tool decisions
advance generated-position history without inventing audio frames. Prompt prefill
is absent from generated-position history. The pure live resolver receives the
current RNNT evidence and sampled tokens before effective feedback reaches TTS.
Failed, abandoned or repeated resolution changes neither host state nor events.
Step-aware admission checks acoustic and generated-position limits before native
submission, including perception-only and fresh/retained tool paths. Reminder and
drain work remain admissible when those unrelated budgets are exhausted.
After shared publication, `observe_committed` checks the captured admission,
invocation and predecessor frontiers, then adopts the same deterministic policy
successor. Duplicate, mismatched and interrupted observations fail without adoption.
Committed RNNT barge-in emits EOS and advances the existing tool-channel identity.
During active tool work, fresh perception reaching the consecutive nonblank
threshold interrupts serialization, waiting or response injection. The same
commit clears the wait and substitutes EOS/PAD neural feedback; perception-only
interruption adds no model position. Dropped or rejected candidates preserve the
original feedback, tool ticket and wait. Late results fail their original ticket
validation, including when a newer turn is already running.
Before accepting another live frame, interruption drains the retained projected
features through `ReplayAudio` work in the same scheduler. Replay advances only
backbone positions and effective PAD history; its audio and RNNT frontiers,
transcription, speech and random state remain unchanged. No hidden clock is read.
The first replay establishes EOS/PAD feedback for both automatic and explicit
interruption, and subsequent replay uses PAD/PAD. Dropped candidates retain the
same next frame for retry. Queued features and the replay cursor are included in
snapshots and forks; capacity exhaustion is typed. Normal completion of tool work
resumes live audio and releases its unreplayed queue.
Explicit interruption invalidates tool waits and responses while preserving
already committed acoustic/model progress.

The composed `VoiceChatConversation` stages those observers together before
neural publication. An admitted transition attaches to the shared prepublication
gate; it validates the architecture output, full tool payload and reserved
publication capacity before the scheduler commits. A malformed EOTC cannot adopt
otherwise valid turn evidence. Reminder cursor advancement is part of the same
candidate; its receipt must match the actual spoken token retained by architecture
execution and validated by host observation. `VoiceChatPublicationState` pairs
the private staged successor with the ordinary neural semantic branch. Its commit
adopts host state and enqueues output immediately after successful neural commit;
applications cannot independently extract or commit the staged host successor.
Failed mapping, failed neural commit, cancellation and discarded candidates leave
canonical conversation state unchanged.

This owner permits one outstanding admission and returns `Busy` for external
results, interruption, wait advancement or another admission until that work is
committed or discarded. Applications retain rejected submissions and retry them;
no audio or result is silently dropped. Recorded timing controls wait and
playback policy. The session retains this admission with its shared scheduler
and includes the conversation state in snapshots.

Before enqueueing an invocation, applications reserve a publication permit from
the canonical publication owner and retain it with queued work. Queue exhaustion
therefore rejects admission without failing the active scheduler request. The
branch consumes the permit before native submission. Reservations remain charged
through completion and cancellation retirement; commit replaces them with exact
queued payload charges. An independent receiver delivers each queued publication
once and releases its charge, including after the scheduler removes a cancelled
or failed request. Queue slots are preallocated and charge validation precedes
neural commit, so host adoption and enqueueing add no fallible post-commit work.
Dropped scheduler progress does not discard committed speech or tool calls.

Agent-text and RNNT transcription deltas share this publication boundary and
have independent byte allowances in each permit and the aggregate queue. Failed
observation, insufficient text capacity and failed neural commit consume neither
decoder carry nor text. Tokenizer lookbehind is explicitly bounded and copied
with the host candidate; split UTF-8 sequences remain private until complete.
RNNT whitespace trimming retains only a bounded trailing suffix, including across
perception-only frames during tool waits. Interruption resets unfinished agent
text without rewinding the acoustic transcript. Drain checks agent byte completion
and finalizes RNNT trimming. The containing architecture frame attributes agent
text to its model/audio frontier and transcription to the source frame's sample
interval; these timestamps do not claim word-level acoustic alignment.

The optional PCM ingress port accepts arbitrary chunks while a branch is pending;
it does not mutate the scheduler's canonical frontend. Applications reserve an
input receipt before enqueueing fresh-audio work and pass it with the output
permit to `admit_input`. Live, perception-only and fresh-tool invocations consume
one input frame. Reminder, retained-tool and silence-tool invocations retain queued
speech. The frame remains capacity-charged until semantic commit. Completion or
policy failure leaves all accepted PCM available for retry, including chunks
submitted after the failed frame was reserved.

Finish is explicit. A final partial frame transfers only original samples and
then closes the branch frontend with padding enabled; its transcription retains
the original `valid_samples`. Empty input, or finish declared after the last full
frame was reserved, uses a finish-only receipt on Drain without advancing audio
time. Drain rejects unread or unfinished ingress. Cancellation or request removal
closes the external sender; the existing scheduler still controls retirement of
an outstanding input receipt. Accepted PCM is not consumed by cancellation.
The publication owner's `prepare_prefill` and `prepare_next` methods combine that
port with transactional invocation selection. Returned `VoiceChatWork` holds an
exclusive conversation admission before scheduler enqueue; native submission
consumes it once through `work.admit`. The adapter attaches the resulting admission
to its ordinary prepublication gate. Dropping queued work releases both permits
and the host admission without consuming PCM or changing committed policy.

`work.submit` combines that admission and attachment with portable invocation
assembly. It passes prepared speaker conditioning and silence features to the
ordinary architecture executor, checks prompt counts before admission, and uses
the selected live/tool/reminder rendering policy. It returns owned
`VoiceChatSubmittedWork`, whose completion error is always `BackendFailure` and
whose gate retains exact native completion and one-shot host validation. The MLX
binding delegates to this same method. A synchronous submission error still
requires ordinary branch discard and native recovery; an observation failure
does not authorize early release of submitted resources.

`VoiceChatAdvance` supplies recorded time and tool-audio policy.
The default tool-audio input comes from `PreparedSilenceConditioning`: one second
of zero PCM through fresh full perception, retaining its first projected frame.
Preparing this input does not consume live PCM, initialize RNNT, advance model
positions or draw random values. The native preparation method establishes exact
completion before the embedding is borrowed through `VoiceChatExecutionInputs`.
Only accepted work advances the committed timing frontier; blocked or discarded
attempts do not trigger timeouts or advance reminder cursors. Prefill must commit
before automatic selection. Active function serialization and response injection
default to prepared silence conditioning, alternating with queued perception/RNNT.
The explicit real-audio policy uses fresh or retained features, with silence before
the first encoder frame; pending external results admit perception-only work.
The selected initial/started reminder finishes
before early responses are injected. A due reminder alternates with available
perception work, with at least 80 ms between committed reminder tokens. Playback
completion remains an explicit application event. Drain never bypasses tool work,
unread input or an uncommitted finish policy. It is selected automatically once
the declared input and continuation finish and pending tool work resolves.

`VoiceChatSession::start` registers prepared state and queues prefill without
executing native work. `advance` submits at most one invocation; completion of
pending work returns before selecting its successor. `poll` retires work without
submission. The `drive` iterator repeatedly calls this same advancement with
caller-recorded ticks, without sleeping or reading a clock. Exhausting ticks leaves
pending work owned by the session. Input and output handles remain independent.

`BoundVoiceChatSession` retains the execution mechanism and those endpoints.
Its `advance` and recorded-tick iterators call the ordinary session driver;
`drive_with_endpoints` allows bounded output draining and PCM submission between
yields. A shared mechanism's pending work causes `Busy` before admission, preserving
queued work and permitting retry at the same recorded timestamp. Polling the
current session can retire its own completion before this availability check.
`session_mut` exposes the existing snapshot, restore and host-policy operations.
`fork_with_executor` explicitly binds compatible prepared resources to the ordinary
accounted fork; selected MLX sessions provide `fork_shared` for the same model.

The independently selected TTS tensor-parallel component borrows caller-owned
local cache branches through `forward_parallel_with_state`. It shares the ordinary
unit leases, retained-value completion and error cleanup. This component entry
point does not establish a distributed VoiceChat session or authorize independent
sampling and turn advancement on each rank; the public bound loader remains a
single-device composition.

The TTS partition contract keeps each stage's cache slice caller-owned and
transports the original causal position or character key visibility with hidden
states. Final normalization owns no cache. Partition entry and completion use the
shared layered partition driver; auxiliary boundary tensors retain their declared
scalar types independently of activation transport precision.
`LayerwisePartitionRuntime` borrows the same unpublished state across all groups
of a local stage, including blocks followed by normalization. A failed later unit
aborts provider work and leaves the candidate state with its caller for discard;
it does not publish or rewind canonical state. Native stage adapters share this
path under resident, host-layerwise and disk-streamed policies.

The cross-process TTS component binds an architecture-validated partition,
manifest and execution plan to the existing distributed graph driver. Each rank
supplies an owned candidate cache vector, recorded invocation descriptor and
commit epoch. Owned and borrowed cache adapters execute the same equations;
normalization-only stages retain no cache, and compound block/normalization
stages retain one forward context across local group callbacks. Shared transport
moves the typed boundary and the shared publisher broadcasts the selected output.
Execution-phase agreement prevents a failed component execution from reaching
output broadcast; bounded completion retains native resources until their
disposition is established. The adapter returns a
`DistributedCommitOutcome`; callers adopt the candidate and expose output only
for `Committed`. Final agreement compares exact integer epoch IDs across ranks;
the enclosing scheduler retains authority for issuing those epochs. Failed
candidate state is discarded, and poisoned communication
or a noncommitted outcome requires recovery before runtime reuse.

This component path does not select independent RNG decisions, turn policy or
tool execution on individual ranks. The full coupled VoiceChat graph and public
bound session loader remain single-device compositions; distributed conversational
commit, snapshots and externally delivered speech are not established by the
component's commit agreement.

The `eredu realtime` command advances this bound session with wall-clock or recorded timestamps,
retries bounded PCM submission and drains committed publications to speakers,
WAV and JSONL. Pending microphone input is distinct from EOF. In offline mode,
pending native completion holds recorded time fixed. System playback receipts
wait for native playback timestamps; interruption invalidates queued older speech. Live frames advance the input clock;
tool serialization retains its independent substeps. Scripted tool results bind
to committed batch tickets and are discarded after interruption or stale-result
rejection. After WAV EOF the application submits bounded silence while committed
tool work remains active. `VoiceChatSession::has_tool_work` observes committed
call serialization, waiting and response decoding through EOTR, without exposing
unpublished changes. Completing or interrupting tool work resets the CLI's
configured response tail; ordinary live frames consume that tail before input
closes and the ordinary drain runs. The JSONL stream records synthetic input
separately from WAV source samples. This application policy does not change the
public input sender's explicit finite finish contract. A step limit reports
exhaustion instead of resetting state.

`VoiceChatStartOptions::{acknowledgement, initial_reminder}` use
`VoiceChatPhrasePolicy::{Disabled, Fixed(text), Streaming { seed }}`. Streaming
selection uses the released 20-phrase acknowledgement pool and ten-phrase initial
reminder pool, each with an independent retained host RNG. Preparation admits all
BOS/text/PAD-tail/EOS sequences against the text limits. The unpublished SOTC
successor selects both phrases; only committing that successor advances canonical
generators. Rejection and discard leave the next choices unchanged. Snapshots and
forks preserve buffered randomness, selected indices and cursors, with shared
pool storage charged to each retained owner. Host streams are independent of
neural sampling and read no clock or OS entropy.

Function-call collection sends acknowledgement tokens to TTS independently of PAD
agent feedback. Each committed Tool invocation consumes one token; interleaved
perception and failed or discarded work consume none. Collection end or
interruption clears its unconsumed suffix. Response injection does not replay it.
The initial reminder remains selected through EOTC. A per-call
`VoiceChatInitialReminder::Configured` uses that choice; `Disabled` suppresses it
and `Text` supplies an explicit override. An early result stays queued until the
initial phrase and its recorded playback finish. Interruption or EOTR clears the
selection. These semantics apply to controlled advancement and the ordinary
iterator over the same session.

Retained startup validates host options and prompt framing before native work.
The native binding prepares Aria/silence under completion ownership, initializes
selected joint caches and RNG, then queues prefill without executing it. Startup
assets and parameter providers remain outside snapshot state; snapshots retain all
mutable generation, conversation and delivery state through the existing owners.

Session operations require nondecreasing recorded time, including blocked or
capacity-rejected attempts; only committed work advances the conversation's policy
frontier. Cancellation and failure retirement use supplied scheduler timestamps.
Host interruption, response submission and playback updates require a settled
invocation boundary and return typed `Busy` while work is outstanding. Queue
capacity rejection preserves input and permits retry after output delivery.
Committed terminal drain closes ingress and finishes the request; committed events
remain deliverable after finish or cancellation. Native and neural-commit failures
retain their original source through the portable session error.

`VoiceChatInputSender::finish(PcmFinishPolicy)` closes PCM admission and declares
final-frame handling plus an explicit finite silence suffix. VoiceChat accepts
zero padding or complete-frame enforcement. Each synthetic frame uses ordinary
advancement, completion, publication budgets and cancellation; it carries
`PcmFrameSource::Continuation` with zero original samples. Source-sample counts
exclude this suffix while frame timestamps include it. Repeating the same finish
policy is idempotent; changing it is rejected. Snapshots and forks preserve the
remaining suffix without allocating the entire zero waveform. Finite continuation
does not guarantee TTS EOS for an arbitrary response. The retained-session loader supplies prepared state to this same
construction path.

For example, two seconds of continuation at the declared 80 ms cadence uses:

```rust,ignore
use eredu::api::realtime::{FinalPcmFrame, PcmFinishPolicy};

input.finish(PcmFinishPolicy {
    final_frame: FinalPcmFrame::ZeroPad,
    trailing_silence_frames: 25,
})?;
// Continue advancing with recorded timestamps until the session finishes.
```

The selected RNNT rules use first-user speech confirmation independently from an
agent greeting, a bounded prior-token window and frozen speech density while the
agent speaks. Main-loop PAD gating reads prior committed speech counters, leaving
generated waveform separate from next-step feedback. Active tool work suppresses
RNNT BOS/EOS overrides, matching the pinned wrapper. Automatic RNNT cancellation
inside tool waits uses the same sustained-speech threshold and committed
interruption transition. Session snapshots retain the host counters and histories.

The VoiceChat RNNT component returns candidate state for explicit encoder-frame
advancement, retaining recurrent values, hypothesis and frame identity. Repeated
advancement, restored continuation and forks share the same bounded label loop.
A capacity failure discards tentative recurrent updates. The coupled speech
transaction retains RNNT candidates through completion and publishes them with
the matching conversation state.

RNNT resident, host-layerwise and disk-streamed providers use those same
transitions. Priming and token updates lease only embedding and recurrent units;
joint evaluation leases its projections, retaining the encoder projection once
per frame. Unit completion protects replacement hidden/cell tensors before a
parameter lease can be released. A failed later unit leaves the borrowed parent
unchanged, and preflight rejects invalid input and hypothesis bounds before
acquisition. Changing residency does not advance audio time or establish a new
commit boundary.

The RNNT coordinated component entry point consumes explicit request, turn and
invocation identities. Ranks agree the parent frame frontier, complete hypothesis,
geometry and limits before acquisition. One designated participant chooses every
greedy symbol; all replicas consume the same completed integer broadcast. Phase
agreements fence mismatched control and local failures before another neural
phase. Returned recurrent/hypothesis state remains an unpublished candidate;
replay from an unchanged parent uses the same greedy transition. This component
coordination does not select a distributed coupled session or publish transcript
events independently of its enclosing scheduler.

The predictor pipeline advances one explicitly identified recurrent step through
stage-local LSTM layers. Its immutable parent includes only the selected global
layer range and the final prediction readout. Boundaries validate both the layer
cut and the complete 64-bit step; token substeps do not advance audio time.
The native entry point appends that step to the caller's seven-word descriptor
for agreement across every pipeline and tensor rank before learned execution.
The shared pipeline owns transport, failure agreement and completion under the
caller-supplied commit epoch. The returned branch is adopted only for a committed
outcome. Snapshot copies retain local hidden/cell tensors and the published row,
without duplicating the final owner's hidden tensor or copying remote cells.
This component entry point does not assign epochs, choose symbols, publish
transcripts or schedule a complete speech session.

The pipeline RNNT frame driver supplies predictor invocations from an explicit
caller-owned `RnntPipelineCursor` and delegates symbol decisions to the existing
coordinated greedy loop. It reserves two priming slots for an unprimed parent plus
the configured symbol bound, or just the symbol bound for a cached parent.
The reservation advances step/epoch frontiers even on later failure, disagreement
or early blank; insufficient capacity and arithmetic overflow reject before any
predictor submission. Replaying a conversational snapshot uses fresh authority
from the same cursor. The cursor is neither cloned nor stored in `RnntState`.
Internal committed readouts remain candidate state until the enclosing frame
commits, and aborted or indeterminate predictor outcomes cannot enter the joint
network. This driver does not publish host transcript events or advance live audio
for token-only predictor substeps.

The paired perception/RNNT pipeline transition reserves the RNNT frame before
perception parameter reads, and checks the same acoustic frontier in both parent
states. One perception execution returns both feature branches. An aborted or
indeterminate perception outcome cannot enter RNNT; a later RNNT failure discards
the entire paired candidate without refunding its recurrent reservation. The
caller supplies a fresh perception epoch and retains the RNNT cursor outside
snapshots. Snapshot and fork copy only the selected stage's caches, retained
prediction and transcript. Internal component commits do not publish transcript
events or install either state into the enclosing session.

PCM advancement uses one `InputExecution` contract for ordinary and stage-local
input caches. The same coupled state transaction owns frontend carry and that
selected cache type. A borrowed pipeline input adapter receives explicit request,
turn, invocation and perception-epoch identity plus the caller's recurrent cursor.
It remains usable across incomplete PCM submissions without spending either
authority. A ready attempt consumes its perception epoch even if later preparation
or execution fails; replay uses a fresh adapter. Ranks agree availability and host
feature content before input tensor creation, preventing one rank from returning
for more PCM while another enters perception. Finish padding and continuation
silence retain the same sample attribution and host drain policy as ordinary input.
Readiness checks still use bounded communication authority; snapshots do not refund
those agreement resources.

VoiceChat frontend advancement consumes at most one 1,280-sample cadence frame
into a candidate PCM/mel state. Input chunks may cross arbitrary frame boundaries;
full queues return backpressure without consuming the rejected input. Finishing
either requires a complete cadence frame or explicitly pads the last frame,
retaining its actual sample count. This host drain does not imply TTS/codec drain.
The perception transition returns both feature branches and new per-layer caches
without modifying its parent. Audio and encoded-frame clocks remain distinct from
RNNT token-only updates. Component fixtures compare whole and segmented execution,
retry, retained snapshots and forks; no component clone establishes native session
lease, completion, event-delivery or external tool-execution ownership.

Perception uses the same architecture-owned subsampling, Conformer and projection
units under resident and bounded parameter providers. Each invocation starts from
an independent candidate over the borrowed parent caches. A failed acquisition,
neural operation or unit completion cannot install intermediate Conformer state
or advance the parent's frame frontier. The shared layerwise guard retains each
unit through its exact completion, including candidate attention/convolution
roots and the raw encoder capture used by RNNT. `invoke_with_context` returns that
capture with the projected output from the same traversal; transcription never
requires a second encoder pass. Startup silence uses the same provider path with
fresh caches and does not advance the live frontend or RNNT.

MLX perception construction consumes cold restricted per-unit bindings and the
existing resident, host-layerwise or disk-streamed policy. These choices alter
parameter residency without changing frame admission, candidate-cache ownership
or the enclosing scheduler's publication boundary. `ResidentRuntime::from_units`
also accepts previously bound units after validating their graph counts, so
resident assembly preserves the original parameter objects. Perception residency
alone does not establish full speech-stack residency or distributed conversation
commit.

Tensor-parallel perception uses the same candidate transition and shared unit
completion. Attention and both feed-forward outputs are reduced before updating
full-width residuals or producing successor caches. Each rank retains the entire
normalized-input and convolution history, with the same frame frontier; local
projection geometry does not shrink this state reservation. A selected partition
requires its matching communicator before execution. The enclosing owner still
coordinates rank failures and commits the coupled conversation; component output
alone does not establish that distributed publication boundary.

Perception pipeline candidates retain canonical block ranges and component
ownership, including the distinction between stateless subsampling and projection
stages. Branch construction and ingress reject state belonging to another stage.
Each boundary's exact frame frontier must match the receiving candidate before
local units execute. The sequence dimension reflects subsampling and overlap
removal. All stages advance the candidate's encoded-frame frontier only when
their local transition finishes; no boundary advances live audio or tool time.
The shared distributed driver publishes both feature branches from one encoder
invocation and establishes native completion before returning its commit outcome.
Failed candidates are discarded, and externally visible publication remains the
enclosing session's responsibility.

Shared pipeline traversal uses architecture-declared shapes for each directed
boundary and for final output. Intermediate sequence lengths and channel order
may differ from request ingress. The driver retains request dimensions, checks
resolved schemas before receive allocation, and uses one declared publication
shape on all ranks. The output owner validates its result inside final-stage
execution, so the existing failure agreement rejects a shape mismatch on every
rank before publication. These geometry hooks preserve the existing parameter-lease,
failure-agreement and commit lifecycle; they do not advance conversational time
or select a codec pipeline for a session.

Nemotron-H supplied-embedding target requests select vocabulary logits or normalized
hidden values through the same resident/layerwise/partitioned drivers. The readout
choice travels as validated I32 control data in pipeline boundary version 2;
malformed shape, dtype or value is rejected. Hidden readout skips vocabulary
projection while preserving normalized-readout observations. VoiceChat's agent
and function heads can consume that result independently; neither projection
advances model caches. The enclosing VoiceChat transaction owns conversational
commit, and external applications execute published tool calls.

The generic function-head provider participates in that same coupled readout for
ordinary and controlled invocation. A shared dual-head method uses the retained
normalized tensor once for each vocabulary projection; it neither reruns Nemotron
nor advances its state. Shape, width and floating-dtype preflight precede head
acquisition. The function graph has one learned unit and no pinned static head;
bounded execution retains its input and logits until exact unit completion.
Provider errors retain their original source through `BackendFailure` and follow
the enclosing invocation's completion/discard boundary. Publication and tool-call
commit belong to the enclosing session transaction.

Prepared head bindings retain the admitted dense or affine matrix format, use
F32 floating weights/companions and charge bounded conversion storage. Ordinary
and bounded construction consume those same bindings. Tensor-parallel providers
validate exact vocabulary ownership against communicator rank and degree before
acquisition and gather complete logits through the shared traversal. Input and
gathered output remain
completion roots until the head lease is released. Projection consumes no RNG and
does not advance a model or conversation frontier. Native TP construction retains
bounded failure authority and validates the declared vocabulary-gather capacity.
Guarded head invocation agrees input metadata before acquisition, uses shared
traversal agreements around provider work, and agrees leased-unit identity before
projection. The backend guards its local multiplication before gathering. A
failed participant prevents peers from crossing those boundaries; acquired leases
follow ordinary abort/completion handling. Logical rejection permits a fresh
provider retry, while transport failures remain governed by the retained
communication authority. These component calls do not establish distributed
conversational commitment. The retained speech-provider driver validates summed finite
parameter-pool allowances and conservative materialization reserves before any
binding callback, including RNNT's selected independent parameter pool. It assembles the
same learned components consumed by `execution`; input carry, neural branches,
RNG and undelivered events retain their existing owners. Failed construction
returns no aggregate, and native completion protects binding cleanup.

The native coupled adapter adopts independently selected speech providers and the
selected Nemotron layer runtime in the same completion owner. Complete parameter
admission combines backbone device/host allowances and loading reservations with
the independent speech pools before binding. Resident, host-layerwise and
disk-streamed backbone blocks use the same prefill and cached invocation. The retained report remains inspectable
after adoption. It covers logical parameter tiers and loading scratch, excluding live
state, activation/spectral workspaces and source caches. Provider selection does not
create a session scheduler, change snapshot ownership or establish native distributed
validation. Nemotron embedding, final normalization and agent-head statics remain
pinned and are included in the selected backbone device allowance.

EAR-TTS RVQ refinement returns a new partial-code state for each active slot.
Its ordinary generation helper repeats that same transition. Parent states
remain unchanged after failed draws or neural operations; cloning a partial
frame preserves its next slot, assigned prefix and PAD suffix. Sampled refinement
uses the shared realtime branch's retained RNG for uniform and Gaussian draws.
Canonical RNG adoption remains at the ordinary completion/commit boundary.
Zero-depth iteration slots request no draws. This component-level replay does
not capture Gemma guidance caches, codec overlap or externally delivered speech.

The EAR-TTS causal transformer accepts existing attention-cache branches supplied
by its execution owner. Layer frontiers, sliding retention policy and positional
capacity are checked before any append. A failure after a neural submission
requires the existing runtime's completion/rollback handling; this component does
not turn a polling failure into completion. Conditional and unconditional lanes
occupy distinct batch rows in every layer cache. Character encoding is an
independent masked batch without recurrent state. These component calls do not
establish a complete conversational snapshot or tool-wait driver.

Resident and bounded TTS transformer providers execute the same block and
final-normalization units. The shared runtime borrows the supplied cache branch
for one call without moving its parameter owner or creating another scheduler.
Unit completion retains exact K/V roots and masks before releasing leased weights.
Both guidance rows travel in those same cache tensors. A late acquisition or
neural failure may leave the unpublished branch partly advanced; the enclosing
transaction discards that branch and retains completion ownership before retry.
It does not rewind or replace the canonical parent inside the component call.
Character encoding uses the same provider contract with explicit stateless input.
The generator, sampling and coupled speech paths accept these providers, so
controlled advancement and ordinary repeated advancement share their equations
and commit boundary. Component residency alone does not establish a full
speech-stack residency schedule or native distributed execution.

MoG prediction and selected-component evaluation share the existing parameter
traversal through `MogExecution`. Prediction executes residual blocks and the small
normalization/logits head; selected evaluation leases only the component bank with
explicit mixture IDs and Gaussian draws. It does not rerun prediction, append an
attention position or advance refinement/RNG state. The forward context retains
guided hidden values, decisions, draws and all moment outputs until each unit's
submission is safe. Malformed draw geometry is rejected before acquisition;
provider and neural failures use the ordinary traversal's completion handling.

Resident, host-layerwise and disk-streamed MoG providers enter the same generator,
RVQ refinement, speech sampling and coupled execution paths. Parameter residency
therefore changes neither explicit iteration selection nor the enclosing candidate
and RNG commit boundary. Retained predictions and partial refinement frames contain
values rather than parameter leases and can resume through the selected provider.
Tensor-parallel MoG execution uses the same traversal and explicit request modes.
Residual MLP reductions precede post-normalization; selected low-rank contractions
are reduced before adding replicated residual and noise terms. Every participant
consumes the same supplied mixture IDs and Gaussian draws; no rank chooses new
random values. The retained layout is checked against the communicator before
acquisition, and guided hidden, decisions, draws and moments remain completion
roots while the selected lease is live. A complete speech-graph residency schedule,
coupled distributed commitment and released-checkpoint session parity remain
outside this component support.

TTS RVQ feedback uses the same resident or bounded unit traversal through
`RvqExecution`. Code sums and residual quantization acquire the canonical stacked
bank; embedded feedback additionally acquires the shared projection. Ingress and
refinement use that same provider. Refinement candidates retain code values and
schedule position independently of parameter leases. Failure while loading or
executing the bank after sampling leaves the supplied frame unchanged; the
enclosing shared random-state transaction restores its candidate draws. Retry
uses the same slot without appending a causal attention position.

Tensor-parallel RVQ providers validate coherent bank/projection ownership against
the communicator before acquisition. Per-rank latent distance contributions are
summed before each code decision, preserving global first-index ties. Each active
slot starts from its supplied fresh latent; previous prefix codes are retained,
not subtracted from that sample. Local residuals stay within the invocation and
only the complete replicated code candidate escapes it. Depth sums and embedded
feedback return full-width results through shared completion. These component
calls retain the enclosing transition's candidate and RNG ownership; they do not
establish coupled distributed conversational commitment.

Learned TTS conditioning uses three independently leased units for character
parameters, ingress and gated fusion. `ConditioningExecution` runs character
lookup and pooling/flags around the selected character transformer, and runs
ingress after RVQ feedback has completed. No conditioning lease spans those other
providers. Empty character selections skip lookup, transformer and pooling.
Both guidance lanes remain completion roots. Conditioning and fusion finish
before the causal transformer advances the supplied cache branch; an acquisition
failure there leaves its attention position unchanged and permits retry through
the ordinary candidate/commit path. These component providers do not add another
session scheduler or random-state owner.


Tensor-parallel conditioning preserves those lease and completion boundaries.
Character lookup and pooling each return a full-width tensor before the next
provider runs; ingress returns both complete guidance lanes. Fusion reconstructs
its projected features before the global normalization. Exact communicator
rank/size is checked before unit acquisition. The selected parallel provider
implements the same `ConditioningExecution` contract without a separate session
scheduler or mutable conversational state.

EAR neural decoding uses `DecoderExecution` for resident and bounded providers.
Each upsampling, ConvNeXt and output-projection unit consumes an unpublished
candidate, with exact convolution carry retained through unit completion. A late
acquisition or computation failure leaves the parent decoder state reusable.
`AudioDecoder` composes that candidate with spectral overlap before returning
PCM and replacement state; replay and fork use the same transition. Neural finish
loads no parameters, and spectral drain emits retained overlap without another
code frame. Codec quantization and lookup likewise lease selected codebook depths
through shared traversal. Temporary residuals and code decisions remain private
to an invocation; late acquisition or numerical failures leave the input reusable
for retry. Codebooks are shared with fixed-Aria startup encoding, and drain
performs no quantizer invocation. The finite encoder also uses independently
leased learned units. These component providers do not establish a complete
conversational residency schedule.



Tensor-parallel EAR neural decoding keeps the same immutable parent and candidate
state transition. Each rank retains the declared full depthwise convolution carry;
partitioned projections finish their collective before a unit completes. A late
failure discards the candidate, leaving the parent available for replay or fork.
Geometry-only neural finish and spectral drain acquire no learned parameters.
The selected decoder uses the same waveform driver and does not advance its own
audio frontier or conversational policy.

Tensor-parallel quantization retains local residuals and replicated code decisions
inside the same invocation. Score collectives precede each code selection; the
final selected unit retains all full-width result tensors through completion.
Only requested depths acquire parameters. Decode likewise reconstructs its
ordered mean sum before publication. Communicator mismatch fails before provider
acquisition, and failed candidates do not change the caller's input. These
component operations do not advance conversational policy or own random state.

MoG and TTS RVQ tensor-parallel providers use retained failure authority during
the same ordinary layered traversal. They agree invocation policy and refinement
position before acquisition, agree each local lease result, and check prepared
projection or score contributions before entering a reduction. Asymmetric local
failures return on all participants without changing the parent refinement state.
A repaired invocation can use the retained parent after its owner establishes
safe completion. Mixture IDs and Gaussian draws remain explicit caller inputs;
these provider agreements do not allocate randomness or commit speech output.

The composed TTS generator uses that same cache contract. `begin_frame` runs one
causal invocation against unpublished branches and returns the two guidance
hidden lanes plus an all-PAD refinement candidate. `advance_frame` proposes one
active iteration; `generate_frame` repeats that exact transition. Retrying or
forking a retained refinement frame never appends another attention position.
The frame's model position counts Gemma positions, independently of live PCM
time and tool substeps. A complete conversational snapshot must retain the
associated attention branch and candidate RNG together with the partial frame.
Caller-supplied draw callbacks retain responsibility for their own RNG rollback.
Selected causal execution validates local state ownership and its explicit
position, including pipeline stages with no attention slots. Replaying copied
state preserves this position and requires fresh invocation authority. A causal
pipeline adapter admits one scheduler-issued epoch; it never allocates epochs or
clones the transaction's mutable caches. Character conditioning uses its own
selected provider. Failures before causal submission require the enclosing
distributed owner to agree preparation outcomes before any rank enters the
pipeline. The adapter's component agreement does not coordinate turn policy or
random decisions for the whole conversation.
`advance_sampled_frame` and `generate_sampled_frame` instead wrap the requested
slot or remaining slots in one branch RNG transaction. Synchronous draw or
dependent MoG/RVQ failure preserves both the parent frame and pre-call RNG,
including when earlier slots in the same call succeeded. Each active slot draws
uniforms followed by Gaussian noise even when noise scale is zero. Complete
generation consumes no draws; submitted branches reject further sampling.
This rollback does not refund other budgets or establish native completion.
Deferred native failures still require the enclosing submission owner's
completion handling before canonical publication or resource release.

Sampled generator and speech operations accept optional
`SpeechSamplingCoordination` over retained replicated communication. Its recorded
request, turn and invocation identity is agreed with the operation mode, geometry,
policy and active slot before dependent work. Only the designated owner consumes
RNG; followers receive mixture IDs and Gaussian values. Phase disagreement and
synchronous peer failure reject the candidate, and final agreement occurs before
the RNG transaction adopts its state. Rendering gates output preparation before
codec execution. Authority is consumed even on failure; retry or snapshot replay
uses fresh authority with an unpublished branch. Authority is not snapshot state
and cannot undo delivered audio. Internal collective providers still own their
local failure gates, and native completion remains the enclosing owner's duty.

TTS EOS input substitution uses the nested speech-generation configuration.
Output-feedback EOS substitution is a distinct top-level streaming policy;
the pinned checkpoint alone does not enable it. The main speech path preserves
raw generated waveform codes before changing next-step feedback. Output policy
explicitly chooses whether a silence decision also changes rendered
audio. PAD permissions remain per-row facade decisions, and codec control
replacement operates on individual codebook entries.

Streaming EAR decoder pipelines branch only the convolution caches owned by
their stage. Stage identity includes the decoder geometry, namespace and global
unit range; restore or replay validates that identity before learned execution.
Budgeted snapshot copying retains that identity, local layout metadata and only
the owned carry, forwarding the reservation to each tensor copy.
Boundary request metadata preserves the original latent geometry across
upsampling. Native completion retains local candidate carry and published
spectra through the shared distributed transaction. Only a committed candidate
replaces the parent. Neural finish clears local carry without learned reads;
it emits no artificial tail and leaves overlap-add drain to spectral execution.
These component operations do not advance conversational turn policy.

The waveform pipeline surrounds codebook lookup, neural traversal and spectral
reconstruction with participant agreement. It returns one unpublished candidate
containing the local convolution state, spectral overlap and frame/sample clocks.
Step and drain use explicit invocation identity and one-use authority; retry and
snapshot replay require fresh authority. Copying waveform state does not refund
communication resources. Drain releases local convolution carry and emits the
retained overlap without loading codebooks or advancing generated-code frames.
The ordinary and pipeline providers implement the same waveform execution contract
used by speech refinement and the coupled transaction.
Tensor-parallel codebook providers retain their cohort authority and agree on
acquisition and local computation success before reductions. The enclosing
waveform agreement handles component failure after those internal gates; it
does not replace them or refund their communication budgets.
Guarded native reductions complete before subsequent control agreements. A
deadline preserves the existing native completion quarantine instead of releasing
the reduction's resources or treating a polling failure as completion.

Fixed-Aria preparation encodes a bounded finite silent carrier once and retains
immutable prompt inputs for independent prefill branches. Carrier samples and
batch positions have explicit capacity limits; expansion reuses the encoded
single row. The finite encoder has no mutable streaming cache. Resident and
bounded providers run the same finite traversal; input rejection precedes weight
acquisition, and a late failure leaves no encoder carry to reset. The final
layout conversion completes with the last learned unit before borrowed
codebook quantization begins. Tensor-parallel encoding uses this same traversal:
each projection or downsampler reconstructs global channels, and residual-block
reduction precedes replicated bias addition. Actual rank and retained parameter
placement are validated before acquisition. Failed finite encoding starts no
quantizer work; replay needs no convolution-state reset. Preparation does
not allocate a persistent state range for encoder pipeline stages. Each stage
retains its global unit interval; boundary metadata validates the completed cut
and original spectral request before execution. The shared distributed driver
publishes latents after the final frame-major conversion, completes native roots,
and returns the recorded epoch's commit outcome. Only a committed result can
enter subsequent quantization or prompt publication. Preparation does
not admit a session or establish native completion: retries after submission
still require the execution owner's completion handling. Prompt positions count
model warmup and do not advance the live-audio frontier.

VoiceChat invocation plans retain independent component progress under the shared
generation transaction. A plan checks capacity before component work and validates
observed frontend, perception, RNNT, backbone, guidance and codec progress before
adopting its candidate schedule. A plan from an earlier boundary cannot replace
a later schedule. Tool steps using cached audio leave the input frontier unchanged;
reminders also leave Nemotron unchanged. The first backbone decode uses PAD
feedback even after prompt prefill or earlier perception-only work. Explicit drain
closes the schedule, including an empty codec stream; text EOS is a speech-policy
input rather than a session-close operation. Session snapshots retain this
schedule alongside conversation, publication and native execution state.

PCM input composition retains preprocessing carry and paired perception/RNNT
caches as one candidate. Arbitrary input chunks are admitted into the bounded
host queue; advancement consumes at most one 80 ms frame. Retrying a failed
perception or transcription operation begins from the unchanged host and neural
parent. Explicit final padding preserves the count of actual source samples.

Speech composition retains feedback codes and the selected codec carry separately from the
owner's attention-cache and RNG branch. Beginning a frame advances the causal
transformer once; explicit refinement and uninterrupted refinement share the
same transition. Rendering requires a complete refinement and matching component
policy, then returns waveform and feedback as an unpublished candidate. Codec
finish does not advance TTS positions. A failure after attention or random work
still requires the owner's completion and rollback handling; these components
do not independently commit caches or authorize replay of delivered audio.
`render_sampled_frame` extends the branch-local RNG rollback boundary through
feedback and waveform rendering. Failure after all refinement draws preserves
the pre-call RNG and predecessor codec carry. A completed retained frame can be
rendered without another random draw or causal-cache append.

VoiceChat output assembly binds its observed receipt, sparse channel payloads
and host projection to one admitted invocation. Every channel uses the invocation
coordinate, while PCM retains an independent absolute emitted-sample offset and
22.05 kHz rate. Tool and reminder invocations therefore emit PCM without advancing
live-input time. Drain emits only the declared nonempty codec tail; an unused
codec emits no tail. Host validation rejects malformed or nonfinite payloads,
misattributed coordinates and disagreement between native and retained decision
IDs before publication. The session combines these validated frames with
semantic tool events at the shared commit boundary.

The coupled VoiceChat component executor consumes the same unpublished generation
branch for ordinary and controlled advancement. Its model state delegates both
Nemotron and TTS caches to one neural transaction, then adopts host input,
perception/RNNT, speech/codec and feedback candidates after the joint commit.
Input readiness includes explicitly padded final PCM and is checked without
feature extraction. `NeedsInput` leaves an unclaimed candidate reusable.
Once neural execution starts, the candidate accepts no further invocation;
an error requires exact completion and discard, and retry starts from a fresh
canonical branch. This prevents a late character/codec failure from retaining
earlier perception, backbone or sampling progress in a local retry. Native
submission lifetime remains the enclosing owner's responsibility on every path.

The released-checkpoint Metal WAV run uses public `BoundVoiceChatSession`
advancement through this same driver: eight prompt positions, 396 audio frames,
404 backbone positions and terminal drain. Committed waveform offsets cover
698,552 samples, including the eight-sample codec tail, and a finished event
follows drain. [Recorded execution evidence](voicechat-validation.md#released-checkpoint-single-device-execution)
covers ordinary single-device publication. A separate pinned tool run commits the
complete call batch, processes nine perception frames during an 800 ms recorded
wait, accepts the matching ticket, resumes live speech with the scripted result
and drains. Tool-only steps preserve the audio frontier; perception-only steps
preserve the backbone frontier. An overlapping-input run with a three-frame
threshold commits interruption during a silent tool wait, rejects its delayed
result as stale, and drains. Its 103 replay invocations advance the backbone
without advancing audio time or repeating transcription. The resumed model
answers the new question with “The sky is blue.” These runs do
not establish released-weight restore/fork, active-reminder interruption, timeout
or distributed-session parity.

The coupled transition accepts a selected `BackboneExecution` provider and its
caller-owned state. Ordinary and TP providers share the existing layered runtime,
and both agent and function projections consume one normalized hidden result.
The function head selects its own provider and communicator. Preflight and final
schedule observation ask the backbone provider to validate its actual frontier;
the coupled driver does not inspect or require remote cache slots. The compact
two-rank conformance path compares cached live/tool work, reminders and drain
with ordinary execution, including clocks, shared carry and bounded layer leases.
This equation-level coverage does not establish native distributed conversation
submission or completion.

The selected TP backbone optionally agrees fused ingress before entering shared
layered traversal. Channel admission compares prompt/audio operation identity,
exact prompt policy and IDs or dense/categorical geometry and dtype, and fusion
weights. Local prompt construction and each vocabulary lookup are followed by an
agreement before the next lookup or fused output is used. Failure at these
boundaries returns the original local source and prevents backbone cache or
provider advancement. The enclosing native completion owner still retains any
submitted lookup resources. Later block preparation and collectives require
their own coordination; ingress agreement does not establish that guarantee.

Shared MLX row-parallel projections use the selected group's retained authority to
agree local projection/observer errors, reduction admission and output/bias metadata
before reducing. Uneven input shards remain valid. Bias is added once after bounded
reduction completion, followed by local-result agreement. A rejected preparation can
retry with the same authority; a transport failure retains its existing completion
ownership. These operator gates do not cover preceding normalization, recurrent or
attention preparation, and do not provide distributed panic recovery.

An optional `ExecutionCoordination` drives owner selection through this same
coupled transition. Its recorded request, turn and invocation identity is
separate from model snapshots. Admission consumes the authority even when local
preflight fails; exact descriptor agreement precedes branch claim and learned
work. The descriptor includes schedule policy, expected and predecessor counters,
feedback, invocation kind, prompt IDs and masks, tensor geometry and speech output
policy. Followers validate their common execution state but need no RNG and do
not call the conversational resolver. Channel and speech authorities borrow the
same retained coordinator in sequence, preserving the existing sampling drivers
and their RNG transactions.

Component agreements prevent a local preparation or component failure from
silently advancing peers to the next component. Output assembly is agreed before
the candidate schedule is adopted. A failed nested authority fences the coupled
authority without issuing a recovery collective. Retry uses fresh authority and,
after execution has started, a fresh unpublished branch; exact native completion
and discard remain the enclosing owner's responsibility. Agreement never commits
speech, tool events or neural caches.

This behavior applies to replicated ordinary providers. The component gates do
not make unguarded internal TP collectives safe: providers must agree their own
fallible preparation before entering those collectives. The native cold loader
selects ordinary execution, so this authority does not establish a
native distributed conversation session.

MLX supplies a joint hybrid/key-value transaction owner for these cache families.
It admits one isolated candidate at a time, charges bounded copy work before
copying, and validates owner identity, generation and layout before atomically
adopting both states. Discard does not refund cumulative copy usage. Copy
completion is established before exposing the candidate; subsequent model
submission and publication still belong to the shared enclosing completion
owner. The separate `IsolatedStateCopy` mechanism releases source execution
admission after completing an independent copy. `IsolatedStateSnapshot` retains
that copy under the original `SnapshotBudget`; restoring makes a provisional
independent copy before atomic installation. Failed attempts consume copy
allowance, and unresolved native recovery retains logical storage reservations.
MLX's isolated joint owners share cumulative copy accounting across descendants.
Neither this component snapshot nor a copied cache owner represents a public
VoiceChat session snapshot or fork: conversation, external ingress, queued
publications and external-delivery history require facade composition.

The shared realtime generation owner implements isolated snapshot composition when
its model supplies `IsolatedStateCopy` and its schedule, identity, controls,
samplers and RNG supply explicit `SnapshotValue` copies. Native copies finish
before adoption. A failed model restore cannot replace any temporal or sampling
value, and successful restore increments the live publication revision instead of
reusing the saved one. Snapshot reservations include dynamic sampler history and
native RNG storage; copy failures retain the original concrete error source and
never refund copying. VoiceChat schedule/history and the MLX two-word key implement
these value contracts. Native CPU checks compare restored draws and model caches
under those policies without claiming a complete conversation snapshot.

VoiceChat's architecture state also supplies complete component copies: internal
PCM/mel carry, perception caches, retained RNNT hypotheses/LSTM state, TTS feedback,
codec convolution/overlap state and retained audio. `IsolatedStateSnapshot` can
capture the low-level MLX generation owner at a completed boundary. Replaying from
that saved model boundary includes final partial PCM and codec drain. The saved
owner retains copied mutable values and shares immutable processing tables/policy.
It has no authority to rewind externally delivered audio or tool execution.

`VoiceChatSession::snapshot` composes these model copies with independent host
conversation state, PCM ingress, undelivered publications and recorded time under
the shared `SnapshotBudget`. `enable_snapshots` configures limits exactly once;
`snapshot_usage` reports retained storage and cumulative copy work. The source
remains runnable after capture. Tool collection/injection, RNNT/agent text carry, turn histories, reminder receipts and
explicit wait timers are retained. Diagnostics require `SnapshotValue`; the shared
runtime supplies exact copies for `RealtimeDecisionDiagnostics` without refunding
observation budgets. Immutable tokenizer/schema resources keep their shared
preparation lifetime; mutable carry and retained response/phrase/vocabulary data
are charged separately.

Capture and restore require an active idle session after prefill has committed.
Restore rejects snapshots from another owner, newly accepted PCM or finish,
outputs delivered since capture, and accepted external tool/wait/playback or
interruption actions. These typed rejections prevent replaying externally executed
tools or delivered speech and prevent silently discarding accepted input. They
occur before copy admission. Recorded time rewinds only on successful restore;
scheduler telemetry, publication revisions and cumulative copy allowances do not.
Terminal sessions have released their canonical state and reject restoration.
`VoiceChatSession::fork(request)` copies the current idle frontier into a new
session using the same scheduler implementation, without prompt replay. It returns
independent PCM/output endpoints and source-to-child mappings for any pending tool
ticket or playback receipt. Already delivered events are absent from the child's
queue; inherited undelivered tool events use the new request identity. Original
receipts cannot directly change the child, and applications decide whether to
apply an already executing tool's result to both continuations. No tool executes
inside this operation.

All descendants share the original snapshot budget and native cumulative copy
accounting. `max_branches` bounds retained child states; dropping or cancelling a
session does not refund its branch slot while its endpoints or unresolved native
work retain copied resources. Delivered payload ownership transfers to the
application. Restore preserves the child's live retention lease. A shared lineage
rejects previously admitted request IDs, including failed copy attempts, and keeps
bounded metadata reservations for those identities. Independently constructed
sessions still require caller-unique request IDs. Longer child tool IDs must fit
both invocation and aggregate queue bounds before copying. Source and child can
advance independently; each native executor retains its existing submission
admission, so sharing one executor requires serialized submissions.

Fork retention also admits growth of the child's committed state. PCM ingress
reserves its configured capacity at creation. Conversation updates, queued output
(including diagnostic payloads) and completed neural candidates admit their full
logical size before installation. Shared high-water charges survive restore,
delivery and failed subsequent transitions, and remain owned by native retirement
holders of the same lease. A retention-limit error commits neither the oversized
neural candidate nor its audio consumption or publication. These retained-state
charges are separate from temporary execution workspace and snapshot copying.

Input submission and output delivery are fenced through snapshot estimation,
copying and native installation, including callbacks from custom copy mechanisms.
`try_pop_output` reports `SnapshotBusy`; `pop_output` reports no available output
while fenced. All host storage is copied before the sole fallible neural install.
Only then are existing sender/receiver contents replaced, so failed restore
preserves live queues, policy and time. Retired diagnostic destructors run after
host installation; any new audio they admit remains queued.

Cold generation preparation checks the released BOS/EOS/PAD identities and
admitted capacity for mandatory speech startup. It initializes the existing
generation owner at the unprefilled boundary with PAD/PAD feedback and empty
effective-token history. Random state is explicit and required even when agent
sampling is greedy, because speech generation remains stochastic. Input carry
and the initialized joint neural owner enter the same canonical state. Cache
initialization checks the exact selected layouts before allocation. Transaction
support is guaranteed by the shared selector; initialization does not select or
advertise complete session capabilities.

The low-level MLX coupled adapter enters shared native recovery before executing
equations, including eager RNNT and sampling work. Architecture preflight occurs
first; `NeedsInput` submits nothing and leaves the candidate reusable. An equation
error or unwind attaches failed completion, poisons execution and prevents
publication. Recovery retains component tensors, model resources, paging owners
and joint copy admission independently of the discarded candidate. Successful
output uses the same prepublication wrapper and architecture host observer as
portable execution. A caller releases the completed frame and completion handles
before requesting the next candidate; safe native retirement releases admission.
The facade session supplies public advancement and conversational policy around
this adapter.
Compact MLX CPU conformance compares explicit branch advancement with the ordinary
shared scheduler across prefill, live audio, perception-only waiting, retained-audio
tool speech, reminders and drain. Output, component caches, both TTS guidance rows
and RNG agree at each committed boundary. The test includes partial PCM and a
retryable `NeedsInput` boundary. This component-level check does not establish
released-checkpoint numerical parity.

VoiceChat's host observer produces owned speech and transcription only after the
shared prepublication wrapper verifies exact completion. It checks source-frame
presence, coordinates and unpadded sample counts against the same schedule
receipt as the output channels. The scheduler commits matching cache, feedback
and RNG state only after observation succeeds; rejected PCM, attribution or host
conversion cannot publish a candidate. Cancellation retains submitted work until
completion and returns no delivered output. This observation boundary does not
authorize replay of already delivered speech or external tool execution.

Backbone composition accepts an unpublished Nemotron state and performs one
ordinary resident, bounded or collective invocation for both output heads.
Failure during either readout still leaves completion and branch disposal with
the enclosing transaction. Prepared pipeline ingress retains the normalized
readout directive and validates the declared token representation.

Schedule receipt observation reads the actual component frontiers. Declared
stateless Nemotron slots do not constrain the token position; all selected
stateful slots must agree. Retained speech state must agree with every TTS cache,
and codec frame/sample counts come from codec carry. Receipt validation does not
authorize publication, substitute for cache/RNG ownership, or aggregate remote
rank state. The enclosing session owns registration, scheduling and publication.

EAR waveform decoding returns neural and spectral carry together as an
unpublished candidate. Arbitrary nonempty code chunks use the same transition;
cloning its parameter-free state preserves both histories and their frontiers.
Finish consumes no new code frame, drains spectral overlap, releases carry and
rejects subsequent input. Finishing an empty stream yields no audio; repeated
finish yields no additional samples. The released decoder emits 1,764 samples
per code frame, with eight samples of startup delay and an eight-sample flush
tail. Offline and streaming boundary normalization differ in the pinned
implementation. A caller must use the chosen mode's declared timing and cannot
discard the streaming prefix or tail without accounting for it. Codec candidate
state does not own native completion, previously published audio or the complete
VoiceChat session commit boundary.

Shared-backbone self-speculation resolves exact verification within the selected
completion deadline. MLX admits quarantine-until-complete disposition before model
work, retains live native roots on timeout and fences further generation after
failure. Cancellation observed while waiting for exact sampling completion prevents
controller commitment and mode acknowledgement. Ordinary target decisions update
the same capture commitment coordinate in AR and self-speculation. Instrumented
causal context may retain every physical score row; sampling consumes only its final
conditional row. Verification capture covers the complete causal transaction, while
accepted decisions and snapshot boundaries retain independent committed positions.

Distributed switchable generation agrees controls before the next physical action.
A pending mode from one rank propagates to every participant; differing pending
commands reject before cache work. Active mode, committed position and remaining
allowance must match exactly. Cancellation from any rank discards prospective
changes. Command exchange uses the retained bounded session transport and consumes
monotone session credits, independent of model snapshots. A complete refinement
canvas must fit the partition's admitted sequence bound at load time.

Composite ordered pipeline placement can retain media-only stages with no decoder
cache. They participate in command agreement and output publication without
asserting a local text frontier. Vision/projector processing occurs for admitted
native image input; cached continuation uses its retained causal context and skips
inactive media producers. Complete invocation visibility and adapter scope travel
with the decoder boundary.

Composite pipeline stages with no decoder state participate in the same bounded
verification-lane and snapshot preparation agreements. Their lanes and control
slots contain explicit stateless realizations, and ownership exchange requires
both sides to have no mutable layout. Snapshot copies preserve prompt-input and
invocation metadata, while cache-shard identity remains absent. Restore and fork
retain the shared cumulative copy and transport charges on every rank.

World-participating boundary routes include idle stages in framing and exact
completion. Only the named consumer installs the received invocation boundary;
idle ranks discard its output. Preparation or source-completion failure is agreed
before the world transfer, so a rejected local tensor cannot strand peers in the
broadcast. These routes remain serialized across restored or forked continuations.

`ExecutionPlan::with_draft_adapter` retains a `DraftAdapterSource` at load time.
`SpeculationOptionsRequest::draft_adapter` selects its use for that invocation.
Adapter parameters execute only during shared-backbone proposal refinement;
causal context, causal verification and ordinary diffusion use the shared base.
A request enabling an adapter absent from the retained selection fails before
altering active execution. Snapshot mode options preserve the adapter selection.

Settled generation forecasts preserve active and pending decoding policies.
They include the complete diffusion canvas even for a one-token horizon,
committed block tokens awaiting causal append, and shared-target verification
overshoot. Current sampler, semantic, capture and snapshot retention remains
charged when projecting a mode handoff. Recomputing facade retention uses the
same mode-aware estimate and excludes completed loading and prompt execution.
Ordinary and common request-table forecasts with zero additional output still
include an admitted handoff's uncached committed suffix. Ordinary shared-backbone
handoffs include isolated append and rollback cache copies; a pending
diffusion block retains complete refinement geometry while reaching that boundary. A previously settled next-token
row adds no second cache position. Forecasting leaves the queued command and
all cumulative budgets unchanged.

Prepared requests retaining an external, embedded or shared-backbone draft
source accept an explicit initial AR mode. They use the same request table and
can enter speculation through a later admitted command. Their startup forecast
keeps the retained parameter owners and causal verification work, with zero
proposal width and no draft or optimistic workspace until speculation applies.

Shared-backbone requests in `GenerationRequestTable` admit initial diffusion as
well as AR and self-speculation. Initial diffusion keeps the context's causal
next-token row without committing it. Numerical advancements retain the full
canvas without publishing tokens; a finalization advancement publishes one constrained
selected position. Controlled step records expose the remaining tentative
canvas separately from committed IDs and speculative proposals. Position captures
carry `RefinementPosition`; causal prefill and verification captures carry
`CausalConditional`. Finalization uses current controller restrictions without
applying causal penalties, temperature or truncation to selected diffusion tokens.
A forced replacement discards its dependent tentative rows before the next
advancement refines that suffix.

Snapshots preserve the partially finalized canvas, its position rows and cursor,
the finalized suffix awaiting causal append, the active and pending modes, and
independent proposal and ordinary-refinement streams. A queued mode change drains
the active canvas and appends its finalized suffix once before acknowledgement.
A second command receives `ModeChangePending`. Cancellation is agreed before
publication and takes priority over a tentative terminal candidate. Snapshot,
copy, observation and transport reservations remain cumulative through restore
and fork. Refinement counters report full numerical canvas work even when EOS or
a shortened output budget ends commitment inside the block.

Switchable executions retaining causal target sampling accept its standard
sampling settings even when the initial mode is diffusion. Those settings remain
available for later AR or verification; diffusion finalization applies only its
position restrictions. A strategy requiring causal conditional scores still
rejects initial or pending diffusion before changing execution. Diffusion-only
families retain their position-score sampling admission rules.

A controlled request starting in diffusion establishes TTFT on its first
position commitment. Active prefill and refinement work contributes to that
metric; time spent paused between advancements and output callbacks does not.
Before a first commitment, snapshots retain the completed active work allowance.
Once established, TTFT remains available through restoration and branch exchange.

Controlled shared-backbone requests admit internal activation capture and tensor
interventions through the same scoped plan as causal and embedded drafting.
Schema version 3 reports each ordinary refinement forward as `Refinement` and
each masked draft forward as `MaskedDraft`, with its iteration index, complete
physical canvas and assumed-prefix provenance. These records describe tentative
computation; committed token IDs remain separate. A constraint correction causes
new refinement invocations for the dependent suffix. Reusing saved position rows
performs no new forward and emits no duplicate activation record. Snapshot and
fork restore observer authority without refunding capture allowances or rewinding
invocation identities.

Batched verification and incremental shared-backbone verification use the same
neutral `VerificationCursor`. It freezes one target choice, reports the exact
accepted input count, and advances after successful commitment. Pending and
finished cursors reject further target decisions with typed generation errors.
Sampling and controller state stay with their existing owners; an explicitly
advanced native cache failure leaves the chosen target frozen. Cursor snapshots
copy writable progress while native verification evidence remains immutable.

Image capture uses physical invocation media geometry, independently of text
sequence and context extents. Shared-backbone prefill and ordinary controlled
prefill obtain packed raw and projected counts from prepared media admission.
Pixtral internal hooks execute during image context preparation; later mode
changes reuse that causal context and do not rerun its tower. Original and
effective values occupy separate capture paths, and edits affect the value
consumed by the next operation. Capture schema version 3 includes these extents
in invocation records and paid snapshots retain the existing authority ledger.

The common request table advances shared refinement one numerical invocation at
a time. A settled iteration produces no committed token; its complete canvas,
selected rows, pinning, schedule counters and independent RNG remain in the
request seed. Snapshots can pause at this boundary. Finalization takes a separate
advancement and preserves selected identities. Mode commands remain pending
through the numerical schedule and left-to-right commitment of the whole block.
Paused numerical storage contributes to snapshot retention and copy allowances.

Independent causal assistants retain ordinary target refinement separately from
their proposal source. `TargetRefinement` supplies the target's declared policy,
request-owned numerical continuation and snapshot estimates. Causal proposals
retain likelihood-ratio verification in every mode. Diffusion refines only the
target, then its causal append advances both target and assistant frontiers before
a mode acknowledgement. Its numerical randomness uses the request's independent
refinement substream; snapshots preserve it with both causal prefix seeds.

Entropy and masked refinement share the common request-table lifecycle while
retaining distinct finalization semantics. Entropy feedback selects the highest
allowed position score and reports `ConstrainedFinalization`; masked refinement
preserves admissible selected identities and reports position scores. Only a
changed masked selection invalidates its dependent suffix. Single-model requests
retain the complete numerical continuation with their immutable causal-context
prefix and independent random stream. Declared refinement-only executions reject
AR and speculative modes before changing request state.
