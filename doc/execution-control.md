# Completed-token execution control

Controlled and uninterrupted generation share preparation, sampling, commitment,
semantic decoding, termination, metrics and observation. Control exposes explicit
advancement and ownership of a settled continuation. It does not select a second
inference engine. The [architecture guide](backend-architecture.md) describes crate
ownership; [bounded inference](bounded-inference.md) defines the admission scope.

## Public prepared-chat workflow

Compile a chat against the loaded model's retained tokenizer and template sources
with `prepare_chat`. Pass that `PreparedChat` and its admitted capacity through
`PreparedChatRequest::new(&chat, settings)` to `start_prepared_chat`. This returns
an optional `PreparedChatSession`: cancellation before startup returns no session.
The session's consuming `advance` and `run` operations use the same committed-token
cursor and receive the same semantic event consumer.

`PreparedChatPrompt` selects the rendered prompt, exact canonical token IDs, or an
authenticated media input. Exact IDs are checked against the tokenizer domain
without decoding and re-encoding. `prepare_chat_input` authenticates host media
against the exact render and selected execution, then supplies the existing
`OriginalModelInput` producer. Media and text use the same request/controller
assembly, prefill driver, decoder and event publisher.

`start_controlled_chat(request, trace_limits, control, emit)` adds bounded control
records to that session. The `GenerationControlHandle` carries cancellation and
control requests; record limits must be positive. Startup emits its prepared
identity before prediction. A backend without the required control capability
returns a typed rejection. The runnable
[controlled example](../eredu/examples/controlled_generate.rs) uses this public
surface. Ordinary prepared-chat tools require no speculative backend or drafter.

The request accepts admitted instrumentation in `options`, or borrowed raw
`capture` and `intervention` declarations. Raw declarations are compiled after
actual prompt geometry is known and cannot overlap already admitted declarations
of the same kind. Source, tokenizer, template, execution and logical session
identity are validated before affected work.

## Completed boundaries and commands

Prediction zero is prompt prefill; later predictions consume cached state. A
committed output token waiting to become the next decode input remains explicit
pending input. Pausing, copying or restoring must not insert it twice.

| State | Meaning |
| --- | --- |
| Prepared | Admitted initial state, before prediction. |
| Running | Prediction, completion, semantic commitment or associated delivery is unfinished. |
| Paused | The current boundary is completed and drained. |
| Completed | Normal terminal outcome; later advancement publishes no duplicate events. |
| Cancelled | Cancellation settled at a safe boundary; it cannot resume generation. |
| Failed | Execution or delivery failed; reuse requires the appropriate teardown/reset path. |

Pause requests take effect at completed boundaries. A pause does not flush a
partial tool name, argument, stop prefix or reasoning delimiter. Explicit
advancement resumes the same parser and native state. Cancelling is fallible:
parser, funding or backend failure is reported rather than converted into a
successful terminal record.

`force_next_token` checks canonical vocabulary membership and the active
controller's permitted choices. Validation consumes neither a prediction nor an
RNG draw. The next ordinary sampling/commitment step applies the pending choice
once. Clearing it leaves committed history unchanged. Grammar and forbidden-tool
constraints remain active under forcing.

Sampling overrides use the shared sampling-control contract. Supported changes,
including explicit reseeding, are validated before installation. Saved and resumed
configuration comes from the sampler actually installed by the backend; requested
settings cannot substitute for its RNG, adaptive history or penalty state.
Backend capability reports distinguish unsupported changes before mutation.

## Managed prepared-chat controllers and tool completion

Text, reasoning, forbidden-tool, active-tool and automatic-tool policies share
one prepared semantic assembly. Controller prefixes, token masks, grammar state,
stop sources, decoder state and event destinations retain their original paid
sources. Automatic activation transfers into the same grammar/controller worker
used by an explicitly active tool.

Semantic output includes visible text, reasoning boundaries/content where the
selected protocol declares them, incremental tool-call information, complete tool
identity and arguments, and terminal attribution. Only committed tokens reach
that parser. Arguments split across tokens retain parser state and publish once;
callers do not need to interpret model delimiters.

Capture records carry an explicit model-transaction outcome. Low-level capture
may report `Untracked`; an omitted serialized outcome is rejected. This contract
is shared by owned and custody-retaining records.

Completion validation uses the selected schema draft and original compiled
schema. Current public cases cover default modern schemas and explicit Draft 7,
including partial arguments and cancellation. Invalid arguments or unknown
required storage return their typed failure. Neither a permissive schema nor a
text-only controller replaces a refused tool policy. Arbitrary callbacks still
need an explicit bounded contract.

`PreparedChatOutputMode::Text` is a permitted literal-output policy over the same
cursor. It does not imply that semantic tool declarations can be ignored. A
retained template with no recognized semantic protocol reports that capability
rather than silently flattening tool output.

## Snapshot, restore and fork

`PreparedChatSession::snapshot` composes native state, controller, semantic parser,
cursor and lifecycle in the existing nonrefundable copy transaction.
`restore_prepared_chat` and `fork_prepared_chat` admit a fresh continuation from the
saved source. Active-session `restore_snapshot`, `fork_snapshot` and branch
`exchange` use the same state-placement contracts.

The recorded wrapper requires explicit snapshot limits and a complete estimate
before advertising snapshot, restore or fork support. Its journal joins the same
host-copy transaction. An unknown native or host copy bound refuses before the
destination producer; enabling stepping alone does not imply snapshot support.

A complete saved boundary includes the following state:

| Owner | Saved state |
| --- | --- |
| Native execution | Applicable KV/recurrent/other mutable state, pending input, selected configuration and exact source lineage. |
| Sampling/controller | RNG, adaptive and penalty history, grammar/forbidden/automatic state and pending forced choice. |
| Semantic cursor | Committed IDs, decoder/stop/parser prefixes, tool state, attribution and terminal status. |
| Lifecycle/journal | Prediction coordinate, record context, timing and reconciliation state. |

Snapshotting is allowed at Prepared, Paused and normally Completed boundaries.
Running, Failed and Cancelled boundaries are not copy sources. State placement
may exchange a separately valid branch into a settled cancelled owner, but does
not revive the cancelled branch. Native quiescence is checked independently of
these portable lifecycle rules.

Copies preserve source identity and immutable paid aliases. Mutable destinations
receive their own funding. Snapshot metadata that escapes independently keeps its
host destination and snapshot reservation alive. A failed copy leaves the live
source unchanged, while work already attempted remains spent.

Restore and fork never refund observation, transport, copy or attempt consumption.
Record sequence/restore epoch provide consumer reconciliation; restoring model
state does not erase externally delivered history. `PreparedChatResumeSettings`
may shorten the remaining output allowance and admit a replacement execution
policy within the same managed domain capacity. It cannot extend saved output
credit or replace saved sampling history.

Terminal snapshots remain terminal after admission. Their future execution geometry
is empty StateOnly work with no output allowance. Restoring one does not schedule
a phantom prediction, reinsert the last token or publish another terminal event.
Read-only `sampling_state()` remains available at a healthy completed boundary;
sampling mutation still rejects completion. `preparation_report()` borrows the
retained admission report, including a restored terminal branch's empty geometry.
It describes that admission rather than current process memory or available pool
capacity, and querying it creates no new authority.
Numerical equality of unchanged branches additionally requires deterministic native
execution; copying RNG alone cannot make nondeterministic kernels deterministic.

## Capture, interventions and records

Completed capture frames are delivered before the associated token's semantic
events. A failed or cancelled prediction may deliver an attributed frame without
a committed token. Delivery is fallible and exactly once; pending frames retain
native and host custody until the delivery transaction settles.

A shared frame, semantic string, record or error keeps the payer for the storage it
actually owns. A fixed error with no owned payload does not need an unrelated
allocation owner. Dropping the session cannot release funding still held by
escaped data. Borrowed progress views do not fabricate owned snapshots.

Host-summary capture uses the admitted row bank and actual evaluated source values.
Dense and uniform sources enter the same selection/reduction worker, including
nonfinite values; result/error handoff consumes its slot once. Summary support is
not established by a fixture returning a preconstructed row or counter.

Raw intervention validation, canonical digesting and paid projection share the
portable producer. Backend transformations consume the admitted plan. Dynamic
callbacks require their own declared storage and attribution contract. Parallel
raw capture and intervention declarations construct exact partition/support
sources through the shared funded worker, including fresh Unix artifact identity.
[Partition-source evidence](bounded-source-contracts.md) separates these
producer guarantees from native distributed execution validation.

## Controlled speculative generation

`PreparedChatSpeculativeRequest` consumes the same prepared chat and rendered,
exact-ID or authenticated-media input variants. `generate_prepared_chat_speculative`
and `with_controlled_prepared_chat_speculative` share lane preparation and the
runtime scheduler. Batch preparation agrees on all lane policies before prompt
work. Speculation extends the common semantic preparation with actual proposal
and verification state; it is not required for ordinary tools.

A speculative control step advances a scheduler action. Proposals are tentative
and remain separate from committed output. Verification commits the accepted
prefix plus any replacement/bonus at the declared boundary. The shared parser,
controller and publisher see that committed sequence once. Rollback cannot refund
observation or copy spending, and failure retains partially completed native work.

Target and independent drafter have distinct authenticated prefill roles and span
origins. Media must retain its real prepared payload through encoder ingress and
subsequent decoder chunks. A semantic token projection describes coordinates; it
cannot replace image/audio content. Captured composite quotation includes the
original input roots and traces the shared full forward. Target and independent
drafter compile distinct semantic bindings from the same authenticated host media
and their actual retained caches. The current shared-input source requires the
same selected stream and a single topology. Full/minimal metadata and 27 neutral
source/account tests pass. The actual single-device image/tool fixture also passes
ordinary/manual and independent speculative parity for Required and Auto.

## Distributed execution and failures

Tensor/pipeline parallel execution uses the same boundary-aware cancellation,
admission agreement and completion driver. All participating ranks agree before
publishing a transition or submitting refused work. A locally failed rank cannot
return while peers continue a collective that needs it.

Native completion remains independent of host status. A polling error, cancellation
request or callback failure cannot authorize resource release. Unresolved native
work keeps its lease and resources until positive completion, terminal settlement
or safe teardown. Where the backend lacks an abort primitive, failed resources
may remain quarantined. See [backend architecture](backend-architecture.md) for
this lifetime boundary.

Topology-specific restrictions describe the selected implementation. In particular,
an independent rank-local request cannot claim the coordination of a non-replicated
world. Missing pipeline or combined-topology behavior is an implementation gap,
not an inherent limitation of the model family.

## Realtime frame boundary

Realtime control operates on `RealtimeInputFrame` encoded audio tokens. Managed
prepare/submit and ordinary scheduling share frame state, distributed agreement
and publication. Source preparation precedes submission in that same transaction;
partial rank admission cannot leave another rank dispatching a refused frame.

Standalone Mimi PCM encode/decode calls are application composition outside this
frame request. Cancelling, releasing or restoring a token-frame session does not
rewind external codec convolution tails, overlap buffers or transformer caches.
`personaplex_full_path_bench` identifies its external codec mode and does not prove
a managed bound for the complete PCM-to-PCM application.

Realtime pipeline/combined TP/PP and snapshot/fork were already absent in the
baseline implementation. Their typed restrictions remain current implementation
limits. Release/resume of an existing frame session is a distinct operation and
does not imply state-copy support. Ordinary inference media encoders selected
inside a request remain covered by that request's admission and completion rules.

## Timing and validation

Time to first token measures active work through first commitment; user pauses
and event-consumer time are excluded. Required capture transformations remain
execution work. Snapshots preserve accumulated timing rather than restarting
it, and external wall time still includes pauses and application processing.

[Public conformance](prepared-chat-conformance.md) covers ordinary tools without
speculation, reasoning, partial arguments, provider errors, manual/run parity,
exact speculative prefixes and source refusals. The portable facade has 38
passing cases with two external fixtures ignored; backend conformance has 126.

Seven public native tools/media cases and six paged Device/Host/Disk cases cover
snapshots, restore, fork, exchange, child sampling, preserved parent state and
Host-backed Device-view retirement. TP2/PP2 media uses an 8 GiB capacity.
Six Dense/Mova distributed CLI cases cover Resident/Host/Disk with both samplers,
unfinished branch exchange, Hybrid reset and cumulative capture refusal under a
64 GiB ceiling and 120-second deadline. Native tests use an explicit 64 MiB Rust
test-thread stack and the build profiles in [validation](bounded-inference-validation.md).
Default 2 MiB debug-stack execution is not established.

[Released validation](prepared-chat-validation.md) separately covers the pinned
text/image sensor request with Required and Auto. These scoped results do not
establish arbitrary prompt behavior or downstream application memory policy.

New inference functionality must use these shared controlled drivers by default.
Any exception needs a concrete state, completion, attribution or accounting reason,
a typed capability/rejection and a documented limit here. Related consumers are
described in [component analysis](component-analysis.md) and
[the inspector control guide](lm-inspector-execution-control.md).
