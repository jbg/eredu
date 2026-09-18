# Bounded inference

Eredu admits framework-managed inference allocations against an explicit shared
capacity before preparing or submitting the affected work. One execution engine
serves ordinary and controlled sessions; speculation consumes the same preparation
and commitment mechanisms with additional proposal/verification state. The
[architecture guide](backend-architecture.md) defines crate ownership and
[execution control](execution-control.md) defines continuation semantics.

<a id="current-integration-status"></a>

## Current integration status

The breaking API and implementation consolidation is in progress. The ordinary
public prepared-chat path supports bounded semantic tools with an ordinary backend
and no speculative capability. The final portable facade run passes 38 tests,
with zero failures and two external-checkpoint cases ignored. All 126 neutral
backend-conformance tests also pass. Native validation is scoped by execution
path: all seven public native cases pass, including ordinary tools and child
sampling, combined image/audio tools, single-device image snapshots/speculation,
and TP2, PP2 and combined TP2/PP2 media. The earlier
combined-topology second-request capacity failure is closed. The distributed
CLI and focused native fixture results are tracked separately in
[the current overview](bounded-followup.md); this public result does not certify
every residency or application combination.

Current work, concrete removals and remaining purposeful mechanisms are listed in
[the consolidation overview](bounded-followup.md). In particular, the released
ordinary sensor-recording request now completes under Required and Auto at the
enforced 64 GiB capacity after reducing the actual collector destinations.
[Released tool validation](bounded-followup-released-tools.md) records this
success and the repaired grammar funding refusal on the earlier generic action
request. That request now completes its 48-token allowance but truncates without
a tool call. Authenticated single-device media branches and independent
speculative parity now pass, as do the exercised distributed media paths. These
results do not establish external application memory policy or unrun native
configurations.

Historical native evidence covers selected resident/Host/Disk, TP/PP/EP,
saved-state, capture, realtime and model-family combinations. Those results retain
their exact source and fixture scope. They do not establish the current whole
matrix after consolidation. The [evidence index](bounded-inference-evidence.md)
preserves 205 implementation records and the
[public conformance record](bounded-followup-public-conformance.md) identifies
current portable results separately.

## Public request and source identity

The public sequence is to compile retained tokenizer/template sources, prepare a
chat at an explicit capacity, then start `PreparedChatRequest` through
`start_prepared_chat`. `PreparedChatSession::advance` and `run` share the same
cursor. `start_controlled_chat` adds a bounded record journal to that session.
See [prepared chat](prepared-chat.md) and
[the ordinary integration example](../eredu/examples/prepared_chat_generate.rs).

The prepared chat owns its render, output/controller policy, original compilation
receipt and capacity. Startup validates tokenizer/template/model/execution identity
and resolved settings. Equal token IDs, matching hashes, copied diagnostic values
or a caller-supplied range do not substitute for authenticated ownership.

Rendered text, exact canonical IDs and authenticated media are input variants of
the same request. Exact IDs are validated without a text round trip. Media comes
from `prepare_chat_input`, which composes architecture framing and semantic
coordinates with the original host/native input producer. Foreign, stale or
mismatched sources refuse before execution.

The capacity supplied by the application is a limit, not an estimate. Source
compilation reserves its own concrete destinations before expensive tokenizer,
template, grammar/schema and semantic preparation. Runtime quotes the selected
execution separately and atomically reserves its remaining shared-domain demand.
The public error boundary stays backend-neutral and preserves typed causes.

## What the bound covers

| Domain | Admission and lifetime |
| --- | --- |
| Host preparation | Source reads/compilation, tokenizer/template outputs, grammar/controller/parser state, masks and concrete report/record destinations are paid before construction. |
| Execution | Actual selected equations, backing, intermediates, native graph/descriptors, sampling, materialization, transfers and retained outputs are quoted before submission. |
| Observation | Capture transformation, transport, host result and record storage are admitted with their actual schedules and geometry. |
| Continuation | Saved native/controller/semantic state and journals require complete copy estimates and independently funded destinations. |
| Shared storage | Exact source identities count a physical allocation once while all aliases retain its owner. |

Persistent model/cache storage, transient execution workspace and source metadata
have different lifetimes. A scalar total is not enough to claim credit for an
existing allocation: the retained source must authenticate its identity and
capacity. A narrow view keeps its backing allocation's full capacity charged.
Replacement state includes old/new overlap until completed retirement.

Framework-managed bounds are distinct from process RSS, physical footprint and
system memory. Application buffers and copies, allocator caches, unrelated tasks,
and opaque platform/compiler/driver internals need separate policy. Native memory
telemetry can validate a mechanism but cannot create an enforceable upper bound.
A safety reserve likewise cannot turn an unknown required contribution into a
known one.

## Selected readout and prefill

`OutputDemand` distinguishes StateOnly, final-position and full-sequence consumers.
The shared readout worker selects required hidden rows before vocabulary projection.
StateOnly skips that equation and retains the dependencies needed for completion.
Pipeline boundaries preserve full hidden geometry; the output owner applies the
final selection. Observers and prediction consumers retain their declared demand.

`PrefillDriver` selects intermediate chunk demand, semantic boundaries, completion,
retirement and cancellation. Intermediate ordinary chunks normally update state
without producing scores. Every selected span must complete before retirement;
absence of scores does not eliminate state, token-validation or observation work.
Media encoders run at their semantic ingress and are shared across decoder chunks.
Uneven final chunks retain exact coordinates rather than padding the logical prompt.

Candidate selection does not assume workspace decreases monotonically with chunk
size. It prices the requested chunk and permitted smaller candidates through the
same worker. Explicit incomplete-workspace or capacity rejections may select a
smaller candidate; identity, native and unrelated failures remain terminal.
Each accepted candidate owns its reservation against concurrent work.

Cancellation is sampled at the actual settled boundary and agreed across required
participants. Cancelling does not prove native completion or refund already spent
attempts. A backend without the necessary completion contract cannot enter bounded
chunk execution.

## Quotation and admission

Architecture equations run on neutral metadata tensors using the selected
mechanism's exact allocation facts. Cold selection/inspection do not create a
native device, stream, tensor, group or completion. Quotation follows the selected
parameter formats, auxiliary tensors, state geometry, physical residency and
readout demand; a model-family name alone cannot determine workspace.

`RuntimeStateEstimate` separates persistent-state completeness from execution
workspace. `ExecutionWorkspaceEstimate` binds activations, attention, vocabulary,
state update, materialization and retained domains to exact batch, cached/input/
output positions, chunk size and output demand. Every required contribution has
a finite bound with assumptions or an explicit unknown reason.

Tracing accounts for unique new backing and scratch through completed spans,
including displaced opening state. Aliases share allocation identity; metadata
handles and native descriptors remain separately priced. All output positions in
the allowance must be covered, including nonmonotone decode growth. Independent
span peaks cannot be added or discounted without their actual lifetime relation.

The admission policy compares borrowed scalar requirements. Owning diagnostics
are then constructed through the same funded report destination as quotation.
Incremental admission no longer uses an uncharged clone of its full state report.
The planner returns the reservation and quote; admission diagnostics are borrowed
from the reservation and cannot escape as an independently funded copy.
Escaping diagnostic errors keep that producer account; fixed scalar refusals need
no unrelated payload owner. The focused report tests and the 244-case
residual-admission checkpoint cover refusal before reservation, final-owner
retirement and unchanged smaller-candidate classification; commands and scope are
in [public conformance](bounded-followup-public-conformance.md).

A complete incremental quote may credit exact already registered decoder roots
and fixed shared controller sources. It does not authorize subtracting an old
byte total from an unrelated peak. Runtime independently validates pool, execution,
geometry, retained sources and any authorized capacity handoff before the atomic
reservation. A successful report alone grants no submission authority.

## Native realization and residency

Core issues move-only submission authority after exact session comparison. Native
work retains that lease alongside its allocations and completions. It releases
authority only after successful completion, safe terminal settlement or teardown.
A failed poll, dropped iterator or missing event is not completion evidence.

Resident, host-layerwise and disk-streamed execution keep their actual mechanisms.
Layer acquisition uses shared preparation, retirement, transfer, lease and
population policy with finite attempts. Foreground I/O, background prefetch,
different storage formats and native CPU/Metal kernels retain their independent
completion and allocation facts. Enforcement cannot trigger an ordinary retry
outside the admitted allowance.

Tensor/pipeline/expert parallelism consumes architecture-owned placement and
semantic boundaries. Rank-local native facts do not permit backend family branches
or reconstructing checkpoint policy. Admission, cancellation and failures require
the declared participant agreement; one local failure must not leave peers
submitting a refused collective.

Unknown native worker storage remains a typed refusal. Source-derived graph,
collector, pipeline and host-control costs can dominate small requests even when
tensor storage fits. The released sensor tool request now succeeds under the
unchanged 64 GiB limit. Combined TP2/PP2 repeated requests also pass at their
unchanged 8 GiB per-rank capacity after removing only inactive installed control
aliases. Escaped owners retain their charges; neither result uses a larger
allowance or an unenforced fallback.

## Semantic output, observation and saved state

Ordinary and speculative consumers share prepared text/reasoning/tool policy,
stop sources, grammar/forbidden/automatic controllers, semantic decoding and
publication. Events describe only committed output. Speculative proposals remain
separate until verification; ordinary tool requests never need a dummy drafter.

Capture frames precede associated semantic events and may exist for a failed step
with no token. Pending, failed and escaped results retain their actual owners.
Raw capture/intervention plans are compiled after prompt geometry is known.
Their source compilation and later native transforms are separate admitted
producers. Parallel partition discovery now constructs its exact source under
funding, including fresh Unix artifact identity. Its source/refusal tests and
native execution limits are recorded in [partition-source evidence](bounded-followup-partition-source.md).

Snapshots preserve native state, pending input, RNG/adaptive history, constraints,
semantic prefixes, cursor, timing and lifecycle. Restore/fork use the same copy
and placement contracts. Cumulative observation, transport, copy and attempt
spending is not refunded. Terminal snapshots admit empty future execution and
publish no duplicate token or terminal event. Unknown copy bounds refuse before
mutation. Details and exceptions are in [execution control](execution-control.md).

Realtime managed requests accept encoded token frames. Standalone PCM codec calls
remain application composition outside that account. Releasing or restoring a
frame session cannot promise to rewind an external codec. Existing realtime
pipeline/combined-topology and snapshot/fork gaps are implementation limits and
remain explicitly documented.

## Numerical and performance evidence

The [released Qwen readout/chunk measurements](bounded-qwen-measurements.md)
preserve independent full-score comparisons, pinned dependencies, original
commands and tolerances. At 3,000 prompt positions, ordinary 128-position chunking
recorded an MLX peak of 4,813,825,208 bytes versus 22,433,750,408 for the full chunk.
A later allocator-corrected managed run recorded 4,811,351,232 bytes. That result
belongs to executable SHA-256
`d0dcefcf2ff567ac0e78c9c39014382e7345bdb1596149f21ac252564dd4e5b2` in
[the public Qwen record](validation/bounded-public-qwen-2026-09-16.json).
These measurements include model residency and are not process-memory bounds.

The separate graph-construction index optimization preserved best-fit allocation
semantics while reducing its recorded 3,000-token run from 65.796 to 4.705 seconds
with identical token IDs. Its distinct executable and measurements are recorded
under `graph_construction_class_index` in the same evidence file. Its timing must
not be combined with the earlier executable's peak or treated as current tool
request performance.

Current tokenizer parity/throughput and ordinary tool behavior are documented in
[public conformance](bounded-followup-public-conformance.md) and
[tokenizer consolidation](bounded-followup-tokenizer.md). Current native mechanism,
released numerical and tool-admission results have separate records; successful
ordinary numerical inference does not establish bounded tool admission.

## Verification and remaining work

Focused verification follows each changed producer or shared lifecycle. The
portable facade uses neutral backends. Native metadata builds establish type and
feature compatibility; actual device tests establish numerical/lifetime behavior.
Released checkpoints stay outside the tracked tree with pinned provenance.

The exercised native image/tool, independent-media and branch/terminal cases
have recorded passing results. Released text and authenticated image requests
pass both Required and Auto policies with freshly verified pinned artifacts.
These results retain their exact fixture and hardware scope. The current
distributed CLI and focused native follow-up verdicts remain in the overview;
fresh funded file opening outside Unix remains an explicit qualification gap.
See [the work overview](bounded-followup.md) for current status and
[the evidence index](bounded-inference-evidence.md) for prior results.

Historical implementation notes remain in repository history at
`77e08045401912c502e6ab1600e8b622c3563478`. Their intermediate startup/record names and
old pending statements are superseded by the current contracts here. Reproducible
commands and measurements remain in the linked evidence rather than being
repeated as claims of current completeness.
