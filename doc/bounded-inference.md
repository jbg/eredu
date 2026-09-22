# Bounded inference

Eredu admits framework-managed inference allocations against the limits of their
physical memory domains before preparing or submitting the affected work. One execution engine
serves ordinary and controlled sessions; speculation consumes the same preparation
and commitment mechanisms with additional proposal/verification state. The
[architecture guide](backend-architecture.md) defines crate ownership and
[execution control](execution-control.md) defines continuation semantics.

<a id="current-integration-status"></a>

## Current integration status

Ordinary prepared chat supports bounded semantic tools with an ordinary backend.
The portable facade and neutral backend-conformance suites pass; native coverage
includes seven public tools/media cases and six distributed CLI residency cases.
Required and Auto tools share ordinary, manual and recorded execution. TP2, PP2
and combined TP2/PP2 media use an 8 GiB capacity; distributed CLI cases use 64 GiB.

The [design guide](bounded-inference-design.md) describes mechanism ownership.
[Validation](bounded-inference-validation.md) identifies tested profiles,
commands, results and platform limits. The released text/image sensor request
passes Required and Auto under a 64 GiB ceiling; this establishes the stated
request and hardware configuration, not arbitrary application memory policy.

## Public request and source identity

The public sequence is to compile retained tokenizer/template sources, prepare a
chat with `MemoryLimitDeclarations`, then start `PreparedChatRequest` through
`start_prepared_chat`. `PreparedChatSession::advance` and `run` share the same
cursor. `start_controlled_chat` adds a bounded record journal to that session.
See [prepared chat](prepared-chat.md) and
[the ordinary integration example](../eredu/examples/prepared_chat_generate.rs).

The prepared chat owns its render, output/controller policy, original compilation
receipt and domain limits. Startup validates tokenizer/template/model/execution identity
and resolved settings. Equal token IDs, matching hashes, copied diagnostic values
or a caller-supplied range do not substitute for authenticated ownership.

Rendered text, exact canonical IDs and authenticated media are input variants of
the same request. Exact IDs are validated without a text round trip. Media comes
from `prepare_chat_input`, which composes architecture framing and semantic
coordinates with the original host/native input producer. Foreign, stale or
mismatched sources refuse before execution.

Low-level native upload constructors authenticate uploaded storage and its
custody. They do not create an inference preparation or a native execution
permit. Execution consumes the original prepared input producer with its finite
output envelope; raw uploaded arrays lacking that proof are rejected under both
finite and unlimited limits.

Each application limit constrains the total live charge in its physical domain,
including existing storage and concurrent work. Unspecified domains default to
`MemoryLimit::Unlimited`; `Finite(u64::MAX)` remains finite. Unlimited admission
uses the same ownership, completeness, bounded-output, reservation and completion
contracts, and every arithmetic operation remains checked. Source
compilation reserves its own concrete destinations before expensive tokenizer,
template, grammar/schema and semantic preparation. Runtime quotes the selected
execution separately and atomically reserves its remaining per-domain demand.
The public error boundary stays backend-neutral and preserves typed causes.

The CLI accepts one `--memory-limit <domain>=<bytes|unlimited>` entry per
physical domain. `--verbose` reports coherent per-domain configured and effective
limits, current charges, historical peaks, and accounting categories before loading
and after generation. Placement allowances include their conservative basis and
are distinct from measured residency. Diagnostic report containers belong to the
caller outside admitted execution storage; observing the ledger grants no
allocation authority or process-memory ceiling. The `host` domain includes physically
unified accelerators; separate MLX accelerators have names such as `mlx-gpu-0`.
For a topology reporting both names:

```sh
eredu --model /path/to/model --verbose \
  --memory-limit host=8589934592 \
  --memory-limit mlx-gpu-0=17179869184 \
  "Explain physical memory domains."
```

On a unified host/accelerator topology, configure only `host`; a device name
does not create another physical domain. Omitted domains are unlimited, and an
explicit entry such as `--memory-limit host=unlimited` has the same capacity
semantics. Unknown names and repeated declarations are rejected. Limits apply
to total live domain charges across sessions sharing the process ledger.

## What the bound covers

| Accounting scope | Admission and lifetime |
| --- | --- |
| Host preparation | Source reads/compilation, tokenizer/template outputs, grammar/controller/parser state, masks and concrete report/record destinations are paid before construction. |
| Execution | Actual selected equations, backing, intermediates, native graph/descriptors, sampling, materialization, transfers and retained outputs are quoted before submission. |
| Observation | Capture transformation, transport, host result and record storage are admitted with their actual schedules and geometry. |
| Continuation | Saved native/controller/semantic state and journals require complete copy estimates and independently funded destinations. |
| Shared storage | Exact source identities count a physical allocation once while all aliases retain its owner. |

The process ledger uses backend facts to map host and accelerator locations to
physical domains. Unified locations consume one domain's capacity; separate host
and accelerator memory have independent limits. Independently allocated buffers
remain distinct charges in either topology. Domain limits, allocation publication
and ceiling succession update all participating domains in one transaction.

CUDA managed allocations retain their full capacity as a conservative allowance
in each distinct candidate domain. These allowances state a placement estimate;
their sum is not measured residency. Transfers retain the source charge while
reserving staging and destination charges for their overlapping lifetimes.

Persistent model/cache storage, transient execution workspace and source metadata
have different lifetimes. A scalar total is not enough to claim credit for an
existing allocation: the retained source must authenticate its identity and
capacity. A narrow view keeps its backing allocation's full capacity charged.
Replacement state includes old/new overlap until completed retirement.

Framework-managed bounds are distinct from process RSS, physical footprint and
system memory. Registered MLX backing stays charged through allocator cache
retention and retires on physical eviction. Application buffers and copies, unrelated tasks,
and opaque platform/compiler/driver internals need separate policy. Native memory
telemetry can validate a mechanism but cannot create an enforceable upper bound.
Additional headroom identifies its physical domain and cannot turn an unknown required contribution into a
known one.

`MemoryOverheadPolicy` remains a separate neutral evaluator. Inference admission
uses workspace completeness, authenticated storage and execution evidence; an
unlimited limit does not replace any of those requirements.
Low-level text calls without the required admitted request return a typed
completeness error before native input construction, observer callbacks or acquiring
operation ownership. Controlled and uninterrupted generation enter through the
same admitted text driver.

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

The admission policy checks borrowed geometry and completeness facts. The ledger
compares domain requirements with the complete live charge under one lock.
Owning diagnostics are constructed through the same funded report destination
as quotation, including the retained state report for incremental admission.
The planner returns the reservation and quote; admission diagnostics are borrowed
from the reservation and cannot escape as an independently funded copy.
Escaping diagnostic errors keep that producer account; fixed typed refusals need
no unrelated payload owner. Report and residual-admission tests cover refusal
before reservation, final-owner retirement and smaller-candidate classification;
commands and scope are
in [public conformance](prepared-chat-conformance.md).

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
tensor storage fits. The released sensor tool request succeeds under the
64 GiB limit. Combined TP2/PP2 repeated requests also pass at their
8 GiB per-rank capacity. Completed-session cleanup retires inactive installed control
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
producers. Parallel partition discovery constructs its exact source under
funding, including fresh Unix artifact identity. Its source/refusal tests and
native execution limits are recorded in [partition-source evidence](bounded-source-contracts.md).

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

[Released Qwen measurements](bounded-qwen-measurements.md) cover readout and
chunk selection with pinned reference versions and full-score tolerances.
[Prepared-chat validation](prepared-chat-validation.md) covers ordinary tool
requests and authenticated image ingress. Numerical inference, bounded tool
admission and whole-process memory are separate claims.

[Tokenizer measurements](bounded-text-processing.md#storage-and-search-cost)
record workload-specific throughput and storage tradeoffs. Native functional
runs establish behavior and custody; their wall times are not isolated benchmarks.

## Verification and limits

Portable tests use neutral backends. Metadata builds establish type/feature
compatibility; actual device tests establish numerical and lifetime behavior.
Released checkpoints stay outside the tracked tree with pinned provenance.

The [validation guide](bounded-inference-validation.md) records native profiles,
stack requirements, tested hardware and unqualified configurations. Fresh funded
file opening outside Unix and downstream memory/cache policy require separate
validation. Unknown native bounds and unsupported prediction-extension producers
return typed refusals; they are not inherent model-family limitations.
