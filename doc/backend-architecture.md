# Language-model backend architecture

Eredu separates portable model semantics and execution policy from native
realization. [Repository rules](../AGENTS.md) define dependency ownership and
feature boundaries. This document describes the current contracts; validation
status is maintained separately in [bounded inference](bounded-inference.md) and
[the design overview](bounded-inference-design.md).

## Library distribution

Eredu libraries must support distribution through crates.io with ordinary
versioned Rust dependencies on unmodified upstream releases. Dependency patching,
modified vendored dependencies and private forks are prohibited by the
[distribution rules](../AGENTS.md#library-distribution-and-dependencies).
Native MLX patches are the explicit exception: Eredu controls their application
through the `safemlx-sys` native build, including when building the published crate.
Downstream consumers must not need an Eredu checkout or dependency overrides.
The local Rust dependency forks and root patch table do not satisfy this requirement.

## Dependency direction

Production dependencies point from applications toward these roots:

```text
foundations:          eredu-collections / eredu-gguf / eredu-nn-macros
storage and text:     eredu-checkpoint / eredu-text
portable contracts:  eredu-core / eredu-nn
portable mechanisms: eredu-media / eredu-runtime
portable families:   eredu-codec / eredu-architectures
native realization:  eredu-backend-*
facade:               eredu
applications:         eredu-cli / examples / downstream applications
```

A shared row does not authorize a dependency between its members. Portable crates
never depend on a concrete backend, the facade, `safemlx`, or another accelerator
runtime. `eredu-evaluation` owns fixtures, metrics and comparison policy; native
examples/tests may use it as a development dependency, production mechanisms may
not. `eredu` without default features remains a portable facade. Backend adapters,
accelerator options and host image/audio dependencies stay feature-gated; model
families and codec families remain available without a backend.

`eredu-collections` is a dependency-free, `no_std` foundation for the shared safe
ordered-map worker. Its prospective insertion callback reports the exact node
layout before allocation and preserves the caller's typed refusal. The local
JSON fork consumes this same worker. Admission policy, funding accounts and
payload custody stay in the consumers; this collection grants no source credit
or native authority. It inherits the workspace prohibition on unsafe code.

`eredu-backend-mlx` uses `safemlx` for native arrays, operators, devices, streams,
collectives, completions and resource owners. The safe wrapper owns native API
and lifetime boundaries. Eredu parameter materialization, cache storage and native
operator composition belong in the backend; family semantics do not.

Every workspace package inherits `unsafe_code = "forbid"` except `safemlx`,
`safemlx-sys` and the `eredu-ios` C-ABI example. The iOS exception is limited to
foreign entry points, callbacks and pointer ownership. Pinned parser forks are
portable workspace members with the same prohibition. The llguidance fork exposes
only its safe Rust API; its C ABI/header generator is excluded. The regex fork's
nightly `Pattern` adapter is likewise outside its compiled feature surface.
Licenses and archive provenance remain in `third-party/parser-upstream.json`.
No additional unsafe exception is introduced by bounded inference.

## Ownership boundary

| Owner | Responsibility |
| --- | --- |
| `eredu-gguf` | Framework-independent GGUF reading/writing, exact canonical tensor encodings, validation and bounded conversion. |
| `eredu-checkpoint` | Backend-neutral schemas, layout recipes, exact prepared source stores, restricted views, leases, provenance, cache policy and resolution guards. |
| `eredu-text` | Tokenizer/template utilities, protocol parsing and shared text transformations without backend resources. |
| `eredu-core` | Portable model/session contracts, exact admission comparison, move-only submission authority, generation semantics, discovery, bounded observation and host-metadata funding errors. |
| `eredu-nn` | Neutral tensor/parameter/operator contracts and symbolic workspace mechanisms. It may use core's funding errors; core does not depend on NN workspace policy. |
| `eredu-runtime` | Plan normalization, load/residency policy, capability synthesis, shared execution/lifecycle drivers, sampling, scheduling, resource admission and observation delivery. |
| `eredu-architectures` | Family configuration, checkpoint naming/schema/topology, module construction, state geometry, semantic parallel plans, processors, selected execution and layer/embedding/output equations. |
| `eredu-codec` | Neutral neural audio codec families, released schemas, topology, recipes and typed artifact construction. |
| `eredu-media` | Optional portable host audio/image/video processing and validation. |
| Concrete backend | Native tensors, materialization, caches, streams, transfers, collectives, completions, exact mechanism facts and final typed binding/erasure. |
| Facade | Retained tokenizer/template composition, application requests, semantic tool orchestration, termination and public controlled records. |

The facade's `LoadedModel<B>` and related application operations expose neutral
errors, including `BackendFailure`, without native error type parameters. Provider
conversion preserves the original typed cause. Backend implementation traits may
retain their concrete errors internally. Public types have one owning crate;
the facade does not reproduce backend module trees or dependency-owned aliases.
Native factories select the implementation for the generic facade.

## Loading and session creation

Cold selection normalizes public policy once and validates the exact artifact,
family, residency and topology against side-effect-free mechanism facts. It must
not create a device, stream, tensor, group or completion. Core retains the neutral
preparation and selected backend target together; realization cannot substitute a
new selection after resources exist.

Architectures drive total prepared construction. Backends provide native contexts,
parameter-binding visitors and executable adapters. They consume retained source
roles and exact prepared artifacts rather than reopening artifacts, redispatching
source formats or reconstructing family branches. External-assistant compatibility
and source selection also precede target/draft resource materialization.

Prepared layerwise loading keeps one move-only manager in the shared binding
slot. The selected dense, routed or composite partition route, including
prediction and media finalizers, moves it into the existing native mechanisms
before policy construction. Local and pipeline dispatch preserve that same
owner. Disk policy therefore consumes its prepared controller and source identity;
it does not replace them with an ordinary background controller during binding.
The cold dense, routed and composite partition visitors use the same addressed
unit collector and physical binding tasks. They retain complete constructor slots
separately from the selected parameter exclusions and materialized lease rows.
Independent-bank exclusions come from the selected bank catalogs, including idle
local owners; off-rank static exclusions come from the selected static tasks.
Neither missing bindings nor an empty local member list establishes exclusion.
Composite construction completes the retained model graph before validating its
boundary, including routed destinations. Publication of the direct composite
equation source remains a separate selection gate.

Checkpoint preparation validates indexes, admitted shard/header identities,
recipes and requested ranges. A prepared store may reopen an authorized range
under its resolution guard; an executor may not replace that store with independent
artifact discovery. Native materialization allocates tensors from the retained
recipe and reports actual source/cache ownership. GGUF IQ operators consume the
public typed codebook contract, not generated private tables.

Architecture vocabulary geometry comes from family configuration and tensor
admission. Tokenizer membership independently limits executable generated IDs;
sparse holes and padded logits are excluded. Facade tokenizer, EOS and template
policy does not become backend weight-admission policy.

Session capabilities describe the selected route, including cache, observation,
copy and intervention support. Device facts alone do not imply session support.
The realized session must match retained admission. Discovery projects those
family declarations through runtime policy and backend collector facts; it is
neither execution authority nor evidence of a finite workspace by itself.

## Submission and completion

Core compares exact session admission and issues move-only submission authority.
The backend retains that lease with native resources until successful completion,
terminal failure with safe retirement, or teardown establishes that release is
safe. A failed poll, missing public event, drained host record or cancelled caller
cannot establish native completion.

Distributed readiness has an inner communication event and an enclosing native
submission role. Completion requires both the event and the sealed role to settle,
including retirement of the role's exact records. A healthy pending role or a
busy retirement guard remains pending. Bounded waiting uses one deadline across
both stages; expiration preserves unresolved recovery custody through the existing
quarantine path. Neither a ready inner event nor a failed poll releases authority.

Prepared scopes, native work and completed outputs preserve distinct ownership.
New native descriptors and backing need their own admitted producers. Immutable
aliases retain the existing owner and receive no new allocation credit. Returned
errors retain every partial resource they expose; bookkeeping shells retire before
their funding owners. Native callbacks enqueue prepared retirement capsules rather
than invoking arbitrary framework destructors while native locks are held.

`synchronize()` waits for retained work and reports failure while preserving state.
`reset()` establishes fresh request state only after settlement and keeps loaded
parameters/selection. Dropping an iterator may settle work but cannot report its
failure; callers use synchronization or reset before reuse. Neither method promises
allocator-cache flushing or immediate process-wide memory reclamation.

After stream settlement and the existing idle/health proof, synchronization also
drains ready retirement across host owners, ordinary owners and native callbacks.
The same public native reclaimer repeats only while actual retirement progresses;
pending or quarantined recovery alone cannot keep it spinning or authorize release.
This closes nested publication custody on the owning thread. Escaped outputs still
retain their charges, and final cleanup after their drop must run before that
owner thread exits.

Synchronization repeats the existing health/idle check and expired-view cleanup
after that drain. Deferred destruction can retire the last strong prefill owner
after the first check; its installed weak projection still owns a priced control
header and the request's host account. Only expired projections are removed, then
their deferred custody is reclaimed. Live prefill owners and escaped projection
aliases retain their independent charges. Cold admission and bounded reset do not
perform this ordinary synchronous cleanup.

The same idle housekeeping also removes lexically closed parallel-control
installations. It acquires both control-slot loans before taking either retired
view, then drops them outside the session and slot loans. Removing an inactive
installation does not certify native completion or refund cumulative grants;
escaped projections and unsettled recovery retain their own original custody.
Active parallel controls remain installed.

A successful reset may retire the previous nonstate publication when its remaining
registered sources exactly match the executable's still-retained initial
publication. Runtime compares live pool identity, storage keys and capacities;
equal byte totals provide no authority, and additional source custody prevents
the match. The backend performs this comparison before mutation and moves the
covered owner into its already funded reset-retirement node. New or uncovered
sources retain their existing owner. Native, host and metadata aliases keep their
own attached custody, and reclamation follows the existing quiescent retirement
boundary rather than the reset return alone.

Paid reset uses the exact selected local state across resident, host-layerwise,
disk-streamed and distributed sessions. Its closed preparation account compares
source/transport metadata and independently admitted native readiness producers
against the same operation limit without reserving their storage twice.
Distributed reset retains the existing coordinator and agrees preparation before
installation, then publication before reuse. Failed publication fences the
session. Both stages consume the coordinator's existing cumulative attempt and
transport counters; reset does not refund model, capture or communication work.
State installation and displaced-state retirement use the shared typed worker.
The canonical `ResidentTableResetState` contract validates placement for each
actual component, so paged attention and device-resident fixed children can share
one Hybrid layer. KV and Hybrid resets use the same source-authenticated,
independently funded empty-manager constructor. The final table constructor moves
that prepared manager even when the local partition has no paged attention rows;
layout, global layer indices, child roles and physical pool identity remain exact.
Old pages and escaped fixed tensors keep their original owners. This constructs
empty metadata and tables without copying tensors or refunding cumulative work.
The six-case Dense/Mova Resident/Host/Disk CLI matrix covers distributed reset;
its exact profile and limits are in [validation](bounded-inference-validation.md).

The portable residency plan's device depth remains the maximum moving-unit
window. Ordinary and admitted native acquisition settle the preceding consumer
before using the same persistent-source-aware trim worker. Dense transfer
windows own group-local units; a group switch retires only exhausted windows and
rejects still-live transfers. A roomy byte budget does not enlarge the selected
window. Completed persistent alias owners retain their explicit exemption.

The neutral native-storage mechanism can also report that a physical backing
already has a retained attachment. MLX supplies this proof from a complete
initial publication after loading authority retires, or from a sealed successful
source attachment whose original custody remains healthy. Its existing Array and Host
inventory nodes use one shared owner/receipt representation and become scalar
allocation-generation/capacity receipts in the exact pool domain. They retain no
native payload or registration pin, and publication allocates no receipt table. Current backing
validation remains required. A matching receipt avoids constructing another
registration and attaching the current request's control account to an already
covered parameter. New allocations, foreign domains and missing proof use the
ordinary checked publication worker. The enlarged owners and receipt-lookup
controls are prospectively funded; no admission allowance is discounted.

The same publication mechanism receives borrowed native roots. MLX distinguishes
an actual Array from an actual immutable Host owner without manufacturing an
Array or cloning payload custody. Checked Host observations authenticate physical
identity, capacity and native constructor provenance before the existing
registry transaction and attachment. A prepaid alias may outlive its original
run and enter a later request in the same pool; it retains the donor's original
charge and never becomes a new birth. Key-only source inventory keeps its stricter
same-request rule. Shared Host/Array backing is one physical root after both
descriptors match; an independent CPU General-copy allocation and its Host source
remain two roots. The actual borrowed-root enum, observations and attachments are
included in the existing finite publication control census.

A prepared immutable Host constructor proves physical provenance, not its payer:
load-time managers and independently admitted saved copies use the same constructor
as prepaid source banks. Their checked Host witness therefore emits the neutral
`ExistingPhysical` observation. This remains strictly existing-only: the canonical
publisher validates generation, capacity, live ownership and original account health,
then preserves the full ordinary, copy-funded or prepaid origin. Missing or foreign
rows cannot become fresh births, and duplicate classifications must agree. Array
native-birth and immutable-source observations retain their stricter origin checks.

Lazy Host sources and canonical device parameter cells share one private scalar
proof of successful attachment. The Host source transaction returns its sealed
result only after initialization, canonical registration and checked attachment;
a zero-capacity source carries no attachment proof. A canonical Array cell records
its proof only after a checked originating attachment whose actual callback payer
matches that cell's original metadata custody. Its once-only slot accepts the same
proof idempotently and refuses a different generation or capacity. A prepared cell
alone grants no receipt, and a published cell cannot be reset as unpublished.

Later publication snapshots each receipt alongside its actual borrowed native
witness at both validation points. Reuse requires equal generation/capacity and a
healthy original account in the same pool; healthy closed donors remain usable.
Foreign, quarantined, asymmetric or stale proofs cannot fall back to load-time
receipts. Plain Array aliases collected before canonical manager cells upgrade the
same physical row, retaining the displaced handle until outside the manager loan;
that handle retires before original cell custody. Shared Host/Array backing still
deduplicates, while an independent CPU copy retains its separate physical charge.
These receipts retain no new registry pin, payload owner or source allowance, and
avoid attaching later request accounts to persistent source allocations. Owners,
iterators, checked handoffs and validation frames use the existing exact census.


An optional, prospectively priced receiver on the same prepared allocation-owner
attachment permits exact owner retirement inside a bounded host worker. The
native callback publishes only that attachment's completed node to its private
atomic slot; it never executes Rust payload destruction. Its nontransferable
receiver can reclaim that node on the preparing thread outside the native lock,
using the ordinary reentrancy and unwind exclusion. Abandoning the receiver sends
an already-retired node or the future callback to the existing ordinary queue.
Live native aliases still retain the attachment; no unrelated queue is drained.

Prepared composite startup retains the admitted input's metadata destination
before selecting serial or tensor-parallel mechanisms. Qwen-VL and conditional
Qwen share startup and part construction; Gemma's adapters use its existing
common worker under the paid error scope. Tensor parallelism changes vocabulary
lookup and local state geometry. Host vectors, decoder rotary preparation and
forward metadata retain their original funding. Completed encoder tables remain
attached through the authenticated media ingress source.

Completed media publication visits both the current decoder and its retained
encoder ingress after actual completion and recovery retirement. The private
typed source erasure exposes only this read-only root traversal. A later decoder
span may install an alias created at ingress; its completed witness therefore
retains the original allocation identity, capacity, budget and account throughout
ingress ownership. This uses the same funded source publisher and creates no
replacement registration or execution grant.

MLX transport timeout or indeterminate completion poisons the affected authority.
Where native abort is unavailable, unresolved resources remain quarantined until
positive native evidence permits reclamation. Thread exit cannot turn missing
completion evidence into safe destruction. This can retain memory after an
unrecoverable device failure.

## Text and multimodal generation

`prepare_chat` compiles a borrowed request against retained tokenizer/template
sources into `PreparedChat`. Its render, policy, source receipt and capacity remain
bound to the loaded execution. The facade's `prepare_chat_invocation` validates
that source, resolves settings, authenticates the selected input and prepares
controller, channel, stop and semantic state once.

`PreparedChatPrompt` selects rendered text, an exact token prefix, or authenticated
media. Explicit IDs undergo domain validation directly. `prepare_chat_input`
uses the existing original model-input producer, joining the actual render and
semantic preparation to the original host/native source. Architecture-owned
processor declarations define framing, placeholder replacement, expansion, video
coordinates and decoder ranges. Runtime compares every ordered rendered text
value; a digest or caller range never authorizes skipped text or foreign input.

Ordinary `start_prepared_chat` constructs `PreparedChatSession` around the core
machine and committed-token cursor. Its manual and uninterrupted operations use
one sampling, readiness, commitment, decoding, stop, metrics and publication
worker. The recorded control wrapper adds commands and a funded journal to this
session. Ordinary tools require no drafter or speculative capability.

Distributed token and completed-media inputs construct the same source-funded
readiness control. Exact media packet, cache, pool and request authentication stays
in shared admission, so a refusal participates in the admission agreement before
prompt or sampler construction. Readiness itself grants no media authority.
Admitted sampling retains the parameter version authenticated by its quotation
before prefill; pending snapshots and branch exchange use the same strict version
checks as completed states.

Prepared media quotation uses one interval loop for serial and partitioned
execution. The selected architecture source supplies the exact local state,
constructor and partition executor; the native adapter supplies the retained
communication source and matching workspace/recipe recorder. Encoder-cut roots
and borrowed observation paths use the existing partition worker, including saved
pending media. A rank-local projection is validated against its retained local
realization rather than the serial global layout. This preparation grants no
invocation or completion authority.

Prepared text and retained media execute under the same transaction's admitted
model and parallel-control context loans. Their execution callbacks share the
same boundary issuer, checkpoint, completion vote and rollback ordering. Context
guards restore the prior bindings on success, error or unwind; media cut validation
remains inside execution before publication.

Inactive pipeline participants submit each architecture-declared reduction wave
through the same retained model context as active participants. Runtime validates
all input contracts before the backend consumes any occurrence. The backend
funds the exact output and root tables, constructs every once-only reduction, and
then completes the whole wave; waiting after the first lazy reduction could
prevent its peers from submitting later members. An absent retained source uses
the ordinary communication mechanism. An invalid retained source is an error.

Architecture block workers preserve tensor geometry and parallel coordinates
independently of observation. Qwen-VL dense blocks and DeepSeek routed blocks use
the same worker for ordinary and observed execution; instrumentation controls
hooks rather than selecting a second equation or reshape sequence.

The same `PrefillDriver` selects intermediate StateOnly output and final-position
readout, while full-sequence consumers retain their declared demand. Prepared
media is encoded once at its semantic ingress and shared across decoder chunks.
A chunk retires only after actual completion and required observation delivery.
Family equations and causal/row declarations remain architecture-owned.

Prepared prefill sources expose only their existing shared cache-identity owner.
The final committed chunk installs that owner; the trait has no owned-identity
conversion or allocating default. The cancellation-aware source gateway accepts
explicit admission policy. Score progress, state-only output and custom span
operations retain their distinct contracts over the same lifecycle worker.

Literal text, reasoning, forbidden tools, active tools and automatic activation
are output/controller policies. Protocol selection, declarations, EOS/structural
IDs and capability decisions share one facade producer. Unknown templates cannot
silently replace requested semantic output with literal text. Tool callbacks
receive committed events, including incremental arguments and validated completion;
no backend owns delimiter parsing or tool-execution policy.

## Speculative generation

`PreparedChatSpeculativeRequest` takes the same chat and input variants. The common
invocation worker supplies semantic preparation; the actual drafting strategy
adds proposal/verification state and its larger event window. Single and batch
requests share lane preparation and vote on all policies before prompt work.
`TokenIdsInputPlan` is the backend prompt contract for exact IDs and authenticated
encoded input. Input custody survives execution and escaping failure.

Controlled and uninterrupted speculation use the same scheduler. Proposals remain
separate from committed output; an accepted prefix plus replacement/bonus commits
atomically at the scheduler boundary. Snapshots preserve target/draft state, RNG,
controller, cursor and cumulative accounting. See [execution control](execution-control.md#controlled-speculative-generation).

Captured media quotation uses the architecture's original prepared input and
ordered semantic-token declaration. Existing token parts are borrowed, declared
markers use the shared U32 fill worker, and multiple segments use one concatenation.
Those tokens are provenance alongside the real media, never its replacement. The
source union includes original input plus completed model/numerical roots before
tracing the shared composite forward. Independent speculative chunked ingress
binds that same authenticated source to each role's actual cache. Its native
single-device acceptance tests pass Required and Auto image/tool generation,
including ordinary/manual and independent speculative parity at the unchanged
8 GiB limit. Native TP2, PP2 and combined TP2/PP2 image/tool tests also pass
Required and Auto through ordinary, manual and recorded execution at that limit.
The public validation record retains their exact artifact and commands.

## Bounded inference ownership

A memory ceiling configures admission over shared mechanisms. It must not select
another parser or inference engine, adopt preexisting unpriced output, replace an
unknown bound with a multiplier, or retry without enforcement. Arbitrary external
callbacks need an explicit bounded contract.

Source compilation, execution workspace, observation quotas, transport credits
and snapshot retention/cumulative copying are separate resource domains. Plans
quote actual destinations before allocation or submission, including grammar and
schema growth, token masks, native graphs, event payloads, host transfers, caches,
failed prefixes and retained outputs. Cumulative work already spent is not refunded
by cancellation, rollback or branch exchange. Live shared allocations are counted
once by exact source identity and retained until their final aliases retire.

The direct dense Qwen-VL partition constructor retains its exact configuration,
TP/PP geometry, complete local state, global parameter declaration and graph.
Cold construction aliases these paid architecture owners and invokes the same
static-module workers. Borrowed parameter declarations authenticate the retained
owner's identity; equal values cannot substitute another source. Native adapters
provide mechanisms and binding, with no family-specific construction branch.

Workspace tracing executes architecture equations with neutral metadata tensors.
The selected backend supplies exact operation and host-mechanism facts; all
required spans and nonmonotone decode growth remain covered. Quote aliases share
one immutable funded diagnostic report; mutation of a shared report first pays
for its independent copy through the existing report worker. Capacity estimates distinguish physical
backing, native descriptor/graph storage, host preparation and selected residency;
scalar equality cannot replace source authentication. Unknown required facts
remain typed refusals before the affected producer.

Incremental/residual planning returns the reservation and accepted quote. Admission
diagnostics are borrowed from that reservation, so a separately returned report
cannot outlive its payer. Diagnostic construction uses the existing funded report
worker; retaining a failure does not change candidate retry policy. A terminal
saved-state placement has no equation spans: its quote preserves existing state
backing and any missing coverage, and residual composition still requires the
exact installed metadata selection and independently retained registered owner.
Copying and enclosing work remain separately admitted.

`HostMetadataFunding` is core's neutral account/handle/error contract. NN extends
it with NN-specific allocation and parameter-copy operations; there is no transparent
second funding wrapper. Source constructors reserve concrete capacity, then retain
that payer through result, error and final shared-shell destruction.

NN errors have one canonical typed-source owner. Construction and cloning retain
the original error without eager diagnostic formatting. Aliases share its control
and source; the final control allocation retires before the concrete source and
its payer. `retained_source_construction_bytes` describes that worker's actual
Box/shared-shell allocations and fixed constructor transports. Paid callers use
that quote or the same `metadata_source` worker before erasure. `Message(String)`
holds already produced diagnostic text and grants no funding or source authority.

Native kernel-name construction also uses one worker for ordinary and admitted
execution. It preserves by-value argument snapshots and appends them by borrow,
removing recursive suffix copies. The qualified selector query prices the actual
entry values, numeric temporary and references while preserving lookup counts,
name heap bounds and cache ownership. [Host evidence](bounded-native-execution.md)
records pristine parity and the derived storage reduction separately from whole
request cost and device execution.

Tokenizer matching and encoding use one implementation with explicit cache policy.
`ControllerDeclarationData::admission_bytes` supplies the fixed planning allowance
for opaque compiled declarations. Their `owned_capacity_bytes` remains unknown;
exact filter and byte-buffer capacities retain their existing contracts. Runtime
deduplicates admission by the same immutable source identity and retains its
charge until final retirement. This allowance is not a measured heap inventory
or a dependency memory ceiling.

The pinned dependencies own parser/table growth and their qualified allocation
facts. Grammar, schema and tokenizer sources share immutable paid graphs; mutable
parser/history/copy state is independently admitted. Compilation receipts bind
actual outputs to the original account, tokenizer and execution. Completion-schema
validation preserves the selected draft and original typed errors; no permissive
fallback follows a funding refusal.

The parser dependencies also own prospective fixed-frame accounting. Derivre's
borrowed operation scope retains the live recursive peak and reuses capacity only
after child frames retire; heap requests remain cumulative. llguidance uses that
same mechanism for mutable parser work, compiled-grammar/condition/regex
inspection, schema construction and copying, and JSON/Lark grammar emission.
Nested workers borrow their operation's scope rather than opening a scope at
every recursive call. Neither a guard nor its payer can escape that lifetime.
Retained-capacity inspection is paid before traversal; a future-copy quote does
not authorize earlier inspection. Compiled copy plans retain their inspected
lexer plan, and failures preserve the first funding cause and original account.

Framework-managed bounds do not describe total process or system memory. Application
buffers, event copies, allocator caches and opaque driver/JIT internals need their
own declared policy. [Bounded inference](bounded-inference.md) defines the public
scope; [released tool validation](prepared-chat-validation.md) records
current Required/Auto text and image successes, exact capacity and process-memory
measurements and validation limits.

Native candidate refusals retain scalar component diagnostics beneath the neutral
`BackendFailure`. They describe initial, minimum and final candidates without
granting capacity or source authority. Their own account survives rejected
candidate retirement; the final error enclosure is funded before allocation and
preserves the original typed planning failure.

Neutral workspace events retain LayerNorm's exact optional weight and bias
roles. Native scalar facts follow the selected worker's type promotion and
bias-only cast. They preserve known operand roles and derive physical precision
from the actual source representation. The inline role flags allocate no storage
and grant no collective authority.

CPU paged attention consumes the shared neutral blockwise descriptor and the
existing native accumulator. Its stage recipes describe actual CPU primitives,
output aliases and buffer births; the CPU completion recipe includes the exact
nested page settlement. Row-wise F32 maximum reduction is a side-effect-free
native capability query, revalidated against the actual primitive, dtype, axis,
strides and output at evaluation. Scalar facts alone create neither cache
ownership nor completion authority.

CPU reshape admission validates the physical stride span, so GQA broadcasts may
repeat a smaller backing. The same borrowed reshape planner determines and prices
aliasing or copying from the actual shape and strides. Integer unit-axis and
transpose views retain their exact scalar dtype and source backing without
floating stride evidence. Transpose facts require the canonical retained axis
permutation; a name-only event supplies no native source. Integer reshapes reserve the existing General-copy
branch when strides are unknown; the storage report also retains the possible
full input alias. Ranked gather quotes therefore include the actual expand,
concatenate and flatten sequence. Empty reshapes use the native zero-element
alias path with no new backing. Rank, dtype, shape and readable-span checks
remain attached to their native producers. CPU GatherAxis permits a selected
source extent distinct from the index/result extent only after checking both
actual readable stride spans; queued indices retain their bounds check. F32
LogAddExp uses the same named floating task in ordinary and admitted execution,
with its existing weak input/output owners, fixed iterator source and rounding.

An explicitly selected F32 tiled CPU GatherMM uses the same SIMD matrix worker
as selected Matmul. Ordinary and admitted execution share its checked matrix
geometry and queued task; the task retains four actual inputs and validates
indices before reading either bank. The cold source accounts for those owners,
inline shape/stride controls, fixed tiles and one output birth. Its frontend
uses the existing typed I32/U32 unit-step range and cast producers; neither a
BLAS allowance nor a materialized expert-bank copy substitutes their costs.
Platform-default GatherMM keeps its existing implementation.
The CPU grouped quote composes that selected worker with the actual validation,
sort, gather, activation, overwrite-scatter and weighted reduction sequence.
Whole and split Units/Finish traces share the native chunk schedule and retain
each independently rounded child allocation, callback destination and optional
TP bias correction. Sequential group-order reduction keeps its integer sort,
broadcast-index gather and ordered additions. The quote requires positively
represented F32 dense banks; unqualified physical layouts retain their typed
source refusal rather than borrowing Metal or platform BLAS facts.

CPU I32 addition, subtraction and multiplication use the existing typed binary
worker, including shape-checked broadcasts. Integer layouts retain their logical
dtype without floating representation evidence. Changed operand shapes contribute
their actual Broadcast alias producers before the binary task; native evaluation
still checks strides, physical spans and signed loop limits. The logical peer
dependency keeps its typed zero, multiply and add sequence unchanged.

The neutral `zeros_like` trace delegates to the existing typed zero constructor,
using the prototype's authenticated scalar precision and shape without claiming
its values as graph inputs. Generic host initialization remains a separate
contract. CPU empty reshape and squeeze use their actual shared-buffer workers:
they produce no task, Data owner or physical backing, and retain the complete
input owner even when a zero-element view holds a nonempty allocation. Exact
shape, inferred-dimension and singleton-axis checks still precede admission.

CPU Slice preserves the native distinction between a partial empty result and
a whole-shape identity. The partial empty worker calls `allocate_data(0)`: its
funded Data owner has no physical backing birth or task and does not retain the
input backing after graph retirement. Identity slices retain the input owner.
Slice and empty Broadcast share the zero-Data constructor census; normalized
coordinates, scalar precision and fixed-rank source checks remain required.

CPU populations keep frontend `construction_entries` separate from Eval
`primitives`. The row-movement source reserves identity casts and unchanged-shape
broadcasts in the frontend constructor bank without adding an executed node;
tape entries, input edges and backing births use its actual Eval sources. Sequential composition
adds each population independently, while alternative branches take their
respective maxima. Addressable completion preserves that distinction when it
builds the shared graph bank and the final traversal.

The beta-scaled CPU softplus quote follows the ordinary widened F32 arithmetic,
Boolean threshold and Select sequence, then returns the original scalar precision.
Its source charges each native cast/broadcast/task envelope and the three actual
F32 scalar constructors; Boolean storage remains distinct from floating storage.
Returned storage preserves the shared logical F32 byte floor for half-precision
results while their representation retains the actual native dtype.
The resulting dtype and contiguous output proof can feed the existing collective
source checks. Missing physical input evidence still refuses before that boundary.
These sources retain their native fixed-rank limits. A pointwise constructor
quote alone does not qualify an entire high-rank CPU pipeline. The recorded
rank-11 to rank-12 Broadcast exceeds the native alias source's rank-five limit
and retains `GraphFailure::invalid_layout` under a real physical budget, followed
by positive record retirement. No larger budget changes that source requirement.

CPU host spill and reload use the existing native General-copy workers and the
same admitted Host source owners as their ordinary counterparts. Their exact
CPU query and evaluation guard validate scalar type, rank, source backing and
copy geometry. A Host descriptor seed is distinct from a new device allocation;
store roots and their nested completion occurrences remain in the composed
recipe. Preserving a scalar annotation alone cannot authorize a host transfer.

The neutral indexing worker retains normalized coordinates for pure range
operations in the existing static-slice contract. Native CPU planning consumes
those coordinates and the actual indexing frontend's storage facts; a generic
index event or unchanged rank alone does not establish a slice producer. This
preserves the scalar and stride evidence of paged keys without changing native
indexing equations. Composed CPU recipes retain internal nested completions
alongside capture reads and preserve the largest reached root/capture population.

## Capture, interventions and state control

Core owns versioned logical observation declarations, raw plan validation and
closed host records. Runtime combines selected support facts, reserves the finite
schedule and delivers each completed frame once. Backends own native transforms
and exact source/completion binding. The facade composes delivery with normal
commitment and text termination. The only backend drain is fallible
`try_take_text_capture`; the shared frame retains custody without a compatibility
carrier. Pending or failed frames remain owned until delivery or safe teardown.

`ActivationObserver::observes_activations` describes activation collection,
generated evidence and intervention interest. Unknown observers default to true;
the no-op observer returns false, and borrowed/error/cut adapters preserve the
fact. A false value omits activation-only paths and hooks in the same partition
equations with the same paid context. It does not disable lifecycle, cancellation,
transport, or retained-media cut callbacks; those remain at their existing
completion boundaries.

Raw capture/intervention declarations are admitted after actual request geometry
is known. Core's intervention validator and copy producer are shared by ordinary
and funded admission. Runtime's paid support projection copies requested targets;
architectures authenticate family points and retained partition placement. A
backend supplies mechanism facts rather than semantic branch logic.

`CaptureSourceConstruction` is core's borrowed prospective descriptive-copy
policy; it grants no admission or submission authority. Architectures publish a
closed `PreparedComponentPartitionSource` from the exact retained selection,
including parameter placement, coordinates, rank ownership and catalog/support.
The backend joins transport and collector facts, then publishes a closed shared
source before compiling original capture or intervention declarations. Ordinary
and funded construction use these same workers; an ordinary cached source is
rebuilt under funding rather than adopted as paid storage.

Precompiled capture inputs qualify that same source during original admission,
before semantic revalidation. Capture admission and quotation retain their actual
pool/execution metadata account. Borrowed declaration comparison uses fixed
errors; owned rejections and existing-storage pin inventories are funded before
construction. Native prepared-path validation receives the same metadata policy.

Checkpoint owns the stable-file fingerprint worker; core owns the versioned
identity reduction and shared resolution cache. Fresh Unix identity construction
funds read buffers, members, paths and errors before their producers, without an
ordinary cache warmup. The safe `rustix` filesystem dependency is host operating
system support and introduces no unsafe-code exception. Fresh funded opening on
other hosts still requires qualification. [Partition-source evidence](bounded-source-contracts.md)
records the exact behavior, refusal/custody tests and native validation scope.

Synchronous host summaries use the same once-only original capture claim. A closed
borrowed F32 source describes dense values or a uniform scalar. Runtime validates
shape/length, applies the original strided selection and common finite/nonfinite
reducer, and retains the actual bank through success or error. Caller-computed
summaries confer no completion evidence or native readback authority.

Snapshots compose all mutable state through one transaction: native model/cache,
pending input, sampler/RNG, controller, semantic parser, cursor and optional journal.
Fresh destinations need original admission; immutable sources may be aliased.
Read-only saved/installed projections report actual sampler and source facts so
facade metadata reflects the state that was installed. Capture, intervention,
transport and copy spending never rewind. Terminal copies use empty execution
geometry and cannot create another prediction. See [execution control](execution-control.md),
[bounded capture](bounded-capture.md), [interventions](interventions.md) and
[component analysis](component-analysis.md).

Hybrid snapshot inspection and copying use the same grouped source worker for
ordinary and prepared entry. Every actual fixed-role child table is retained
and funded, including an empty table on an all-KV layer. The worker preserves
resident versus paged storage and each exact manager; a KV-only outer-table
copy path cannot replace that topology during source-identity inspection.

Prediction-lane execution borrows its complete mutable state through the neutral
session's scoped state-loan worker. It validates the selected state layout and
agreement boundary, preserves each state's actual revision during placement, and
restores ownership on return, error or unwind. Numerical publication still
advances the installed lane revision; a loan never reinstalls a saved revision,
establishes completion, clears a fence or refunds custody. Persistent branch/state
exchange continues to invalidate prior source bindings.

## Cache, residency and distributed execution

A completed Host-backed native array has separate immutable-source and Device-view
lifetimes. The MLX adapter authenticates its Host identity and capacity plus the
nonrecycled native `Data` generation before attaching a prepared Device-residency
owner to that exact `Data`. Array clones, shared-buffer views and native recovery
pins keep the Device charge alive; an immutable Host source or a later independent
reload does not. This positive view witness grants no source or mutable-birth
authority. Failed authentication returns the unchanged prepared owner, and fixed
inspection/attachment controls are paid before preparation. The existing specific
retirement receiver releases a charge only after that native `Data` dies, allowing
repeated completed demotions under the same finite Device cap without releasing
an escaped view's charge or draining unrelated callbacks.

Architectures declare residency units and parameter/state geometry. Runtime owns
selected-task sizing and model-independent residency policy; backends implement
physical materialization, cache storage, transfer and completion. Layer acquisition
shares window preparation, draining, transfer order, lease construction and
population. Admitted attempts are finite. An ordinary recovery policy can retry
only after its required pending owner actually retires. Foreground reads,
background prefetch and resident borrowing retain their distinct I/O witnesses.

Runtime resolves selected transformation source recipes without native tensors.
It checks the complete admitted derived output before applying the retained
rank's additional and primary placements. MLX materialization uses that same
resolver, then verifies the resulting shape and dtype against its native source
slots. Recipe resolution reads metadata only and grants no storage admission.

MLX cold quantization translates architecture-projected destination slots through
the same scalar binding descriptions as native parameter traversal. Both routes
use one exact-task planner for format groups, complete target consumption,
companion identities, packed output shapes and compatible dtypes. Working-set
sizing uses metadata only; bounded payload acquisition belongs to conversion.
Ordinary native loading additionally validates the actual source tensor handles
before using that plan. Destination projection and planning grant no admission.
Conversion chooses row and leading-matrix tiles using metadata alone. It checks
the selected tile alongside pending submissions before acquiring bounded-read
proofs or materializing source bytes. Rejected sizing candidates perform no
payload reads. Conversion retains bounded-read refusals and pending completion
ownership on failure.

Completed quantization can retain its exact plan and immutable source identity
in a move-only handoff. Residency preparation may borrow the resulting overlay;
native adoption uses the same source/destination validation as ordinary loading,
compares the retained plan and source, and returns that same overlay and report.
Adoption reads metadata only. It neither requantizes the payload nor grants new
storage admission; output buffers retain their original constructor custody.

Writable tensor buffers in `eredu-checkpoint` retain constructor custody before
allocating their metadata and payload. Publication moves those bytes into the
immutable memory store; leases, detached readers and weak storage identities
retain the same control. Allocation failures and duplicate-name rejections preserve
their custody and actual constructed prefix. MLX quantization writes into these
buffers. Buffer construction itself grants no admission; the caller also owns
funding for catalog construction and subsequent reader/selection controls.
Quantization preflights every selected target's source geometry and minimum
working set before allocating any final output. Conversion retains those exact
destinations and uses the same bounded tile worker, including its one- or two-slot
completion window. Before invoking any destination allocator, allocation borrows
an existing stream to qualify every target's workspace for its device. This also
applies to pool-funded final buffers. It creates no conversion streams or tensors
and acquires no payload leases; bounded reads belong to the conversion stage.
Affine tile sizing includes the original scales and biases while casts produce
the selected companion precision, including F16/BF16 casts of equal byte width.
Cold minimum-row checks, complete targets, leading batches and row candidates use
that same live-output contribution. Final encoded output telemetry excludes these
temporary cast sources.
The tile's completion owner reserves output slots before quantization and retains
all native siblings before converting either companion. Each successful cast
replaces its retained root; a later failure leaves the accepted prefix and its
sources with the same recovery owner. Original materialization owners use their
prepared vector capacity and reject growth. They can submit a caller-supplied
finite traversal on an explicit stream through that same owner; source preparation,
native fit and producer admission remain the caller's separate obligations.
The CPU affine tile entry authenticates that owner and validates its completed
input and selected storage quote before constructing quantization or companion
casts. It uses the shared tile producer with separate affine and cast constructor
banks, then submits the quote's finite traversal through the retained owner.
Lazy inputs and mismatched quotes are rejected before producing native outputs.
An admitted encoded checkpoint read can fill the final native input buffer
directly. Its source account covers the metadata arena, physical backing, one
prepared array alias and synchronous read scratch before payload I/O. Exact shape
and dtype checks precede admission; only a complete successful read publishes an
immutable source. Escaped array aliases retain the source account independently
of the constructor wrapper, and failed reads retain their account with the typed
error. Retained read metadata and runtime initialization are separate prerequisites.
Shared constructor failures can separate their uncalled plan from an owned
failure. The failure retains typed causes, any completed output or failed prefix,
and its original account. This allows borrowed planning inputs to retire without
refunding surviving resources. Output access remains borrowed, and retained
native owners continue to determine the failure's thread-safety requirements.
Cold conversion can admit its fixed materialization slot through the source
account before a text request exists. Checkout transfers the one prepared node;
its source account stays with the same completion/recovery owner after the slot
wrapper is dropped. Request-funded slots retain their text control guard through
that same driver. Input capacity is checked before source production, and input
aliases move directly into the final prepared vector without a temporary vector.
Cold native invocations admit their selected graph, record, physical-buffer and
recovery storage through the source pool before creating native owners. They use
the shared native-role completion driver with that source custody, including on
callback failure and deferred retirement. An admission rejection retains the
uncalled invocation; escaped native outputs retain their original accounts.
Runtime initialization, streams, retained read metadata and dynamic result
storage remain separately funded prerequisites. A cold root rejects an existing
original scope; nested work requires the explicit parent relation.
Submission seals an invocation's scope and retains its native owners without
waiting. Its result may be borrowed for independently qualified readback;
explicit finish establishes completion, while dropping an unfinished invocation
transfers its owners to the same recovery queue. Queued conversion tiles need
independent scopes because a materialization owner observes its entire scope.
Synchronous native roles submit and finish through this same implementation,
preserving their deadline across the callback and final completion observation.
Bounded conversion shares tile selection, the cross-weight completion window,
writeback and telemetry across producers. An ordinary producer supplies its
existing materialization owner; a cold producer supplies an independently scoped
submission. The queue keeps either owner through output readback and retires the
cold submission after successful writeback. A later submission or writeback
failure drops the same queued owners through their existing recovery paths.
The common driver takes the selected device fact without constructing device or
stream wrappers; producer resources and pipeline metadata need separate admission.
The queue propagates the producer's concrete failure type, including thread-local
construction owners. A cold callback failure retains its typed cause and pending
native role together; separating success from failure neither waits nor releases
that role. Encoded affine construction admits its input and fixed materialization
slot within the role and returns their original failures. Earlier queued tiles
retire independently when a later constructor fails, without publishing a partial
converted store. Runtime, stream and retained read metadata funding remains a
separate prerequisite of this constructor.
Encoded recipe preparation also offers an uncached entry through the same
contiguous/projection compiler. It infers against metadata retained by the read
batch, so temporary inference does not consult or populate the source's
persistent cache. Construction metadata, the retained read and read scratch
still require their own admission; disabling inference caching grants none.
Bounded conversion's geometry, selection search, peak sizing and source-byte
telemetry use a borrowed uncached catalog view. Recursive inference follows the
same recipe algorithms but leaves temporary candidate recipes and metadata with
the caller, without retaining them in the source's persistent inference cache.
Repeated inspections recompute metadata. Physical bounded-read checks and each
producer's materialization ownership remain separate; the view itself supplies
no reservation for temporary construction or retained read metadata.
A CPU tile resource owner composes the admitted process allocator and scheduler
with two distinct registered source streams and their admitted workers. Fixed
composition controls have their own source account; each native child keeps its
existing constructor account. The owner exposes borrowed runtime and stream
access, validates the original pool and retains successful prefixes on later
construction failure. Foreign domains and active unquoted work refuse before
runtime construction. Native registrations and worker threads retain their
accounts for their actual process lifetime after wrapper retirement.
The completion window stores its two slots inline and retires abandoned entries
in submission order, including after wraparound. The admitted pipeline entry
reserves these fixed queue controls and the allocator-cache cleanup node before
running the shared tile driver. Cleanup retains its original account through
ordinary deferred retirement; returning from conversion or dropping an error
cannot refund a still-queued cleanup node. Cache clearing remains restricted to
explicit host boundaries, with the same active-working-set and retained-cache
limits. Source/read, recipe and overlay metadata remain separately owned.
Stream qualification includes CPU MXFP4's composed quantizer payloads: floating
codebook distances, reduction/index values, scale intermediates, constants and
a possible input compaction. Sizing sums potential destinations without assuming
donation or early retirement, and limits submissions for the 16-way expansion.
Direct GPU quantization keeps its own payload profile. Model-load minimum sizing
covers the CPU fallback; final output telemetry still reports only encoded bytes.
These logical payload contributions exclude allocator rounding and native
graph/worker controls, which require their own producer admission.
CPU Abs and floating Round expose the same unary task storage source used by
their native evaluator. Their cold queries cover task, descriptor, output and
strided-iterator storage; physical backing remains separately admitted. Integer
range admission accepts nonnegative integral coordinates and positive integral
steps whose final increment remains within I32. Compact U32 final-axis sums at
ranks two through four share the ordinary typed reduction task and SIMD order.
Both sources retain separate physical-buffer admission and completion custody.
Half-precision row maxima and typed F16/BF16/F32 arg-reductions expose their
ordinary comparison workers, including SIMD NaN handling and first-index ties.
Their source checks authenticate the reduction axes, output dtype, shape and
readable backing before admitting the task and its separately funded output.
The largest arg-reduction row index also fits the worker's signed location cursor.
CPU affine conversion authenticates the ordered packed-weight, scale and bias
siblings and shares one Eval task and cleanup across all three outputs. Its
storage query includes each output allocation, weak descriptor and optional
General-copy input temporary; cleanup retains that temporary through completion.
The source accepts F16/BF16/F32 inputs, group sizes 32/64/128 and bit widths
2/3/4/5/6/8 with checked positive geometry and nonnegative readable strides.
Affine graph construction has its own bank for one primitive, its retained
fallback, three sibling descriptors, shape/stride storage and three C result
handles. Stack-held shape/dtype metadata and a reserved output vector avoid
ordinary allocator calls during that construction. Fixed C result handles feed
the same safe Rust quantization API for ordinary and prepared calls; failure
leaves each published prefix under its result guard. Graph quota extents include
alignment headroom; they are not a process-wide memory ceiling.
A shared CPU affine submission layout composes that constructor, optional scale
and bias casts, all three output roots, the completion Synchronizer and finite
traversal storage. It includes possible input compaction and prices each physical
allocation with the selected allocator without assuming donation or early release.
Completed detached input custody, arena and failure owners, runtime/stream/cache
preparation and final encoded host destinations require separate admission.
If a CPU primitive fails after retaining encoder temporaries, its prepared cleanup
is queued behind accepted work on the same stream. Refusals without temporaries
enqueue no cleanup work. Cleanup submission failure preserves encoder custody and
the original primitive error.
CPU reshape planning scans borrowed dimensions in collapse order, with no
rank-sized temporary shape table. Its admitted alias/copy layouts include
out-of-line destination strides, and the General-copy worker retains its existing
rank-dependent storage quote. Ordinary and prepared evaluation use the same
planner, including higher-rank matrix banks.

The backend quantizer passes the retained input geometry directly to the native
constructor. Packing changes the final dimension; leading bank dimensions remain
in the weights and companion outputs. Native allocation queries describe that
same geometry, with source preparation and companion dtype conversion funded
separately.
Quantization tile writers borrow output views from the materialization owner
after its exact completion wait. Ordinary owners retain their existing evaluation
behavior; original owners validate completed storage through their retained
observer, including after the scope is sealed, without submitting another Eval.
The encoded destination copy consumes that view and cannot initiate evaluation.

CPU MXFP4 construction uses the same resident bank, with a source-derived bound
for binary/Select casts and broadcasts, unary operators, views, reductions and
range construction. Its six eager constants expose individual payload requests;
their physical capacities are admitted separately with the selected allocator.
The shared payload inventory also lists possible evaluation allocations,
including input compaction, both codebook-distance destinations and integer
packing intermediates. Physical capacity sums the selected allocator's request
layout for each entry, without relying on donation or early retirement. The
bounded converter's logical temporary workspace uses this same inventory;
source input custody and final encoded host buffers remain separate.
The bank also covers the two fixed C result handles. Identity elision may consume
less than the reserved population. Eval, physical buffers, source preparation
and streams require separate admission. Failed allocation prefixes retain their
accounts without publishing failed outputs as completed values.
A shared CPU MXFP4 submission layout composes the constructor population with
existing typed worker queries, exact output roots, the completion Synchronizer
and finite traversal storage. It prices every potential frontend candidate with
the largest applicable worker layout, so identity elision and donation cannot
increase its allowance. Fresh Graph and Record capacities cover one detached,
completed input and two output roots on the same CPU stream. Physical buffers,
input custody, arena owners, runtime/stream/cache setup, failure ownership and
encoded host outputs remain separate contributions. Completed output access
uses the original scope's read-only completed view rather than another evaluation.
Prepared host-transfer storage follows the selected native allocator: CPU
backing includes its size header before page rounding, while shared Metal
backing has no in-band header. Immutable Rust metadata and source witnesses
retain that actual storage kind, physical capacity and constructor provenance.
CPU transfers accept CPU or shared Metal storage; GPU transfers require shared
Metal storage. Host output destinations are prepared before original submission,
and completed source descriptors are validated by their retained observer before
copying. Escaped source arrays keep the source arena and attached custody alive.
CPU-only event dispatch quotes use the actual named wait and signal tasks,
including their retained Event owners and Graph allocator extents. No GPU
handler allocation is included in that CPU-only backend's query.
Empty CPU operator outputs reserve the selected allocator's real zero-byte
backing: the CPU allocator retains a charged header/page, while Metal has no
physical allocation. Shared empty-output layouts preserve that distinction and
retain each physical account until its final owner is released.
Completed tiles copy logical native-endian values directly into the final encoded
buffers. This copy checks the exact destination length and supports signed strides,
broadcast and unaligned views without allocating an intermediate byte payload.

Runtime's original memory-tensor constructor reserves qualified fresh payload
capacity before allocation and authenticates it through private source custody.
Its quote separately exposes configurable metadata headroom for names, shapes,
sharing controls and catalog publication; that contribution is an estimate.
Subsequent recipe caches, readers, selections and native conversion need their
own funding. Settlement allows the later native load owner while payloads,
independent readers and weak source identities retain the original reservation.
Same-pool inventory registration recognizes those prepaid payload bytes;
foreign pools charge the complete capacity and ordinary buffers gain no credit.

Layerwise cold binding and the selected native unit populator share one
immutable independent-parameter exclusion owner. The pre-load manager source
constructor prices and copies its exact sorted names before publication; its
closed shared owner retains the original source account through the final
native or cold alias. The native handoff checks the actual selected names
before taking that owner. Empty selections require no name allocation. Both
ordinary map binding and paid source-row binding leave only those explicitly
selected slots with their independent owner; absent rows never create an
exclusion. Missing unit parameters, unexpected rows and incompatible
representations still refuse before replacement. The retained policy and
exact names establish source identity; the owner outlives policy retirement.

The MLX Host and foreground Disk parameter sources retain the exact selected
stream's scalar identity and device. CPU and Metal destinations use the same
paid source-manager constructor and residency lifecycle; CPU selection does not
substitute an ordinary manager. Ordinary and admitted Host snapshots share one locked comparison of the retained
stream and immutable source owners before pinning; validation does not reconstruct
a source snapshot. Foreground read descriptors remain
source-only, while their owning identity and bound recipe also match the
selected destination. Copy quotation selects the actual CPU General-copy and
completion controls or Metal dispatch controls from that retained device.
Completed source descriptors enlarge traversal storage without duplicating the
equation worker population. CPU source shapes additionally prove positive
signed-index geometry; rank and dtype remain subject to the native CPU copy
query. The existing Transfer/MetalShared Host-backing requirement is preserved
for both destinations, independently of which device executes the copy.
Layerwise parameter construction uses that same destination witness for CPU
and Metal. Each named unloaded slot contributes its actual scalar seed and
lazy native constructor nodes; strict binding replaces those nodes before
evaluation. Cold destination projection supplies the complete per-unit slot
inventory, including placeholders populated by independent parameter banks.
The original manager prices and retains its exact constructor counts, dtypes
and maximum ranks separately from the smaller lease-row inventory. Model-equation
and owned-child quotes compare placeholder counts and dtypes with that retained
constructor source. A lease-only source can certify construction only when it
owns every slot; excluded slots without their complete declaration refuse.
Supplementary sources retain their own slot inventory and never index the main
execution units' constructor table through a shared manager.
Sampling spans on
the same recorder remain separate and do not reconstruct model parameters.
CPU and Metal descriptive reports share the exact eager dtype-sized scalar
backing and zero disjoint Host-payload facts for unloaded slots, keeping tensor,
Host and residual totals complete. These facts do not certify a native CPU
operation: its plan remains absent, and only the retained constructor inventory
authenticates the equation trace and prices the replaced lazy graph.
Retained boundary events, copy-frontier expansion and Graph quotation share
one source-derived stream census: CPU model and collective streams stay
distinct, as do the Metal model, CPU router and collective streams. Scheduled
source-copy aggregates authenticate the exact CPU or Metal stream and execute
only their new Synchronizer. CPU aggregates share the nested-event settlement
worker, which waits for the actual signal tasks and Records before restoring
the resident construction bank; Metal aggregates retain asynchronous completion.
Refusals preserve accepted Record, Graph and source custody and do not certify
completion. Finite invocation and wait counts remain unchanged.

Original unit-operation banks retain request sources and cumulative slots across
branch exchanges. Each text submission separately activates its existing bank
and retains that guard through the actual completion/error owner. Rebinding
requires healthy inactive registries with no registered roles, pending leases,
preparation loans or active background forward. Deactivation neither clears
registrations nor refunds counters. Speculative and realtime banks retain their
existing single-invocation activation and failure custody.
Prediction roles keep their registration in the same native recovery node.
Successful retirement destroys its payload, drains that exact observer's records,
then removes the live registration without restoring the cumulative role count.
Deferred observed payloads keep the registration in their existing cleanup node.
Retirement refusals fence the bank: synchronous callers receive the original
native or fixed cause, while abandoned cleanup retains that cause for the next
fallible observation. Failed or unresolved roles remain registered.

Parameter traversal loans canonical metadata and retained values. Owning consumers
explicitly fund copies. Dynamic execution graphs are retained at construction and
borrowed through one declaration contract; a single-group declaration can remain
allocation-free. Native immutable parameter tables share one sorted representation
for ordinary and funded construction. Mutable slot replacement still requires
quiescent authority.

Architecture state layouts, identities, unit paths, group counts, media ingress,
partition schemas and collective declarations use one producer with an explicit
metadata destination. Checked callers pass their actual context; a refusal never
selects ordinary allocation. Parameter descriptions borrow retained declarations
or construct through that same context. An owning consumer explicitly moves the
owned result or funds a copy of the borrowed graph, layout and parameter names.
Physical parameter expansion shares one format worker across both destinations.

Observation rebinding receives the same typed metadata destination as its
equation constructor. Regenerated paths, row declarations and ordered routing
nodes use that destination. Routing declarations share an immutable paid owner;
adding a bank to a shared declaration funds its copy. Binding identities retain
their original payer through strong and weak aliases. Existing metadata refusals
cross observation and session wrappers without another funding callback.

Architectures own TP/PP/combined placement and transport across semantic boundaries.
Runtime validates complete communication manifests, groups, routes, bounded
completion policy and agreement before native realization. Native backends supply
groups and operations; a communicator does not own application scheduling or
family placement. Stateless pipeline ranks still participate in control votes.

Completed routed partitions retain the selected resident-provider source or the
actual independently cached banks and their policy. The cold quote borrows that
same completed construction; it cannot substitute resident expert weights for a
cache-backed local invocation. For resident partitions, the execution strategy
lends the executor's actual resident policy through its parameter-owner contract.
Cold quoting and request installation therefore use the same retained registry;
the strategy neither constructs a second bank nor infers authority from the
residency enum alone. Bounded modes keep their selected bounded policy.
The executor and strategy separately declare the actual group-submission
mechanism. Shared layered-graph execution contributes its selected group
boundaries; manual composite/pipeline traversal contributes no unexecuted group
submissions or consumer waits. The backend consumes this declaration in quote,
source facts and installation. Resident addressable registries and bounded final
output completion keep their independent authority and accounting.
The executor separately declares its group-submission mechanism. Direct and routed
layered drivers retain their ordinary initial/group completion source; manual
composite and pipeline traversals reserve only the policy completions they call.
This distinction preserves bounded final-output completion and the registered
resident owner needed by indexed children, without adding nonexistent graph
boundaries merely because a parameter policy is available.
Expert-region declarations carry the exact local
physical members, compact-bank byte policy, tensor partition and finite received-row
ceiling. Direct and exchanged Units observations use the same physical addressable
layout producer. These descriptions grant no member identity, read, lease or native
execution authority. The backend must consume the accepted indexed source and its
actual completed row count through the existing addressable role, including the
zero-row occurrence; unused finite constructor capacity is never reissued.
The existing partition-source owner retains the adapter's exact physical member
map once the checked provider constructor has verified every selected local key
and its actual bank bytes. Both the direct quote and exchanged local declaration
borrow that map, including during decode. Pre-partition checkpoint recipes do
not determine the physical chunk width after tensor partitioning. A missing
completion refuses; a genuine empty local bank completes with an empty map.
Repeated or competing constructors adopt the winning published source and
execution owners. An identical completion is idempotent, while changed physical
facts or policy refuse. The map moves into the existing owner without a second
member map or a native payload pin.
The local numerical envelope inspects a bounded set of outer-chunk tails and
native grouped-kernel transitions. It combines the completed numerical equations
before joining the authentic maximum-row source-copy population once. Copy and
aggregate producers retain their separate Graph/Record/kernel and completion
costs; sharing the constructor arena does not add them to the CPU equation tape.
It shares immutable parameter descriptions and specializes only the actual
completed row geometry under separately paid
metadata controls. A nonempty local invocation enters the existing indexed
native role directly; it does not add a resident-weight child. An authenticated
zero-row invocation consumes its source occurrence and retires unused child
grants before executing the existing empty numerical branch. A lexical loan
keeps the accepted source channel and activation alive through the caller's
original typed result; failure or unwinding fences the occurrence. Runtime row
specialization charges the accepted request while retaining the immutable source
account independently.

State/cache replacement stages detached candidates and publishes only after the
required all-rank votes. A transport failure or indeterminate commit fences later
work; an observed global abort permits the specified rollback. Routed failure
agreement has one narrow terminal recovery vote under the retained bound and never
clears original poison. Logical member-only groups need exact independent status
support; disconnected Ring membership cannot silently require inactive peers.

Global observations require architecture-derived ownership, bounded transport,
complete receipts and shared commit. Ordinary rank-local inspection does not
implicitly gather. Raw and precompiled enforced capture use the funded retained
layout/support source above. Prior lazy discovery cannot authorize that storage.
Native Ring fixtures separately exercise source publication and compare actual
distributed capture with an ordinary local reference; their execution results
are recorded in the partition-source evidence.

## Realtime and codec composition

Realtime scheduling consumes encoded token frames through one transactional
prepare/submit/completion/publication driver. Managed frame preparation adds its
actual source, state-copy, numerical, transport and host-output admission to that
same driver. Release/resume retains the existing session; it is distinct from
snapshot/fork. The selected topology must support coordinated admission and
completion before dispatch.

Standalone PCM codec calls are application composition outside the encoded-token
frame account. Their convolution tails, overlap and transformer caches do not
rewind when a token-frame session is cancelled or released. Codec families remain
neutral; native backends provide ordinary tensor/materialization mechanisms.
Existing realtime PP/combined and snapshot gaps remain implementation limitations,
not inferred architectural impossibilities.

## Guarantees and verification

Use compiler visibility, manifests/features and behavioral conformance to enforce
these boundaries. Portable facade tests use neutral backends. Family integration
must include applicable released source formats, bounded residency, TP/PP,
controlled inference, snapshots, discovery and independent numerical validation.
Do not police source spelling, paths or dependency graphs with text-scanning tests.

Run the relevant checks listed in [AGENTS.md](../AGENTS.md). Native build setup is
in [the MLX backend guide](../eredu-backend-mlx/doc/README.md). Detailed discovery
contracts are in [architecture discovery](architecture-discovery.md), and public
control integration is in [the Inspector guide](lm-inspector-execution-control.md).

Reproducible commands, checkpoint revisions, artifact hashes and tolerances are
in the [validation guide](bounded-inference-validation.md) and its result records.
The [design guide](bounded-inference-design.md) describes execution ownership;
validation distinguishes portable contracts, native mechanisms and released
requests with explicit platform and test-environment limits.
