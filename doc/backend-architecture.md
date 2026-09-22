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
Rust dependency admission uses public APIs. The [memory contract](#memory-contract)
requires explicit estimates or identified unknown overhead for internal allocations
whose sizes those APIs do not expose.
Archive verification and fresh consumers exercise the packaged first-party chain;
staging registry configuration stays outside the product manifests.

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
not. Shared K2 Horizon conformance data is embedded in `eredu-evaluation`, with
source hashes and publisher provenance, so backend and facade test archives do
not require sibling source directories. `eredu` without default features remains
a portable facade. Backend adapters,
accelerator options and host image/audio dependencies stay feature-gated; model
families and codec families remain available without a backend.

`eredu-collections` is a dependency-free, `no_std` foundation for the shared safe
ordered-map worker. Its prospective insertion callback reports the exact node
layout before allocation and preserves the caller's typed refusal. Portable
source producers consume this worker. Admission policy, funding accounts and
payload custody stay in the consumers; this collection grants no source credit
or native authority. It inherits the workspace prohibition on unsafe code.

`eredu-backend-mlx` uses `safemlx` for native arrays, operators, devices, streams,
collectives, completions and resource owners. The safe wrapper owns native API
and lifetime boundaries. Eredu parameter materialization, cache storage and native
operator composition belong in the backend; family semantics do not.

Every workspace package inherits `unsafe_code = "forbid"` except `safemlx`,
`safemlx-sys` and the `eredu-ios` C-ABI example. The iOS exception is limited to
foreign entry points, callbacks and pointer ownership. Published third-party
dependencies retain their upstream implementation and safety boundaries.
Their public APIs do not weaken the prohibition in portable workspace crates.
Cargo.lock records the selected releases and archive checksums; each dependency
is distributed with its upstream licensing and provenance. Native MLX patches
remain confined to the published `safemlx-sys` build.

## Ownership boundary

| Owner | Responsibility |
| --- | --- |
| `eredu-gguf` | Framework-independent GGUF reading/writing, exact canonical tensor encodings, validation and bounded conversion. |
| `eredu-checkpoint` | Backend-neutral schemas, layout recipes, exact prepared source stores, restricted views, leases, provenance, cache policy and resolution guards. |
| `eredu-text` | Tokenizer/template utilities, protocol parsing and shared text transformations without backend resources. |
| `eredu-core` | Portable model/session contracts, exact admission comparison, neutral memory contributions and pure policy evaluation, move-only submission authority, generation semantics, discovery, bounded observation and host-metadata funding errors. |
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

## Memory contract

Core's physical-domain descriptors distinguish location, accounting ownership,
and backing allocation identity. `MemoryTopology` contains immutable domains and
the backend-declared mapping of host and accelerator locations to those domains.
Every topology has a distinct process-local identity; identically named domains
in independent topologies are not interchangeable. Unified locations share one
physical domain only when backend facts establish that physical sharing.

`MemoryLimit` distinguishes `Finite(u64)` from `Unlimited`, including at
`u64::MAX`. `MemoryLimits` resolves a complete vector, defaults unspecified domains
to unlimited, and rejects unknown names, duplicate declarations, and foreign
identities. Pointwise intersections describe the tightest configured and live
constraints. Raising a finite limit to unlimited is a relaxation; a pure limit
comparison grants no ceiling succession authority.

`MemoryPlacement` describes fixed placement or a finite, source-labelled set of
candidate domains. A candidate set reserves a conservative full-capacity allowance
in each distinct physical domain. Locations mapping to one domain collapse;
independent allocations do not. Candidate allowances retain their estimation
basis and describe no measured residency. `validate_access` rejects a location
outside that set but grants no native work authority.

`DomainMemoryRequirements` separates accounted allocations, candidate-placement
allowances, finite overhead estimates, and domain-specific headroom. It checks
all arithmetic before changing a descriptor, including under unlimited limits.
Descriptions expose their controlled backing capacities for host metadata
quotation. Descriptor construction does not reserve or fund those capacities.
Its capacity comparison consumes existing live requirements and an increment;
it does not coordinate concurrent operations or register storage.

`SharedStorageAccountingId` identifies an attachment's accounting owner rather
than physical placement. Shared immutable backing carries one attachment for
each accounting identity. `same_ledger` compares runtime accounting-owner
identity and does not infer physical sharing. Allocation identities continue to
establish backing sharing independently of these identities and diagnostic names.

Shared host attachments use one ordered-map node per accounting owner. The
portable collection worker reports the prospective node layout before allocation;
the closed provider admits that node and its concrete custody owner before either
is constructed. Runtime host metadata uses its existing source lock for both
fixed prepared slots and ordinary nodes. Reuse needs no new node or funding.
Final retirement frees the node and accounting identity key before releasing its
attached owner. The source constructor separately covers source and lock controls;
attachment funding grants no native execution authority.

`MemoryLedger` coordinates all registered physical domains in one process. Dense
domain balances, one storage directory and funding accounts share one mutex.
The [physical memory validation guide](physical-memory-validation.md) separates
neutral conformance, native mechanism evidence and platform qualification gaps.
Reservation, publication and ceiling succession validate every participating
domain before committing any counters, identities, owners or peaks. An ordinary
publication rejection preserves the complete ledger observation taken after
its separately admitted preparation. Commit performs
no allocation, provider callback, native work or payload destruction; retired
owners are released after unlocking.
Ceiling succession has a read-only preflight that rejects an ineligible live
predecessor before planning metadata is constructed. It uses the reservation's
eligibility worker and grants no authority; the transaction repeats that check
under its commit lock.

Prepared-chat resume forwards replacement physical-domain request limits into
that same saved-source admission. The facade validates domain declarations and
the retained semantic source; it does not require equality with historical
request limits or mint successor authority. The semantic source's independent
metadata account keeps its accepted limits, and host-copy limits intersect those
constraints. Public failures retain the neutral domain or metadata-funding cause.

The coordinator owns a funded dense transaction scratch buffer. Admission checks
borrowed requirement descriptions and all domain ceilings in that buffer before
allocating account vectors, report copies, source-pin containers or execution
identities. The accepted pending account covers those constructors; concurrent
constructors wait on the coordinator's funded condition variable. A constructor
failure releases its pending charge while preserving the already observed peak.
Ordinary admission rejection changes no counters or funding identifiers.

Exact host-container quotations use the published runtime's
[compiler memory profiles](../eredu-runtime/compiler-memory-profiles.md), including
standard-library backing and target-specific mutex and condition-variable storage.
These profiles authenticate unmodified compiler and library artifacts; they do
not establish a process-memory ceiling.

Host metadata includes topology descriptors, domain vectors, account/report
controls and storage-registration containers. `StorageMetadataFunding` admits
registration metadata before construction and authenticates its ledger identity.
Cold shared constructors obtain `prepare_construction_metadata` before building
their plans. This host-only planning owner uses the coordinator's construction
identity and preserves configured and live account ceilings without adding a
request ceiling. It retains the charge through plans and owning errors and grants
no native allocation or submission authority. Its move-only
`PreparedConstructionMetadata` seals a completed immutable plan: the coordinator
closes further allocation through every funding alias and ends the construction
exclusion without changing charges, peaks, identifiers or active limits. The
returned custody retains the same charge until the last metadata owner retires;
later consumers fund their own new allocations. Native constructor temporaries use
the separately admitted allocation scope of the same shared-constructor account.
Its scoped physical observer marks newly born backing for physical retirement at
the final alias instead of cache retention. The shared cache worker preserves an
existing cached root's original payer, identity and cache policy when reused;
entering a constructor neither relabels that root nor changes its charge. Normal
observers keep the ordinary cache policy. Bounded ordinary request Work uses
the same observer and allocator workers with an explicit fresh-backing policy:
it bypasses cache reuse and retires newly created roots after their last physical
alias. This selection is independent of finite or unlimited limits and device
topology. Older cached roots keep their existing identity and charge until their
own physical eviction; the request neither borrows nor relabels them. The
same rule preserves the original payer and cache policy when a native operation
aliases or donates an already live external backing. The
selected tensor capacity envelope includes fresh allocator rounding, and every
new root still enters the existing assigned physical admission callback. Deferred
accounting release follows actual physical destruction, including escaped aliases
and asynchronous workers; logical paging reservations attached to those roots
follow that same destruction.
After certified completion and retirement of all temporary native resources,
that account may retain only its authenticated fixed host metadata owner. Its
request exclusion then closes while the same fixed charge, domain limits and
account controls remain live through the last storage alias. The private owner
identity prevents an escaped native allocation from substituting for fixed host
storage; incomplete or quarantined construction retains the exclusion.
`StoragePublicationLayout<K>` quotes a bounded inventory without allocating.
Its `fund` operation reserves the complete constructor on that metadata owner;
`fund_from` converts an admitted account's assigned fixed host allowance without
increasing the total charge. The resulting move-only
`PreparedStoragePublication<K>` publishes or pins the exact backing descriptors.
It can accompany an existing unquoted loading owner but grants no execution or
native allocation authority. Metadata custody survives through the last returned
owner and container; native completion governs the separate backing lifetimes. Snapshots distinguish fixed baselines, registered
storage, reservations, placement allowances, estimates and headroom in every
domain. Registry metadata and reservation controls are identified subsets of
those charges. Historical peaks include their simultaneous physical charges.
Unquoted owners pay for their shared host ownership records before construction.
Their aliases preserve both the admission exclusion and this charge; the final
owner releases the shared allocation before refunding its host controls. Their
unknown payload remains excluded from inference admission. A full host domain
can refuse the owner record even when no execution is active.

Shared native constructors declare fixed host storage separately from temporary
physical allocations. When a constructor creates temporary native storage, one
ledger transaction admits both contributions and their accounting metadata.
The fixed host owner converts its allowance into retained custody; the producer
takes the temporary allocation scope once and uses its ordinary metadata and
allocation funding. Publication preserves the total charge. Native completion
and independent storage custody are required before certification, including
under unlimited limits. An abandoned scope retains quarantine, while completed
temporary allocations and surviving fixed host owners retire independently.

Scheduled capture source pins use the original program's prepaid host controls.
The consuming claim authenticates the native account, original control custody
and canonical chunk before constructing its bounded source pin. This operation
keeps the source backing and native backing-control record alive without a second
capacity reservation. Ordinary admitted transfers use their account's assigned
host allowance for the same pinning mechanism.

Placement-allowance snapshots identify the conservative full-capacity-per-candidate
rule. Allocation descriptors and retained requirements preserve the supplying
mechanism's basis and candidate set; these allowances are not residency telemetry.
Protected original host payloads remain payload charges even when an account
retains them in its control floor. Only quoted reservation bookkeeping is reported
as reservation controls.

The MLX coordinator attaches one accounting owner to each native backing root,
including roots already present when its observer is installed. An allocation
callback first reserves a pending backing allowance and its registration controls.
An admitted ordinary native scope carries an accounting-only handle to its
already assigned allowance. The exact scope and scheduled CPU task transport
that handle to allocation; the pending backing consumes that account's allowance
without increasing the ledger's total charge. The handle cannot grant execution
authority or create another scope, and certification, abandonment or quarantine
fences further allocation. Successful backing publication moves reserved bytes
to registered storage atomically. Allocation failure restores the same account's
allowance; surviving aliases keep their backing and host controls charged.
Views and cached buffers preserve the root identity. Reuse carries its existing
charge, while eviction retires the backing and controls. Ordinary Metal buffers
and CUDA device buffers use independent native allocations so their physical
retirement matches those roots.

Ordinary graph and submission allocators retain the accounting handle captured
at construction. Container growth, delayed task destruction, descriptors and
shared data owners use that same source even when another scope is current;
source refusal cannot fall back to the current scope. Their host allocations
publish through the existing physical observer and retire with the last owner.
CPU source queries describe the selected frontend, Eval, dispatch and consumer
wait populations separately from C wrapper shells and numerical payloads.
The dispatch allowance conservatively converts the same native worker's request
extents, including unused framing; the observer charges actual live capacities.
Opaque platform-event overhead remains a separate contribution. These queries
neither reserve capacity nor grant original execution authority. Caller metadata
comes from the selected safe/C signatures, including dense lookup validation and
mask construction, fused CPU attention, and each Default/YaRN RoPE batch, slice
and final join. Native primitive counts do not substitute for that caller census;
a selected worker without a matching caller source remains unqualified.

Each selected CPU completion frontier receives an Eval allowance for its reachable
graph. Container growth bounds include conservative deque blocks and hash-node
attempts; each possible observed allocation also receives its ledger ownership
allowance. Repeated nested frontiers can therefore produce a reservation much
larger than tensor storage or measured live native allocations. These allowances
remain balances in the admitted domain account; they do not eagerly allocate a
table for every possible control occurrence. Actual allocations consume their
assigned allowance and retain their individual ownership records.

CPU Eval source queries include the actual cross-stream Fence owners, one wait
per crossing edge and one update per crossing output, including siblings. The
CPU-only build retains the existing condition-variable worker. Ordinary CPU
producers in a Metal build select the Fence's existing Event worker independently
of limits, avoiding an unrelated GPU synchronization buffer. Both retain the
actual physical payer through queued tasks and final aliases. Ordinary GPU
Fence shared owners also retain that payer through the selected fast-buffer or
Event worker. Original Fence sources keep their prepared mechanism; CPU-only
facts do not qualify a mixed-device graph.

The Metal census combines the shared ordinary Eval containers with the selected
GPU prologue, numerical worker and encoder allocation sources. A retained router
CPU partition uses its separate source stream and actual ArgPartition bank; its
two-stream Eval query includes the crossing Fence and Event population. The
single-stream query remains separate. A retained Ring source contributes its
actual CPU constructors, dispatch population and completion frontiers before
the enclosing Metal graph's Eval allowances are computed. Each such frontier
can reach lazy GPU predecessors. Its two possible fast Fence U32 buffers have
separate default-allocator placement, allocation counts and host controls in the
collective scratch trace; slow Event mode retains the same conservative allowance.
The query's typed frames, worker controls and nested traversal metadata consume
the preparation account even when the requested native source is unqualified.
A graph containing both a router CPU stream and a distinct Ring CPU stream
requires a three-stream source and remains unqualified by the two-stream query.
The Eval contribution owns its Synchronizer event;
the GPU worker owns the async event, handlers, receipts and fences. Their
platform-event count is composed once. Consumer boundaries retain their actual
wait population without replaying the numerical DAG. Funded receipts and encoder
fences use the same captured physical observer as other graph controls; commit
leaves no idle successor receipt retaining the completed request. Prepared
fixed-kernel callers quote their actual borrowed-input vector reserve and
returned C handles separately from their native graph and numerical workers.

Retained CPU worker and GPU stream loans authenticate the same admitted
Scheduler identity during ordinary and original execution. GPU loans also
check the Device, stream and constructor token. This read-only source validation
creates no resource and grants no numerical or completion authority; each
producer validates its own execution context. Native constructors continue to
require an empty submission context.

Scope observer custody includes its quoted deferred retirement node. Native
release enqueues that node; the shared reclaimer destroys the Rust owner outside
native locks, after the native bridge and shared allocation have retired. The
same reclaimer retires completed submission records before draining owner queues,
so their scope and account custody can retire without another inference request.

Ordinary host-transfer buffers use the same backing observer. Their public C++
shared allocator reports the concrete rebound control allocation before either
the control or payload is allocated. Admission reserves both together. An array
view of a transfer buffer preserves its full backing identity, capacity and
placement. Completed staging buffers and independent destinations retire
separately; an alias keeps the original backing and charge alive.

CUDA managed roots keep full-capacity allowances in each backend-reported
candidate domain for their entire lifetime. The prepared CUDA mechanism uses
host-accessible pinned host storage with fixed host placement, including its
funded native descriptor. It uses the same original-buffer account and completion
custody as the other prepared mechanisms. This selection may cost device access
performance; it does not reinterpret managed storage as fixed host memory.
Native constructors still require their selected C++ ownership-layout profile.
An unqualified profile is an attribution failure under either limit mode.
The current native preparation profile qualifies Apple clang/libc++ 210106.
GCC and MSVC qualification is incomplete for the original allocator, scheduler
and main-thread guard, stream-registration storage, worker TLS and submission
registry. Cold Original workspace placement uses the immutable strategy witnessed
by actual prepared allocator initialization, including pinned Host storage. It
remains unqualified before that source exists or if a later initialization
contradicts it. Ordinary CPU and GPU allocation candidates retain their separate
allocator facts; selecting a CPU worker does not replace those candidates with
Host placement. The fixed snapshot belongs to the admitted process baseline.
CUDA backing placement and managed-domain allowances are implemented;
they do not establish those separate preparation and execution facts. A CUDA
inference path requiring that profile rejects with incomplete attribution under
finite and unlimited limits. This is an implementation qualification gap.

Host access cannot silently relocate an attributed CUDA device backing while
retaining its old identity and placement. A host read requiring new storage uses
an explicit host-transfer allocation and retains source and destination custody
through completion. Direct slice access to fixed device backing rejects the
implicit relocation; host-accessible prepared and managed backing retain their
respective fixed or candidate-domain attribution.

The neutral tensor readback methods return `HostTensorBuffer<T>`. Its full host
backing capacity and concrete accounting owner remain together through slice
access, truncation and consuming iteration. The backing deallocates before the
accounting owner retires; there is no conversion that detaches the same vector
from its charge. MLX copies a completed source synchronously into the admitted
host destination without constructing another execution graph. General exports
reserve that new retained output under the live domain ceilings. Selected expert
and capture readbacks use their existing prepared source and destination funding.
CUDA device-to-host readback uses the runtime's documented synchronous copy
contract, which completes host access before returning; an asynchronous polling
error provides no such evidence. Native copy errors retain their CUDA status.
See NVIDIA's [API synchronization behavior](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html).
A completed adopted Metal host buffer can supply its readable extent from the
retained native buffer even when its foreign allocation identity is unknown.
That read-only extent witness does not change placement or capacity attribution,
reserve storage, or authorize an admitted source; descriptor-based admission
continues to require its independent allocation evidence.
Copying values into application-owned storage creates an independent allocation
outside this retained owner and does not establish a process-memory ceiling.
Ordinary tensor observations use `SharedTensorObservation`, so cloning an
`ObservationValue::Tensor` retains the same admitted host backing. The raw
`TensorObservation` remains a caller-owned data-transfer value. Serialization
keeps that value's wire representation; deserialization creates independent
caller-owned storage and grants no execution or allocation authority.
Parameter reads and projections return `SharedParameterValues` and
`SharedParameterProjectionValues`. Their strings, geometry and numerical buffers
share one owner, so aliases retain the original producer's paid host custody.
The raw parameter DTOs remain explicit caller-owned exports. Final retirement
frees shared controls and payload storage before releasing that custody.
The neutral `NeuralBackend::ParameterPreparation` associated loan carries the
backend's preparation context through the ordinary parameter owner and residency
policy. The MLX loan retains its actual source, allocator, selected stream and
metadata payer. Nonresident inspection uses the same selected-window construction
and completion worker, with a standalone numerical account that atomically admits
native backing, source reads and constructor controls. It supplies no token,
prediction or realtime execution authority; target and supplementary source
identities remain distinct.


Configured limits and live account constraints intersect independently in each
domain. Raising a predecessor's constraint, including to `Unlimited`, requires
its move-only succession authority; unrelated constraints remain effective.
Unlimited skips capacity comparison only. Arithmetic, completeness, allocation
identity, source pins, ownership and completion checks use the same paths.
Core admission checks context and completeness and retains the domain declarations
for this transaction. Aggregate hardware-availability observations do not impose
another inference ceiling; cold mechanism selection may still use those observations.

Cold workspace inspection uses descriptive inputs through the same
`PrefillDriver` traversal as execution; it constructs no `InferenceRequest`,
execution identity or submission authority. Workspace lifetimes are reduced in
their resolved domains. `WorkspaceBound::PerDomain` preserves complete domain
requirements when an aggregate diagnostic cannot be represented; `Unknown`
continues to mean missing evidence.
The eager F32 and U8 scalar producers retain the default allocator's physical
placement. A typed zero or Full operation quotes its eager scalar scratch on
that same source, separately from its GPU fill output. Original execution keeps
the placement established by its retained prepared allocator. Aliases preserve
the full original backing and its controls through the shared lifetime reducer.
Generated F32 initialization owns its fixed Rust staging vector separately from
one eager native upload backing; its stream Copy retains that backing. The cold
CPU and Metal producers use the same typed upload facts. Geometry-only generic
initialization retains its numerical envelope but supplies no allocator placement,
including under an Original allocation strategy. Parameter coefficients use their
actual borrowed F32 slice producer instead of that geometry-only declaration.
Cold contexts created before native topology publication retain numerical facts
and incomplete physical reports; source preparation does not initialize native
resources or turn absent topology into an operation failure.
Compound native fact emitters retain actual default-allocator constructors and
complete mutually exclusive scratch envelopes before any domain reduction.
Their lexical companion resolves simultaneous source rows, then takes domain
maxima. Child projections carry their source alternatives into parent equations;
scalar-sized casts remain on the selected execution allocator. Repeated grouped
chunks preserve each complete first/down projection choice before reduction.
Attention keeps query and value dtype choices consistent across all tiles. Its
bounded source witnesses preserve the maxima for the default allocator, execution
allocator and their shared domain before physical reduction. Completed key-block
phases release eager page inputs while retaining their accumulator and query
outputs; normalization, value and final-concatenation populations remain distinct.
Absent constructor counts remain explicit missing controls even when the source
backing bytes are finite. Recording and
ordinary workspace contexts consume the same source factory and lifetime reducer,
including direct Metal facts and retained resident mechanisms. Lexical source
construction uses the caller's metadata recorder and account, so recorded cold
quotes include the same source tables funded execution constructs. Source preparation
errors retain their metadata payer separately from allocation-free equation errors.
The router's existing cold equation worker retains eager scalar, mask and ID
constructors and CPU cutoff partitions separately from execution-stream child
buffers. Its lexical source companion resolves each simultaneous scratch branch
before taking per-domain maxima. Returned partition and partial-update backings
keep their actual allocator candidates; a full forced-ID replacement retains
its eager U32 source. Escaping-output candidates and completion scratch retain
independent conservative placement allowances, including their source basis,
rather than claiming a data-dependent placement was measured. Their raw output
rows preserve aliases and allocation controls in the same lifetime graph.
Ordinary Broadcast quotes its actual retained publication group and three-array
completion population: input, rank contribution and returned output. The non-root
contribution remains in the shared numerical trace; publication uses the existing
Sum and completion workers without creating an Original execution source.
Ordinary token-mask caller quotation uses the shared expanded Bool upload,
logits clone, scalar and Select worker; its Rust mask payload remains in the
separate Host workspace source.

The selected communication mechanism also supplies a descriptive completion
strategy. Cold model collectives use the ordinary per-operation fallback when
the native mechanism owns separate submissions; mechanisms borrowing an admitted
model submission record their nested completions. Both follow the same partition
driver. Local communication dependencies retain a distinct worker description,
so their caller and completion charges do not consume layerwise completion
evidence. An enclosing-submission selection cannot use a direct dependency
submission without a separately qualified source. These scheduling facts create
no native completion or execution authority.

The same span and sampling reports retain their closing allocation populations.
Ordinary storage publication combines these populations with the actual opening
source inventory to quote finite collector and registry storage. Its assigned
host allowance funds the collector owner before constructing any collector
vectors or cloned storage descriptors. Publication retains that owner through
the resulting receipts. This host custody supplies no original-source or native
execution authority. Repeated sampler outputs that alias independent score
backings retain those complete backing charges; their simultaneous domain totals
remain subject to checked arithmetic.

`eredu_core::memory` provides borrowed memory contributions, checked aggregation,
pure policy evaluation and source-labelled warning records. The declaration for
one reservation scope separates:

| Contribution | Meaning | Budget charge |
| --- | --- | --- |
| Accounted | Allocation capacity and lifetime Eredu controls. | Declared accounted bytes. |
| Estimated | Finite inclusive estimated range with an identified source and basis. Neither endpoint is a guaranteed bound. | Upper estimated endpoint. |
| Unknown | Range `[0, ∞)` with an identified source and reason no finite estimate is available. | No fabricated finite amount; apply the unknown-overhead policy. |

`FiniteMemoryEstimate` validates ordered `u64` endpoints. A point estimate and an
explicit zero estimate remain estimates. `MemoryContribution` distinguishes them
from accounted capacity and unknown overhead. `MemoryRequirementReport` retains
every contribution in input order and separately exposes accounted bytes, the sum
of finite estimated ranges, their upper-endpoint allowance, additional headroom,
the budget charge and all unknown sources.

The charge is the sum of accounted bytes, finite estimate allowances and additional
headroom. It is not total process memory. A configured budget constrains that
charge; fitting it does not establish a dependency-wide or process-memory ceiling.
Unknown overhead remains `[0, ∞)` even when the finite charge is zero or headroom
is large. Neither allocator telemetry nor a reserve converts an estimate into an
enforceable allocation bound.

`evaluate_memory_requirements` checks descriptions and all finite sums before
making a policy decision. Overflow and malformed declarations are errors under
both policies. A charge above the supplied budget produces `BudgetExceeded`,
including when unknown entries are present. Equality fits. An absent budget skips
only that comparison; it does not disable the selected overhead policy.

| `MemoryOverheadPolicy` | Behavior when finite charges fit |
| --- | --- |
| `AllowUnknownOverhead` (default) | Return `Permitted`, preserve unknown contributions, and expose a warning for each unknown source. |
| `RequireFiniteEstimates` | Return `FiniteEstimateRequired` if any required contribution is unknown. Finite estimates remain estimates. |

Every well-formed evaluation retains its report, including rejected evaluations.
The `warnings()` iterator contains one `UnknownMemoryOverhead` record per unknown
contribution only when the evaluation permits proceeding. Consumers must emit
those warnings when proceeding; the contract chooses no logging, stderr or callback
transport. Unknown entries remain available from the report independently of
warning delivery or rejection.

The evaluator allocates no report storage and borrows its descriptions. A runtime
retaining a report must supply the corresponding funded owner. Diagnostic source
labels are not authenticated storage identities. Repeated labels neither deduplicate
charges nor merge warnings. Callers establish disjoint contributions for the actual
reservation scope; shared-allocation credit and lifetime overlap remain matters
for identity validation, planning and reservation. `Permitted` reserves no capacity
and grants no submission authority.

The architecture contract places checked arithmetic, contribution types, policy
comparison, warning descriptions and typed failures in core. Runtime owns context
policy propagation, contribution composition, atomic reservation, report custody
and diagnostic delivery. Backends supply native facts, estimates and identified
unknown overhead while retaining native resource ownership and completion safety.
Applications select policy and present diagnostics through facade orchestration.
Policy is inherited from the execution context and shared by controlled and
uninterrupted execution; it must not select a feature flag, alternate parser or
inference engine. Both choices preserve input/cache/concurrency limits, reservation
lifetimes and source custody. Budget exhaustion, invalid identities, unsupported
mechanisms and unsafe completion states remain errors under both choices.

The pure evaluator is independent of current inference admission. Execution
contexts do not apply `MemoryOverheadPolicy`; inference uses `WorkspaceBound`,
completeness requirements and the memory ledger's unquoted-owner exclusion.
`DependencyMemoryPolicy` supplies scalar host estimates to its existing reservation
mechanisms. These producers and reports do not use the three contribution variants,
and the runtime does not deliver the evaluator's warnings. `UnknownBound` also
appears in ownership and identity validation: accepting overhead uncertainty must
not bypass those failures.

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

Parameter discovery and residency-scoped queries lend immutable tensor values.
They preserve prepared observation bindings. Replacement publication remains a
separate operation that invalidates bindings and requires fresh preparation;
callers cannot replace parameter slots through inspection visitors.

Prepared configuration loans project the typed configuration from the retained
artifact and verify its identity. Execution eligibility belongs to cold selection;
borrowing a selected routed configuration does not require dense-model eligibility.
Composite workspace inspection uses the same input validation as execution.
Contexts that enforce a metadata allowance require the counted constructor;
ordinary diagnostic contexts use ordinary admission and confer no funded authority.

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

Ordinary residency acquisition can borrow its request's host-metadata account
for the same prepared controller, canonical closure, lease collection, named
destinations and transfer-publication workers. The ordinary recovery scope keeps
that payer through completion; cached named values retain it through their own
retirement. This host custody supplies no original-operation observer or source
receipt. Source-derived constructor queries describe those concrete controls;
checkpoint recipe/read preparation, native copies and completion require their
own contributions before the complete operation has a finite estimate.

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

The shared session describes each output transaction from its actual demand,
observer participation and selected strategy control operations. Its one-use
cursor distinguishes prediction-state loan entry and return, admits only the
driver's conditional phase order, and keeps failed prefixes separate from
successful commit. Named callback factories supply the actual session and
prepared-group callback types to backend metadata queries; these descriptions
create neither native resources nor submission authority. The existing shared
transaction and prediction-state loan remain the execution workers.

Model-internal partition agreements are recorded by that same partition driver in
the ordered workspace trace, with their selected group and boundary route. These
markers carry no tensor storage or native authority. A funded shared source table
retains their actual operation ordinals after numerical trace reduction. The
selected session visits the same outer-context and required-group callback
factories used by execution; only the inner agreement contributes native capacity.
A separate one-use cursor checks these model events without consuming session
lifecycle occurrences. Source-table aliases share their original host payer, while
the admitted native owner funds cursor construction and preserves failed-prefix
and completion custody.

Text and speculative model controls share the same group authentication,
collective callbacks, and native submission worker. The closed native owner lends
either the accepted text bank or the actual speculative role's retained budget;
it does not construct a second request. Speculative recovery retains that owner,
while lexical projection closure prevents later submissions. A retained root may
lend a current descendant only when the existing child collector authenticates
its explicit parent identity. The shared worker prices those checks and retains
its actual completion custody. Expert regions retain their separately recorded
provider-vote sources; their suppressed cold callbacks add no duplicate model
phase or native allowance.

Autoregressive, embedded-target and external-equation roles authenticate their
actual issuance account through the same request custody comparison. Embedded
recovery owns the completion roots and retained parallel invocation; the session
borrows a weak projection only for the existing equation callback. The projection
checks execution identity, current scope and the captured compute stream before
lending its group. Initialized expert integers are a distinct source-program
component, separate from parameter storage. Phase-funded workspace metadata and
addressable descriptors keep their own payer while the retained communication
source preserves its original native quotation and control payer. Recorded model
callbacks use a separate prepaid child sealed to that same source issuer; the
model-only context hook lends it through the shared agreement worker. Expert
and neural-collective callbacks keep the original source payer. The lexical
phase binding spends its own prepaid phase child, and neither child is charged
again to the native role. Weak projections retain the account that pays their
shared control header without retaining the native root payload.

Distributed prompt-cache controls use the same phase callback and prepared-group
visitor. Their neutral source retains the selected save preflight, preparation,
and publication votes, or the load preflight and preparation votes. A move-only
cursor accepts only that order and requires every selected vote before successful
completion; actual failure may end a reached prefix. The existing reversible save
and provisional load drivers perform all cache work. The source and callback query
carry no cache, communication, or submission authority.

MLX retains the initialized agreement source in one cache-control owner and
installs only a weak projection during the shared save/load call. Each exact
phase consumes the neutral cursor before using the shared physical native-role
worker, which admits the source's backing, graph, record and callback controls
and preserves its completion timeout. The accepted construction account funds
the owner, projection header and typed failure transports. Refusal, an invalid
phase or unwinding closes the owner; completion and recovery retain their own
native custody independently of the lexical installation.

`SharedPromptCacheManifest` retains one immutable manifest, its construction
account and any separately admitted dependency estimate account. Its prepared
shell is charged before publication; field constructors pay their own allocations
through the construction account. Cloning shares the backing, and the final
shared shell and payload retire before their funding. The plain manifest remains
an independently owned schema for inspection and deserialization;
neither representation authenticates file contents or grants cache restoration.

The public per-session cache methods require a rank-local state manifest and
return `PromptCacheError::RankHasNoState` for a stateless rank. The neutral
distributed driver still participates in the selected agreements and explicitly
returns `None` for that rank; it does not import a manifest or establish a new
logical frontier there.

Admitted prompt-cache persistence separates controlled metadata from dependency
estimates. Exact buffers, paths and shared shells use the session's construction
account. Each bounded metadata input admits a source-labelled finite estimate for
stock parsing, validation containers and filesystem scratch through the same
ledger transaction. The upper estimate appears as estimated overhead in the host
domain; retention links remain accounted metadata. Failed admission leaves every
domain and funding identifier unchanged. All successful estimates remain charged
through the final result or failure alias, including intermediate parser storage.
The default manifest input limit is 64 MiB; it bounds bytes read from the opened
file, independently of the estimate and physical-domain ceilings. Low-level
persistence constructors accept an explicit finite input limit. These allowances
do not establish a process-memory ceiling or apply `MemoryOverheadPolicy` to
inference admission.

The MLX autoregressive row retains a separately quoted control owner from its
actual partition source. Its admitted model role funds the existing native child
worker for each selected control occurrence, independently of capture transport.
The lexical mechanism binding covers prediction-state entry, the shared model
transaction, and state return. Actual callback factories supply both loan and
binding layouts before admission; typed failure transports retain the same
metadata account. Transaction and loan completion advance the one-use cursor,
while failed prefixes and unwinding preserve native recovery custody.

A successful reset may retire the previous nonstate publication when its remaining
registered sources exactly match the executable's still-retained initial
publication. Runtime compares live ledger identity, storage keys and capacities;
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
one Hybrid layer. Generation also permits fixed-only sibling layers beside an
actual paged KV layer. The complete hybrid source worker validates each fixed
role, dtype, extent and backing; the paged source retains its independent manager,
file, transfer and completion proofs. KV and Hybrid resets use the same source-authenticated,
independently funded empty-manager constructor. The final table constructor moves
that prepared manager even when the local partition has no paged attention rows;
layout, global layer indices, child roles and physical pool identity remain exact.
Old pages and escaped fixed tensors keep their original owners. This constructs
empty metadata and tables without copying tensors or refunding cumulative work.
Resident key-only and compressed latent/rotary caches use their exact inline
empty constructors. Reset preserves the key-only/window configuration and the
compressed allocation step, checks their actual component placements, and leaves
snapshot-owned backings charged until those snapshots retire.
Pooling state uses the same prepared table publication and retirement worker. Its
local-key window and pooling ratios come from the retained layer policy; paged
local keys use a source-qualified empty manager. The portable `DeviceState`
adapter forwards the actual layer constructor plan and funded host context,
while displaced local, pooled, partial-window and overlap arrays retain their
original custody.
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
allocation-generation/capacity receipts for the exact ledger. They retain no
native payload or registration pin, and publication allocates no receipt table. Current backing
validation remains required. A matching receipt avoids constructing another
registration and attaching the current request's control account to an already
covered parameter. New allocations, foreign ledgers and missing proof use the
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

A prepared immutable Array/Host constructor or foreign native-budget witness proves
physical provenance, not its payer: load-time managers, conversions and independently
admitted saved copies share constructors with prepaid request banks. Their checked
witnesses therefore emit the neutral
`ExistingPhysical` observation. This remains strictly existing-only: the canonical
publisher validates generation, capacity, live ownership and original account health,
then preserves the full ordinary, copy-funded or prepaid origin. Missing or foreign
rows cannot become fresh births, and duplicate classifications must agree. Array
births authenticated against the current budget retain their stricter origin checks.

Checkpoint sources likewise retain their first canonical charge. Same-pool source
constructor custody is validated even when ordinary loading registered the full
payload without prepaid credit. Later controlled publication preserves that full
ordinary charge; it cannot promote the row, create a missing row or substitute
an unrelated execution account for source funding.

Host source construction receipts retain only the actual accepted accounting
hold after a finite debit. The producing bank keeps the full source-validation
guard through construction and publication. Escaped native buffers and owning
refusals retain their paid storage without retaining an earlier request's source
witnesses; repeated disk materialization therefore does not form a chain of
completed request accounts. Receipt identity still compares the original account,
and a spent receipt cannot issue or refill construction capacity.

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

Restoring a completed snapshot retains the selected neural and layerwise-transfer
source descriptions even when its equation schedule is empty. The same source
matching checks apply; its submission, transfer and wait populations remain zero.

Prepared text and retained media execute under the same transaction's admitted
model and parallel-control context loans. Their execution callbacks share the
same boundary issuer, checkpoint, completion vote and rollback ordering. Context
guards restore the prior bindings on success, error or unwind; media cut validation
remains inside execution before publication.

Tensor-only pipeline schedules retain the selected routed bank invocation order
and each bank's declared reductions even when expert parallelism is disabled.
The bank-order validation is shared with expert-exchange scheduling; absence of
an expert communication group does not turn a sparse model unit into a dense one.

Inactive pipeline participants submit each architecture-declared reduction wave
through the same retained model context as active participants. Runtime validates
all input contracts before the backend consumes any occurrence. The backend
funds the exact output and root tables, constructs every once-only reduction, and
then completes the whole wave; waiting after the first lazy reduction could
prevent its peers from submitting later members. An absent retained source uses
the ordinary communication mechanism. An invalid retained source is an error.

The shared local-expert callback distinguishes genuine completed Original row
receipts from ordinary receive-row values. An ordinary MLX callback borrows the
indexed owner installed by its admitted Work, selected by the exact current
physical observer. Its existing bank channel authenticates the declaration and
actual row count, and retains the source through completion or failed-callback
retirement. Cold preparation carries a separate ordinary local bank program;
branch alternatives share the same source census, and the parent joins their
lazy graph and validation roots into its completion. Descriptive Original
communication sources never grant an ordinary bank an Original role or arena.

Admitted ordinary communication retains that same Work source through its
prepared recovery, resource registry, quarantine destination and housekeeping
registration. Its caller census uses the actual retained array, count-buffer,
group and stream populations, including the selected group's nested metadata.
Completion polling observes each array's own signalled event before applying
strict backing and completed-read checks. It performs no evaluation or wait and
creates no new recovery scope. Escaped scalar agreement and flag results retain the prepaid host owner
until their final alias retires. Boundary header comparisons use fixed readback
buffers sized from the actual expected bytes; an escaped mismatch retains the
same host owner. Bounded-wait refusals also retain that owner while native work
remains quarantined, and independently until the final escaped error retires.
The route's declared tensor limit bounds the bundled completion root vector
before allocation. Native Eval remains part of the enclosing lazy
graph census; host-result buffers and route metadata have separate producers.

Original status agreements retain their scalar arithmetic and native leaf
descriptions with the actual prepared communication source. These descriptions
own native Group handles and paid host metadata, but no input Array, callback,
producing scope or communication-owner backreference. Each admitted vote pays
its own execution metadata and authenticates the same retained group itinerary.
The source keeps preparation-account identity separate from the current
execution payer; lending a funded source changes neither its origin nor its
native authority.

Completed communication readback first observes the source array's own
availability, including a signalled event still attached after a pair wait.
Both ordinary paths use that same nonwaiting worker before immutable backing
inspection. Pending arrays remain refused; observing readiness supplies no
allocation identity or execution authority.

Ordinary paged publication borrows the same authenticated Work channel. Its
companion retains canonical manager sources and prepares independent catalog and
record-metadata destinations from the exact append schedule. Installation checks
the retained manager generation and population before mutation. Original and
ordinary appends share the numerical traversal and canonical publication and
rollback workers; each supplies its own source checks and completion mechanism.
Rollback identities come from the consumed publication program and cannot remove
unrelated existing blocks. Published records retain their own metadata payer.
Synchronous cache-evaluation failures preserve the native cause and prepaid host
custody. A completion wrapper quote alone does not qualify scan, tier-transfer,
visible-window or truncation bookkeeping; those need their own source-owned
prepared destinations.

Ordinary paged Host transfers retain the actual cold store/load itinerary beside
that companion. Each destination has independent metadata funding, immutable
source identity and one-use submission state. Logical cache reservations are
prepared without occupancy and committed only after canonical source and capacity
checks. Admission separately includes each retained Host allocation's actual
capacity and placement, observed constructor controls, and the shared native
copy population. Hot reads and completed promotions use the same scan traversal;
canonical publication and eviction use the same policy workers with ordinary
source proofs. Pending arrays, buffers and events remain in their destination
through failure, and escaped Host aliases retain their allocation metadata payer.
A completed Disk-read promotion attaches its replaced Host and transfer reservation
to the exact staging backings. An independent copied Device destination does not
retain that Host occupancy; a Host-backed Device view does until its final backing
alias retires. The later Device return uses a separately prepared empty reservation
identity. Both publications preserve aggregate usage and peaks, and specific
retirement handles release only the token whose backing has actually retired.
A failed checkout restoration quarantines the complete transfer owner. Disk
itineraries require their own qualified reader, writer and publication sources;
Host transfer support alone cannot admit them.

The ordinary Disk itinerary binds those sources to the same per-manager I/O
worker and canonical write/read transitions. Read destinations retain their
actual ordinary allocator capacities and placements. They activate prepared
logical reservations only for the selected load, then attach the prepaid source
owner to fresh exclusive Host writers before lending bytes to the I/O worker.
The creator context stays outside the Send payload; a closed identity token
joins the completed body back to that context. A successful read moves its
completed body into the promotion witness, so the operation does not retain a
second staging owner. Completed writes retain their durable file and canonical
pin independently from Host staging. Their source witness authenticates every
Host alias released by the itinerary; the replaced logical token remains
attached to the actual backing until its last alias retires. The same selected
destinations supply the finite caller-control envelope before their one-use
read and write descriptors move into execution.

Disk completion, occupancy and retained-failure mutexes include their qualified
platform storage in the Host quotation. Each mutex is initialized before it is
shared, so its allowance covers one platform allocation. Completion errors retain
that funding alongside the actual file, buffer and source-pin owners.

The enclosing ordinary state checkpoint has one host source per actual model
forward. Its cold state hook retains the actual shared layout, outer and nested
slot-table identities, and existing inference request and lease handles. The
accepted text step supplies the only request that may be added to that retained
source. Before copying, the native worker validates those identities against
its borrowed live state and consumes one checkpoint row. Outer layer tables,
hybrid fixed-role tables and retention directories use the current Work's
metadata funding and the existing host-table constructors. Retention copies
keep their payer after their directories and revision, independently of state
field order. This host source grants no native numerical copy authority.

Ordinary paged checkpoints consume a separate destination from that same
retained step source. It captures the actual canonical IDs and tail frontier
before mutation, and the saved pager retains its immutable native tail aliases,
thread-safe metadata owner and prepaid host custody. Restoration authenticates
both current and saved pagers, removes only later append blocks through the
shared rollback worker, and restores the saved tail directly. It does not copy a
partial sealed replacement at the saved token offset. Prepared sliding-history
entries preserve the existing discard predicate; both strong checkpoint aliases
and weak manager entries retain the payer until their own allocations retire.
The neutral `PagedTruncatePlan` also drives ordinary arbitrary cuts in the same
slice, completion, contiguous-copy and publication order. That separate crossing
copy still requires its own numerical source under an admitted scope; checkpoint
custody does not grant it permission.

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

A memory budget configures admission over shared mechanisms. It must not adopt
preexisting unpriced output or retry without enforcement. Accounted capacity,
estimated allowances and unknown overhead have the distinct meanings specified by
the [memory contract](#memory-contract). Arbitrary external callbacks need an
explicit resource and ownership contract.

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
scalar equality cannot replace source authentication. Missing identity, ownership,
geometry or execution evidence remains a typed refusal before the affected
producer. Current workspace admission also rejects incomplete memory coverage;
the pure overhead-policy evaluator does not change that admission behavior.

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

The copied-U32 input producer exposes descriptive layout facts through the same
native shape/count/allocator query used by its borrowed-value plan. Inspection
reads no payload, constructs no tensor or arena, and grants no allocation
capacity. The returned facts retain no source pointers; actual construction
still requires the initialized runtime, exact borrowed values and admitted arena.
CPU static integer-axis indexing uses the existing Slice and Reshape sources.
Its possible reshape copy keeps both destination allowance and the full input
alias identity; erased axis/stride information never establishes output layout
or physical floating precision.

Distributed parameter operations and scheduler exchanges share the prepared
portable-word transport. Its native source, staging, destination and exact
completion retain their admitted physical charges. Neutral coordination borrows
completed words during validation, prepays its framing and decoding vectors,
and retains the metadata payer in escaped typed coordination errors. The same
operation worker performs admission, source preparation, delivery and reversible
publication; logical communication quotas do not authorize physical allocation.
Partition executors forward structured parameter reads and publication traversal
to these shared workers. Borrowed reads and publication preparation preserve the
current observation binding; raw mutable architecture exposure still invalidates
it, as does finalization after successful publication. Read-only parameter inspection
uses the same native guard and recovery worker: a callback error is returned with
preserved state only after successful settlement. Failed or unobservable native
completion still fences execution and retains its source custody.

Parameter publication exchanges a checked generation counter with its prepared
values and retained replacement sources. Snapshot and media bindings retain the
session's existing identity owner plus that generation; overflow rejects during
preparation, and rollback restores the prior counter with the prior values.
Finalization invalidates observation bindings after publication locks are released
and creates no identity allocation. It retains the immutable path source and
clears only the session token. The next explicit cold preparation revalidates
that source and funds a fresh binding before construction. Old fingerprints
remain stale; raw mutable architecture exposure leaves a populated stale token
and cannot request this publication-specific rebind.
The same transaction prepares the complete empty state through the original
source-bound reset constructor. Its host account includes domain vectors, limit
metadata, source pins and retirement controls. Commit swaps the prepared state,
prompt identity, parameter handles, future-loader sources and counters without
constructing or destroying payloads. The displaced state remains in the prepared
slot for rollback or completion-safe retirement. All participating parameter-bank
locks remain held across final validation and exchange.
Prepared native handle clones use a separately paid metadata arena and the
existing closed prepared-handle destructor. Dropping a prepared or published
handle retires its arena charge even while the source backing and other handles
remain alive; failed preparation attaches no payer to that source backing.

Standalone numerical work uses `NumericalSourceRequirements` to reserve its native
backing, native wrapper metadata, finite source-construction population and Host
controls in one ledger transaction. `OriginalNumericalSource` issues separate,
once-only source, native and metadata claims. A native claim authenticates the
accepted backing capacity and placement; the backend supplies mechanism and
completion evidence. Source receipts, native outputs and escaped failures retain
the same account through their final owner. Once all native producers and
submissions close, completion-certified occupancy releases unused allowance and
each independently retired backing under the coordinator lock. Completed-source
inspection aliases retain native identity without retaining allocation authority
or a finished staging buffer's charge. Managed backings retain their full
allowance in every candidate domain until that backing retires; occupancy here
describes live allocated backing, not physical residency. Host controls and live
account ceilings remain charged through their final owners. Invalid retirement,
poisoning and unwinding quarantine the remaining charge. This source grants no
inference or text-generation execution authority.
`SharedOriginalBufferInspection` can retain that non-producing identity across
threads: reference counts and occupancy are atomic, descriptor inspection takes
the native runtime lock, and final custody enters the unlocked retirement queue.
It exposes no producer or Scope binding. Canonical cached parameter cells retain
their exact completed source without retaining a cell, directory or array through
that source, so cache eviction and final array aliases determine backing retirement.
Selected parameter sources complete their actual transfers before a separately
admitted numerical consumer begins. The completed source loan retains its unit,
residency pins and original cleanup accounts throughout that consumer.
Completed parameter results retain their actual native budget and account. The
backend authenticates each full backing against that budget before importing its
capacity, placement and host controls into workspace planning. The ordinary
registered source pin also retains these completed accounts, including when its
source union contains a prepared input. Binding adds no physical registry entry
or second allocation charge. Native publication preserves the same distinction:
completed numerical aliases receive account-only validation in the ordinary
publication transaction. The backend reauthenticates their native owner and
retires transient publication controls outside the coordinator lock. The native
budget already retains the original accounting custody, so repeated publication
adds no backing sidecar. Array-handle controls retain their own metadata payers.
Fixed prompt-cache imports publish authenticated completed numerical backings
through a separate canonical publication plan. The existing directory retains
full allocation identity, capacity, placement and the original numerical account;
native Host-control rows use that account's separate metadata allowance. This
publication reserves only its exact directory and attachment metadata. It neither
charges the payload again nor refunds it: the original native budget observer
retains that responsibility. A move-only receipt attached to the actual backing
marks its directory rows retired after the final physical alias disappears.
Failed attachment disarms that physical-retirement signal and withdraws only
its fresh publication, so surviving directory pins reject the abandoned source.
It neither refunds numerical payload nor invalidates an existing canonical row. Existing registered-copy and capture discovery can therefore pin
imported allocations without retaining a materializer or a second storage bank.

Session publication also retains the exact completed sources of displaced
parameters kept for removal or rollback. Those observation owners authenticate
only their selected roots and do not enlarge the installed workspace source.
Per-domain capacity
and placement-allowance checks preserve the original estimation basis, and every
later admission validates the
retained account under the coordinator lock. Standalone numerical custody cannot
serve as a speculative model or numerical-phase role.

Completed parameter loans retain the successful filled-buffer attachment receipt
when an exported parameter aliases Host storage. The receipt records the exact
native allocation facts and its accounting-only source custody; it holds no
payload and grants no construction authority. Parameter publication selects these
receipts by the actual retained roots, including displaced originals needed for
overlay removal. Later tensor aliases validate the same receipt and ledger before
reusing the backing's existing attachment, so repeated requests do not accumulate
registration owners on an unchanged Host allocation.

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
`eredu-text::tokenizer::ModelCachePolicy` configures stock Hugging Face model
caches through public APIs. The capacity limits entries per model cache; BPE
keeps a separate cache on each participating thread. Zero prevents new entries.
It does not bound process memory, reclaim retired thread-local BPE generations,
or limit the transient allocations made by JSON parsing and model construction.
Configuring an existing tokenizer clones and replaces its model because upstream
exposes no mutable model accessor. Both copies coexist during that operation.
An Eredu tokenizer snapshot records a configured policy; direct mutable access
or an unmanaged import makes that observation unknown. Frozen recipes explicitly
record and restore a policy, using disabled caching for unknown configurations.
The first snapshot or vocabulary view builds an owned, sparse index with stock
public APIs. Snapshots and borrowed `TokenizerVocabulary` views share that index. The facade
prepares it during load so `LoadedTokenizerView` queries allocate no storage. Raw
tokenizer mutation invalidates the wrapper's cached index without affecting older snapshots. Its cold construction
is separate from Eredu's exact fixed-buffer decoder compilation requirements.
Prepared tokenizer construction includes the index in its dependency headroom.
Decoder construction rejects distinct model spellings that share one ID, while
model/added-token aliases retain upstream added-token precedence.

`TokenizerPlan` uses stock deserialization with model caches disabled, followed
by Eredu's fixed-buffer decoder compiler. `TokenizerMemoryEstimate` defaults to
64 KiB plus 128 bytes per serialized source byte for construction and 64 KiB
plus 512 bytes per UTF-8 input byte for encoding. These configurable estimates
include upstream work and output, without claiming allocation bounds. Encoding
retains the complete upstream `Encoding` with its ID slice; private offset and
scratch capacities are not exposed as exact measurements. Runtime retains each
operation reservation, source identity, and completed output or error through
settlement and retirement.

Input-prefix normalization uses the same portable text worker for ordinary and
prepared sources. A changed prepared source owns a full model copy, vocabulary
index, and decoder program while retaining its original source identity. Its
admission estimate charges a full construction footprint for that copy. Identity
normalization aliases the original source. Configuration comparison uses public
serialization plus nonserialized Unigram inference fields and the special-token
splitting flag; this cold check can allocate temporary JSON values.

Runtime serialization funds each output-buffer growth before allocation and
retains its reservation with partial output on failure. Separate upstream scratch
headroom defaults to 64 KiB plus 256 bytes per vocabulary entry and can be set
through `compile_tokenizer_source_for_generation_with_serialization_headroom`.
Token lengths and serializer internals can exceed that estimate. Library-owned
buffer growth, admission concurrency, and source/error lifetimes remain enforced.

`OriginalJsonValue` validates complete JSON with stock serde_json under
`DependencyMemoryPolicy` headroom. A temporary upstream value supplies ordinary
numeric and nested-string validation. Decoded root strings additionally reserve
their first-party immutable destination; both successful results and trailing-input
errors retain the original payer. `parse_with_memory_policy` configures the
dependency estimate independently of that output reservation.

`OriginalJsonTree` retains a stock immutable JSON value and its admission account.
Borrowed nodes and child iterators preserve ordinary duplicate-key, numeric and
object-order semantics; `value()` lends that same value to stock validation APIs.
Its configurable dependency headroom covers upstream construction. A trailing-input
error retains a completed root, while upstream retires incomplete parse temporaries.
Capture receipts use a streaming serde visitor to deliver every field, including
duplicates, to their fixed semantic reader. Borrowed `RawValue` nodes validate
syntax before callbacks and typed scalar deserialization preserves numeric
events even when a consumer enables serde_json `arbitrary_precision`. The
receipt adapter limits nesting to 128; repeated subtree validation costs up to
O(input bytes × depth) without building an owned JSON tree. Default dependency
headroom is admitted before parsing; first-party output buffers, identity/shape/charge validation and
spent-claim custody remain separate. A parse failure does not establish native
completion or authorize another delivery attempt.

The ordinary chat `tojson` filter uses serde_json's public formatter API for
Hugging Face's Python-compatible presentation, including Unicode, key ordering,
separators, indentation and floating-point notation.

Prepared chat compiles and renders with stock MiniJinja using the same source
normalization, Python-compatible methods, JSON filter and request clock as ordinary
chat. Shared range arithmetic uses wide intermediates and limits ranges to 100,000
elements. Slice normalization uses the public parser API to replace expression
slices with a Unicode-aware compatibility filter, preserving omitted negative
bounds and empty slices. MiniJinja is pinned because that public parser API carries
no semver guarantee. Normalization parses once before upstream compilation; it
does not introduce a separate template interpreter. The source adapter and slice
filter allocate temporary storage covered by dependency estimates.

`ChatMemoryEstimate` defaults to 64 KiB plus 128 bytes per serialized source byte
for construction, and 64 KiB plus 128 bytes per serialized request-input byte for
rendering. Each render also reserves two first-party output buffers. Configurable
`ChatRenderLimits` default to 8 MiB of serialized input, input depth 128, 1 MiB per
output variant, 10 million VM instructions per variant and recursion limit 256.
Planning validates borrowed input without executing the template. Rendering runs
each variant once with one clock observation. Failures retain partial output under
the original reservation. Fuel and output limits do not bound intermediate values,
filter allocations or elapsed time; dependency headroom remains an estimate.

Token-trie construction packs tokenizer-derived bytes, validates stock toktrie's
compact representation, then calls its public constructor. Ordered EOS aliases
are preserved; absent EOS uses the upstream invalid-token sentinel, which matches
no valid vocabulary token. `TokenTrieMemoryPolicy` defaults to 64 KiB plus 128 bytes
per lexical payload byte and per vocabulary slot, in addition to first-party
destinations. Multiple EOS aliases require upstream to clone the trie, so both
copies receive headroom. Upstream construction uses infallible allocations and
recursive traversal. A configurable 1,024-byte token-length limit bounds its input
depth; raising it requires adequate caller stack. Prefix geometry validation checks
the actual node and parent-pop limits, including duplicate and shared prefixes.
It does not construct a substitute trie. Runtime exposes matching estimate and
construction methods with this policy and retains source identity and reservation
lifetimes through failure and final alias retirement.

Prepared grammar byte tokenization calls stock toktrie's public UTF-8 fallback,
special-token and marker workers. A scoped fallible encoder stops encoding after
its first failure, retaining completed source-owned encodings and partial scalar
IDs with the error. Dependency scratch receives separate input-derived headroom;
the final fixed ID buffer is admitted before construction. No fork tokenization
visitor or alternate tokenizer implementation is required.

Tagged tool calls use stock JSON parsing and serialization in the shared text
worker. Runtime enforces each semantic parser's total input-byte limit and
reserves exact pending/event buffers separately from dependency headroom.
`DependencyMemoryPolicy` defaults to 64 KiB plus 128 bytes per allowed input byte;
callers can select different fixed and proportional estimates through
`OriginalSemanticChannelParser::prepare_with_dependency_memory`. Each independent
snapshot reserves the same headroom for its own mutable session. Copies preserve
the consumed-input counter and never refund the original account. JSON containers,
parameter indexes and their transient allocations are covered by this estimate,
not by dependency allocation callbacks. Schema preparation reserves separate
input-derived headroom alongside its validator. Syntax, declared types, required
fields, insertion order, and validation failures share the ordinary parser path.

Admitted prompt-cache persistence uses the same manifest parser, shard validator
and atomic generation publication workers as caller-managed inspection. It funds
controlled input buffers, paths, formatting destinations and shared manifest
shells before construction. Opened manifest files are read only through their
admitted extent; the default input limit is 64 MiB, configurable through the
low-level funding constructor. Safetensors headers retain their separate 1 MiB
limit. Stock serde/safetensors containers, catalog validation and platform path
scratch receive separately classified, input-derived `DependencyMemoryPolicy`
estimates. Each allowance is admitted through the same ledger transaction with
its source and basis; it is not an exact dependency or process-memory bound.
Returned shared manifests and typed errors retain both accounts until their final
alias retires. Reversible publication prepares rollback metadata before changing
visibility. These persistence allowances do not apply `MemoryOverheadPolicy` to
inference admission or replace file identity, schema, digest or completion checks.

Imported decoder layer and fixed-role tables are constructed by the ledger's
closed host-slot builder. It admits their exact boxed extent before allocation
and publishes the actual table identity in the existing storage directory. The
persistence construction account pays and retains metadata separately; table
payload is charged once and stays attached through its final table owner. The
next request's opening census and source pins consume these same canonical rows,
including nested hybrid fixed-role tables, without inferring custody from a
metadata byte allowance.

Imported cache blocks retain a distinct persistent-file source: the opened file,
its observed version, actual encoded header and offsets, verified payload digest,
and both metadata accounts. This source neither owns a live-writer Disk reservation
nor unlinks the persistent shard. Canonical cache reads select the retained Live or
Persistent origin and use the same positional reader; persistent reads validate
the header and digest before publication. File-to-file saves use that reader's
fixed 4 KiB buffer and authenticate the completed source version, without a
tensor-sized Host staging allocation. Failed reads retain their source handle,
written prefix and funding until retirement. Paged projection retains the same
closed file source through cold planning, promotion and backed Device return;
append validation compares that exact source with the canonical row. Persistent
imports use their authenticated shard layout without acquiring live-writer
reservation or unlink authority. Their selected loads prepare and fund the shared
Disk read worker and Host staging even when the manager disables live writes;
write destinations remain conditional on that separate policy.

Fixed-state imports authenticate a single declared tensor on the same opened file,
then use the retained executable, ledger and prepared allocator through the shared
native numerical constructor and completion worker. File bytes and decoded typed
Host values each have an independent storage account; both retire after completed
construction while native aliases retain their own physical allowance. Exact
completed allocation witnesses publish payload and native Host-control identities
into the canonical source directory without charging those bytes again. The
backing owns the directory receipts, so ordinary capture and copy discovery remain
valid after the importer retires and aliases release their registrations only with
the actual backing. No state-specific source bank or opaque array attachment is
used as a substitute for canonical identity.
The qualified constructor source covers Float32, Float16, Bfloat16, Int32 and
Uint32. A persisted Float64 state currently returns a typed missing-constructor
source refusal; it is not represented as Float32 or treated as a hardware limit.

Facade tool schemas compile with stock `jsonschema::Validator` and validate the
runtime's retained `serde_json::Value` directly. The preparation account admits
configurable dependency headroom before compilation; each validation invocation
admits separate parsing and validation headroom. Source aliases and failures
retain their original account. Facade-owned names, rows and grammar text use
prospective buffer reservations without dependency allocator callbacks.

`ControllerDeclarationData::admission_bytes` supplies the fixed planning allowance
for opaque compiled declarations. Their `owned_capacity_bytes` remains unknown;
exact filter and byte-buffer capacities retain their existing contracts. Runtime
deduplicates admission by the same immutable source identity and retains its
charge until final retirement. This allowance is not a measured heap inventory
or a dependency memory ceiling.

Stock llguidance owns compilation, lexer/table growth, forcing and token parsing.
The facade retains a private unstarted `TokenParser` as its closed declaration
template because the public factory has no token-parser constructor from an
already compiled grammar. The template is never advanced or exported. Startup
and snapshots use `deep_clone`; upstream shares compiled grammar data while each
session owns its mutable parser state. Original-source validation binds the
template to the exact recipe, tokenizer and compilation receipt before startup.

Each live session has a separate tokenizer diagnostic slot and destination
account. Its public token-level environment performs fallible original encoding;
upstream Earley and slicer environments borrow only the same immutable trie.
Cold compilation serializes access to its own diagnostic slot. Session creation,
copying and operations admit dependency estimates. The template estimate uses
serialized grammar size and packed vocabulary-mask width; copies add an estimate
for their accumulated output bytes. Shared trie and slicer storage stays with
its original source allowance. `LoadedModel::prepare_chat_with_grammar_memory`
selects the managed compiler, schema-validation and grammar-session headroom;
snapshots and activation tokenization retain the selected policy under their
own accounts. Ordinary `prepare_chat` uses 64 KiB plus 128 bytes per logical
input byte. These settings do not change tokenizer/trie, template-rendering,
native or first-party buffer admission. Upstream parser limits bound grammar
size, lexer states, row size and per-step fuel. These are resource controls
and planning allowances, not allocator interception or exact dependency capacities.
Eredu-owned history, source controls and masks retain their ordinary admission and
lifetime checks. A failed operation owns its parser and sources until diagnostic
retirement. Completion-schema validation preserves the selected draft and typed
errors; funding or tokenization refusals cannot select the syntax-only fallback.

Framework-managed bounds do not describe total process or system memory. Application
buffers, independently copied events and opaque driver/JIT internals need their
own declared policy. Registered MLX backing remains charged while its allocator
cache retains it. [Bounded inference](bounded-inference.md) defines the public
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

Retained sampling input plans preserve the source row's physical representation,
backing envelope and allocation population. Sampling revisions reconstruct that
same descriptor under their metadata account; unknown layouts remain unknown.
The backend consumes these retained facts when pricing changed sampling settings
for a live or restored controlled session.
CPU optional token filters price the same expanded Boolean mask, scalar and
seven native constructors as a required filter, while retaining the possible
full input alias when all tokens are allowed. Both paths preserve known F32
row layout; missing source evidence still prevents bounded admission.
CPU repetition, frequency and presence penalties compose their selected binary
and Select workers, with separate Boolean, F32 row and scalar backing capacities.
Their host estimate includes both vocabulary buffers and the selected history
suffix. A zero-length history window preserves the configured native graph;
it does not remove its constructor or completion records.

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
floating stride evidence. Floating squeeze and expand-dimension views preserve
the exact nonunit source strides across their changed rank, including sliced
rows. Their existing native alias workers validate the readable source span and
retain the complete backing without another allocation. Transpose facts require the canonical retained axis
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

Prepared spatial rotary on CPU uses the existing native rotary worker and its
selected row layout. Its census includes I32/U32-to-I64 widening, signed
saturation, strided column aliases, eight-byte integer intermediates, supplied
frequency seeds and Copy aliases, concatenations, F32 cosine/sine and both output
roots. The native alias and concatenate consumers authenticate the same I64
backing spans and byte widths as the unchanged copy worker. Original row storage
uses its inspected Graph bank; ordinary row storage retains the selected local
container capacity. These facts do not authorize unprepared frequency generation.
The shared CPU trigonometric worker checks its reduced argument before using
the SIMD polynomial. Lanes outside that range use the platform scalar sine or
cosine with the original value; the source includes those fixed call frames.

Caller-provided rotary embeddings use the shared explicit application worker.
Its CPU source covers full-width F32 split halves with dense rank-two or
rank-three cosine/sine tables and exact batch, position and feature broadcasting.
Input head views may retain gaps between rows when their final axis is proved
contiguous. The census includes the actual batch/head views, dtype casts, half
slices, negative scalar seed, concatenate, products and sum. The tables remain
independent source roots; their geometry does not grant frequency-construction
authority.

CPU equality with an I32 scalar retains the eager signed scalar source and the
native integer promotion. U32 inputs and that scalar widen to I64 before the
Bool comparison, with both eight-byte cast destinations included in the
allowance. I32 inputs keep their dtype. Boolean unions use the existing logical
worker; none of these operations converts integer identifiers through floating
point storage.

CPU biased tensor-linear operations retain their distinct AddMM source. The
selected F32 tiled path accounts for the real bias copy and matrix tasks, their
three-input cleanup, weak copy owners, and one possible output backing. The
frontend quote includes weight transpose, bias broadcast, and any batch flatten
and restore views. Native evaluation validates the exact AddMM primitive,
immutable CPU selection, matrix interiors and readable bias span; constructed
linear modules retain their separate Matmul-plus-Add sequence.

Borrowed F32 and I32 tensor slices retain their exact host constructor in the
neutral trace. MLX realizes it as one eager backing and a stream-bound Copy
alias. CPU admission prices the actual alias evaluation and its cleanup without
a numerical task or second payload; native evaluation checks the same shape,
dtype and readable source extent, including empty aliases. Generic scalar-fill
initialization remains a separate operation and requires its own source facts.
Signed and unsigned integer fills retain their typed scalar constructor and the
shared Broadcast/Full worker, including the eager seed backing. Prepared spatial
rotary execution preserves widened I64 coordinates through slicing and reshape;
its CPU source uses the actual column strides and borrowed reshape planner before
the existing arithmetic, concatenation and trigonometric workers.
Ordinary text prompt preparation has its own final-shape U32 upload descriptor:
the native worker creates one eager backing without a lazy Copy, while the
report separately retains the caller's actual host capacity and text identity
controls. This descriptive upload quote does not supply original token-input
authority or waive the ordinary request's reservation and source checks.

Completed parameter replacements remain explicit sources when a bounded policy
loads a later unit. The layerwise snapshot retains their immutable owner and
copies descriptor facts under the funded preparation context. Its borrowed rows
perform no native calls. Retained rows require the same actual backing installed
in the parameter table; native allocation generations join aliases with static
parameters, including full capacities and physical placement. Prospective lease
destinations remain separate materialization contributions. The existing native
binding worker chooses one fallible handle per row, and invocation checks reject
a different replacement owner before any unit slot is populated.

An escaping standalone numerical replacement keeps its completed arena and
numerical account. Publication selects those sources by authenticated native
budget, allocation generation, full capacity and placement, then exchanges them
alongside the immutable parameter values. Workspace bindings combine these
completed roots with existing registry pins and, where applicable, prepared media
roots in one checked selection. The source account remains in the same retained
registration through residual quotation and copying; it creates neither another
payload charge nor a registry entry for an arena-backed root. Restoring original
parameters restores their source subset, allowing unrelated replacement arenas
to retire. Standalone numerical custody grants no speculative request role.

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

CPU affine projections use the ordinary packed QMM worker. The native storage
query and evaluation guard check transposed rank-two U32 weights, matching
floating scales and affine biases, contiguous source spans, packing divisibility
and signed loop limits. Positive rank-two through rank-four inputs retain their
leading geometry. The quote includes promotion casts, the worker's output birth
and five weak array captures, fixed unpacking controls, and optional output-bias
broadcast and addition. It does not materialize a dequantized weight matrix.
Floating parameter replacements use the shared dense projection planner; their
retained packed companions do not participate in arithmetic or precision promotion.
Layerwise packed U32 weights use the existing typed General CPU host-copy worker;
the copy recipe includes them alongside floating companion tensors and authenticates
the retained host allocation before loading.

CPU I32 addition, subtraction and multiplication use the existing typed binary
worker, including shape-checked broadcasts. Integer layouts retain their logical
dtype without floating representation evidence. Changed operand shapes contribute
their actual Broadcast alias producers before the binary task; native evaluation
still checks strides, physical spans and signed loop limits. The logical peer
dependency keeps its typed zero, multiply and add sequence unchanged.
Checked CPU axis-zero gathers retain the selected complete trailing slice at
source ranks one through four. Their quote uses the existing signed-index
validation, native Gather and Squeeze sources, including all trailing dimensions
and the native worker's combined-rank limit.

Ungrouped learned-offset RMS normalization uses the shared CPU RMS recipe. The
explicit F32 offset promotes the learned gain before normalization, so the recipe
counts that cast/add sequence, its scalar source and vector backing, and the F32
result. Width-one means retain their actual identity-cast reduction. Grouped RMS
keeps its distinct reshape and final input-precision cast sequence. Its caller
source follows the actual cast, grouped reshape, weightless RMS, output reshape
and optional gain operations. Positive Metal rows select the F32 row kernel;
CPU and empty rows select the arithmetic fallback. Shape vectors and returned
handles are included independently of numerical backing.

CPU LayerNorm follows the same native fallback as ordinary execution. Its source
retains both F32 mean reductions, centering and variance arithmetic, actual
optional affine casts, and the final selected precision. Eager mean divisors,
epsilon and absent affine defaults have separate backing births. Every primitive
uses the selected stream, and the named fallback reports its own control frames.
Unknown operand precision or unsupported row geometry remains an identified
missing numerical source.

The explicit CPU attention recipe follows the shared score-policy worker:
K/V head expansion, score-precision casts and products, optional soft-capping,
Boolean or additive masks, optional sink logits, F32 softmax, and the final
query/value-precision product. It uses retained scalar and stride facts for
reshape and matrix-copy admission. Its source population is distinct from the
native fused SDPA fallback and from bounded query/key recurrence.

The fused CPU SDPA fallback accepts proved contiguous final-axis rows with gaps
between them. Its query-scaling product creates a dense input for the first
matrix product; a key or value that retains its gapped layout contributes the
existing matrix worker's compaction source. A precision-changing cast already
creates a dense operand and does not require that additional copy. Missing row
evidence remains an unavailable source.

Sliding-window CPU planning composes each actual query tile from three source
views, the shared causal-mask recipe, and its selected attention recipe. The
completion retains the exact chunk-table capacity and joins the final transpose
and reshape, including any physical copy required by the head/token layout.
Cache offsets and retained source strides determine each tile's key geometry.

The CPU tanh-GELU recipe preserves the input-precision cube, the native scalar
square root, and the eager F32 scalar operations that produce its F32 result.
Unbounded gated products compose the selected activation recipe with the shared
multiplication recipe using the activation's resulting precision. SiLU and sigmoid
keep their F32 internal arithmetic and final F16/BF16 cast.

CPU scalar multiplication composes its actual eager F32 scalar with the same
binary worker used by tensor multiplication. Complete strided rows retain their
readable backing geometry through the native source checks. F16 and BF16 operands
use the quoted promotion copy before the F32 result; scalar and output storage
remain separate births.

The neutral `zeros_like` trace delegates to the existing typed zero constructor,
using the prototype's authenticated scalar precision and shape without claiming
its values as graph inputs. Generic host initialization remains a separate
contract. CPU empty reshape and squeeze use their actual shared-buffer workers:
they produce no task, Data owner or physical backing, and retain the complete
input owner even when a zero-element view holds a nonempty allocation. Exact
shape, inferred-dimension and singleton-axis checks still precede admission.
CPU transpose preserves exact input strides through its selected permutation,
including slices with gaps between rows. A reshape that only inserts or removes
unit axes retains its nonunit steps after the native planner proves an alias.
Singleton coordinates carry canonical strides. A later noncontiguous reshape
that removes a leading singleton prices both the existing alias and General-copy branches,
retains either backing owner, and preserves only their common scalar fact.
These alias facts do not establish a dense layout or replace complete backing
custody with the selected region's byte count.

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

Contiguous CPU reshape uses the native source's checked arbitrary-rank layout
witness. Its out-of-line shape and stride controls are priced by that source;
the view retains the complete backing allocation without another storage charge.
This witness requires proven row contiguity and does not extend the separate
fixed-rank limits of Broadcast, strided views or other native workers.

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
The architecture workspace trace selects the final sampling row with an exact
sequence interval followed by removal of its singleton axis. The native text
output adapter uses the same slice and squeeze operations, preserving the
selected values and backing ownership while exposing their separate CPU facts.

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

Independent autoregressive and Embedded model capture share the same neutral
request-owned cumulative ledger and funded host collector. Each retains its
own genuine occurrence role. An autoregressive role admits the sum of its exact
recorded prefill-frame plans, including each physical width and global window,
then constructs an owned frame bank before native execution. The same source
readout predicate permits a Sequence output demand only when the selected
prefill capture or edit requires it; all other role geometry remains exact.
The bank issues
frames in the same order as actual prefill-span claims; sequence decode consumes
one frame. Cloning a role, restoring state, or abandoning a callback does not
restore a spent frame or refund capture usage. Frame outputs and typed failures
retain the accepted account through their final aliases. These host contracts
supply no native observer, scope, numerical source, or completion authority.

Independent partition equations borrow the selected constructor's exact local
state and retained communication declaration through the same partition worker
and collective recorder as ordinary text. Every recorded prefill span, or the
single sequence-decode span, retains its own parallel invocation. The admitted
AR completion lends that invocation and its actual root owner to the shared
native callbacks. A lexical session binding removes the loan on success, error
or unwind, including when execution fences the session; teardown does not claim
completion, restore a spent occurrence or release pending recovery custody.
Each retained model row quotes its actual expert integer publications and admits
those constructors as a distinct component of its source program. That bank moves
once into the same model control owner, which authenticates the accepted source
account before lending a constructor. Sampling, parameter materialization and
cache construction retain their separate banks and lifetimes.

Embedded target and prediction quotes consume the retained partition constructor,
its communication declaration, and the selected rank-local state. Target-only
construction preserves the extension's retained placement without materializing
prediction modules. Prediction quotes apply the same typed target operations and
prediction equation worker as execution. Target hidden capture remains independent
of vocabulary demand and follows the ordinary partition capture/publication path.
Request-funded reconstruction reuses the exact load-time store-validation witness;
a symbolic communication description or metadata account cannot replace it.

Independent-target discovery projects the ordinary architecture descriptor and
target hook support without requiring a prediction-extension catalog. It omits
prediction-only scopes. Execution revalidates those same borrowed declarations
against the actual selected autoregressive schedule, source identity and role;
the descriptive discovery report grants no invocation authority. Partitioned target revalidation borrows the actual loaded layout and collector phase report, checks their artifact and execution identity against the selected source, and uses the same declaration validator. An ordinary single-source report cannot qualify that route.

Before/after intervention fragment plans retain their exact companion storage identity, operation ordinal, side, physical shape and logical window. The model frame admits both destinations with its existing role, consumes the pair once, and keeps that role account alive through the final Host alias. A text prefill causal selection is required only for the text traversal; explicit model invocation evidence uses its retained invocation coordinates. The backend evidence companion retains the loaded component coordinate maps and sealed per-side scalar facts, then binds the consumed Model Hosts through the same two-side program constructor used by Text. Source-owned metadata queries cover those actual map copies, selected row tables and label copies. Borrowed receipt descriptors supply the later protocol census without constructing another receipt or admission.

The funded partition receipt worker retains explicit invocation axes through
native geometry, decoding and the final assembly charge. Its final destination
must belong to the same source, phase, prediction and invocation. Owned fragment
quotations retain their immutable receipt and are consumed once by the admitted
model frame; the shared destination constructor preserves the actual Text or
Model account custody. These quotations do not grant transport or native work.
The shared native frame constructor borrows the retained loaded partition
publication and consumes those same funded fragment slots. Its immutable
protocol census comes from the receipt exchange's checked payload width and
fixed frame kinds. Source-only transport demands retain each scheduled source
vote, the enclosing coordination checks, receipt writer, and each rank's byte
decoder scratch, including inactive ranks. Native gather quotation uses the
actual retained Group, copied-U32 input layout, completion source and the same
caller-control queries as execution; it constructs no placeholder tensor.
Single and batched independent requests retain the target session's exact
partition publication and communication source before entering its model
operation. Each target role borrows that retained pair under its own admitted
budget. Draft roles keep their separate source identity and cannot borrow the
target's transport by default. The local autoregressive unit bank accepts a
partitioned execution only when its actual equation recipe retains one common
communication source across every span. The same local policy and selected
stream construct the unit bank; the communication worker separately validates
the retained group and active scope. Publication settles its intermediate leaf and
closes the model construction bank. Autoregressive row readout retains its
actual Index trace; captured prefill retains the exact final-row Index operation
from its observed model span, using the local output on its producer and the
received publication on other ranks. A selected publication quotes an independent
construction bank from those source facts before converting cumulative extents
to Graph capacity. The same role and observer open that bank once, retain it
through failure or recovery, and use the existing merged final completion.
Explicit physical axes and logical row windows remain separate. Receipt context
retains both the logical invocation and the exact physical interval; canonical
identity and decoding distinguish equal-width windows with different starts.
The existing window geometry worker intersects the selection before spatial
producer projection. Complete and partial windows then use the same funded
fragment destinations, assembly, delivery votes and evidence publication.
Final destinations authenticate the original source, physical axes and window,
including on failure, and retain the accepted account until the last owner drops.

The runtime's prospective partition constructor queries use the same fixed
controls, exact UTF-8 label formatter and selected-row allocation workers as
execution. Projected rows share their actual table; routed rows retain the
additional hook table. Epoch-label queries describe the same run-name formatter
without binding an epoch. These queries reserve no capacity and grant no source
or execution authority. Receipt preparation, protocol callbacks and delivery
remain separate contributions to the accepted model role. Receipt execution
queries retain the actual complete, contiguous or routed source mode. They
compose reconstruction, evidence, allowance and the same closed encoder/decoder
workers, including the existing parser estimate at the receipt's maximum wire
size. The program separately quotes its once-only evidence table and fixed entry
controls. Source-labelled queries neither fabricate a context nor execute a
worker to estimate its storage.

The shared text and media session workers accept an explicit observer alongside
their existing completion callback. Observation failure and completion failure
use the same transaction rollback. Independent-decoder orchestration carries
the collector's scheduler origin, delivery and snapshot authority through its
existing prefill and cached-sequence invocation workers. A cold prospective
capture description retains the source's funded metadata without beginning an
invocation or spending capture usage. Its physical prefill span must match the
recorded input range, cache position and output demand; the native adapter must
separately authenticate the genuine model role and completion resources.

Raw capture/intervention declarations are admitted after actual request geometry
is known. Core's intervention validator and copy producer are shared by ordinary
and funded admission. Runtime's paid support projection copies requested targets;
architectures authenticate family points and retained partition placement. A
backend supplies mechanism facts rather than semantic branch logic.

Partition intervention transcripts retain the exact physical model invocation and
its optional logical row window alongside the source-owned component projection.
Equal physical widths at different logical offsets have distinct source identities.
The local allowance validates those coordinates against the original frame claim
before spending edit credits; the final outcome revalidates them on return.
Descriptor metadata queries describe the same paid copy worker and grant neither
a claim nor native execution authority.

The intervention claim bank retains two activation evidence rows or four routing
rows: expert IDs and coefficients before and after the operation. Each field
has its own ordered, nonrefundable claim within the same account. Unsigned IDs
use a fixed U64 host destination; floating coefficients use F32. Native source
types must match those declared categories, and host quotation uses the actual
element width. Prefill keeps these targets across every canonical chunk, including
chunks that contribute no values to a bounded Preview. Missing or abandoned hooks
prevent successful delivery. Logical completion does not certify native completion.

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

Capture preparation resolves content identity under its source metadata account
for both raw and precompiled declarations, sharing the architecture's identity
worker with partition construction. The facade reads that retained identity from
the prepared chat session. Controlled capture validates it before
emitting `Started`; record construction does not resolve or reopen artifacts.

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

Synchronized public reset passes a move-only core claim for the current session
to the backend. It adds no per-operation ceiling; existing resource-domain and
live-account ceilings still apply. MLX sessions with quoted resident KV or hybrid
state use the same source-bound reset constructor and publication as explicit
admitted reset. Tokenizer sources and completed outputs retain their independent
charges across reset. Unquoted native sessions retain their existing allocation
authority; a reset does not convert or revoke that authority. Embedded prediction
state has no whole-state admitted reset profile and retains its ordinary reset
path.

Native-state discovery and atomic exchange are neutral primitive contracts.
Allocating snapshot and copy operations require the complete prepared continuation
source and its host/native reservations, including when configured limits are
unlimited. The primitive state interface has no unadmitted allocating copy hook.

Saved-array and pending-input copy admission selects native resource requirements
from the retained execution stream. CPU copies use the shared CPU copy worker
without requiring the Metal feature, with no GPU pipeline lookups. Source
completion, physical-buffer admission and destination custody remain required.

An empty paged state still owns a cache manager. Autoregressive startup and
state copying lend the inspected KV or grouped source to the same independently
admitted paged-manager and tail-copy worker, even when there are no array
operands. Grouped copies preserve every parent and fixed-role child table;
present fixed values use the same completed numerical copy worker and remain
under its custody. Saved grouped sources retain their own exact manager and
table identities. The host-only empty-table constructor remains limited to
sources with no additional manager construction.

Autoregressive paged execution retains those actual manager sources beside its
invocation recipe. It prepares the same canonical catalogs, append programs and
Host itinerary, then binds them to the admitted speculative account and exact
span plan. Each claimed prefill span, or the sole decode invocation row,
authenticates both that account and the current native scope before lending the
source bank. The existing recovery owner retains the bank through completion or
failure; weak lookup storage does not own it or authorize another occurrence.
The per-span source-facts table retains its original construction-metadata payer
when moved into the role. Admission reserves its additional source-construction
allowances without charging that already paid table backing again.

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

For replicated CPU affine loading, the source-prepared residency manager carries
completed conversions in selected format-group order. Native construction consumes
and validates each conversion before adopting its overlay; the manager refuses
adoption if a conversion remains unconsumed. Conversion eligibility checks the
complete encoded source/recipe selection and native accounting domain before
creating native resources. Once
conversion starts, resource, payload, or residency errors propagate with their
typed causes; they cannot trigger ordinary requantization. Parallel, auxiliary,
computational-recipe and other-device/format selections use ordinary construction
when this producer cannot prepare the complete selection. Both routes retain the
shared exact-task planner and bounded tile driver. Host-copy snapshots validate
the actual storage kind against the destination: CPU transfer buffers are
accepted for CPU workers, while MetalShared transfer storage retains its native
copy qualification. Foreground disk rows use the allocator's exact single-buffer
request geometry, including the CPU allocation header, for their retained copy
capacity.

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
`EncodedRecipeReadView` borrows an admitted read's exact output and source records
while its typed custody remains with the owning read. Reads with different custody
types use the same detached-source constructor. Detachment copies funded metadata
and retains authenticated backing identities; shape or content equality does not
establish source identity. Detached read failures retain their own constructor
custody after the detached source retires.
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
Finite recipe inference can retain its inspected recipe/catalog plan and move
its actual output metadata to the caller without cloning. The tile metadata
constructor admits that worker's storage and native dimension array before
inference, retains the original account with the completed metadata, and keeps
inferred metadata and any converted shape prefix with typed construction errors.
Its bound covers recipe output metadata and native dimensions; source/header
admission, encoded-read range construction and native payloads remain separate.
SafeTensors selection reads and encoded recipe projections share one counted
range-destination plan. It borrows the validated selection and shapes, counts
coalesced ranges without cloning indices, and fills only an exactly sized caller
destination. Encoded projection allocates that destination after counting, so
range storage follows the selected geometry rather than tensor payload size.
The plan itself supplies no reservation: read-batch metadata, mapping arrays,
source/header retention and their construction lifetimes still need admission.
Per-source projected spans also use a borrowed count/fill plan over the actual
read intersections. File and memory batches allocate exact span destinations,
retain their original source owners and restore physical source ordering after
fill. Counting validates disjoint logical source coordinates and checked span
layouts; filling refuses wrong destination lengths before writes. This geometry
plan does not reserve storage or cover source-batch and recipe-mapping creation.
Recursive byte-preserving recipe mappings likewise bind source, selection or
interleaved-child coordinates in a counted destination plan. Counting and filling
share the same adjacency coalescing and slice traversal. Child mappings remain
borrowed during output construction; exact range storage replaces geometric
growth, with an additional geometry pass. Runtime implements its shared initializer directly for the checkpoint mapping
plan, comparing before the exact requested range-vector allocation. Source,
selection and interleaving use this same worker in ordinary compilation and
admitted construction. Completed mappings retain their original account after
borrowed children retire; rejected construction leaves existing child accounts
intact. Construction retains typed allocation errors and publishes no native
alias. Child-reference arrays remain borrowed prerequisites. This mapping
admission does not cover source metadata, inference, projected read spans or
child-array construction, and does not establish whole-compiler admission.
Selection-range destinations likewise use the shared physical-read geometry
worker. Public encoded planning validates into caller-owned shape scratch before
counting; compiler-internal planning uses its already inferred geometry. Both
construct the same exact requested range vector. Runtime implements the existing
shared initializer directly for that destination plan, preserving reserve errors
and keeping the owned ranges charged after input shapes and selections retire.
Mapping construction can borrow these admitted ranges in the same pool. Shape
scratch, child arrays, projected source spans and whole-compiler construction
retain separate admission.
Encoded recipe reads share one source-key traversal for structural qualification,
counting and construction. The ordered occurrence list preserves duplicates and
uses exact requested vector/string capacities. Runtime implements its existing
shared initializer directly for the checkpoint-owned plan, comparing before
construction and retaining custody with the move-only keys or failed prefix.
Unsupported computational recipes invoke no source read constructor. Geometry
inference, source-batch admission, projection storage and payloads remain separate;
the key plan is not a whole-recipe or native-conversion admission grant.
Contiguous and projected encoded reads share a metadata catalog constructor.
Its borrowed plan sizes one exact requested index allocation from all source
occurrences, sorts indices in place and retains the last duplicate name. Ordinary
construction supplies unit custody; runtime implements its existing shared
initializer directly for the same plan and compares before allocation. The
move-only catalog retains caller custody until its index retires, and reserve
failure retains custody without a completed allocation prefix. Tensor metadata
and source custody remain borrowed prerequisites. Owning lookups clone their
selected records under separate policy; borrowed lookup allocates nothing and
supports finite inference over the same retained entries. This index admission
covers neither metadata inference nor complete encoded recipe compilation.
Runtime implements shared initialization directly for the checkpoint-owned
memory and SafeTensors read plans. Callers inspect the plan through its existing
source constructors, then compare and construct through the pool; there is no
second initializer object. Source visibility and prepared-header requirements
are enforced by the checkpoint plans before admission.

Encoded reads use the same finite inference worker when operating on a borrowed
batch catalog. Persistent source-cache selection remains explicit; cached reads
retain their existing cache policy. Projected compilation uses one checkpoint
traversal with a construction policy for inferred metadata, selection ranges,
byte mappings and a single array of child owners and chunk widths. Ordinary
construction and runtime admission execute those same plans. Source metadata is
borrowed without temporary shape or dtype clones. Batch-catalog compilation binds
inference to its actual source occurrences and rejects inconsistent duplicate
metadata before traversal.

Runtime's construction policy admits each actual plan in the original pool and
retains each account alongside its output or typed constructor error. Child-array
ownership transfers with its original account before recursive population; the
array cannot grow past the planned child count. Temporary children and inference
scratch retire synchronously. No native work or independent alias escapes.
Completed metadata and mappings retain their accounts after recipe/catalog
inputs retire. Recipe validation precedes byte-read support decisions; numerical
transforms still require numerical execution. Fixed selection failures preserve
their typed causes without allocating a stringified store error. Machine stack,
source birth, recipe declarations, subsequent read scratch/output and complete
producer construction have separate limits and admission.

Runtime can consume an initialized encoded batch to compile and assemble one
recipe-read owner. It validates the original pool, admits a batch catalog,
invokes the shared recursive compiler, admits projected spans and moves the
inferred metadata with its account into the completed read. No successful
metadata or source record is cloned. A catalog failure retires borrowed scratch
locally while retaining its diagnostic account; projection failures keep actual
constructed prefixes. Numerical recipes return no encoded read and release the
consumed batch without ordinary preparation.

EncodedRecipeRead carries caller custody after its metadata and read records;
cloning is available only when that custody permits it. Existing ordinary reads
use unit custody. Assembly checks the inferred byte length and retains both
owners on disagreement. The native encoded-input constructor and affine cold
submission accept this same read type with either custody, using the same
borrowed read worker and source checks. Native input aliases retain their own
input account after compiler/read storage retires. Runtime and source birth,
recipe declarations, native shape storage, producer scheduling and prepared
manager handoff remain separate construction responsibilities.

File and memory encoded-read constructors return one move-only prepared read
owner. Memory reads retain their direct, scratch-free copy worker; file reads
retain original file validation and diagnostics through the borrowed worker.
Runtime can move a completed read out of its initialization result only by
transferring that result's actual account into the read's custody. No generic
unfunded output extraction is exposed.

Projected reads consume this exact owner. Planning validates coverage and counts
new spans without allocation or payload reads. Construction reserves each exact
span destination and shares the ordinary projection worker, retaining replaced
and untouched source records with both the original and projection custody on
failure. Ordinary recipe compilation uses the same constructor. Source handles
and tensor metadata are moved, not cloned; errors retain typed geometry or
allocator causes. Runtime compares before construction and preserves the read
inside a rejected plan. Completed projections remain charged after source stores
and mappings retire. Keys, recipe/overlay storage, later read scratch and full
producer construction retain separate admission.

Memory-backed encoded reads inspect their actual immutable store and ordered
keys before constructing metadata records and source spans. Ordinary reads and
the runtime-admitted constructor share that worker. Runtime compares the original
constructor contribution with the cold pool before allocation; completed batches
and failed reserve prefixes retain its custody after all metadata and source
handles. The move-only result reads into caller-owned storage without staging or
read scratch. Existing source payloads and keys retain separate admission; this
constructor does not establish file-header, recipe-compiler or whole-model fit.
Admitted memory-read plans can also select through retained built-in source
views. Selection authenticates each concrete owner, preserves restricted/resolved
visibility and the composite single-child batch rule, and retains the selected
memory store before comparison. Prepared views require the concrete fixed-catalog
promise, including off-route composite dependencies. Unsupported or forwarded
owner identities return no plan without invoking ordinary read construction.
The selected plan needs no routing callbacks during construction; rejected plans
retain their source until retirement, and completed batches retain the original
payload handles after the surrounding views and catalog store retire.
SafeTensors discovery retains configurable encoded index and header limits.
The default index limit is 100 MB; the header limit defaults to the reader's
existing 100 MB ceiling and can be lowered. Oversized index files are rejected
before reading their contents, and a bounded read also detects growth after the
length check. Header limits are checked before allocating or reading the JSON
body and remain attached to shared shard admissions. Indexed stores still read
headers only on demand; strict discovery validates every shard. These are input
limits, separate from decoded-metadata estimates and lifetime admission.
Shard-set clones share one immutable catalog containing canonical paths, tensor
locations and admission handles. A shared clone allocates no new path maps or
strings. Owned path export moves the catalog's vector when uniquely owned and
copies it when aliases remain. Header admissions and leased file/payload owners
keep their own lifetimes after the last catalog map owner retires.
SafeTensors discovery and the weight store can inject a neutral admission policy
for header creation.
It receives the checked JSON length, byte-buffer request and path length before
the body is allocated or read. Accepted custody stays with the shared header;
parse, index-consistency and geometry failures retain it in the cached error.
Error clones share that failure and its custody, preserving the typed source.
Concurrent initialization invokes admission once per shard. Header limits and
file-identity checks still run, and ordinary opening uses the same reader.
The reservation completion callback runs once after header construction,
before publishing its result. It ends active construction while retaining bytes;
completion errors retain any completed metadata and preserve a simultaneous
construction error as the primary source.
Runtime supplies header admission through the memory ledger.
Its quote separates the encoded buffer, configurable metadata/dependency
headroom and qualified fixed reservation/error controls. The metadata portion is
an input-derived estimate, not a dependency-wide or process-wide ceiling.
A policy has a finite lifetime header-initialization allowance, sized for the
source inventory. Each attempted initialization consumes one slot, including a
refusal; aliases of an initialized shard consume none. Fixed policy controls and
refusal wrappers are prepaid, and calls beyond the allowance share one retained
limit error. Accepted byte reservations remain through the last header/error
alias, while active-construction exclusion ends after parsing and validation.
Policy and failure custody use weak pool references to avoid registry cycles.
This header policy does not reserve discovery/index/catalog storage, later
metadata clones, caches or read payloads; those owners need separate admission.

Runtime also provides original SafeTensors source opening under the existing
pool. It reserves initial policy controls and path/discovery headroom before
filesystem discovery, then extends the same contribution for the measured index
buffer and decoded-index estimate before reading. The retained index handle is
checked against its observed version before and after the exact-extent read;
a stack-owned byte probes growth without expanding the buffer. The index is
decoded once, and its distinct shard count sizes header admission before any
header is prepared. A separate estimate over borrowed tensor names and paths
precedes store-map and fixed diagnostic construction. Indexed headers stay lazy.
Source construction ends its active exclusion when opening returns, including
failures. Shared catalogs, independent file identities, cache controls and read
telemetry retain the source contribution; detached file identities preserve
that custody too. Incremental admission failures retain the account in their
shared typed cause, so cloning a refusal cannot refund its diagnostic storage.
Path, decoded-index and catalog estimates remain configurable planning policy;
they are not enforceable metadata or process ceilings. Payload/read buffers,
subsequent cache contents and independently exported metadata have separate
admission requirements. Ordinary opening and runtime-funded opening share the
same discovery, index and header workers.
Runtime can move a funded SafeTensors source into a typed closed retained
source owner. It checks the actual private store-policy type and owning pool
before admitting the outer allocation; an ordinary source, caller-defined policy
or foreign pool supplies no such origin. Refusals retain the original source.
Strong aliases share the closed source, while opaque weak identities keep the
outer allocation and its custody after nested source storage retires. These
owners keep no strong reference back to the pool. Retaining a source prepares no
headers and reopens no artifacts. Explicit metadata preparation borrows the
retained header through its original policy, and ordinary owned metadata uses
that same worker. The typed owner supplies the existing authenticated file route
for prepared reads without copying headers; generic closed provider ownership
does not supply a typed acquisition route.

Core inspection accepts a neutral SafeTensors source policy before discovery;
runtime can supply it from a memory ledger without core depending on
runtime. Strict inspection uses the same discovery and header workers, admitting
the metadata map and tensor-descriptor map before copying them. Tensor-catalog
clones share one immutable map and its retained custody. Equality and serialized
values exclude custody; deserialization constructs an ordinary independent map.
Runtime ends construction exclusion after inspection returns and retains the
accepted contribution with either the inspection or its error. Configuration and
architecture-plan allocations remain separate policy responsibilities.

A fresh runtime store over retained shards checks their private discovery policy
and owning pool, then admits independent store/cache controls. It neither
rediscovers the directory/index nor reparses the retained headers. Independent
stores retain their selected cache limits while sharing immutable shard and
header data. The store's policy identifies its own construction contribution;
ordinary store construction over funded shards does not establish that origin.
The input shards and accepted fresh contribution remain owned on refusal.
Model loading invokes a neutral backend/factory inspection hook before source
preparation. The MLX adapter installs the runtime policy through that hook;
loading an already inspected artifact consumes its retained inspection directly.
Both entries share preparation. An unavailable admission bound before any
construction permits ordinary inspection with the same encoded input limits.
Budget refusals and failures after accepted construction propagate without a
retry. Foreign-pool inspection custody is rejected, and ordinary caller-supplied
inspections are never promoted to funded origins. GGUF loading retains its
ordinary route; explicit catalog-pool preparation is independently available.
Payloads and subsequent recipe storage require separate admission.

The MLX execution-plan factory selects admitted execution and source streams
from the actual device before ordinary stream construction. CPU stream and
worker ownership uses that same factory without requiring the Metal feature.
An unavailable native stream layout permits ordinary construction; a failure
after admission propagates. Stream ownership alone does not establish complete
workspace coverage for the selected allocator and operators.

Cold workspace selection also supports CPU-only Apple builds. Allocation facts
query the compiled allocator through the native public layout API, independently
of the selected operator stream. CPU bounds include the size header and page
rounding used by original allocations and a conservative ordinary cache-reuse
envelope that also covers fresh request allocations. Ordinary Metal
bounds retain their own rounding rules; original Metal backing uses separately
queried physical capacities. The canonical backing census retains ordinary
allocator-cache charges through cache reuse and releases them only on physical
eviction. Bounded ordinary request sources bypass those cached roots and retire
their own fresh backing at its final physical alias. These charges do not
establish a process-wide memory ceiling. CPU and Metal equations retain their
respective mechanism dispatch, and missing operator facts still reject admission.
Shared sampling-constructor controls take the selected device explicitly. CPU
workers retain their operation-specific layout accounting; GPU sampling also
requires Metal random-worker controls. Eager random-key preparation uses the
host constructor and issues no device evaluation.
A recipe with a certified zero GPU-kernel population creates no pipeline cache.
Unknown populations remain unqualified, and positive populations require the
existing native cache layout and admitted construction.
The shared native allocation-population query uses the retained allocator to
bound the sum of independently rounded births. CPU populations include a size
header for every allocation attempt, including empty payloads; Metal empty
payloads have no backing. Finite payload and birth bounds retain every generation
without relying on donation, cache reuse or early retirement.

Runtime can pin an admitted SafeTensors leaf to the inspected metadata and
selected checkpoint contract. Ordinary and admitted preparation share the same
descriptor conversion, metadata comparison and provenance construction. Runtime
reserves configurable metadata/provenance/contract-map estimates and the fixed
outer wrapper requests before making their copies. The resulting typed prepared
and resolved owners preserve the existing authenticated file-read route and
contract checks. Strong aliases and opaque weak identities retain the wrapper
custody without keeping the pool alive; failure retains the original leaf and
accepted contribution. Metadata estimates do not cover future recipe entries or
payload buffers.

Architecture source preparation with an explicit catalog pool uses this pinned
SafeTensors path and requires shards inspected under that same pool. It also
retains the existing GGUF catalog construction path. Ordinary source preparation
continues through its shared constructors without claiming pool admission.
Architecture graph snapshots, projected views, manager storage and native
materialization have independent ownership and admission responsibilities.

Automatic planning errors retain a neutral backend failure with its typed source;
clones share its ownership. Native stream-construction errors dispose local
wrappers before crossing this thread-safe boundary and preserve typed constructor
and accounting diagnostics. The shared initializer account outlives that disposal.
Native process registrations retain their own custody independently; wrapper
disposal neither establishes completion nor refunds surviving registrations.

File-backed encoded reads have a sized constructor over already retained shard
headers. Ordinary reads perform their lazy header preparation before using that
same constructor. The admitted entry refuses missing headers and retains actual
header-error owners without cloning diagnostics or retrying them. Tensor metadata
and source-order scratch are sized by occurrence count; shard slots are bounded
by both occurrence and admitted-shard counts, with a conservative maximum
selected-path allowance.
Sorted source spans coalesce in place. Runtime compares before construction and
retains the completed batch or failed prefix through its original account.
Touched-metadata and payload diagnostics share fixed flags over one source-owned
admitted-path table. Construction and reads mark existing rows without extending
a path set. Header/source construction and later read scratch and output buffers
remain separate admission requirements; file batches keep exact admitted file
identities and use the existing validated borrowed read worker.
File and memory read selection share the same retained-view authorization worker.
File selection borrows an authenticated concrete source through the original root;
the root and ordered keys remain borrowed through comparison and construction.
This preserves exact source-owned header errors and contract identities without
allocating route records or invoking ordinary read preparation. Completed file
batches own their file identities and metadata, so the views may then retire.
Memory selection retains its selected store independently of the enclosing views.
Both paths preserve prepared-catalog qualification and composite batch ordering.
File inspection errors own the existing header or authorization handle and expose
a fixed kind plus borrowed diagnostic access. Header causes retain the original
StoreError identity after the root and keys retire. Authorization failures keep
the exact view and its source alive until error retirement, preserving the
contract name without allocating a copy or invoking a new source callback.

Runtime composes key admission, closed memory/file routing, read construction and
recursive compilation through the same pool. Unsupported routes return no read;
inspection and admission failures propagate with owned diagnostics and prefixes.
No ordinary read or lease callback serves as a fallback. The tile driver retains
the original source handle through submission. Each producer returns its validated
logical input byte count with its completion for shared telemetry. Ordinary
producers own ordinary lease preflight and metadata inference after the tile fits;
admitted encoded producers validate/read through their funded source records.
Source/header birth, candidate recipe and overlay storage, numerical producers
and complete prepared-manager handoff retain separate responsibilities.
Encoded CPU affine materialization uses the admitted resource owner and the
shared tile driver. One prepared tile retains its compiled read and native shape
until input construction returns; each queued native completion then keeps its
own input account. Input and companion dtypes select the actual native layout.
Failure-injection fixtures use this same tile preparation and submission worker.

Completed tensor replacement is owned by checkpoint's `MaterializedCheckpointSource`.
Its memory tensors override matching original keys while both physical stores
remain retained for storage accounting. Ordinary reads and concrete prepared
routes select the same source; enclosing restrictions still authorize every key.
The MLX `QuantizedCheckpoint` result moves that neutral source and its conversion
report into loading, without implementing a second checkpoint source. Output
buffers retain their payload reservations and estimated catalog metadata headroom;
this source type does not itself grant allocation authority.
Completed overlays use the retained source owner's typed materialized constructor.
Separately admitted metadata custody follows source aliases, retained acquisition
routes and opaque identities until their retirement. A detached encoded memory
read retains its validated tensor owners and can release the enclosing catalog;
authorization errors retain the source view needed for their diagnostics. This
lifetime contract does not certify admission of an overlay's original source,
catalog or output payloads.
Prepared encoded reads authenticate each concrete source and inspect the immutable
leaf before comparing enclosing catalogs from inner to outer. Catalogs over a
materialized source compare metadata and scalar provenance with the same borrowed
predicate as ordinary reads. This does not require a recipe cache or invoke an
ordinary read callback. Missing entries and catalog mismatches preserve ordered
occurrence diagnostics; unqualified custom sources remain unsupported by this
closed constructor.

The CPU encoded conversion initializer borrows the source and transform plan
until host metadata admission succeeds. Its configurable estimate uses concrete
catalog names, materialization provenance, recipe declarations and selected
borrowed metadata; it does not open lazy headers or read tensor payloads while
sizing. Shared catalog branches may contribute repeatedly to this conservative
input measure. The estimate covers cold clones, collision/inference scratch,
conversion provenance and retained plan/root overhead. Output-buffer publication
metadata is reserved separately with each buffer. These are planning estimates,
not retained-byte measurements or process memory ceilings.
The initializer runs the common cold and tile workers. Its completed handoff and
materialized source share metadata custody, including surviving opaque identities;
failure diagnostics retain the initializer account. Native resources, tile work,
original source/header birth and model-manager construction retain their own
admission requirements.
Cold conversion errors retire thread-local constructor outputs and transfer
pending native work to the existing recovery owner before entering the shared
backend error envelope. Typed causes and their original diagnostic accounts
survive that conversion; error transfer does not assert native completion.
The shared initialization failure owns this retirement operation, so callers
separate any rejected plan locally and use the same operation for constructor
and settlement failures.

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

Ordinary indexed-bank construction retains its originating host payer and checks
the actual bank revision and chunk census. Compact construction uses the same
physical row binder for acquired leases and published replacement views. Its
host quotation composes the concrete source visitor, finite row destinations,
prepared bindings, grouped specification copies and selected module constructor.
The constructor reads the same physical-field declarations used by its quote;
companion copies include their authoritative weight identity. Replacement-view
controls use the largest actual candidate rank for each possibly selected field.
Discovery, acquisition, materialization, native allocation and completion retain
their separate source requirements; a compact-constructor quote grants none of
those authorities.

Ordinary acquisition funds the same prepared controller, closure, named array,
lease and recovery destinations through its request's host account. This funding
does not supply an original-operation observer. Retained immutable Host backups
serve both Host publication and Device copying through the shared transfer worker;
Host publication preserves the existing backing owner and receipt. Actual source
pins and metadata custody remain with the invocation through completion and
deferred unpinning. Cold acquisition facts report request metadata, physically
observed host controls, Host staging and Device destinations separately. Existing
pinned backing is not charged as a fresh allocation. Recipe materialization or
transfer execution without a qualified source contribution remains incomplete.

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
signed-index geometry; the native CPU host-copy source covers ranks one through
five for F32, F16, BF16, I32 and U32 through the existing General-copy worker.
Its rank-dependent iterator and weak-descriptor storage comes from that worker's
checked source query. The existing Transfer/MetalShared Host-backing requirement is preserved
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
The same cold acquisition callback records the ordered unit ordinals for each
equation span. Its paid visit lists move into the native recipe with the
constructor census. Background Disk preparation consumes the generation subset
of those exact records, excluding the conservative final decode. Each forward
keeps its own selected windows and their original Host lookahead; decoder spans
do not acquire encoder units omitted by their actual traversal. Runtime checks
the first and subsequent ordinals against that prepared list and requires every
selected window to complete before joining the worker. These descriptive visits
grant no source ownership or execution authority.
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

Materialization borrows the enclosing role's native backing budget and accepted
host metadata account. Text registries retain the selected native bank and
validate its control guard before lending that budget. Nested and standalone
numerical roles lend the budget retained by their actual native role context.
Speculative and realtime banks retain the same explicit loan from the accepted
role, and authenticate its registered observer before every materialization.
Embedded invocation binding receives the actual recovery owner's budget and
metadata account before graph construction.
Inactive, stale or borrowed sources refuse before invoking the materializer.
These loans do not create another reservation or infer authority from the current
stream. Ordinary materialization requirements follow the recorded layerwise
visits and their selected windows; physical temporary allowances join the request's
domain vector, while caller metadata enters the corresponding work owner's host
allowance. Existing source storage remains pinned under its own identity.
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
CPU and Metal capture use the same selection, readout, source publication and
completion workers. The stream retained by the resident plan selects the CPU or
Metal operation recipes and compiled allocator facts. Completed positive strides
remain explicit in projected source metadata, including row gaps. Empty CPU
slices reserve the compiled allocator's zero-size backing independently of the
input storage. A supported stream alone
supplies no admission: the enclosing work retains the exact native scope, source
pins, completion roots and host destination claim. CPU-only builds can use those
workers without enabling Metal; unavailable operation bounds still refuse the
request before native submission.

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

Composite workspace populations retain the source's distinct new-allocation identities
and original physical placement descriptors through the shared domain reducer.
The fact producer lends this immutable population before scalar summation; its
fixed placements and candidate-placement bases stay separate. Existing borrowed
inputs do not become new child scratch. Repeated occurrences represent independent
populations, and the same metadata account pays retained source and destination
descriptors. Alternative completed paths resolve those original descriptors before
the core per-domain peak comparison; a scalar-largest branch does not substitute
for the largest requirement in every physical domain. Charges preserve accounted
allocation versus candidate-placement allowance categories and all retained bases.
The MLX movement producer retains its actual zero, unit, and maximum-row cold
branches through this loan; executing its ordinary worker does not copy planning
descriptors. Expert, count, provider, reorder and transport composition retains
each selected numerical child and native CPU communication allocation source.
Retained outputs use the same allocation and alias graph: each raw backing row
keeps its placement, allocation count and host controls, while a zero-byte union
root retains those rows and any actual input-alias alternatives. Pipeline frame
buffers and decoded tensor buffers therefore remain separate through closing,
state displacement and borrowed-source exclusion. Resolved alternative maxima
cannot substitute for output backing identities. The descriptive loans grant no
storage credit or native permission.
