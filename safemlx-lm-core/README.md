# safemlx-lm-core

`safemlx-lm-core` contains backend-neutral language-model contracts,
transactional scheduling, validation, and portable planning schemas. It does
not depend on MLX, SafeMLX, Metal, CUDA, or another accelerator runtime.

Most applications should use `safemlx-lm`, which reexports canonical core
types at its facade root and supplies the default MLX backend. Backend authors
can depend on this crate to implement whole-model preparation and session-level
prefill/decode without implementing a primitive tensor algebra.

Core owns the production weight-residency policy and ownership state machine,
not a parallel summary schema. `OffloadPlan` validates stable unit identities,
tier assignments, budgets, prefetch depth, and eviction policy.
`ResidencyLedger` atomically admits batches, reserves and publishes copies,
tracks leases and protected windows, chooses deterministic eviction victims,
and resolves exact transfer generations. Its transitions explicitly return the
copies whose concrete storage a backend must release. `OffloadTelemetry` and
`OffloadReport` account for transfers, prefetch, eviction, process memory, and
backend allocator observations. Deserialization cannot bypass plan invariants.
Backends own only resource materialization, native transfer/completion objects,
and destruction of the storage selected by ledger transitions.

Core also owns the process-wide live-cache admission boundary.
`CacheResidencyPool` registers independent cache managers, publishes their
concrete occupancy, and issues exact RAII reservations for device, host,
in-flight transfer, and live-disk resources. Concurrent admission is atomic and
fails closed per resource axis. Backends retain the cache tensors and native
completion objects; dropping a reservation or manager membership releases only
that owner's accounting.

Cache semantics are core-owned as well. Stable block/rank identities, logical
tiers, per-layer attention-state geometry, fixed-state roles,
symbolic shapes, dtype families, and mutable-state residency classes have one
canonical definition in this crate. Constructors and deserialized policies use
the same validation. A backend maps these contracts to its arrays and storage;
it cannot redefine cache geometry or fixed-state ownership.

`CacheBlockLifecycle` is the canonical live-block ownership catalog. It owns
registration, exact lease counts, access clocks, protected-prefix state,
deterministic LRU/LFU selection, transactional truncation replacement, and
mutable-tail frontiers. `CacheBlockStorage` separately owns the legal physical
phase protocol for a block: device demotion, host promotion and rollback,
backing writes, backing reads, and direct release of backed resources. Device,
host, backing, and completion values are opaque type parameters. Exact operation
keys are core types and are bound to one block and generation. A backend owns
the values but cannot construct contradictory resource/phase combinations.

Reusable prompt-cache identity and catalogs are also canonical here. Core owns
model and prefix identity, rank-local topology, the versioned manifest,
architecture and token fingerprints, block/state descriptors, compatibility,
coverage, geometry, ordering, and safe relative-path validation. A backend may
publish those catalogs to its chosen storage and materialize their payloads,
but does not repeat or reinterpret their semantic validation.

The production Moshi/PersonaPlex realtime scheduler uses the core request state
machine. Its MLX adapter supplies opaque work, session branches, submissions,
and exact completions while core owns fairness and transactional publication.
Distributed pipeline requests use that same state machine, including poisoning
and exact failure retention. Core also owns protocol framing and fail-closed
schedule, disposition, and completion agreement. Backends implement only the
portable `ConsensusTransport` word collective.

See [the backend architecture guide](../doc/backend-architecture.md) for the
ownership boundary and backend implementation checklist.
