# safemlx-lm-core

`safemlx-lm-core` contains backend-neutral language-model contracts,
transactional scheduling, validation, and portable planning schemas. It does
not depend on MLX, SafeMLX, Metal, CUDA, or another accelerator runtime.

Most applications should use `safemlx-lm`, which reexports canonical core
types at its facade root and supplies the default MLX backend. Backend authors
can depend on this crate to implement whole-model preparation and session-level
prefill/decode without implementing a primitive tensor algebra.

Core owns the production weight-residency policy rather than a parallel summary
schema. `OffloadPlan` validates stable unit identities, tier assignments,
budgets, prefetch depth, and eviction policy. `OffloadTelemetry` and
`OffloadReport` account for transfers, prefetch, eviction, process memory, and
backend allocator observations. Deserialization cannot bypass plan invariants.
Backends own resource materialization, transfer submission, and exact transfer
completion.

The production Moshi/PersonaPlex realtime scheduler uses the core request state
machine. Its MLX adapter supplies opaque work, session branches, submissions,
and exact completions while core owns fairness and transactional publication.
Distributed pipeline requests use that same state machine, including poisoning
and exact failure retention. Core also owns protocol framing and fail-closed
schedule, disposition, and completion agreement. Backends implement only the
portable `ConsensusTransport` word collective.

See [the backend architecture guide](../doc/backend-architecture.md) for the
ownership boundary and backend implementation checklist.
