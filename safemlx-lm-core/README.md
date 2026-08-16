# safemlx-lm-core

`safemlx-lm-core` contains backend-neutral language-model contracts,
transactional scheduling, validation, and portable planning schemas. It does
not depend on MLX, SafeMLX, Metal, CUDA, or another accelerator runtime.

Most applications should use `safemlx-lm`, which reexports canonical core
types at compatible paths and supplies the default MLX backend. Backend authors
can depend on this crate to implement whole-model preparation and session-level
prefill/decode without implementing a primitive tensor algebra.

The production Moshi/PersonaPlex realtime scheduler uses the core request state
machine. Its MLX adapter supplies opaque work, session branches, submissions,
and exact completions while core owns fairness and transactional publication.

See [the backend architecture guide](../doc/backend-architecture.md) for the
ownership boundary and backend implementation checklist.
