# eredu-core

`eredu-core` provides backend-neutral contracts and orchestration for
language-model runtimes. It has no dependency on MLX, `safemlx`, Metal, CUDA, or
another accelerator runtime.

Most applications should use [`eredu`](../eredu/). Use this crate
directly when implementing an execution backend or reusing its portable
planning and scheduling components.

## What the crate provides

- Artifact inspection, resolved model configuration, tensor catalogs, and preparation plans
  for SafeTensors and GGUF checkpoints.
- Whole-model backend, session, prefill, decode, completion, capability, and
  optional distributed contracts.
- Generation, cancellation, speculative decoding, realtime scheduling, and
  transactional request state machines.
- Attention schedules, parallel topology, execution plans, admission reports,
  and telemetry schemas.
- Weight and cache residency policy, budgets, accounting schemas, eviction
  rules, and transfer contracts.
- Portable decoded media descriptions and ordered multimodal requests.

Backend contracts operate on associated opaque types. Implementations retain
their own tensors, queues, executables, caches, samplers, and completion
objects; core coordinates complete model/session operations rather than
defining a tensor algebra.

## Execution model

A backend prepares a model and consumes it to create one stateful session.
`ModelRuntime<B>` owns that backend and session, and submits prefill and decode
work through the same interface for every implementation. Exact completion
objects keep outputs and transactional resources alive until the submitted work
is safe to observe or release.

Capabilities fail closed. Optional facilities such as multimodal preparation,
speculative decoding, realtime execution, transfers, and collectives are
separate contracts attached to the selected backend or session.

See the [backend architecture guide](../doc/backend-architecture.md) for the
ownership rules and backend implementation checklist.

## License

Licensed under either Apache-2.0 or MIT.
