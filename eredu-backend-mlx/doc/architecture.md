# MLX backend architecture

This guide describes the MLX-specific realization of Eredu's
[backend-neutral architecture](../../doc/backend-architecture.md). Model-family
configuration, checkpoint schemas, parameter topology, execution equations,
and state geometry remain in `eredu-architectures`; this crate binds those
declarations to MLX storage and execution.

## Public surfaces

Reusable mechanics are rooted at `eredu_backend_mlx::backend::{nn, runtime,
...}`. The `native` module is an explicit escape hatch for backend tooling that
needs composition-owned sessions, execution contexts, or random state. Raw MLX
arrays, streams, devices, operations, and distributed groups retain their
canonical `safemlx` paths instead of being re-exported by the backend. The flat
crate root exposes the narrow adapters used by the `eredu` facade.

- `backend::MlxBackend` implements complete model/session contracts and owns
  its execution and weight-materialization streams.
- `backend::nn::shared::MlxNeuralBackend` is the stateless selector that
  implements neutral neural, parameter, submission, and transfer traits.
- `MlxTensor` is a transparent, zero-copy wrapper around `safemlx::Array` and
  implements `eredu_nn::Tensor`.
- `native::MlxModelSession` owns the executable model, cache, processor state,
  and optional distributed context. `backend::MlxModel` exposes its neutral
  capabilities and telemetry while keeping the architecture-erased executable
  private.

The crate never depends on the `eredu` facade. MLX arrays, streams, devices,
events, groups, and exceptions do not appear in neutral core or facade
signatures.

## Materialization and execution

Runtime modules implement checkpoint materialization, sampling, cache storage,
residency workers, media processing, and collectives. They consume exact
architecture-owned parameter identities, bindings, recipes, execution-unit
layouts, and parallel plans; they do not reconstruct family policy from names,
shapes, or configuration fields.

Public checkpoint runtime types are the MLX-owned materializers, materialized
tensors, errors, and explicit conversion options and reports used by backend
tooling. Raw SafeTensors visitation, shard discovery, GGUF metadata copying and
format lowering, and strict module assignment remain crate-internal composition
helpers; portable catalogs, stores, selections, recipes, and I/O errors use
their canonical `eredu-checkpoint` and `eredu-gguf` APIs.

GGUF family selection and portable structural admission remain architecture
registry concerns. MLX composition consumes the admitted family and schema,
then validates only native encoding, operator, placement, and device
compatibility. Resident, tensor-parallel, pipeline, and expert-parallel paths
therefore share the same architecture declarations.

Architecture-owned typed dispatch selects direct, routed, composite, or narrow
extension construction before MLX materialization. The MLX binder receives the
selected rank-local tasks, parameter recipes, state factory, opaque
communication manifest, and generic residency requirements; it does not carry
an exhaustive family realization matrix. Every active expert-parallel
combination uses the same neutral routed provider and addressable-bank
mechanisms.

Strict module loading performs no prefix rewriting, unused-prefix exemptions,
or implicit parameter-name expansion. Selective and distributed loading use
the exact checkpoint keys and logical targets retained by the preparation
plan. MLX completion events retain arrays and other Rust-side resources needed
by submitted work.

Host-resident weights, expert-cache bindings, and sealed live-cache blocks use
frozen `safemlx` host-transfer buffers. Residency reports charge their physical
allocation capacity rather than only the logical tensor length.

## Facade and feature integration

The facade's `mlx` feature selects this adapter. Its weak `metal`, `cuda`,
`image`, `audio`, and `nccl` features configure the adapter but do not select
it. Direct CUDA facade builds therefore use `--no-default-features` with
`--features mlx,cuda`; `metal` and `cuda` are mutually exclusive. The
`eredu-cli` `metal` and `cuda` convenience features both select `mlx`, and
`nccl` selects `cuda`.

This crate has no default features and disables `safemlx` defaults, so direct
users explicitly select native execution support. Enable `image` or `audio`
for the corresponding processor support and `codec` for the MLX Mimi binding.

Application clients should use portable `ExecutionPlan`, `LocalLoadOptions`,
and `LocalInspectionOptions` values through `eredu::api`. Backend tooling may
instead pass a checked neutral `ParallelRankTopology` and a process-local
`native::DeviceAssignment` through the root `MlxLoadRequest` adapter. Selection
projects the architecture's exact groups and routes into an opaque manifest
before payload access. MLX binds that manifest to the local rank/device context;
materialization receives only the selected preparation and resolved
composition-owned construction values. No public MLX parallel-plan or
family-pipeline construction surface remains.

Use `safemlx::distributed::device_for_local_rank` for the process-local device.
A global rank is not a local GPU index: launchers commonly restrict each
process with `CUDA_VISIBLE_DEVICES`, making the visible local index zero.

## Capabilities and observation

Device discovery reports only device-scoped capabilities. Loading derives
session capabilities from artifact inspection, residency policy, and topology,
checks requested capabilities before payload materialization, and verifies the
same report on the realized session.

Completed output is observed through `BackendSession::observe_output`.
Instrumented execution binds architecture-named activation points to
`InspectableBackendSession`; distributed observations remain rank-local and
only the logits-owning rank reports `model.logits`. Comparison policy,
statistics, and evaluation drivers remain in `eredu-evaluation`.

Collective support is reported only when the backend instance has an attached
world communicator. Realtime Moshi and PersonaPlex routes report
persistent request caches and portable output observation, but not named
activation inspection.
