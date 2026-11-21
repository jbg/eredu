# MLX backend architecture

This guide describes the MLX-specific realization of Eredu's
[backend-neutral architecture](../../doc/backend-architecture.md). Model-family
configuration, checkpoint schemas, parameter topology, execution equations,
and state geometry live in `eredu-architectures`. Portable composition binds
selected models using the MLX storage and execution mechanisms supplied here.

## Public surfaces

Reusable mechanics are rooted at `eredu_backend_mlx::backend::{nn, runtime,
...}`. The `native` module is an explicit escape hatch for backend tooling that
needs mechanism bindings, execution contexts, or random state. Raw MLX
arrays, streams, devices, operations, and distributed groups retain their
canonical `safemlx` paths instead of being re-exported by the backend. The flat
crate root exposes the narrow adapters used by the `eredu` facade.

- `backend::MlxBackend` provides native resources for model and session
  binding and owns its execution and weight-materialization streams.
- `backend::nn::shared::MlxNeuralBackend` is the stateless selector that
  implements neutral neural, parameter, submission, and transfer traits.
- `MlxTensor` is a transparent, zero-copy wrapper around `safemlx::Array` and
  implements `eredu_nn::Tensor`.
- The portable text driver constructs and owns the installed executable,
  together with the exact native cache and distributed context. MLX supplies
  borrowed binding resources and realizes communication from the agreed
  manifest. The portable rank visitors retain the realized handles directly.

The crate never depends on the `eredu` facade. MLX arrays, streams, devices,
events, groups, and exceptions do not appear in neutral core or facade
signatures.

## Internal module organization

The implementation follows the ownership boundary above without making its
private layout part of the public API:

- `backend::nn` contains reusable MLX neural mechanisms. Its `shared` module
  separates parameter adapters, ordinary operators, submission and
  communication, the core neutral-backend implementation, and additive
  grouped or hyper extensions. Packed grouped operations and native
  quantization are divided by operation and lowering responsibility.
- `backend::runtime` contains native storage and execution mechanisms. Cache
  modules supply native state, page allocation, transfer, and completion.
  Portable residency selects the page schedule and publishes staged spills.
  Checkpoint modules separate source leases, materialization, recipes, and
  bounded transformation; residency modules separate catalogs, acquisition,
  movement, and telemetry; distributed modules separate handles, operations,
  completion ownership, manifest realization, and placement.
- `composition::mlx` supplies mechanism adapters such as cold capability
  facts, selected parameter storage, queue and completion handling, and
  native tensor formats. The portable driver owns selection, state, and
  session erasure.
- The facade's feature-gated integration targets exercise native Ring and
  model execution against the portable contracts.

These modules exchange typed neutral plans and concrete MLX mechanisms. They
do not introduce dynamic dispatch into tensor operations or per-unit execution.

## Materialization and execution

Runtime modules implement checkpoint materialization, sampling, cache storage,
residency workers, processed-buffer tensor conversion, and collectives. They consume exact
architecture-owned parameter identities, bindings, recipes, execution-unit
layouts, and parallel plans; they do not reconstruct family policy from names,
shapes, or configuration fields.

Canonical artifact identity, prepared-source exactness, declarative parameter
binding, host media processing, admission/report assembly, replicated-session
construction, and logical distributed placement come from neutral crates.
Public native checkpoint types include MLX materializers, materialized
tensors, and native errors. Checkpoint conversion options, reports, shard
discovery and publication use the portable `eredu-checkpoint` APIs. The MLX
packer handles selected dense tensors through that crate's encoding contract.

GGUF family selection and structural admission use the architecture registry.
MLX receives the resulting exact encoding, operator, placement, and device
requirements. Resident, tensor-parallel, pipeline, and expert-parallel paths
therefore share the same architecture declarations.

Architecture-owned typed dispatch selects direct, routed, composite, or narrow
extension construction before MLX materialization. The MLX binder receives the
selected rank-local tasks, parameter recipes, and agreed communication
resources. Portable construction owns the state factory and residency policy.
Expert-parallel variants use the shared routed provider and addressable-bank
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
for the corresponding processor support. Neutral codec constructors use the
base generic checkpoint and parameter-materialization mechanisms.

Application clients pass `eredu_core::ExecutionPlan` and this crate's
`MlxBackendFactory` to `eredu::api::LoadedModel::load_execution_plan`. Models,
requests, and controls retain their backend parameters. `ExecutionPlan`
provides portable policy for `eredu::api::inspect_execution`.
Backend tooling may
instead pass a checked neutral `ParallelRankTopology` and a process-local
`native::DeviceAssignment` through `eredu_core::ModelLoadOptions`. Selection
projects the architecture's exact groups and routes into an opaque manifest
before payload access. The neutral runtime validates complete-world groups,
routes, capabilities, and realization waves before MLX binds the prepared
descriptors to the local rank/device context;
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
With `tensor-capture`, bounded execution binds architecture-named activation
points to capture plans and delivers distributed values according to their
selected producer layout. Comparison policy,
statistics, and evaluation drivers remain in `eredu-evaluation`.

Collective support is reported only when the backend instance has an attached
world communicator. Realtime Moshi and PersonaPlex routes report
persistent request caches and portable output observation, but not named
activation inspection.
