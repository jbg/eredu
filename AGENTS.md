# Repository architecture rules

These rules apply to the entire repository. They describe semantic ownership
and dependency direction; directory names alone are not an architectural API.

## Dependency direction

Production crates occupy these dependency strata, from dependency roots to
applications:

```text
foundations:          eredu-gguf / eredu-nn-macros
storage and text:     eredu-checkpoint / eredu-text
portable contracts:  eredu-core / eredu-nn
portable mechanisms: eredu-media / eredu-runtime
portable families:   eredu-codec / eredu-architectures
native realization:  eredu-backend-* (currently eredu-backend-mlx)
facade:               eredu
applications:         eredu-cli / examples / downstream applications
```

A later stratum may depend on an earlier one; placement on the same line does
not itself authorize a dependency. Native runtime crates such as `safemlx` are
backend-specific dependencies and never dependencies of the
portable strata. Validation tooling such as `eredu-evaluation` may depend on
portable families, but production facade and backend code must not depend on
it.

- `eredu-gguf` owns framework-independent GGUF reading, writing, validation,
  canonical tensor encodings, and bounded conversion. `eredu-checkpoint` owns
  backend-neutral checkpoint schemas, recipes, exact prepared source stores,
  restricted views, cache policy, leases, provenance, and resolution guards.
  Neither crate allocates a backend tensor or selects an execution mechanism.
- `eredu-text` owns backend-neutral tokenizer and chat-template utilities.
  Facade-owned tokenizer reconstruction, generation termination, and
  application policy may use those utilities, but do not move into a backend.
- `eredu-core`, `eredu-checkpoint`, `eredu-gguf`, `eredu-nn`, `eredu-text`,
  `eredu-media`, `eredu-runtime`, `eredu-codec`, and `eredu-architectures` are
  backend-neutral and must not depend on `eredu`, a concrete `eredu-backend-*`
  crate, `safemlx`, or another native accelerator runtime. Target-specific
  operating-system or host-library support for portable facilities is allowed.
- `eredu-architectures` owns model-family configuration, checkpoint schemas,
  parameter topology, module construction, state geometry, parallel semantic
  plans, processor request policy, cold execution-class selection, prepared
  source roles, total prepared-execution construction, typed state-profile
  dispatch, architecture-aware inspection, and embedding/layer/output
  execution. Backends supply native contexts, typed binding visitors and final
  executable adapters to that construction driver, not semantic branch logic.
  It may use neutral backend traits, but must not import `safemlx`,
  `eredu::backend`, or another concrete backend.
- `eredu-runtime` owns execution-plan normalization, portable load policy,
  selected-task residency sizing, neutral residency telemetry, and
  model-independent mechanism selection. It synthesizes capabilities from
  exact single-candidate support predicates and backend mechanism facts.
  Backends inject diagnostics choices and native observations; they do not
  repeat portable plan conversion or requirement enumeration.
  Cold selection and inspection consume backend capability facts but never a
  native device, stream, tensor, group, or completion object.
- `eredu-core` owns exact session-admission comparison and move-only submission
  authority. Backends retain the neutral lease alongside native completion
  resources and establish safe completion, terminal failure, or teardown before
  releasing it. A polling error alone is not a universal completion signal.
  It also owns versioned logical architecture and observation discovery contracts.
  Architecture-layer projections supply family semantics; runtime support reports
  combine those declarations with backend collector facts and retained selection.
  Bounded capture admission and host records are neutral contracts; runtime owns
  reservation and one-step delivery policy, backends own native transformations,
  and the facade composes capture with ordinary generation and text termination.
- `eredu-codec` owns backend-neutral neural audio codec architectures, released
  checkpoint schemas, parameter topology, layout recipes, and typed artifact
  construction. It consumes general neutral tensor, parameter, and runtime
  contracts; a concrete backend supplies ordinary materialization and neural
  mechanisms rather than a codec-family implementation.
- Every concrete `eredu-backend-*` crate owns its native tensors, operators,
  clients/devices, streams or queues, completion objects, materialization,
  cache storage, transfers, collectives, exact hardware facts, and final typed
  binding/erasure. It may bind architecture-declared parameters and assemble
  native sessions or distributed executables, but must consume the retained
  neutral selection and prepared sources without reopening artifacts or
  reconstructing semantic branches. A backend must not own model-family
  configuration, checkpoint naming policy, layer equations, family-specific
  state geometry, portable inspection state, source-format dispatch,
  tokenizer/generation policy, or application scheduling.
- `eredu-backend-mlx` is the current concrete realization. It owns reusable
  MLX tensors, operators, streams, completion objects, materialization, cache
  storage, transfers, and collectives through `safemlx`. Its MLX composition
  may bind architecture-declared parameters and assemble sessions or
  distributed executables. Its cold adapters translate public options once,
  report exact side-effect-free capability facts, bind native rank/device
  resources, and
  consume the retained neutral selection and prepared sources without
  reopening artifacts or reconstructing semantic branches. Reusable backend
  modules must not own model-family configuration, checkpoint naming policy,
  layer equations, family-specific state geometry, portable inspection state,
  or source-format dispatch. The crate must not depend on the `eredu` facade.
- `eredu/src/api` and `eredu/src/runtime` own backend-independent facade
  orchestration. Production backend code must not import them. A backend's
  public adapter may depend on narrow composition-owned executable or session
  types needed to implement the neutral backend traits; that exception does
  not transfer tokenizer, generation, scheduling, or application policy into
  the backend.
- `eredu-media` owns portable host audio, image, and video validation and
  processing. It may depend on optional host libraries, but never allocates a
  backend tensor, selects a device or stream, or depends on a concrete backend.
- `eredu-evaluation` owns reusable evaluation fixtures, metrics, and comparison
  policy. Concrete backend tests and examples may use it as a development
  dependency; production backend mechanisms must not call into it.
- `eredu-cli`, examples, and downstream applications are consumers above the
  facade or explicitly imported low-level neutral contracts. They do not
  become owners of backend, architecture, or runtime policy merely because
  they configure it.

## Feature boundary

- Public application operations, iterators, token observations, and errors must
  not expose native error types or backend-error type parameters. Translate
  native failures to the neutral `eredu_core::BackendFailure`, preserving the
  original error as its source. Backend implementation contracts may retain
  concrete errors internally; portable policy errors remain typed.
- Language-model-family definitions and neutral execution live in
  `eredu-architectures`; neural audio codec families live in `eredu-codec`.
  Both must remain available without enabling any concrete backend feature.
  Never make an entire family conditional on `mlx`, `cuda`, or another backend
  feature.
- `eredu` with `default-features = false` is the portable facade. Native MLX,
  other concrete backends, accelerator runtimes, and optional host-media
  dependencies must remain feature-gated. The `mlx` feature selects the current
  adapter; `metal`, `cuda`, and `nccl` configure it only when selected. Image
  and audio features enable portable host processing plus conversion in an
  enabled backend, not backend ownership of media semantics.
- `eredu-nn` and `eredu-codec` expose no concrete-backend features, including
  under `--all-features`. MLX implementations belong in `eredu-backend-mlx`.
- `eredu-media` keeps image and audio dependencies optional. Consumers forward
  only the host-media features they need.
- Feature-gate concrete backend adapters and backend-specific composition, not
  the family they adapt. If a family integration currently mixes neutral and
  native code, separate those surfaces instead of hiding the family behind the
  native feature.
- Portable facade and conformance tests must use neutral traits and mock
  backends. Do not make them import a concrete backend implementation.
- New backend capabilities should be expressed as neutral associated types,
  traits, plans, or reports first, then implemented by the backend and wired in
  composition.

## How to enforce these rules

Prefer semantic enforcement that survives refactors:

- Cargo manifests and feature-gated builds for crate and optional-dependency
  boundaries;
- the workspace `unsafe_code = "forbid"` lint, inherited by every package except
  the `safemlx` native wrapper, the raw `safemlx-sys` bindings, and the
  `eredu-ios` C-ABI example;
- compiler type checking and visibility for ownership boundaries;
- backend-neutral conformance tests for public behavior; and
- focused tests for architecture/runtime contracts.

Any additional native-backend unsafe-code exception must be explicit and
crate-local, document the safety boundary here and in
`doc/backend-architecture.md`, and must not weaken the lint for neutral or
unrelated crates.

Do not add tests that inspect the Cargo dependency graph, recursively inspect
repository source text, forbid family names by substring, or assert a particular
file/directory layout. Those tests confuse repository shape with dependency
ownership and become stale during valid reorganizations. Keep dependency rules
explicit in this file and manifests. If a boundary needs stronger mechanical
enforcement, prefer introducing a crate boundary or narrowing visibility.

When changing any boundary above, update `doc/backend-architecture.md` in the
same change. Useful verification commands are:

```sh
cargo check -p eredu-gguf
cargo check -p eredu-checkpoint
cargo check -p eredu-core
cargo check -p eredu-text --no-default-features
cargo check -p eredu-runtime
cargo check -p eredu-nn --all-features
cargo check -p eredu-codec --all-features
cargo check -p eredu-media --no-default-features
cargo check -p eredu-media --all-features
cargo check -p eredu-architectures
cargo check -p eredu-backend-mlx --no-default-features
cargo test -p eredu --no-default-features --test portable_facade
cargo test -p eredu --no-default-features --test backend_conformance
```
