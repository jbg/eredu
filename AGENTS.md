# Repository architecture rules

These rules apply to the entire repository. They describe semantic ownership
and dependency direction; directory names alone are not an architectural API.

## Dependency direction

Keep production dependencies flowing in this direction:

```text
eredu-core / eredu-checkpoint / eredu-nn
                    |
              eredu-runtime
                    |
           eredu-architectures
                    |
          eredu-backend-mlx
                    |
              eredu API
```

- `eredu-core`, `eredu-checkpoint`, `eredu-nn`, `eredu-runtime`, and
  `eredu-codec` are backend-neutral and must not depend on `eredu`,
  `eredu-backend-mlx`, `safemlx`, or another native accelerator runtime.
  Target-specific operating system support for portable facilities is allowed.
- `eredu-architectures` owns model-family configuration, checkpoint schemas,
  parameter topology, module construction, state geometry, parallel semantic
  plans, and embedding/layer/output execution. It may use neutral backend
  traits, but must not import `safemlx`, `eredu::backend`, or another concrete
  backend.
- `eredu-backend-mlx` owns reusable MLX tensors, operators,
  streams, completion objects, materialization, cache storage, transfers, and
  collectives. It also owns MLX family composition, which may bind
  architecture-declared parameters and assemble sessions or distributed
  executables. Reusable backend modules must not own model-family
  configuration, checkpoint naming policy, layer equations, or family-specific
  state geometry. The crate must not depend on the `eredu` facade.
- `eredu/src/api` and `eredu/src/runtime` own backend-independent facade
  orchestration. Production backend code must not import them. A backend's
  public adapter may depend on narrow composition-owned executable or session
  types needed to implement the neutral backend traits; that exception does
  not transfer tokenizer, generation, scheduling, or application policy into
  the backend.

## Feature boundary

- Model-family definitions and neutral execution live in
  `eredu-architectures` and must remain available without enabling any concrete
  backend feature. Never make an entire family conditional on `mlx`, `cuda`, or
  another backend feature.
- `eredu` with `default-features = false` is the portable facade. Native MLX,
  CUDA, media, and codec dependencies must remain optional and enabled by their
  corresponding features.
- `eredu-nn` and `eredu-codec` expose no concrete-backend features, including
  under `--all-features`. MLX implementations belong in `eredu-backend-mlx`.
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

Do not add tests that inspect the Cargo dependency graph, recursively inspect
repository source text, forbid family names by substring, or assert a particular
file/directory layout. Those tests confuse repository shape with dependency
ownership and become stale during valid reorganizations. Keep dependency rules
explicit in this file and manifests. If a boundary needs stronger mechanical
enforcement, prefer introducing a crate boundary or narrowing visibility.

When changing any boundary above, update `doc/backend-architecture.md` in the
same change. Useful verification commands are:

```sh
cargo check -p eredu-core
cargo check -p eredu-checkpoint
cargo check -p eredu-runtime
cargo check -p eredu-nn --all-features
cargo check -p eredu-codec --all-features
cargo check -p eredu-architectures
cargo test -p eredu --no-default-features --test portable_facade
cargo test -p eredu --no-default-features --test backend_conformance
```
