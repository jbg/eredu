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
       eredu composition + backends
                    |
              eredu API
```

- `eredu-core`, `eredu-checkpoint`, and `eredu-runtime` are backend-neutral and
  must not depend on `eredu`, `eredu-architectures`, `safemlx`, or another
  native accelerator runtime. The default `eredu-nn` contract layer is also
  backend-neutral; any concrete implementation feature it exposes must remain
  optional and must not enter an architecture crate's dependency graph.
  Target-specific operating system support for portable facilities is allowed.
- `eredu-architectures` owns model-family configuration, checkpoint schemas,
  parameter topology, module construction, state geometry, parallel semantic
  plans, and embedding/layer/output execution. It may use neutral backend
  traits, but must not import `safemlx`, `eredu::backend`, or another concrete
  backend.
- `eredu/src/backend/<backend>` owns reusable native tensors, operators,
  streams, completion objects, materialization, cache storage, transfers, and
  collectives. Reusable backend modules must not own model-family configuration,
  checkpoint naming policy, layer equations, or family-specific state geometry.
- `eredu/src/composition` is the integration layer. It may select a family and
  a backend, bind architecture-declared parameters, and assemble sessions or
  distributed executables. Family/backend coupling belongs here rather than in
  a neutral crate or reusable backend module.
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
- compiler type checking and visibility for ownership boundaries;
- backend-neutral conformance tests for public behavior; and
- focused tests for architecture/runtime contracts.

Do not add tests that recursively inspect repository source text, forbid family
names by substring, or assert a particular file/directory layout. Those tests
confuse spelling and placement with dependency ownership and become stale during
valid reorganizations. If a boundary needs stronger mechanical enforcement,
prefer introducing a crate boundary, narrowing visibility, or adding a
manifest/dependency check.

When changing any boundary above, update `doc/backend-architecture.md` in the
same change. Useful verification commands are:

```sh
cargo test -p eredu-core --test dependency_boundary
cargo test -p eredu-runtime --test dependency_boundary
cargo test -p eredu --no-default-features --test portable_facade
cargo test -p eredu --no-default-features --test backend_conformance
cargo check -p eredu-architectures
```
