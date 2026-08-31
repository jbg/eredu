# Releasing workspace crates

There is deliberately no workspace-wide package version. Each publishable
crate declares and advances its own version, so a crate can release without
forcing unrelated workspace crates to release. When a crate version changes,
update the corresponding requirement in `[workspace.dependencies]` and any
dependent crates that need the new release; Cargo includes those requirements
when it packages path dependencies.

Every publishable crate must pass the same archive validation before a release:

```bash
python3 validation/validate_release_packages.py
```

The validator copies the Git package candidates to a temporary workspace, then
runs `cargo package` for each crate. It unpacks each archive, compiles its
library unit tests, and runs its doctests with default features disabled before
adding that archive to an ephemeral local registry. After staging a library, it
also checks a new lock-free downstream crate that depends on the staged package
with default features disabled. This lets consumers and the next workspace crate
resolve the exact unpublished version while retaining Cargo's normal package
verification. When `safemlx-internal-macros` is staged, the validator also
compiles the published `safemlx` 0.1.3 release from a lock-free consumer. That
guards its `safemlx-internal-macros = "0.1.1"` requirement against resolving to
an incompatible patch candidate. Package builds use a temporary target directory
that is removed after validation. Nothing contacts a registry publishing API and
no credentials are required.

This catches:

- path dependencies without publishable versions or versions that do not match
  an earlier workspace archive;
- files omitted from the generated archive that cause package, packaged unit
  test compilation, or packaged doctests to fail;
- dependency requirements that fail when a downstream consumer resolves the
  published crate without inheriting the workspace lockfile;
- semver-incompatible macro candidates that break the published `safemlx` 0.1.3
  dependency graph;
- archives above crates.io's 10 MiB compressed-size limit; and
- new publishable crates or dependency changes that are missing from the
  declared order.

The release-package CI job runs this validation on Ubuntu with both the minimum
supported Rust version and stable, using the CPU MLX prerequisites. Archive unit
tests are compiled and doctests run without default features, while Cargo's
normal package verification still compiles each crate's default packaged
targets. Packaging does not need an Apple or NVIDIA runner: target-native Metal
and CUDA coverage remains in the platform workflows.

The Linux build workflow separately denies all workspace Clippy warnings and
checks each weakly forwarded facade feature (`metal`, `cuda`, `nccl`, `image`,
and `audio`) without `mlx` at the minimum supported Rust version. These checks
keep optional facade features from accidentally activating the native backend
or depending on its availability.

The manually dispatched `Native release gate` workflow must pass before
publication. Its macOS job first proves that native MLX execution is available;
failure to initialize Metal is a test failure, never a skip. The same job runs
every self-contained ignored distributed Ring test serially across the
Cartesian-topology, expert-exchange, checkpoint-partition, pipeline, and
realtime suites rather than sampling the representative cases used by
pull-request CI. To run these gates locally on an Apple silicon host, outside a
sandbox:

```bash
cargo test -p eredu-backend-mlx --features metal --lib \
  composition::mlx_architecture_conformance::native_mlx_execution_is_available -- \
  --exact
cargo test -p eredu-backend-mlx --lib \
  _ring:: -- \
  --ignored \
  --skip moshi_ring_tp2_native_model_parity \
  --skip moshi_ring_tp2_personaplex_model_parity \
  --test-threads=1 --nocapture
```

The two production-model realtime parity tests are explicit opt-in gates
because their released model directories are not stored in the repository or
provisioned on the GitHub-hosted runner. Run either test separately with its
documented `EREDU_MOSHI_NATIVE_FIXTURE` or
`EREDU_MOSHI_PERSONAPLEX_FIXTURE` directory when validating those artifacts.

## Publication order

Publish one crate at a time in this order, waiting for each version to become
available in the registry index before continuing:

1. `eredu-gguf`
2. `safemlx-internal-macros`
3. `eredu-backend-mlx-macros`
4. `safemlx-sys`
5. `eredu-nn-macros`
6. `eredu-checkpoint`
7. `eredu-core`
8. `eredu-text`
9. `safemlx`
10. `eredu-nn`
11. `eredu-runtime`
12. `eredu-architectures`
13. `eredu-codec`
14. `eredu-evaluation`
15. `eredu-backend-mlx`
16. `eredu`
17. `eredu-cli`

This is a valid topological order for normal, build, and development
dependencies. In particular, `eredu-evaluation` precedes
`eredu-backend-mlx` because the backend uses it as a development dependency.
The order is intentionally maintained once in the validator and checked
against Cargo metadata so CI fails when it becomes stale.

For a local check of uncommitted source, pass `--allow-dirty`. CI and release
preparation should use a clean checkout and the command without that option.
