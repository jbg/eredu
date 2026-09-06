# Releasing workspace crates

There is deliberately no workspace-wide package version. Each publishable
crate declares and advances its own version, so a crate can release without
forcing unrelated workspace crates to release. When a crate version changes,
update the corresponding requirement in `[workspace.dependencies]` and any
dependent crates that need the new release; Cargo includes those requirements
when it packages path dependencies.

Choose the next version by comparing all pending changes with the latest
version actually published on crates.io, not with an earlier development
phase. Reuse that pending version until publication unless later changes need
a different compatibility level. For these pre-1.0 crates,
breaking public API changes require the next minor version; compatible fixes
use the next patch. Leave unchanged crates at their published version and
start new crates at `0.1.0`. A binary-only crate can take a patch for internal
dependency updates when its command-line interface remains compatible.

## Workspace releases on 2026-09-06

The following pre-release baselines were checked against the crates.io API and published
source archives on 2026-09-06. Archive checksums were verified against the API,
and the source was compared with main at `5b30e9d8`. The `0.1.0` archives below
come from `5d26950c`, as does `safemlx-internal-macros` `0.2.0`. The MLX
`0.1.2`, CLI `0.1.1`, `safemlx` `0.2.2`, and `safemlx-sys` `0.2.1`
archives come from `f3a0b862`. Recheck the registry before publishing.

The 13 changed crates below were published on 2026-09-06 from
`c5b6f080`, including the Windows DLL linkage and C++ exception unwinding
repairs in `safemlx-sys`, repeatable GPU submission recovery, and MLX
partition binding, session completion, and synchronous resource-retirement repairs. The release passed the
[native release gate](https://github.com/jbg/eredu/actions/runs/34055527048)
and [archive validation](https://github.com/jbg/eredu/actions/runs/34055504464).

| Crate | Previous release | Released / retained | Reason |
| --- | --- | --- | --- |
| `eredu` | [0.1.0](https://crates.io/crates/eredu/0.1.0) | [0.2.0](https://crates.io/crates/eredu/0.2.0) | Public load options and backend contracts changed; the local speculative telemetry helper was removed. |
| `eredu-core` | [0.1.0](https://crates.io/crates/eredu-core/0.1.0) | [0.2.0](https://crates.io/crates/eredu-core/0.2.0) | Planning and loading traits have new required items and changed signatures. |
| `eredu-checkpoint` | [0.1.0](https://crates.io/crates/eredu-checkpoint/0.1.0) | [0.2.0](https://crates.io/crates/eredu-checkpoint/0.2.0) | `CheckpointLease::Gguf` now contains a boxed lease, changing public construction and matching. |
| `eredu-nn` | [0.1.0](https://crates.io/crates/eredu-nn/0.1.0) | [0.2.0](https://crates.io/crates/eredu-nn/0.2.0) | `GroupedRelu2Operator` requires a new `spec` method. |
| `eredu-runtime` | [0.1.0](https://crates.io/crates/eredu-runtime/0.1.0) | [0.2.0](https://crates.io/crates/eredu-runtime/0.2.0) | Parameter binding and state-support traits have incompatible signatures and new requirements. |
| `eredu-architectures` | [0.1.0](https://crates.io/crates/eredu-architectures/0.1.0) | [0.2.0](https://crates.io/crates/eredu-architectures/0.2.0) | Public preparation types and execution interfaces were replaced or changed. |
| `eredu-codec` | [0.1.0](https://crates.io/crates/eredu-codec/0.1.0) | [0.2.0](https://crates.io/crates/eredu-codec/0.2.0) | Codec construction changed and public `Error` variants were removed. |
| `eredu-evaluation` | [0.1.0](https://crates.io/crates/eredu-evaluation/0.1.0) | [0.2.0](https://crates.io/crates/eredu-evaluation/0.2.0) | Realtime evaluation now requires a `RealtimeEvaluationDriver` instead of a `RealtimeModel` loader. |
| `eredu-backend-mlx` | [0.1.2](https://crates.io/crates/eredu-backend-mlx/0.1.2) | [0.2.0](https://crates.io/crates/eredu-backend-mlx/0.2.0) | Native adapters implement the new contracts; public composition APIs and the `codec` feature changed. |
| `eredu-cli` | [0.1.1](https://crates.io/crates/eredu-cli/0.1.1) | [0.1.2](https://crates.io/crates/eredu-cli/0.1.2) | Internal API updates and retained automatic inspection preserve the command-line interface. |
| `eredu-media` | Unpublished | [0.1.0](https://crates.io/crates/eredu-media/0.1.0) | First release. |
| `eredu-gguf` | [0.1.0](https://crates.io/crates/eredu-gguf/0.1.0) | 0.1.0 (retained) | Source and dependency requirements are unchanged. |
| `eredu-text` | [0.1.0](https://crates.io/crates/eredu-text/0.1.0) | 0.1.0 (retained) | Source and dependency requirements are unchanged. |
| `eredu-nn-macros` | [0.1.0](https://crates.io/crates/eredu-nn-macros/0.1.0) | 0.1.0 (retained) | Source and dependency requirements are unchanged. |
| `eredu-backend-mlx-macros` | [0.1.0](https://crates.io/crates/eredu-backend-mlx-macros/0.1.0) | 0.1.0 (retained) | Source and dependency requirements are unchanged. |
| `safemlx` | [0.2.2](https://crates.io/crates/safemlx/0.2.2) | [0.3.0](https://crates.io/crates/safemlx/0.3.0) | Adding `Misaligned` and `TooLarge` to the exhaustive public `AsSliceError` enum breaks downstream matches. |
| `safemlx-sys` | [0.2.1](https://crates.io/crates/safemlx-sys/0.2.1) | [0.2.2](https://crates.io/crates/safemlx-sys/0.2.2) | Native lifetime repairs and additional FFI entry points preserve the existing interface. |
| `safemlx-internal-macros` | [0.2.0](https://crates.io/crates/safemlx-internal-macros/0.2.0) | 0.2.0 (retained) | Source and dependency requirements are unchanged. |

The `eredu-ios` example and `safemlx-tests` remain `0.1.0` with
`publish = false`. Already published, unchanged crates need no new upload;
skip them in the publication order below.

## Archive validation

Every publishable crate must pass the same archive validation before a release:

```bash
python3 validation/validate_release_packages.py
```

The validator keeps the invoking workspace's active Rust toolchain for package,
archive, and downstream-consumer checks, including checks in temporary
directories outside the workspace. An explicit `RUSTUP_TOOLCHAIN` takes
precedence; otherwise it preserves rustup's active environment, directory
override, or repository toolchain selection.

Validate the minimum supported Rust version explicitly with an installed
toolchain:

```bash
rustup toolchain install 1.89.0 --profile minimal
RUSTUP_TOOLCHAIN=1.89.0 python3 validation/validate_release_packages.py
```

The validator copies the Git package candidates to a temporary workspace, then
runs `cargo package` for each crate. It unpacks each archive, compiles its
library unit tests, and runs its doctests with default features disabled before
adding that archive to an ephemeral local registry. After staging a library, it
also checks a new lock-free downstream crate that depends on the staged package
with default features disabled. This lets consumers and the next workspace crate
resolve the exact unpublished version while retaining Cargo's normal package
verification. Package builds use a temporary target directory that is removed
after validation. Nothing contacts a registry publishing API and no credentials
are required.

This catches:

- path dependencies without publishable versions or versions that do not match
  an earlier workspace archive;
- files omitted from the generated archive that cause package, packaged unit
  test compilation, or packaged doctests to fail;
- dependency requirements that fail when a downstream consumer resolves the
  published crate without inheriting the workspace lockfile;
- archives above crates.io's 10 MiB compressed-size limit; and
- new publishable crates or dependency changes that are missing from the
  declared order.

The release-package CI job runs this validation on Ubuntu with both the minimum
supported Rust version and stable, using the CPU MLX prerequisites. Its matrix
explicitly selects Rust 1.89.0 or stable for every package, taking precedence over
the repository toolchain pin. To run the stable leg locally:

```bash
rustup toolchain install stable --profile minimal
RUSTUP_TOOLCHAIN=stable python3 validation/validate_release_packages.py
```

Archive test targets are compiled and doctests run without default features,
while Cargo's normal package verification still compiles each crate's default
packaged targets. Packaging does not need an Apple or NVIDIA runner:
target-native Metal and CUDA coverage remains in the platform workflows.

The Linux build workflow separately denies all workspace Clippy warnings and
checks each weakly forwarded facade feature (`metal`, `cuda`, `nccl`, `image`,
and `audio`) without `mlx` at the minimum supported Rust version. These checks
keep optional facade features from accidentally activating the native backend
or depending on its availability.

The `Native release gate` is the single CI entry point. It runs automatically
for code changes on main and pull requests, and can be dispatched manually.
Its native preflight runs ordinary backend tests with normal parallelism,
then all self-contained Ring tests serially, using Metal and Accelerate throughout. Only
after preflight succeeds do the Apple, Linux, Windows, and archive-validation
workflows start. The Apple cross-build matrix therefore cannot delay native
feedback for the same candidate.

Before publication, require a successful **main push or manual dispatch** of
this workflow for the exact release commit. Both include Windows CUDA 12.9.1
and 13.0.2 in the same run, together with the complete platform and archive
checks. Reuse that successful run; do not launch a second release matrix for
an already-validated commit. Pull-request runs cover CUDA 12.9.1 only and are
not the full release gate. The Windows CUDA compatibility workflow retains
nightly and manual coverage of both toolkits. It runs independently of the
macOS queue, so Windows-only retries can start immediately.
Self-hosted GPU execution remains explicit opt-in through the gate's
`run_windows_gpu` and `run_linux_gpu` inputs.

To run the same preflight locally on an Apple silicon host outside a sandbox:

```bash
bash validation/preflight.sh
```

Native initialization failure is an error, never a skip. See
[the development iteration guide](development.md) for focused test commands,
reference-harness filtering, and the independent native build cache.

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
12. `eredu-media`
13. `eredu-architectures`
14. `eredu-codec`
15. `eredu-evaluation`
16. `eredu-backend-mlx`
17. `eredu`
18. `eredu-cli`

This is a valid topological order for normal, build, and development
dependencies. In particular, `eredu-evaluation` precedes
`eredu-backend-mlx` because the backend uses it as a development dependency.
`eredu-media` precedes its architecture consumers. The validator's declared
order is checked against Cargo metadata so CI fails when it becomes stale.

For a local check of uncommitted source, pass `--allow-dirty`. CI and release
preparation should use a clean checkout and the command without that option.
