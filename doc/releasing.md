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

## Git lineage notices

When retiring an API experiment or rewriting a branch used by consumers, preserve
the displaced history under a named archive ref and add a dated
[changelog entry](../CHANGELOG.md) linked from the README. Name the affected public
APIs, the last previously published branch revision, the immutable archive snapshot,
the base or replacement lineage, and the pinning or migration path. Distinguish an
archival checkpoint from a validated release. Publish that notice with the branch
change so consumers do not have to reconstruct the transition from diffs or reflogs.

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

The validator copies the Git package candidates to a content-identified workspace, then
runs `cargo package` for each crate. It unpacks each archive, compiles its
library unit tests, and runs its doctests with default features disabled before
adding that archive to its own content-identified local registry. After staging a library, it
also checks a new lock-free downstream crate that depends on the staged package
with default features disabled. This lets consumers and the next workspace crate
resolve the exact unpublished version while retaining Cargo's normal package
verification. Package builds use a temporary target directory by default, or retain compiler
outputs in an explicit `--target-dir` for subsequent runs. Nothing contacts a registry publishing API and no credentials
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

The `Native release gate` is the single CI entry point. It runs on main and
pull requests, nightly, and by manual dispatch. A small planning job compares the
candidate with the most recent successful **full** main gate; a version-only
follow-up therefore cannot hide the preceding source change. The plan is printed
in the job summary and retained as the `release-plan` artifact. Before the first
new full run, the baseline is `084458c3`, verified by
[the September 7 full gate](https://github.com/jbg/eredu/actions/runs/34130046331).
Expired or missing newer receipts fall back to that older baseline.

Scoped eligibility is intentionally narrow and follows changed source, not the
SemVer number. Changes spanning more than 20 files or 1,000 added/deleted lines
require full validation. The initial reviewed scope covers Qwen hybrid checkpoint recipes,
replicated-text lowering, and the private grouped/binding implementations used by
the FP8 fix. It still runs architecture tests, the ordinary native Metal backend
suite (including FP8 resident/addressable regressions and required native-device
admission), portable facade conformance, and native CLI consumers on macOS,
Linux, and Windows. Changing public declarations, platform/feature attributes,
external dependencies, native/build/toolchain inputs, other production code, or
the scope rules themselves requires the full matrix. The classifier is a
conservative verification selector, not an API compatibility checker.

Full validation retains workspace and Ring tests, all six Apple cross targets,
Linux CPU/MSRV/CUDA/NCCL, and Windows CPU plus CUDA 12.9.1 and 13.0.2. Nightly
runs and the dispatch `full` input always select full validation. Self-hosted GPU
execution remains explicit opt-in through `run_windows_gpu` and `run_linux_gpu`.
Linux, Windows, and archive jobs start alongside native validation. Ordinary
backend tests and ignored Ring cases each run once in the combined macOS job.
Apple cross-builds have at most four concurrent builds, reserving the fifth
standard macOS slot for native validation. Target-mapping checks share the native
job rather than taking another macOS runner. The full matrix can therefore run
alongside native validation without a second serial barrier.

Before publication, require a successful **main push or manual dispatch** gate
for the exact release commit and inspect its release plan. Either automatically
selected scope is valid; do not manually omit a failed required check. Reuse that
successful run rather than launching a second identical matrix. These are
verification workflows; they never publish crates or create release tags.

For a bounded release, validate just its release roots and unpublished workspace
dependencies (normal, build, development, optional and target-specific):

```bash
python3 validation/validate_release_packages.py \
  --packages eredu-architectures eredu-backend-mlx eredu eredu-cli \
  --target-dir target/release-validation
```

Unselected published dependencies resolve from crates.io, rather than being
repackaged from the workspace. Registry lookup errors fail validation; they do
not count as proof that a dependency is published. The CI plan supplies pending crate roots in publication order, comparing each
crate with its own current-version release tag. Previously published scoped
changes therefore do not accumulate in the next release's archive set. Infrastructure-only full checks and
nightly audits validate every package. Both stable and Rust 1.89 archive jobs
remain mandatory. The target directory can persist across runs. Each staged
registry is identified by the archive checksum and its complete index record,
including the exact registry identities of its staged dependencies. Identical
archives and dependencies therefore reuse their identities across runs;
different bytes at the same name/version receive a different identity. Adding
a later crate does not change the identities of earlier dependencies. Staged
source paths are content-identified and their mtimes are normalized, while
downstream consumers still start without lockfiles. A process lock protects
staging, and temporary sources and registries are removed at the end. Cargo's
normal package verification, packaged tests, doctests and downstream checks
remain mandatory on both toolchains.

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
