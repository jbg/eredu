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
adding that archive to an ephemeral local registry. This lets the next crate
resolve the exact unpublished workspace version while retaining Cargo's normal
package verification. Nothing contacts a registry publishing API and no
credentials are required.

This catches:

- path dependencies without publishable versions or versions that do not match
  an earlier workspace archive;
- files omitted from the generated archive that cause package, packaged unit
  test compilation, or packaged doctests to fail;
- archives above crates.io's 10 MiB compressed-size limit; and
- new publishable crates or dependency changes that are missing from the
  declared order.

The release-package CI job runs this validation on Ubuntu with the CPU MLX
prerequisites. Archive unit tests are compiled and doctests run without default
features, while Cargo's normal package verification still compiles each crate's
default packaged targets. Packaging does not need an Apple or NVIDIA runner:
target-native Metal and CUDA coverage remains in the platform workflows.

## Publication order

Publish one crate at a time in this order, waiting for each version to become
available in the registry index before continuing:

1. `eredu-gguf`
2. `safemlx-internal-macros`
3. `safemlx-macros`
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
