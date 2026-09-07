# Development iteration

On Apple silicon, run the shared native preflight before starting a broad
platform build:

```sh
bash validation/preflight.sh
```

This checks native MLX availability, runs ordinary backend tests with normal
parallelism, and then runs every self-contained Ring case serially. All three
commands use `--features metal,accelerate --lib`, so changing test selection
does not rebuild the backend with a different feature set. The two production-model
Ring parity cases still require their external model fixtures and remain
explicit opt-in checks; see [releasing](releasing.md).

For an individual edit, start with the affected test and then run preflight:

```sh
cargo test --locked -p eredu-backend-mlx --features metal,accelerate --lib <test-name>
cargo test --locked -p safemlx --features metal,accelerate --lib concurrent_safe -- --test-threads=1
```

The second command targets only the two SafeMLX concurrency tests. Avoid a
workspace-wide command when only one crate's tests are needed.

## Reference conformance selection

The custom reference harness accepts Cargo-compatible filters, `--exact`,
`--skip`, `--list`, and ignored-test selection through `libtest-mimic`:

```sh
cargo test --locked -p eredu-architectures --test reference_conformance -- --list
cargo test --locked -p eredu-architectures --test reference_conformance typed_extension
cargo test --locked -p eredu-architectures --test reference_conformance -- --skip speculative
```

A filter with no matches executes no cases. Listing also executes no cases.
The eight aggregate suites run serially by default; `--test-threads` can
explicitly override that. The existing `EREDU_REFERENCE_CASE` environment
selector still selects an exact case and intersects with CLI filters. An
unknown environment selector is an error instead of a successful empty run.

## Native build reuse

CI stores the native CMake tree separately from Cargo's `target` directory.
The opt-in local equivalent is:

```sh
export SAFEMLX_NATIVE_BUILD_ROOT="$PWD/.native-build/local"
bash validation/preflight.sh
```

The root must be absolute and should have only one Cargo build writing to it
at a time. Each target, Cargo profile, and native feature configuration has
its own subdirectory. Ordinary builds without this variable retain Cargo's
default build-directory behavior. Changing a Rust package version or
`CARGO_TARGET_DIR` no longer discards native compilation when this cache is
used. CMake still configures the project and tracks compiler flags, sources,
and the content-identified MLX patches. `cargo clean` does not remove this
external native directory; remove the selected `.native-build` tree explicitly
when a completely fresh native build is wanted.

The shared CI cache action identifies the installed compiler, CMake, runner
image, Xcode/Metal or CUDA toolkit, and installed Linux CUDA dependencies.
Native build keys include native sources and build scripts independently of
Rust manifests and package versions. A native tree must be available before
restoring Rust build-script outputs that refer to it.

Native macOS, Apple cross-builds, and stable/MSRV archive validation restore
separate Rust snapshots. Successful main builds save up to 768 MiB for each
Apple cross target, 1 GiB for each archive toolchain, and 1.5 GiB for native
macOS. Pruning removes incremental and packaged outputs, then evicts whole
Cargo units with their fingerprints and dependency files. Native build support
and dependency libraries take priority; test and example executables remain
when they fit, avoiding unnecessary recompilation. Cargo rebuilds any evicted
outputs normally. Native and archive test builds disable
Rust debug information and incremental compilation to reduce snapshot size.
Archive jobs retain `target/release-validation`. Each staged registry has an
identity derived from its exact archive checksum and dependency records, so
identical inputs can reuse compiled outputs and changed unpublished archives
cannot reuse old registry contents. Source paths and mtimes are reproducible;
staging is exclusively locked and cleaned after validation.

Rust keys include the source commit and fall back across compatible dependency
versions, allowing the snapshot to improve after source-only fixes. Gate cleanup
keeps only the newest snapshot per configuration, with a 4 GiB aggregate Rust
budget across all cache revisions, and removes the superseded `rust-v2`
snapshots. The `rust-v4` keys fall back to compatible `rust-v3` snapshots during
migration. Archive subsets have distinct exact keys on the same commit but
share fallback artifacts. Exact hits are not saved again. Cleanup leaves all native
caches untouched. Pull requests restore caches but do not save Rust snapshots or
run cache cleanup. Missing caches only affect performance, never test selection.

## CI ordering

The release planner selects scoped or full verification from actual changes
since a successful full main run; see [releasing](releasing.md). Linux, Windows,
and stable/MSRV archive jobs run alongside combined native macOS validation.
Native backend tests are not repeated in a second macOS workspace job. Full
Apple cross-builds use at most four concurrent runners, reserving the fifth
standard macOS slot for native validation instead of waiting for it to finish. Nightly and explicitly requested full runs retain the entire platform
matrix. A successful main/manual gate validates the exact release commit, so
publication does not need another identical matrix.

Compatible target selections share one Cargo invocation (for example, the
Windows CUDA library, completion-event tests, and smoke executable). Apple
backend and facade builds share a feature configuration. Standalone safemlx,
portable facade, weak facade features, isolated audio, and minimal native
feature checks remain separate. Native, Apple, archive and CUDA jobs retain
Cargo timing reports for seven days; archive commands also report elapsed time.

For a Windows-only retry, dispatch `Windows CUDA compatibility`; it covers both
toolkits without waiting for macOS.
