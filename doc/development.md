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
image, Xcode and Metal or the CUDA toolkit, and Linux CUDA/cuDNN/NCCL package versions. Its
native key also includes the native sources and build scripts, but excludes
Rust crate manifests and Cargo.lock. The toolkit variant includes Windows
CUDA/cuDNN versions and target architecture. Rust artifacts use a separate
key with the Rust toolchain and Cargo dependency versions. Only the shared
macOS preflight build caches Rust artifacts: caching full Rust test trees for
every matrix entry would quickly exhaust the repository's 10 GB budget and
evict the expensive native builds. Native files must be available before
restoring Rust artifacts that may refer to them.

Successful native builds are saved before downstream tests and Rust builds,
so a later failure does not throw away the expensive native compilation.
Preflight finishes and saves its Rust cache before the macOS workspace job
starts, allowing that job to reuse its build as well.

## CI ordering

The Native release gate calls preflight before any of the broad platform or
archive jobs. Windows CUDA 12.9 is built once per candidate; main and manual
release runs add CUDA 13.0 in that same workflow. Apple cross-builds start only
after native feedback has succeeded. A successful main/manual gate already
validates that exact release commit, so publication does not need another
identical matrix. The first build under a new native cache key still needs to
compile the native library; subsequent Rust-only changes reuse it.
