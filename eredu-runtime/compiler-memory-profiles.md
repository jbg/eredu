# Compiler memory profiles

The runtime uses public APIs of unmodified Rust releases. Its exact managed
metadata extents are qualified against the compiler, driver, `liballoc` and
`libstd` artifacts in `build.rs`. A different compiler or standard library
remains an attribution error under both finite and unlimited memory limits.
The profile does not bound malloc-private bookkeeping, kernel resources, or
process residency and does not apply `MemoryOverheadPolicy` to inference.

Rust 1.98.0 profiles cover `aarch64-apple-darwin`,
`x86_64-unknown-linux-gnu` and `x86_64-pc-windows-msvc`. The official
[release manifest](https://static.rust-lang.org/dist/channel-rust-1.98.0.toml)
identifies the distribution archives. `compiler-memory-profiles.json` retains
their URLs and SHA-256 digests, the audited source archive, and the source-file
digests. The compiler release and commit are checked in addition to artifact
hashes. Wrappers, bootstrap overrides, replacement sysroots, external standard
libraries and dynamic standard-library linking do not qualify.

The source audit establishes these managed allocation facts:

- `Global::alloc_impl_runtime` returns the requested layout extent. Fresh
  `Vec::try_reserve_exact` uses `RawVec::grow_exact`; fresh clones use
  `slice::to_vec_in` and request the source length. Spare source capacity is
  not cloned. Arbitrary element cloning and `clone_from` are outside this fact.
- `ArcInner` and `RcInner` have two `usize` counters followed by the payload,
  with `repr(C, align(2))`. Runtime quotations use checked `Layout` composition
  and target-specific payload sizes, including final padding.
- Unix OS strings own byte vectors. Windows OS strings own a `Wtf8Buf` with a
  byte vector and an inline validity flag; fresh cloning follows the same vector
  worker. This qualifies the encoded-byte backing, not arbitrary conversions.
- Linux and the ordinary Windows target select inline futex mutex and condition
  variable implementations. They allocate no additional Rust heap payload.
  Darwin selects `OnceBox` owners around `pthread_mutex_t` and `pthread_cond_t`.
  The ledger charges both payloads and initializes them before sharing the
  coordinator, so competing first users cannot allocate duplicate lazy owners.

To reproduce provenance verification, download each listed archive from its
official URL, verify its SHA-256, extract it outside the repository, and compare
the compiler/driver/library digests with `build.rs`. Extract `rust-src` and compare
the reviewed file digests with the JSON record. Native validation uses the actual
compiler path reported by `rustup which rustc` as `RUSTC`; a rustup shim is not a
qualified compiler artifact. Profile identity is not native hardware validation.

The profiles ship in the published runtime crate. Consumers use ordinary
versioned Cargo dependencies and the selected official toolchain; no patched
dependency, repository checkout, or Cargo source override is required.
