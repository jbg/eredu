# Vendored MLX C

This directory contains the MLX C API used by `safemlx-sys`. MLX C exposes the
MLX array framework to C and provides the native boundary for the SafeMLX Rust
bindings.

SafeMLX builds this copy through `safemlx-sys`; Rust applications should not
configure it separately. Use [`safemlx`](../../../safemlx/) for the safe API or
[`safemlx-sys`](../../) when direct binding-level access is required.

The source retains its component license, code of conduct, contribution guide,
and acknowledgments. Public upstream API documentation is available from the
[MLX C documentation](https://ml-explore.github.io/mlx-c/).
