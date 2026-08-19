# Vendored MLX C

This directory contains the MLX C API used by `safemlx-sys`. MLX C exposes the
MLX array framework to C and provides the native boundary for Eredu's MLX
implementation.

Eredu builds this copy through `safemlx-sys`; Rust applications should not
configure it separately. Use [`eredu`](../../../eredu/) for the model runtime,
[`safemlx`](../../../safemlx/) for direct MLX operations, or
[`safemlx-sys`](../../) when binding-level access is required.

The source retains its component license, code of conduct, contribution guide,
and acknowledgments. Public upstream API documentation is available from the
[MLX C documentation](https://ml-explore.github.io/mlx-c/).
