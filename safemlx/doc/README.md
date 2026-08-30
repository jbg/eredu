# `safemlx` implementation guides

These guides document low-level MLX bindings exposed by `safemlx`. Framework
abstractions—including neural-network layers, quantized modules, mutable
execution contexts, checkpoint materialization, and model composition—live in
`eredu-backend-mlx` or a backend-neutral owning crate. Eredu applications
normally use the facade and do not need these APIs directly.

- [Completion events](completion-events.md): graph submission, host
  observation, and same-device stream dependencies.
- [Asynchronous device timing](device-timing.md): execution-timeline timestamp
  boundaries and nonblocking profiling.
- [Host-transfer buffers](host-transfer-buffers.md): storage selection,
  ownership, and asynchronous copy rules.

Backend integration and native build instructions live with
[`eredu-backend-mlx`](../../eredu-backend-mlx/doc/README.md).
