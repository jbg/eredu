# `safemlx` implementation guides

These guides document low-level MLX primitives exposed by `safemlx`. The crate
wraps reusable MLX operations and neural-network building blocks, while
concrete model architectures and complete encoder/decoder stacks live outside
it. Eredu applications normally use the facade and do not need these APIs
directly.

- [Completion events](completion-events.md): graph submission, host
  observation, and same-device stream dependencies.
- [Asynchronous device timing](device-timing.md): execution-timeline timestamp
  boundaries and nonblocking profiling.
- [Host-transfer buffers](host-transfer-buffers.md): storage selection,
  ownership, and asynchronous copy rules.

Backend integration and native build instructions live with
[`eredu-backend-mlx`](../../eredu-backend-mlx/doc/README.md).
