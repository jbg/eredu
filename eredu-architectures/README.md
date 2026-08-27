# eredu-architectures

`eredu-architectures` provides Eredu's backend-neutral text, multimodal, and
realtime model families. It owns model configuration, checkpoint schemas,
parameter topology, state geometry, parallel semantic plans, and model
execution equations.

Architecture implementations are generic over the traits from `eredu-nn`.
Concrete backends retain control of tensors, storage, graph construction,
kernel fusion, streams, caches, and collectives. This keeps every model family
available without enabling MLX, CUDA, or another native runtime.

Most applications should use
[`eredu`](https://github.com/jbg/eredu/tree/main/eredu). Backend authors and
model-family integrations can use this crate with `eredu-runtime` and
`eredu-nn` to assemble portable execution.

See the [backend architecture
guide](https://github.com/jbg/eredu/blob/main/doc/backend-architecture.md) for
the ownership and dependency rules.

## License

Licensed under either Apache-2.0 or MIT.
