# eredu-runtime

`eredu-runtime` provides backend-neutral model execution and resource
orchestration for Eredu. It coordinates opaque backend-owned values through
portable contracts for parameter binding, mutable state, cache and weight
residency, transfers, collectives, generation, speculative decoding, and
realtime execution.

It owns the canonical declarative parameter-binding and logical placement
plans, communication-manifest validation, and reusable replicated-session
construction flow. Backends supply statically dispatched native
materialization, completion, storage, group, and tensor mechanisms to those
plans.

The crate does not depend on a model-family implementation or a concrete
backend. Architectures declare their execution and state semantics in
`eredu-architectures`; backends implement the capabilities required to realize
those declarations.

Most applications should use
[`eredu`](https://github.com/jbg/eredu/tree/main/eredu). Use this crate directly
when building architecture or backend integrations.

See the [backend architecture
guide](https://github.com/jbg/eredu/blob/main/doc/backend-architecture.md) for
the ownership and dependency rules.

## License

Licensed under either Apache-2.0 or MIT.
