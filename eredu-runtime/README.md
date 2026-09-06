# eredu-runtime

`eredu-runtime` provides backend-neutral model execution and resource
orchestration for Eredu. It coordinates opaque backend-owned values through
portable contracts for parameter binding, mutable state, cache and weight
residency, transfers, collectives, generation, speculative decoding, and
realtime execution. `NormalizedLoadRequest` is the singular portable cold-load
policy: topology, wire, invocation bounds, completion, residency, session, and
drafting intent are validated before any backend resource is selected.

Execution plans normalize through one shared converter with injected diagnostic
choices. Capability synthesis enumerates exact parameter and state requirements
and queries backend support predicates; providers report mechanisms without
repeating that enumeration. Selected-task residency sizing and report telemetry
are portable runtime operations.

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
