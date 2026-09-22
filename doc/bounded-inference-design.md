# Bounded inference design

Ordinary prepared chat supports admitted semantic tools and authenticated media
through the same execution driver as controlled sessions. Speculation consumes
the same preparation and commitment mechanisms and supplies its own drafting
and verification state.

## Request preparation

`LoadedModel::prepare_chat` compiles the loaded tokenizer and template against a
borrowed request. `PreparedChat` retains the render, policy, compilation receipt
and authenticated capacity. `PreparedChatRequest` accepts rendered text, exact
token IDs or an `OriginalModelInput` produced by `prepare_chat_input`.
`start_prepared_chat` returns a session whose `advance` and `run` methods share
one committed-token cursor. Ordinary tool calls require ordinary generation
support, with no speculative backend requirement.

`prepare_chat_invocation` checks source identity, input, capacity and resolved
settings, then prepares the controller, semantic channels and stops. Speculative
single, batch, controlled and observed consumers use this worker. Exact token
prefixes are validated without decoding and re-encoding. Native speculative
media binds each role to the authenticated host source and its actual cache.

See [prepared chat](prepared-chat.md) and the
[integration example](../eredu/examples/prepared_chat_generate.rs).

## Ownership

Source compilation reserves concrete destinations before tokenizer, template,
grammar, schema and semantic construction. Equal-valued copies, hashes or caller
ranges do not confer authority. Immutable render, grammar and declarations share
their paid owners; mutable parser, sampling, cursor and continuation state have
independent construction and copy admission.

Media binding retains completed host/native sources and architecture-declared
coordinates. It compares ordered text tokens and exact framing. Prefill chunks
share the prepared ingress. Capture, observation, transport, snapshot and
attempt quotas remain distinct from host-allocation admission.

Native admission survives until completion, terminal failure or safe teardown.
A polling error cannot establish completion. Escaped events, outputs, errors,
input aliases and snapshots retain their original payers. Restore, fork and
branch exchange preserve cumulative observation, transport, copy and attempt
spending. Terminal restoration uses empty execution geometry and emits no
prediction.

## Mechanism ownership

| Concern | Owner and contract |
| --- | --- |
| Chat, tools, termination | Facade preparation and the shared committed cursor. |
| Tokenization | One added-token matcher and shared model workers with explicit cache capacity. |
| Capture and records | Fallible custody-preserving delivery and paid prompt attribution. |
| Parameters and metadata | Borrowed contracts; consuming boundaries fund their owned destinations. |
| Layer acquisition | Shared retirement, transfer, lease and population workers with explicit recovery policy. |
| Native copy and reset | Grouped state copies and source-authenticated manager construction for KV and Hybrid state. |
| Cancellation | Boundary-aware agreement after retirement, using the exact request and control occurrence. |
| Errors | Typed retained causes with prospective constructor funding and neutral public translation. |

Checkpoint formats, protocol dialects, family equations, devices and residency
policies select real supported mechanisms. Unknown required bounds produce typed
refusals. The [architecture guide](backend-architecture.md) defines crate and
feature boundaries; [source contracts](bounded-source-contracts.md) describe
admission and escaped ownership.

Framework-managed accounting includes registered MLX backing retained by its
allocator cache. Application buffers, copied callbacks and unrelated process
memory remain outside it; unpriced dependency internals remain unknown
contributions. Physical-domain limits constrain the charged allocations and
allowances, not measured RSS. [Validation](bounded-inference-validation.md) records the
tested configurations, performance tradeoffs and hardware limitations.
