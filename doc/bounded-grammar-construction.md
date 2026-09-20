# Grammar and schema memory policy

Prepared chat compiles tool policy before generation. Required and Auto tools,
reasoning channels, tagged payloads, literal stops and semantic publication use
shared committed-token drivers in ordinary and controlled sessions. Tool events
describe requests; Eredu does not execute external tools.

## Stock dependency APIs

Eredu uses the published llguidance parser factory and token parser, Derivre,
jsonschema, and their ordinary registry dependencies. Compiled templates retain
stock parser state. An invocation or independent snapshot uses the public deep
copy operation and receives its own tokenizer-error slot and mutable session.
The immutable trie and upstream compiled grammar may remain shared. Preparation
serializes access to the shared compiler environment so one request cannot
consume another request's tokenizer failure.

Schema compilation and validation use the stock jsonschema validator. URI,
numeric, email, Unicode and regular-expression behavior comes from its selected
published dependencies. Public upstream parser and regex resource limits retain
their normal behavior. Eredu does not inspect dependency-private capacities or
intercept allocations inside these libraries.

## Admission and lifetime

`DependencyMemoryPolicy` reserves estimated host headroom before dependency work.
Its default estimate is 64 KiB plus 128 bytes per input byte. JSON estimates use
the serialized input size without retaining an additional JSON string. Operations
and independently mutable sessions have separate reservations; temporary
operation spending is cumulative within its account.

`LoadedModel::prepare_chat_with_grammar_memory` accepts a custom policy for the
prepared grammar, schema operations, parser sessions and copies. Existing
`prepare_chat` uses the default policy. Both methods preserve the request type
and share the same preparation driver. Tokenizer, template-render, model and
native resource policies remain separate.

These reservations are planning estimates, not measurements or enforceable
limits on dependency allocations or process memory. A checked estimate can
refuse with a typed overflow or funding error before the dependency call. The
estimate can still be smaller than the dependency's actual allocation. Input,
cache and concurrency limits remain necessary; increasing headroom does not
change a dependency's algorithmic limits.

Eredu separately admits the buffers and ordered-map nodes it constructs. The
`eredu-collections` map reports the prospective node layout before allocation;
its consumers own keys, values and reservation lifetimes. Stock serde_json uses
its upstream map implementation. Model weights, KV/state caches, activations,
prefill chunks, transfers and native execution retain their own admission and
completion contracts.

Prepared sources, results and failures retain their original accounts. Parser
copies retain the selected memory policy and independent mutable state. Restore
and fork do not refund cumulative copy or observation spending. Funding,
tokenizer and session failures propagate instead of becoming successful partial
output. The existing schema-grammar fallback is restricted to compilation or
grammar-diagnostic failures; it does not suppress funding refusals.

## Semantic channels and validation

Tagged-tool processing shares declaration validation and payload traversal with
its prepared runtime. JSON object/list and XML-style tool profiles retain their
own syntax and termination rules. Unsupported profiles produce typed refusals.
Tentative speculative output remains separate from committed semantic events.

Behavioral tests cover nonzero tool arguments, malformed declarations and
payloads, parser-copy independence, ordinary/controlled parity, source identity,
headroom refusal and retry, retained error custody, and final account retirement.
The portable facade and backend-conformance suites use neutral backends. Native
prepared-chat fixtures additionally exercise media, capture, snapshots and
local parallel execution.

[Text processing](bounded-text-processing.md) describes tokenizer limits and the
shared template compatibility helpers. [Inference validation](bounded-inference-validation.md)
describes native and released-checkpoint procedures and their limits. Exact
selected dependency versions and archive checksums are recorded in Cargo.lock;
the libraries consume versioned, unmodified crates.io dependencies.
