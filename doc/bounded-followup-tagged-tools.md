# Tagged tool payload consolidation

The shared tagged worker, paid runtime state and public XML path are implemented.
The initial inventory below describes the removed gap; validation follows.

## Initial inventory

The original semantic channel source stored only JSON object/list declarations.
Its `Next::Tool` transition rejected tagged parameter declarations before parsing
any function. The ordinary declarative parser already supported exact function,
parameter, optional type and value delimiters, whitespace between tags, paired
value framing, raw strings and schema-sensitive JSON conversion. Those ordinary
branches lived in facade `dialect.rs`, alongside copied schema catalogs.

The consolidation moved exact tagged transitions and value interpretation
to `eredu-text`, with explicit ordinary/enforced allocation policies over the
same producers. Facade policy projects literal declarations and retains actual
parameter schema sources. Runtime owns paid mutable call state and escaped
semantic events, authenticating the original schema/controller declaration.
Full-object validation remains the existing original schema callback.

The shared worker replaced the duplicate tagged transition/value branches while
preserving function-start timing, missing and duplicate parameter rejection,
nullable/enum selection, type annotations, newlines inside values, argument
ordering, incomplete finish, cancellation and independent saved-state copies.
Source/state/value/event growth is paid at its producer before mutation.

The required checks were shared transition split/refusal tests, ordinary tagged
regressions, the portable XML whitespace/history fixture, and semantic snapshot
and refusal/custody tests. Their results are recorded below.

## Implemented worker and validation

Implemented source split: `eredu-text::semantic_channels::tagged` now owns both
wrapper and parameter transitions, incremental owning names/arguments, value
framing and schema-sensitive raw/JSON selection. The facade's ordinary parser
uses that worker with explicit `Unenforced`. Prepared runtime uses it with the
actual invocation account and source-owned parameter callbacks. The old facade
parameter and wrapper transition algorithms were removed.

Parameter validators are compiled once into the original declaration source.
Their compact type projection avoids retaining a second copy of application
schema values. They preserve the ordinary local-property validation scope;
complete-object validation still uses the original full schema. Actual source
census includes each retained validator and all row/name/type backings.

Argument text uses the existing serde Value serialization worker. Its ordinary
entry and funded wrapper now share recursion and scalar serialization; the
funded writer admits actual output replacements and traversal controls before
reaching them. A single pre-admitted serializer error owner transports an I/O
refusal back to the fixed allocation marker without formatting another error.

Mutable call copies declare actual fresh string lengths, name-vector backing and
exact hash-index layout,
then the same clone producer must consume that exact prepaid extent. The runtime
retains the copy authority after the copied strings, including escaping failure
owners. Incomplete tagged calls remain errors, and cancellation cannot publish a
fabricated call end.

Focused shared-worker tests pass six cases: every UTF-8 split point over framed
values and headers, independent copies with exact observed copy requests,
duplicate/missing/unknown/invalid field rejection, and refusal at every reached
call/serialization request, nullable/declared array types, and 2,048 reverse-ordered
parameters with exact insertion order, independent copy and duplicate rejection.
Names use the pinned prospectively admitted hash table; cold property rows use
binary search. These preserve efficient lookup without retaining duplicate values.
The serializer's expected JSON is independent fixed
text in addition to ordinary-path parity.

The public XML whitespace/history fixture passes on the default test thread stack
(14.78 seconds). Its initial run exposed oversized inherited grammar transports:
cold constructor errors travelled through every mutable token callback, which
also moved the complete 3,304-byte lexer value. Mutable causes now contain only
operation failures, and the existing constructor/copy admission pays a lexer
cell before allocation. The source and funding still retire after the lexer.
No parser algorithm, test-stack setting or fixture expectation changed. On the
same AArch64 debug compiler, inspected prologue reservations fell from 560,208 to
166,448 bytes for activation, 408,368 to 116,736 for forcing collection, and
288,288 to 183,056 for grammar startup. Other cold construction frames remain
larger; this is evidence for the exercised default-stack path, not a claim that
all debug frames are small.

The neutral semantic snapshot/custody test passes with a tagged branch alongside
its existing JSON branch. It checks independent completion/refusal, exact start,
argument and end events, incomplete finish and cancellation. The final facade
literal-output and ordinary tagged filters pass 28 tests, with one existing
external-reference oracle ignored. Their coverage includes whitespace split
points, raw string enum boundaries, nullable types, grammar producer refusal,
profile restrictions, skip-special-token policy, caller stops, and manual/run
session parity. A rooted-property reference regression also passes: an unresolved local
reference defers to the complete source validator, which still rejects a bad
nested value. Allocation/unqualified/reference-transport failures are excluded
from that semantic deferral. The strengthened runtime copy with a completed parameter
index also passes, exercising its prepaid hash backing and error custody.

Reproduction uses the authenticated compiler and no test-stack override:

```sh
export CARGO_INCREMENTAL=0
export RUSTC=/Users/jbg/.rustup/toolchains/1.98.0-aarch64-apple-darwin/bin/rustc
cargo test --offline -p eredu-text --no-default-features --features tokenizer-compiler-test-support --lib semantic_channels::tagged::tests
cargo test --offline -p eredu-runtime --lib channel_semantic_owner_preserves_structural_stop_snapshot_and_escaped_custody
cargo test --offline -p eredu --no-default-features --test portable_facade nanbeige::official_nanbeige_xml_whitespace_values_and_history
cargo test --offline -p eredu --no-default-features --lib -- original_token_input::tests::original_chat tagged_
```
