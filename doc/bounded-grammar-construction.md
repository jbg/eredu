# Bounded grammar and schema construction

## Controller and semantic channels

`ConstraintController` contains plain control, an original registered constraint,
or a shared conditional control state. Prepared chat compiles policy before
generation. Required and Auto tools, reasoning channels, tagged payloads, literal
stops and semantic publication use the same committed-token driver.

Tagged-tool processing shares declaration validation and payload traversal with
its funded runtime. JSON object/list and XML-style tool profiles keep their
own syntax and terminal rules. Unsupported profiles are typed refusals rather
than implicit plain-text fallback. Tool events describe requests; the library
does not execute external tools.

## Allocation ownership

Parser forks own their actual allocation facts and growth checks. Runtime owns
admission policy, accounts and escaped source custody. Cold grammar/schema
inspection, compilation, copy and JSON/Lark emission reserve before constructing
their destinations. Funding a completed object cannot authorize its construction.

The parser holds one borrowed frame workspace per chart operation. Derivre's
shared frame mechanism pays its controls before construction and grows the paid
capacity only when simultaneous recursive frames exceed the retained peak.
Sequential temporary frames reuse that peak; persistent heap growth and
copy/attempt spending remain cumulative. The mechanism has no ambient or
thread-local funding fallback.

Failures retain the first typed cause and the actual failing compiler prefix,
including completed aggregates. Errors, compiled sources and invocation caches
retain their payers through escaped ownership. A refused reserve cannot be
converted into a different regex, partial schema or successful partial output.

## Schema and reference producers

The JSON Schema compiler funds registry construction, crawl/index work,
resolvers, keyword contexts, source text, paths, diagnostics and numeric/format
producers. URI, email, IDNA, normalization, fraction and number conversion use
their selected dependency workers. Ordinary and funded construction preserve
ordering, error precedence and schema semantics.

The local JSON map uses `eredu-collections`' safe AVL worker. Prospective node
callbacks report exact layouts before allocation. Map keys, values, diagnostics
and account lifetime remain with the caller. IndexMap preserves insertion order
and shares its ordinary and funded reservation worker.

## Regular-expression compilation

`jsonschema-regex` owns ECMA translation, AST traversal, capture names, literal
optimization and witness construction. The selected fancy-regex 0.19.0 worker
owns parsing, analysis, compilation, delegated engines and backtracking state.
Its source census includes immutable programs and invocation-owned caches.

Regex syntax funding covers AST/HIR construction, Unicode classes, interval
operations and literal extraction. Thompson compilation funds state tables,
literal and range tries, UTF-8 caches, capture metadata and suffix caches. Dense
DFA construction funds transitions, starts, determinization, minimization,
acceleration and shrinking. Lazy DFA caches fund prospective state/transition
growth and cache reset without discarding cumulative work accounting.

PikeVM and bounded backtracking use their selected search algorithms. Capture
slots, active states, epsilon/restore stacks and UTF-8 empty-match handling have
explicit workspace bounds. Literal prefilters use ordinary Aho-Corasick/memchr
algorithms with prospective construction hooks; no alternate search policy is
selected merely because funding is enforced.

Safe workspace members inherit `unsafe_code = "forbid"`. External collection
and literal-search dependencies retain their documented upstream implementation
boundaries; local allocation hooks are safe Rust. Archive hashes, licenses and
local source inventories are recorded under `third-party`.

## Validation

Behavioral coverage exercises independent matching/format oracles, reserve
failure at reached construction cuts, retained immutable ownership, parser
copies and refusal propagation. The dependency compiler suite has 90 passing
cases and the facade controller suite 47 in the scoped validation record.
These counts do not establish arbitrary custom callback bounds or every regex
feature combination. [Validation](bounded-inference-validation.md) describes
the public and native limits; `doc/validation/*-compare.rs` contains independent
comparison harnesses.

## Independent comparison harnesses

Run `doc/validation/standard-regex-compare.rs` in an isolated Cargo project outside
the workspace. Alias the local `third-party/regex-1.13.1` package as `current`,
a pristine registry `regex = "=1.13.1"` as `reference`, and the local
`regex-automata-0.4.18` as `current_automata`. The local wrapper's dependency path
must resolve to the local automata/syntax workers; the reference dependency graph
must resolve to pristine registry packages. Do not use a global patch that sends
both aliases through the same implementation. Match feature selections and
record `cargo tree` output with results to identify the actual reference graph.

The harness compares acceptance, error variants, captures, replacements, byte
matching and sets, then measures warmed search and construction independently.
Other `*-compare.rs` harnesses declare their required aliases in their imports;
use the pinned versions from the relevant upstream manifest. Check archive
hashes before preparing pristine source copies and keep generated projects and
outputs outside the tracked tree.
