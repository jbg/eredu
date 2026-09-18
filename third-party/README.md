# Pinned parser sources

These workspace members provide dependency-owned allocation facts and checks
before parser-table growth. They are selected with `[patch.crates-io]` in the
workspace manifest. Backend selection, resource custody and memory admission
remain in Eredu's existing neutral contracts and runtime.

| Package | Upstream archive | SHA-256 |
| --- | --- | --- |
| derivre | 0.3.12 | `a892ec3953f4ca893e8d2086ed03d4b58e19c77ffb864a574b332d7d40c175f3` |
| llguidance | 1.8.0 | `207e72ede15e79f1e7a7f0e2683387da031c1931f6de650020afb80c571efb67` |
| regex-syntax | 0.8.11 | `d6f6ff9a378485b298a5286656da665ba74413d36db0979633275d2e708145d4` |
| referencing | 0.52.1 | `d38a014525040cdc9893361b7419bcf1f43b7ba7055eabaf6d91d6b75caa8b3f` |

The crates.io archive hashes match the pinned package versions. `parser-upstream.json` records those hashes, each original
file's SHA-256, and the archived VCS metadata. Archive hashes are authoritative;
llguidance's archived VCS metadata reports a dirty checkout. Each package retains
its upstream licenses, original manifest and source artifacts.

These forks inherit the workspace `unsafe_code = "forbid"` lint. The local
llguidance package builds only its Rust library. Its upstream C-ABI modules,
static/shared C libraries, header generation and header-copy build script are
excluded from compilation. The original C-ABI sources remain as provenance
artifacts; this fork does not expose that API. No portable unsafe-code exception
is introduced. Published third-party dependencies outside these forks retain
their own upstream boundaries.

The llguidance archive includes integration-test sources that import an
unpublished `llg_test_utils` helper, and a benchmark that references a schema
outside the archive. Their targets and unused test dependencies are omitted from
the local manifest; their original sources remain recorded. The self-contained
upstream library tests, local public-parser tests and Eredu's grammar conformance
tests are executable. Derivre's packaged upstream tests remain enabled.

Compiled llguidance grammars have one closed shared owner used by ordinary
parsers and funded Earley state. Publication reserves its shell through the actual
compiler account; final retirement frees the shell before the immutable graph
and its account. The factory accepts that shared source without a deep copy.
There is no infallible deep clone of a compiled grammar. Explicit independent
copies remain fallible operations with their own construction requirements.

The hash-cons table's retained-capacity API describes its owned vector capacity
and hash-table allocation. A prepared table may insert within its fixed
capacities without allocation, including duplicate lookup when full. Preparing
those capacities itself allocates and requires the caller's existing authority;
a requested element count is not a byte-budget certificate.

Incremental insertion reserves explicit scratch headroom and writes directly at
the uncommitted frontier. Successful insertion publishes those same words;
duplicate, rejected, dropped or unwound insertions clear their written scratch.
Write failures remain terminal for that insertion. A deliberately forgotten
guard leaves the owner fenced against further mutation until destruction, while
committed lookups remain valid. Scratch contributes to the reported capacity.

Expression encoding shares one implementation between the ordinary arena and
the prepared writer. The raw API checks encoded length, preserves flags and byte
layout, and returns a table-local `u32` identifier. It does not seed an ExprSet,
validate child references, or supply a complete expression/cache memory bound.

The lexer checks a new state's count and representable table geometry before
inserting its interned description or growing descriptor/transition storage.
Existing states remain reusable at the limit. These are local table guarantees:
derivative computation, tokenization, grammar compilation, mutable parser state,
copies and other caches still need complete accounting before finite managed
grammar admission can succeed.

The state cap is installed before the two sentinel states are built. Limits below
two reject with a typed configuration error before grammar compilation. Initial
states, first-byte warming, optional large-lexeme precomputation, initial parser
rows and skip selection all preserve the original typed cause on failure; a
failed lexer is never published as a successful parser. Independent clones keep
the same per-instance cap. Other compiler/parser allocations still require their
own bounds.

The regex-syntax fork adds borrowed allocation admission to the existing AST
parser and HIR translator. Ordinary construction supplies the unenforced policy
to those same workers. The dependency reports its actual box, vector and string
producers; it does not depend on host accounts or backend resources. Iterative
tree teardown uses explicit geometric stacks, with storage credit reserved
before node creation. HIR move/retirement sentinels and diagnostic formatting do
not allocate. The caller retains its funding owner through syntax, parser and
error retirement. Integration status and focused validation are recorded in
`doc/bounded-grammar-construction.md`.

Matcher failures preserve the known closed configuration/parser causes and their
diagnostic text. They do not retain arbitrary constructor errors that might own
unrelated resources. Diagnostic cause inspection after a caught callback panic
does not block on or recover a poisoned lexer; it preserves the original failure.

TokenParser's ordinary and deep copies use one exhaustive field-copy helper.
Deep copying constructs one independent parser with its history, caches, limits
and errors.
Ordinary copies still share the lexer. Independent-copy allocation authority and complete byte bounds remain separate
requirements.

When updating a fork, verify the replacement archive, refresh provenance, review
the local changes against the recorded upstream files, and run its relevant
upstream and Eredu behavior tests. Do not edit Cargo's registry cache in place.

## Tokenizer source ownership

`tokenizers-upstream.json` identifies the external HF tokenizer 0.23.2 archive,
licenses, upstream files and local source delta. The fork keeps one model
representation and one packed added-token matcher for ordinary and admitted
execution. BPE/WordLevel/Unigram construction, normalization, pre-tokenization,
template processing and ID output retain their original owners. An explicit
`ModelCachePolicy` selects cache entry count; it is not a byte bound.

Split and ByteLevel retain checked regex source ownership and workspace plans.
General arbitrary syntax and Oniguruma remain selected mechanisms. Errors
propagate instead of yielding partial successful tokenization. See
[bounded text processing](../doc/bounded-text-processing.md) for supported
composition, semantics, storage bounds and reproducible measurements.

## Regex and schema mechanisms

The selected fancy-regex 0.19.0 engine serves tokenizer and schema consumers.
Regex syntax, Thompson construction, dense/lazy DFA, PikeVM, backtracking and
literal prefilters report their own prospective allocation facts. Ordinary and
funded callers select the same algorithms. Shared immutable programs and
invocation caches retain separate ownership.

`jsonschema-regex` owns ECMA translation and syntax/witness producers.
`referencing` and `fluent-uri` own registry and URI construction. Numeric, email,
IDNA and Unicode format dependencies fund their selected producers. Caller
runtime accounts own admission and failure custody, never the parser forks.
See [grammar construction](../doc/bounded-grammar-construction.md).

## Collections and literal search

The JSON ordered map uses `eredu-collections`' dependency-free safe AVL worker.
Its exact prospective node-layout callback precedes allocation; keys, values and
account lifetimes remain the consumer's responsibility. `hashbrown` and
`indexmap` remain external dependencies with their upstream unsafe boundaries.
Local growth callbacks are safe Rust and do not weaken portable workspace lints.
IndexMap shares one ordinary/funded reserve worker and preserves insertion order.

`aho-corasick` and `memchr` remain external literal-search dependencies with
upstream search/SIMD internals. Local prospective construction hooks add no unsafe
code. `prefilter-upstream.json` records archive hashes and local source facts.
These boundaries are also specified in [AGENTS.md](../AGENTS.md).

## Validation and provenance

Archive hashes and preserved upstream licenses identify dependencies; local
file hashes and patch series identify modifications. Patch ordering expresses
reproduction dependencies. Regenerate the applicable manifest when changing
vendored sources, and preserve the upstream archive and license inventory.
Upstream distribution documentation is retained with its provenance.

Portable checks include `cargo check -p eredu-text --no-default-features`,
`cargo check -p eredu-runtime` and `cargo test -p eredu --no-default-features --lib`.
Independent source-comparison harnesses live in `doc/validation` and package
`validation` directories. Behavioral tests cover value parity, reached reserve
failures, escaped owners and final retirement; source spelling is not an
architectural enforcement mechanism.
