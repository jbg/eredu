# Dynamic regex allocation evidence

The dynamic constructor, selected lower engines, prospective invocation growth
and exact source census are implemented. The inventory and intermediate
checkpoints below preserve their original scope; they do not describe current
missing constructors. Ordinary and enforced operation enter the same workers.

Inventory before implementation, following the canonical 0.19 engine upgrade.
The closed tokenizer recipe constructor is already measured and funded; it
cannot admit arbitrary schema regex source or ordinary pooled search storage.
No completed general-engine funding claim follows from the version upgrade.

Canonical owner and planned changes:

- `fancy-regex::parse`: original AST vectors, strings, boxes and capture Arcs;
  named-group and name-position maps; dense backreference sets; Unicode property
  rewrites and allocated diagnostics. Pass one borrowed neutral allocation
  policy through the original parser and resolver, with fixed refusal errors.
  Ordinary entry points use explicit `Unenforced` on this same worker.
- `analyze`, `optimize`, `to_hir`, `compile`: original recursive/iterative controls,
  group/subroutine graph tables and sets, expression/HIR clones, VM instruction
  vectors, rewritten pattern strings, class ranges, delegate configuration,
  error boxes and final source owners. Every actual growth must precede mutation
  with its complete prospective destination request. Preserve upstream feature,
  optimization, recursion and match semantics.
- The registry `bit-set`/`bit-vec` representation hides its growing word vector
  and exposes unsafe mutation to its own adapter. The parser only requires a
  dense ascending integer set. Replace this storage with one safe word-vector
  implementation exposing its real growth; retain its dense complexity and
  ascending iteration, rather than adding a separate paid membership algorithm.
  Name/group maps should use the existing local hash table's exact prospective
  layout hook, preserving expected constant-time lookup and insertion.
- `regex-automata`: NFA, DFA, HIR lowering, meta-engine selection, literal/Unicode
  helper tables, captures/group indices and shared control allocations require
  hooks in their original producers. This is an independent lower-level slice;
  copying a completed automaton or quoting pattern length cannot cover it.
- Runtime: a validation invocation can retain one caller-owned scoped search
  workspace with the borrowed policy until its actual buffers retire. It must
  enter the existing VM and selected delegate workers. It cannot charge a
  transient payer for storage retained by the compiled regex's ordinary pool.
  Preserve ordinary behavior through an explicit storage policy, never choose
  a different regex algorithm solely because funding is enforced.

The source callback is `regex_syntax::allocation::Allocation`; its fixed
`AllocationError` crosses dependency boundaries without allocated diagnostics.
The parent compiler retains its concrete refusal and ownership throughout
compiled output/error lifetimes. The dynamic constructor is not exposed as
fully funded until all reached producers above are connected and refusal tests
exercise the original worker at every allocation boundary.

## Implemented front-end boundary

The canonical parser, resolver, optimizer, analysis graph, direct HIR translator
and expression/seek text writers now use one borrowed storage context. Their
ordinary entry points select `Unenforced` and enter those same workers. The
previous external bit set is replaced with one dense ascending `Vec<u32>` set;
name and graph maps use the local hash table's actual prospective layout. Group
name diagnostics borrow their source names rather than cloning them. Expression
cloning and capture-group copy-on-write share explicit admitted producers.

Semantic HIR fallback remains an optional result, while allocation failure is
returned separately and cannot select an unfunded fallback. The regex-syntax
smart constructors expose their existing admitted worker to the direct translator.
At this front-end checkpoint the general builder was not yet funded: instruction and
delegate construction, the lower automata producers and invocation-owned search
storage remain necessary.

Property-name lowercase preserves the selected compiler's Unicode 17 standard
library behavior, including context-sensitive final sigma. It must not reuse the
Unicode 16 matching-property tables in regex-syntax. One two-pass writer counts
its exact UTF-8 destination then fills it after admission. Pinned classification
ranges describe the standard library's effective cased/ignorable context; the
actual scalar lowercase mapping is the same pinned `char::to_lowercase` worker.
`fancy-regex-0.19.0/validation/lowercase-provenance.json` records the compiler,
standard-library archive, generator and formatted table hashes. Reproduce the
tables with `emit-lowercase.rs` and then rustfmt the generated output before
comparing its hash. Exhaustive scalar and sigma-context output comparisons with
standard `str::to_lowercase` pass (2 tests, 7.46 seconds).

Validation at this boundary:

- 306 complete fancy library tests pass, 1 explicit emitter ignored (12.99 s).
- The pristine 0.19 reference again passes 15,507 exact matching, capture and
  diagnostic comparisons across 437 patterns after these front-end changes.
- Analysis: 58 tests pass, including refusal at each reached graph/diagnostic
  destination. Optimization: 32 existing tests plus exhaustive reached-destination
  refusal coverage pass, including shared capture Arc mutation.
- Direct HIR: 4 independent syntax-translation oracle tests and a separate
  every-destination refusal test pass. Semantic unsupported syntax does not hide
  a refused allocation.
- The seek writer retains its original approximation and passes all 14 existing
  behavioral fixtures. Its strings now use the same admitted writers.

The reference runner additionally requires the exact local hashbrown rlib used
by the canonical fork. The pristine crate still uses its pinned bit-set rlib;
this keeps the membership and map representation changes inside the comparison.

## Original compiler and VM boundary

VM instruction emission, group/subroutine controls, native class parsing, direct
HIR/string delegate inputs, expression owners and VM branch/save/stack/capture
buffers now use the same prospective producers under explicit ordinary policy.
The closed tokenizer workspace enters that same VM with growth refused after
preparation. Unicode case-insensitive backreferences borrow the syntax library's
existing simple-fold table instead of constructing a class for each scalar.

All 611 fancy unit/integration tests pass (one explicit emitter ignored), as do
the minimal feature build and five targeted every-reached-allocation refusal
tests. The pristine 0.19 oracle again passes all 15,507 matching, capture and
diagnostic comparisons across 437 patterns after these compiler/VM changes.

The next implementation checkpoint covered regex-automata. A borrowed neutral allocator
now owns concrete vector, string, box, Arc and pinned hash-table admission. The
Thompson compiler/group-info workers, literal extraction and preference trie,
meta-engine selection, prefilter construction and selected engine caches must all
consume it. Ordinary and enforced policy must not select different search
algorithms. In particular, retaining the callback only around a completed NFA or
using a fixed recipe for an arbitrary source would not cover construction.

Literal sequence work includes exact byte copies, cross-product destinations,
preference-trie states/transitions, saved optimization candidates and sequence
growth. Stable sorting of fully ordered literal values can become an in-place
unstable sort because equal values are indistinguishable. Splicing must move the
already admitted values in-place rather than invoke an opaque scratch producer.

At that intermediate boundary, meta/prefilter construction and invocation-owned
search caches still blocked the public funded dynamic constructor. Their closure
is recorded below.

Literal extraction now executes the same post-order traversal with an admitted
continuation vector. Source nesting does not consume recursive Rust call frames.
All 161 syntax library tests pass, including 4,096 nested captures on a 64 KiB
thread stack and refusal at every reached extraction/optimization destination.
The independent pinned syntax reference now compares literal extraction and
preference optimization as well as AST/HIR/properties/diagnostics: all 5,285
patterns pass, for both prefix/suffix extraction and two literal-count limits.

The original one-pass DFA builder now admits its NFA-to-DFA map, traversal
controls, transition/start vectors, remapper and final shrinking destinations.
Its caller-owned explicit capture cache uses the same paid growth/reset worker.
All 10 one-pass tests pass, including every reached construction refusal and
capture-span/reuse checks. The general meta strategy and other selected engine
caches are still required; this does not claim that every low-level search API
has a complete funded error path.

## Independent release measurement

The canonical 0.19 original compiler/VM was compiled alongside the verified
pristine archive using Rust 1.98.0, optimization level 3 and thin LTO. The runner
first compared all 15,507 expected outputs, then checked exact match spans before
each timed search; every sample ran for at least 400 ms after four warmups.
These results precede the later literal/one-pass changes above.

| Case | Input bytes | Matches | Pristine search µs | Canonical search µs | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tokenizer pattern 0, chat | 20,992 | 4,608 | 545.421 | 570.248 | 1.046 |
| Tokenizer pattern 0, Unicode | 17,408 | 4,095 | 645.591 | 623.953 | 0.966 |
| Tokenizer pattern 0, whitespace | 32,769 | 2 | 257.562 | 288.145 | 1.119 |
| Tokenizer pattern 2, chat | 20,992 | 4,608 | 353.747 | 364.018 | 1.029 |
| Tokenizer pattern 2, Unicode | 17,408 | 3,584 | 310.577 | 326.341 | 1.051 |
| Tokenizer pattern 2, whitespace | 32,769 | 2 | 292.217 | 319.340 | 1.093 |
| Schema lookahead | 12 | 1 | 0.162 | 0.158 | 0.977 |
| Unicode backreference | 4,608 | 768 | 1,253.605 | 568.788 | 0.454 |
| Atomic stack | 8,193 | 1 | 80.054 | 84.606 | 1.057 |

Compilation for source lengths 74, 274, 41, 21 and 11 bytes respectively measured
391.026/395.191, 1711.451/1716.146, 139.449/140.853, 3.014/3.280 and 1.092/1.355 µs
(pristine/canonical). The largest relative compile increase is 0.263 µs on the
smallest atomic expression; the representative tokenizer sources remain within
1.1%. The Unicode search improvement removes the original per-scalar temporary
case-fold class. At this checkpoint, general heap census still awaited the complete lower producer
connection; the closed tokenizer recipes retain their measured 425,943 and
2,553,938 source bytes and 56,002,521-byte declared workspace.

Reproduce with `validation/run-reference.sh` under the canonical fancy fork,
setting `REFERENCE_OPT_LEVEL=3 REFERENCE_LTO=thin REFERENCE_BENCHMARK=1` and passing
the verified pristine source plus exact release regex-automata, regex-syntax,
bit-set and local hashbrown artifacts. `validation/performance.rs` contains the
complete fixed inputs and equality checks. Release thin LTO is necessary when
using the workspace's LLVM-bitcode dependency artifacts with the system linker.

The independent one-pass runner additionally passes 630 pristine construction
comparisons and 2,448 exact capture comparisons. Assignment/Unicode/65,536-byte
run search ratios are 1.010/1.003/0.988, with compile ratios 1.053/1.088/0.998.
Their final transition tables occupy 772/900/260 bytes and explicit capture caches
32 bytes each. Cumulative prospective construction admissions are
29,789/338,755/10,270 bytes across 179/137/104 destinations; these include temporary
producers and must not be presented as simultaneous live heap usage. The checked
runner is `regex-automata-0.4.18/validation/run-onepass-reference.sh`.

## Lazy DFA cache and search

The original hybrid determinizer now admits actual state bytes, immutable state
owners, transition/start rows, powerset lookup tables, sparse sets and traversal
controls. The same adaptive worker serves ordinary and enforced callers; cache
clearing retains its original heuristic and reuses the immutable empty sentinel.
Allocation refusal is an inline typed search error, and a partially changed cache
must reset before another search can access its IDs. An overlapping search with
an old saved ID rejects after refusal until the caller restarts that search.

Five focused tests pass, including refusal at every reached source/cache/search
allocation, clearing and reuse after refusal. The pristine hybrid runner verifies
630 construction/error results, 52,008 search results across input ranges and
`earliest` settings, and 113 overlapping results. It checks match equality before
each timing. Rust 1.98.0 release samples use four warmups and at least 300 ms:

| Case | Source bytes | Input bytes | Pristine/current warm µs | Pristine/current cold µs | Pristine/current compile µs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Assignment | 37 | 23,552 | 0.071/0.067 | 4.228/4.109 | 12.564/13.294 |
| Unicode | 19 | 30,720 | 0.090/0.094 | 3.758/3.755 | 18.682/18.986 |
| Long run | 8 | 65,536 | 106.979/116.155 | 110.374/117.965 | 5.802/6.328 |
| Repeated cache clearing | 14 | 6,464 | 894.055/1010.023 | 898.332/1012.740 | 8.199/8.952 |

Assignment and Unicode return the first match near the input start; their full
input sizes do not represent scanned throughput. The long run and cache-clearing
case scan the input. Cache clearing admits many short-lived determinized states;
that actual repeated construction explains its 13% warm-search increase. First
search cumulative admissions are 7,520/7,456/2,856/262,340 bytes across
60/61/50/6,536 destinations. Source construction cumulative admissions are
51,862/363,448/16,472/30,102 bytes. These admissions cover actual prospective
replacement destinations; the unchanged upstream cache-capacity heuristic is not
used as a storage bound or described as exact live memory.

Reproduce with `validation/run-hybrid-reference.sh` under the automata fork,
passing the verified pristine 0.4.18 registry directory and an artifact directory,
with the absolute qualified `RUSTC` and `CARGO_INCREMENTAL=0`. The runner verifies
all recorded upstream file hashes before building both engines.

The hybrid checkpoint still preceded the shared meta build/strategy/cache
connection, reverse-HIR optimization producers and prefilter hooks. The following
checkpoint closes those producers and exposes the paid fancy constructor.

## General source and scoped invocation boundary

The pending source connection above is now implemented. The original meta
constructor admits AST/HIR collections, metadata properties and owners, literal
extraction, selected strategy/engine graphs, closure boxes, initial pool rows
and shells. Reverse-inner/suffix proofs and HIR copying use paid continuation
vectors, preserving original traversal and selection semantics. Ordinary source
constructors delegate to this worker with explicit `Unenforced`.

`fancy_regex::RegexOptionsBuilder::build_with_allocations` accepts a borrowed
pattern before its first owning copy. `Regex::search_workspace_with_allocations`
creates storage bound to that exact compiled regex. Its `is_match`, `find`, and
`find_input` operations enter the same meta strategies and VM loop as ordinary
operations. Cache ownership is explicit: pooled ordinary calls use their existing
persistent pools; scoped calls keep actual delegate caches and VM buffers in the
caller-owned workspace. Either policy can use the scoped interface. No algorithm
is selected based on whether funding is enforced, and no borrowed callback is
stored in a compiled source or persistent cache pool.

Allocation errors cross every optional engine and heuristic fallback unchanged.
Every-reached-request tests cover source construction and invocation growth,
including Unicode, backreferences, atomic groups, lookahead, variable lookbehind,
absent repetition and seek delegates. Refusal stops at its first callback and a
subsequent invocation can reuse the workspace successfully. The meta sweep also
exposed an oversized capture-slot panic; the original Pike worker now limits its
borrowed active slots to actual group geometry and preserves unrelated tail slots.

The complete automata library passes 246 tests. The canonical fancy library and
integration suites pass 619 tests plus 40 documentation tests, with one development
recipe emitter ignored. The minimal Unicode library profile passes 310 tests;
the automata all-feature instrumentation profile passes 237 tests.
The upstream `compile_mem` example intentionally installs an unsafe global
allocator and is not part of these safe library/integration checks. Full feature
library checking passes without weakening the workspace lint.

The independent meta runner verifies 304 construction/error results and 232,848
exact search, capture and boolean results against the pinned pristine 0.4.18
implementation. Profiles include ordinary selection, lazy DFA, bounded
backtracking and Pike-only execution. Rust 1.98 release measurements use four
warmups and at least 300 ms per operation:

| Case | Source bytes | Input bytes | Pristine/current warm µs | Pristine/current cold µs | Pristine/current compile µs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Literal | 6 | 18,432 | 0.024/0.024 | 0.035/0.054 | 2.170/2.178 |
| Assignment | 39 | 24,576 | 0.045/0.049 | 3.204/3.292 | 48.109/45.933 |
| Unicode | 20 | 22,528 | 0.043/0.047 | 5.700/6.001 | 41.325/43.434 |
| Long absent suffix | 7 | 65,536 | 0.633/0.643 | 0.682/0.726 | 11.971/12.382 |
| Reverse suffix | 9 | 40,963 | 1.050/1.095 | 28.578/32.971 | 160.598/173.823 |

Literal, assignment and Unicode return a match near the beginning. The absent
suffix case exercises the retained original literal rejection optimization;
these input sizes are not claims that every byte passes through a regex VM.
Cumulative source destination bytes are 7,933/463,711/469,075/29,798/658,855;
first scoped-cache/search admissions are 16/13,009/21,044/16/81,072 bytes. These
are prospective allocation totals, not simultaneous live heap or an immutable
source census. Reproduce with `validation/run-meta-reference.sh`, passing the
pristine source and artifact directories under the absolute qualified `RUSTC`.

Private ordinary helper adapters have been removed after their callers migrated.
This cleanup also found and fixed the reverse-suffix proof call to use its actual
source payer; every reached-request refusal tests cover the resulting source.

Exact retained source ownership is now implemented through the dependency-owned
visitor described in `bounded-followup-regex-retained-storage.md`. It reports
actual capacities and shared owners; cumulative construction totals and upstream
approximate `memory_usage` summaries are not substituted for that census.


## Whole-engine oracle and current cost

`fancy-regex-0.19.0/validation/run-cargo-reference.sh` checks the canonical fork
against registry fancy-regex 0.19.0 with untouched registry regex-automata 0.4.18,
regex-syntax 0.8.11, aho-corasick 1.1.5 and memchr 2.8.3 dependencies. It verifies
the archived fancy source hashes before constructing the independent Cargo
fixture. All 438 patterns and 28,817 exact construction, span, capture and boolean
comparisons pass, including scoped searches and all six tokenizer regex recipes.

This stronger oracle found a bounded-repetition edge in the iterative Thompson
compiler: direct HIR with a maximum below its minimum must preserve the original
empty optional-repeat loop. Saturating the continuation count preserves that
behavior without an overflowing loop. A focused `a{3,1}` regression now verifies
the resulting match. An oversized bounded-backtracking delegate preserves the
actual `MatchError` in `RuntimeError::DelegateError`; it no longer reports a
workspace identity error. The independent `^.{0,404600}$` probe reaches the same
`HaystackTooLong { len: 0 }` cause as the original engine's panic wrapper.

The final release measurements use Rust 1.98.0, thin LTO, four warmups and at
least 400 ms per sample. Each whole-match loop first verifies exact spans against
the pristine implementation. Pattern rows refer to the checked runner's literal
source definitions, which contain the full inputs and options.

| Case | Pattern bytes | Input bytes | Matches | Pristine/current whole-loop µs | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tokenizer 0, chat | 74 | 20,992 | 4,608 | 607.707/639.639 | 1.053 |
| Tokenizer 0, Unicode | 74 | 17,408 | 4,095 | 662.399/712.537 | 1.076 |
| Tokenizer 0, long spaces | 74 | 32,769 | 2 | 277.286/330.576 | 1.192 |
| Tokenizer 2, chat | 274 | 20,992 | 4,608 | 357.399/401.735 | 1.124 |
| Tokenizer 2, Unicode | 274 | 17,408 | 3,584 | 310.282/351.813 | 1.134 |
| Tokenizer 2, long spaces | 274 | 32,769 | 2 | 287.854/341.355 | 1.186 |
| Schema lookahead | 41 | 12 | 1 | 0.157/0.167 | 1.059 |
| Unicode backreference | 21 | 4,608 | 768 | 1265.858/586.732 | 0.464 |
| Atomic stack | 11 | 8,193 | 1 | 79.184/81.819 | 1.033 |

The long-space case remains about 19% slower through the original adaptive
worker; these measurements do not establish which individual instruction causes
the difference. The former forced fixed-workspace 1.4–3.5x regression is no longer
the general search route. Prospective vector push checks skip the producer on
already available capacity. No alternate production algorithm or payer-dependent
engine switch was introduced to obtain these results.

Construction pristine/current µs for the two tokenizer patterns, schema
lookahead, Unicode backreference and atomic stack are respectively
404.638/438.799, 1624.794/1850.021, 137.433/144.493, 2.762/3.311 and 1.097/1.359.
Exact retained source bytes are 163,020 / 558,278 / 55,518 / 2,405 / 2,273.
Cumulative prospective source destinations are 2,173,512 / 5,385,052 / 853,995 /
13,351 / 4,970 bytes, including temporary construction storage. First invocation
cumulative destinations are 38,901 / 124,653 / 13,845 / 888 / 786,800 bytes;
these are not a simultaneous live-cache census.

## Reusable invocation ownership

`OwnedSearchWorkspace::new_with_allocations` consumes an existing `Arc<Regex>`
alias and retains that exact source with the invocation cache. `source` and
`matches_source` expose shared-owner identity for a caller's heterogeneous cache
lookup; pattern equality does not authenticate a source. Search uses the same
storage and workers as the borrowed workspace. Tests verify exact identity,
source lifetime, retirement and zero new allocation requests on repeated warmed
searches.

Creating fresh scoped storage for every short match is costly: the first two
tokenizer patterns measured 4.087/5.462 µs fresh, against 0.132/0.063 µs when
reused and 0.127/0.057 µs for pristine pooled search. Schema lookahead measured
4.083 µs fresh, 0.155 µs reused and 0.140 µs pristine pooled. The owning handle
allows a validation invocation to reuse caches while keeping its borrowed payer
out of persistent source pools. The schema caller now retains this cache for
each validation invocation; its
explicit ordinary pool remains available through the same search worker. Warmed
same-engine ratios were 0.975, 0.949 and 1.070 for identifier, Unicode and
backreference fixtures. Separate complete funded invocations still pay initial
cache/accounting costs (71.540×, 6.083× and 1.761× ordinary pooled validation for
1, 16 and 128 patterned properties in the later benchmark). These are scope
costs, not a claim of unchanged latency. See
[schema compiler evidence](bounded-followup-schema-compiler.md#standard-regex-wrapper-inventory-before-edits)
and its checked benchmark sources for commands and exact measurement scope.
