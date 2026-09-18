# Adaptive NFA search and cache growth

Inventory before implementation:

- Pike search uses one active-state algorithm but its cache construction/reset,
  capture-slot tables and epsilon/restore stack grow without prospective funding.
  The UTF-8 empty-match path allocates temporary slots for multiple patterns when
  callers supply fewer capture slots. Overlapping search shares the epsilon worker.
- The existing closed Pike workspace independently constructs and initializes
  cache internals, then relies on ordinary search growth remaining within its
  prepared capacity. This duplicates cache geometry/initialization and leaves an
  unchecked allocation path behind a capacity proof.
- Bounded backtracking has the same original branch/restore stack producers and
  insufficient-slot UTF-8 temporary. Its visited bitmap grows from actual state
  count and search span. Its algorithmic visited-size limit is distinct from
  prospective allocation admission.
- Both algorithms mutate capture slots while following branches. A refused push
  must unwind pending restore frames without allocating and leave caches reusable.
- Optional Pike instrumentation owns opaque thread-local counters/maps; this is
  excluded from the selected feature profile and requires an explicit producer
  contract before it can be used under enforcement. Prefilter construction and
  shared source custody belong to the enclosing strategy.

Canonical direction: keep the original Pike and bounded-backtracking search
workers and introduce borrowed prospective funding at their real mutation sites.
Ordinary entry points use explicit unenforced policy. Admit cache/table/visited
and branch-stack growth before mutation, preserve UTF-8, pattern priority, captures,
overlapping and span semantics, and propagate inline typed allocation errors without
fallback. Make the closed workspace use the canonical empty/reset workers and a
no-growth policy during search, preserving its exact source borrow and failure-prefix
custody. The caller retains real authority through caches, errors and source aliases.

Implemented: both original search engines now take borrowed admission through
all adaptive branch/restore stack and insufficient-slot UTF-8 destinations.
Backtracking admits the actual visited bitmap before growing it. Pike cache
construction/reset admits its sparse sets and slot tables; one geometry function
is shared by planning and actual initialization. Ordinary find/capture/slot/match
entry points delegate to the same funded workers with explicit unenforced policy.
The typed inline `MatchError` supplied by the common search contract preserves
refusal separately from algorithmic haystack limits.

A refused epsilon/branch push drains existing restore frames without allocation
before returning. Refusal cannot report a successful capture, and the same cache
can run another search. The closed workspace retains its existing exact source
borrow, seven-buffer reserve failure prefix and capacity proof, then initializes
through canonical cache reset. Its actual search worker runs with a growth-refusing
policy. It no longer duplicates cache construction or slot initialization. Fixed
search, temporary-slot and result control storage is declared at the public producer
boundary and included in the closed workspace's original preparation requirements.

Validation uses pinned Rust 1.98, `CARGO_INCREMENTAL=0` and
`CARGO_TARGET_DIR=/private/tmp/eredu-grammar-check`:

```sh
cargo test --manifest-path third-party/regex-automata-0.4.18/Cargo.toml --lib --offline --no-default-features --features std,syntax,unicode,nfa-pikevm,nfa-backtrack -- --test-threads=2
cargo check --manifest-path third-party/regex-automata-0.4.18/Cargo.toml --offline --no-default-features --features alloc,nfa-pikevm,nfa-backtrack
```

The isolated NFA profile passes 131 tests, including every reached cache/visited/
branch/UTF-8 refusal, nested capture restoration, same-cache recovery, overlapping
matches and the existing closed-workspace lifetime/failure suite. Minimal no-std,
no-syntax compilation also passes. Logs are
`/private/tmp/contracts-nfa-runtime-tests.log`,
`/private/tmp/contracts-nfa-search-final-tests.log` and
`/private/tmp/contracts-nfa-search-minimal.log`.

The standalone consumer `/private/tmp/eredu-nfa-search-reference` compares both
engines with registry regex-automata 0.4.18 (archive SHA-256
`ad8553b9b26413251cbf30e620595c7a41b3887f03da04579c0e6b0d6a06b4b2`). All 111,024
pattern-ID and complete capture-slot comparisons match, including empty patterns,
Unicode boundaries, nested/optional captures, multi-pattern priorities, every
selected byte span, anchored/per-pattern searches, earliest mode and insufficient
slot widths. A release probe of 20,000 searches per workload with reused caches
measured 0.90–1.16 times upstream for Pike and 0.91–1.05 for backtracking across
word captures, Unicode alternatives and failing ambiguous repetitions. These local
samples are regression evidence, not a performance guarantee. Output is in
`/private/tmp/contracts-nfa-search-performance.log`.

The original allocating cache `Clone` API remains an explicitly unenforced public
operation; the funded strategy creates/resets caller-owned caches and does not use
it. Optional thread-local instrumentation/external logging remains unqualified
outside the selected feature profile. Prefilter source construction and meta-engine
selection/custody are separate producer slices; this evidence does not assert their
completion.

The later meta-engine refusal sweep also exercised caller slot arrays larger
than the NFA's group geometry. The original Pike worker now limits the active
prefix to its actual group slots, preserving the caller's extra tail. Its six
focused allocation tests pass, including oversized-slot matching, exhaustive
reached-request refusal and reuse of the same cache after refusal. Reproduction:
`cargo test --manifest-path third-party/regex-automata-0.4.18/Cargo.toml --lib
nfa::thompson::pikevm::allocation_tests --offline`.
