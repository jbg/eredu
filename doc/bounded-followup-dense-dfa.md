# Dense DFA construction and minimization

Inventory before implementation:

- Dense construction creates transition/start tables and accelerator headers,
  runs powerset determinization, optionally minimizes and accelerates states,
  then shrinks retained buffers. Ordinary and prospective construction must
  execute these same selected algorithms.
- Determinization owns compact state keys, a state cache, sparse sets, epsilon
  work stacks, representative-byte rows and pending-state rows. Its cache uses
  unrelated opaque standard hash/tree implementations across feature profiles.
- State shuffling and acceleration use ordered maps/sets keyed by premultiplied
  state identifiers. Their keys are actual dense state indices; source-dependent
  tree-node allocations have no prospective producer contract.
- Hopcroft minimization owns incoming-transition rows, partition/waiting vectors,
  shared Rc partition shells and mutable state-ID vectors, intersection/difference
  scratch, remap tables and initial pattern-list groups. Stable sorting can also
  allocate scratch. No source-size estimate qualifies these producers.
- Shared compact-state and remapping helpers have prospective workers owned by
  the common automata allocation layer. Special-state range metadata is fixed
  inline storage. Optional external logging remains a distinct callback boundary.

Canonical direction: thread the existing borrowed allocation policy through the
original builder, powerset and Hopcroft workers. Ordinary entry points delegate
with explicit unenforced policy. Use the pinned hash collection for state-key
lookup. Replace keyed-by-state temporary trees with stride-indexed optional rows,
whose ordered scans preserve state ordering and whose actual growth is admitted.
Group initial partitions using the pinned hash table and one in-place sorted row
view, preserving lexicographic pattern-list order without quadratic insertion.
Admit Rc shells and vector replacements before construction, propagate fixed
allocation failures without allocating diagnostics, and retain the real enclosing
source authority through scratch, completed source and failure retirement.

This inventory does not claim implementation or validation closure. Source copies,
other selected engines, strategy selection and search caches remain independent
producer slices. An enabled external logger requires its own bounded contract.

Implemented: `Builder::build_with_allocations`, `build_many_with_allocations`
and `build_from_nfa_with_allocations` enter the same original workers as ordinary
construction. Fixed always/never construction does too. Compact state sources,
cache growth, sparse sets, epsilon and pending work, transition/start/match tables,
Rc partition shells, incoming rows, remapping, acceleration and final shrinking
replacements are admitted prospectively. Fixed source/control/result headers are
admitted at their producer boundaries. The initial Thompson configuration is
copied into fresh allocation-free scratch rather than cloning mutable compiler
storage. Allocation errors remain inline and identifiable across NFA/DFA errors;
they are not ordinary heuristic size-limit errors.

State-key scratch uses stride-indexed rows and lexicographic partition grouping
uses the pinned hash table plus one ordered row view. The original powerset and
Hopcroft refinement equations and state permutation order remain unchanged.
Partition ordering uses one iterative in-place heapsort with declared constant
control storage, removing the recursive library-sort dependency. There is no
source-depth recursive DFA traversal. Existing configured DFA/determinizer size
limits remain algorithm-selection heuristics; prospective allocation admission
uses actual producers independently of those historical estimates.

Validation with the pinned Rust 1.98 compiler, `CARGO_INCREMENTAL=0`, and
`CARGO_TARGET_DIR=/private/tmp/eredu-grammar-check`:

```sh
cargo test --manifest-path third-party/regex-automata-0.4.18/Cargo.toml --lib --offline -- --test-threads=2
cargo test --manifest-path third-party/regex-automata-0.4.18/Cargo.toml --lib --offline dfa::dense::allocation_tests:: -- --test-threads=2
cargo check --manifest-path third-party/regex-automata-0.4.18/Cargo.toml --offline --no-default-features --features alloc,dfa-build
```

The complete library passed 228 tests after iterative sorting. The final three
focused tests pass after fixed control admission: every reached request is refused
in turn across minimization/acceleration combinations, fixed graphs and reusable
borrowed NFAs; each refusal propagates before another allocation request. The
minimal no-std, no-syntax construction profile passes. Logs are
`/private/tmp/contracts-dense-full-tests.log`,
`/private/tmp/contracts-dense-final-refusal-tests.log`, and
`/private/tmp/contracts-dense-minimal.log`.

The independent `/private/tmp/eredu-dense-reference` consumer compares complete
serialized DFA bytes with registry regex-automata 0.4.18 (archive SHA-256
`ad8553b9b26413251cbf30e620595c7a41b3887f03da04579c0e6b0d6a06b4b2`). All 1,408
cases match across forward/reverse construction, multi-pattern and Unicode input,
minimization, acceleration, byte classes, special start states, per-pattern starts
and both match policies. This comparison passed after the final control changes.
A release probe of 200 fresh builds per pattern measured 0.96–1.02 times upstream
without minimization and 1.14–1.21 times upstream with minimization across bounded
literal alternatives, Unicode ranges and anchored paths. These short local samples
are regression evidence, not a performance guarantee. Full output is in
`/private/tmp/contracts-dense-reference-performance.log`.

The selected feature profile excludes external logging. Preexisting prefilters
remain shared source aliases whose actual construction and authority are owned
by the surrounding strategy. This slice does not qualify an arbitrary external
logger or adopt preexisting source storage under a new account.
