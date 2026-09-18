# Thompson construction and capture metadata

Inventory before implementation:

- The selected regex-automata Thompson compiler builds the same NFA for ordinary
  and schema/parser consumers, but its state builder, literal trie, forward UTF-8
  cache, reverse range trie and suffix cache grow without prospective allocation
  callbacks. The compiler constructor also creates range-trie roots before a
  build policy is supplied.
- Finalization copies transition and start tables, creates capture metadata,
  traverses epsilon edges with stack/sparse-set storage, shrinks vectors and
  creates an Arc source shell. These are original producers, not storage that
  can be qualified by copying an already built NFA.
- Capture metadata uses an opaque standard hash map (an ordered map in the
  no-std profile), two name/index tables, strings and a shared shell. Captures
  additionally allocate and clone mutable slot storage. Duplicate-name errors
  own diagnostic strings.
- Existing source-copy constructors describe copies of already built automata.
  They do not fund the original compiler or allow a different account to adopt
  its retained workspaces.

Canonical direction: pass one borrowed allocation policy through the original
compiler, builder, trie and finalization workers. Ordinary entry points use the
same workers with explicit unenforced policy. Preserve forward/reverse choices,
UTF-8 compression, literal priority, capture geometry and selected search engines.
Use the pinned hash collection's actual prospective growth facts for capture-name
lookups; remove the profile-dependent opaque capture-name allocation path. Fixed
allocation failures must propagate without allocating diagnostic strings. The
enclosing source owner must retain each real policy authority through compiler
scratch, completed NFA/capture aliases, and failure retirement.

Implemented: original compiler/builder/trie/finalization growth now uses this
policy, including the forward UTF-8 cache, reverse suffix cache and range-trie
scratch. Compiler creation is allocation-free. Final state rows are reserved from
the actual count after empty-state removal; transition slices and source shells
are admitted before construction. Capture-name tables use the same pinned hash
implementation in every feature profile, and obsolete optional map rows are gone.
Duplicate-name diagnostics retain the already constructed name rather than
allocating another string. Fixed always/never NFAs, sparse-set pairs, capture slot
construction and copies use the same funded workers as ordinary entry points.

Source-dependent HIR recursion is replaced by an explicit continuation vector.
The original traversal order, capture entry/exit, literal-priority trie, nullable
star handling and greedy/lazy repetition equations remain in one compiler.
Every continuation growth is admitted, and constant machine/dispatch/result
controls are declared before entering the machine. No pattern-size multiplier
or caller-supplied capacity stands in for actual ancestry. The source owner must
still retain the actual payer across reusable compiler scratch and NFA aliases.

Validation uses the pinned Rust 1.98 compiler, `CARGO_INCREMENTAL=0`, and
`CARGO_TARGET_DIR=/private/tmp/eredu-grammar-check`:

```sh
cargo test --manifest-path third-party/regex-automata-0.4.18/Cargo.toml --lib --offline -- --test-threads=2
cargo check --manifest-path third-party/regex-automata-0.4.18/Cargo.toml --offline --no-default-features --features alloc,nfa-thompson
cargo test --manifest-path third-party/regex-automata-0.4.18/Cargo.toml --lib --offline --no-default-features --features syntax,nfa-pikevm,unicode nfa::thompson:: -- --test-threads=2
```

The full library passes 220 tests at the explicit-stack boundary. The minimal
no-std construction check passes, and the no-std syntax/Thompson profile passes
39 tests. Coverage includes every reached refusal in forward/reverse compressed
Unicode construction and capture metadata, and 12,000 nested captures on a
256 KiB thread stack. Logs are
`/private/tmp/contracts-thompson-full-tests.log`,
`/private/tmp/contracts-thompson-minimal.log`, and
`/private/tmp/contracts-thompson-syntax.log`.

A separate consumer compares 1,890 complete NFA graphs against registry
regex-automata 0.4.18, archive SHA-256
`ad8553b9b26413251cbf30e620595c7a41b3887f03da04579c0e6b0d6a06b4b2`.
It covers forward/reverse direction, both reverse compression choices, captures,
literal priority, nullable expressions and exact/bounded/unbounded greedy/lazy
repetition. All graphs match, including state order and transition equivalence
classes. The independent consumer and lockfile are in
`/private/tmp/eredu-thompson-reference`; output is
`/private/tmp/contracts-thompson-reference.log`.

The selected automata feature profile excludes `logging`; its log macro removes
the diagnostic producer at compilation. An enabled external logger still needs
an explicit bounded callback contract. Meta strategies, prefilters, other
selected automata engines and caches were separate producer slices at this
checkpoint. Their subsequent source/search closure is recorded in
[dynamic regex allocation evidence](bounded-followup-regex-engine-funding.md);
the Thompson tests alone do not establish those bounds.

The final focused refusal suite passes all five tests, including fixed graph
construction and sparse-set pairs (`/private/tmp/contracts-thompson-final-refusal-tests.log`).
A release probe of 1,000 fresh constructions per pattern measured selected versus
pristine times between 0.92 and 1.14 times upstream across named captures, bounded
literal repetitions, nullable lazy repetition, Unicode ranges and anchored paths.
These short local samples are regression evidence, not a performance guarantee;
raw results are in `/private/tmp/contracts-thompson-reference-performance.log`.
