# Original literal-prefilter construction

Inventory before implementation:

- Regex-automata selects byte, substring, packed Teddy and Aho-Corasick prefilters
  through one constructor. Its shared trait-object shell and adapter construction
  are not prospectively admitted. Refused construction currently resembles an
  optional optimization miss and can silently select another strategy.
- The selected memchr 2.8.3 owned Finder converts borrowed needle bytes into a
  retained box. Its actual searcher/SIMD metadata is inline; search does not require
  an alternate algorithm or a reconstructed needle.
- The selected aho-corasick 1.1.5 builder constructs a noncontiguous trie and failure
  links, then its selected contiguous NFA or DFA. These producers own states,
  sparse/dense transitions, match/pattern rows, traversal/remap scratch and shared
  source shells. The automatic engine heuristic is separate from allocation policy.
- Packed construction additionally copies and orders patterns, builds Rabin-Karp
  buckets, Teddy lane/bucket masks and architecture-specific shared searcher shells.
  Anchored regex-prefilter matching builds the selected Aho-Corasick DFA alongside
  that packed searcher. These are supported distinct search mechanisms.
- Both dependencies contain upstream unsafe search/SIMD internals. Prospective
  producer hooks must remain safe changes to narrow external dependency forks;
  portable/parser workspace unsafe forbids remain unchanged. Archive identity and
  license files must be retained before any modification.

Canonical direction: preserve actual prefilter/engine selection and original
construction/search workers, threading one borrowed fixed-error allocation policy
through real retained and temporary producers. Ordinary entry points delegate with
explicit unenforced policy. A funding refusal is a hard typed error before mutation,
never an optimization miss or invitation to allocate through another engine. Retain
source authority in the enclosing owner across adapters, shared sources and errors.

Implemented: memchr's Finder, reverse Finder and borrowed search iterators share
one owned-needle conversion worker. Borrowed bytes are admitted before copying;
an already owned needle moves without another allocation or policy request.
Aho-Corasick's original noncontiguous trie, failure links, dense-state conversion,
contiguous encoding, DFA conversion, automatic engine selection, internal
prefilters, packed pattern copies, Rabin-Karp buckets and Teddy constructors now
use a borrowed prospective policy. The public ordinary constructors delegate with
explicit unenforced policy. Build errors preserve fixed allocation markers inline,
and optional engine selection stops on those markers instead of falling back.

The original breadth-first queue remains a queue. Its growth, case-insensitive
state-membership rows and remapping copies are admitted from their actual
representations. The fixed 256-byte start membership domain now lives inline, so
compiler startup does not allocate that table before a policy is present.
Contiguous encoding reserves each complete actual state row before writing its
first word. Packed ordering uses iterative heapsort with explicit constant
control storage and original insertion precedence for equal-length patterns.
Teddy's at-most-four-nybble keys are inline integers; its membership table is a
sorted vector bounded by the existing packed limit of 128 patterns. Bucket
assignment and search equations are unchanged. Fixed mask and bucket-header
arrays replace temporary vectors. Source/control/result headers, shared shells,
vector replacements and shrinking destinations are admitted at their producers.
There is no source-depth recursive construction traversal in this slice.

Regex-automata's `Prefilter::new_with_allocations(kind, needles, funding)` returns
`Result<Option<Prefilter>, AllocationError>` through the same original choice
worker. Intrinsic optimization misses remain optional; source refusals are hard
errors. The memchr, packed and Aho adapters forward the same authority and the
outer shared prefilter shell is admitted. Prefix/suffix extraction and meta
strategy integration remain owned by their original workers; this change does
not replace them with a separate search algorithm.

Validation uses the pinned Rust 1.98 compiler, `CARGO_INCREMENTAL=0` and
`CARGO_TARGET_DIR=/private/tmp/eredu-grammar-check`:

```sh
cargo test --manifest-path third-party/aho-corasick-1.1.5/Cargo.toml --lib --offline
cargo test --manifest-path third-party/aho-corasick-1.1.5/Cargo.toml --lib --offline --no-default-features
cargo test --manifest-path third-party/regex-automata-0.4.18/Cargo.toml --lib --offline util::prefilter::allocation_tests
cargo check --manifest-path third-party/regex-automata-0.4.18/Cargo.toml --offline --no-default-features --features alloc,perf-literal-substring,perf-literal-multisubstring
cargo check --manifest-path third-party/regex-automata-0.4.18/Cargo.toml --offline --no-default-features
```

Aho-Corasick passes 153 full-profile tests and 133 minimal-profile tests. The new
cases refuse every reached request across all automaton kinds, match policies,
internal prefilters and packed construction, assert no further request after
refusal, and check builder reuse. The regex prefilter test does the same for byte,
substring, packed, DFA and contiguous-NFA choices, including allocation-free shared-source
aliases. Both minimal feature checks pass. Memchr's focused direct-rustc
fixture passes forward/reverse refusal and already-owned transfer checks; its
standalone Cargo test currently lacks offline registry metadata for optional
`rustc-std-workspace-core`. The selected package itself checks successfully.
Logs are `/private/tmp/contracts-aho-final-tests.log`,
`/private/tmp/contracts-aho-minimal-tests.log`,
`/private/tmp/contracts-prefilter-final-tests.log`,
`/private/tmp/contracts-prefilter-minimal.log`,
`/private/tmp/contracts-prefilter-noalloc.log`, and
`/private/tmp/contracts-memchr-direct-tests.log`.

The independent consumer `/private/tmp/eredu-prefilter-reference` compares the
selected forks with pristine registry aho-corasick 1.1.5 and regex-automata 0.4.18.
It passes 4,032 complete automaton debug graphs, 403,200 matches and 44,528
forward/anchored prefilter span results, covering all engine kinds and start
policies, all match semantics, ASCII case folding, byte classes, internal
prefilter selection, duplicates, empty patterns, Unicode bytes and small/large
pattern sets. Reproduce with `cargo run --manifest-path
/private/tmp/eredu-prefilter-reference/Cargo.toml --release --offline`.
The final local release sample measured 0.72–1.05 times upstream for 300 fresh
prefilters per pattern set and 0.99–1.02 times upstream for 100,000 reused searches.
These short samples are regression evidence, not a performance guarantee. Raw
output is `/private/tmp/contracts-prefilter-reference-final.log`.

Validation ran on AArch64 with the selected NEON implementation. The unchanged
SSSE3/AVX2 search workers were not executed on this host. The local Aho fork's
minimum Rust version is 1.65 for safe fixed-array construction and fallible worker
syntax. Its upstream licenses and original archive/file hashes remain in
`third-party/prefilter-upstream.json`.

The selected production profile excludes external logging; enabling an arbitrary
logger still needs its own bounded callback contract. Caller-provided pattern
iterators and byte views must be the already-qualified immutable source traversal.
The enclosing source owner must retain its actual authority through temporary
prefixes, errors, completed automata and all shared aliases. These constructor
checks neither adopt a previously built source nor qualify unrelated deep-copy
APIs, streaming replacement output or complete meta/search-cache integration.
