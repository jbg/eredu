# Tokenizer consolidation evidence

The representation and matcher removals below are complete. The opening
inventory and intermediate timing checkpoints describe the starting state; later
sections record their replacements. Validation totals and timings are scoped to
those checkpoints, rather than the latest complete test run. Current source-prefix
and model coverage is recorded in [prefix sources](bounded-followup-prefix-sources.md),
[Unigram](bounded-followup-unigram.md),
[ordered compositions and feature semantics](bounded-followup-tokenizer-composition.md), and
[retained configuration sources](bounded-followup-tokenizer-source.md).

## Initial inventory

Inspected `77e08045401912c502e6ab1600e8b622c3563478` against
`c513e17c583ad7cbc1caa1c9e8d9a4a2c5f62fa3`. The tokenizers fork was
introduced in that interval; upstream provenance remains in
`third-party/tokenizers-upstream.json` and its original license remains intact.

* Added vocabulary initially stored either hash maps plus two DAAC matchers or
  packed strings plus a separately implemented prefix search. Mutation converts
  packed storage back into maps and reconstructs the DAACs. The canonical target
  is packed strings and one bounded sparse failure-link matcher for construction,
  deserialization and encoding. Normalized spellings, word boundaries, whitespace
  flags and special-token membership remain semantic data. Remove the storage
  enum, DAAC matcher, prefix matcher and conversion worker.
* BPE already shares its merge algorithm but dispatches every lookup between map
  and packed tables. Builders/trainers replace packed tables with map tables.
  Retain one packed table representation; mutable training maps remain temporary
  training machinery. Remove storage dispatch and representation conversion APIs.
* `Legacy` and `NoModelCaches` select cache construction and duplicate untagged
  model deserialization. Replace them with an explicit entry-capacity policy
  (zero disables caches), supplied before construction, and one seeded visitor.
  Tagged/untagged model JSON and string/pair merge JSON are legitimate input
  formats and remain supported through the same resulting model representation.
* Bounded source compilers retain their borrowed-source parsing and failure
  custody. They must construct the same execution representation and quote every
  added matcher buffer. Ordinary serde construction does not itself claim an
  allocation grant. Existing owned vocabulary output is an explicit caller copy;
  borrowed traversal is canonical.

Validation used independent literal expected results, existing upstream
normalization/flags fixtures, source compiler failure/capacity tests, ordinary
cache-policy tests and reproducible matcher timing on chat-like and adverse
overlapping-prefix inputs. Timing and final removals are recorded below.

## Implemented result

Added vocabulary now has one packed representation and one sparse failure-link
matcher. The DAAC dependency, the packed prefix-search algorithm, the
`LegacyAddedVocabulary` implementation, storage dispatch and mutation conversion
are removed. Both normalizer phases use the same matcher. Its root-byte skip
only skips bytes that cannot begin a pattern; it does not select another search
implementation. Token properties, normalized decoding spellings, sticky special
membership and the existing post-match word/whitespace filtering are preserved.
Borrowed traversal is canonical; requested owned maps are explicit copies.

BPE now retains only immutable packed tables. Serde and trainers construct those
tables from their temporary model-building maps; source compilation fills the
same tables directly. Model inference no longer dispatches between storage
representations. Both BPE trainers use the canonical table installer.

`ModelCachePolicy { capacity }` controls model-cache entry capacity before
construction. Zero disables cache construction; the default is 10,000 entries.
The same seeded visitor handles every policy, including tagged and untagged
model JSON. Cache capacity is an entry count, not a byte bound or a promise
about process-wide allocator caches. Existing model-format support is retained.
The upstream archive manifest and licenses are unchanged.

## Storage and search cost

For the admitted added-token source profile, let `N` be raw declaration count
and `B` the total spelling bytes, including duplicate declarations. Construction
reserves exactly five vector targets: `N` entries, `B` bytes, two `N` indexes and
`B + 2` matcher nodes. Arithmetic and layout overflow are checked before reserves.
Every failed reserve retains its actual previously allocated buffers. Duplicate
spellings are grouped by sorting, preserving first ID, final flags and sticky
special membership, without the previous quadratic declaration scan.

On the measured arm64 build, entries are 40 bytes and matcher nodes 28 bytes:
the source-derived buffer bound is `56*N + 29*B + 56`. Fixed construction
controls add 11,844 bytes. Nodes contain the intrusive construction queue, so
building failure links requires no additional heap or recursive stack. Search
has a 56-byte iterator plus fixed local state and allocates no search scratch.
The previous bounded prefix representation used `40*N + B` buffer bytes; the
larger finite representation buys shared ordinary/admitted execution and avoids
restarting the full prefix search at every unmatched input position.

| Vocabulary | Ordinary retained buffers | Compiled buffers and bound | Used nodes |
| --- | ---: | ---: | ---: |
| 256 chat delimiters | 125,488 B | 122,562 B | 780 |
| 32 shared-prefix spellings | 47,448 B | 46,856 B | 97 |
| 63 nested matching spellings | 62,136 B | 62,048 B | 65 |
| Short token plus 256-byte competitor | 7,733 B | 7,621 B | 258 |
| Four Unicode spellings | 788 B | 773 B | 17 |

These are retained buffer capacities, not RSS or allocator metadata. Ordinary
generic normalization can store additional normalized spellings; source
compilation continues to reject unsupported normalization/word/strip profiles
before allocation. That parser/profile distinction does not select another
matching implementation.

## Reproducible measurements

Measured on arm64 with `rustc 1.98.0 (88d9e12ae 2026-08-18)`, release profile,
on 2026-09-18. Each search repeats for at least 350 ms on the same hot input;
these are CPU matcher measurements, not end-to-end generation throughput.
Expected matches come from an independent `str::find` leftmost/longest oracle.
The baseline used the retained upstream DAAC matcher before its removal.

| Input | Bytes | Patterns / matches | DAAC MiB/s | Canonical MiB/s | DAAC / canonical latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Chat prose with delimiters | 33,792 | 256 / 512 | 828.32 | 1,303.57 | 38.91 / 24.72 us |
| Prose without delimiters | 30,208 | 256 / 0 | 1,146.90 | 94,446.39 | 25.12 / 0.31 us |
| Shared prefixes without matches | 32,768 | 32 / 0 | 620.61 | 370.26 | 50.35 / 84.40 us |
| Nested matching prefixes | 32,768 | 63 / 521 | 327.99 | 393.51 | 95.28 / 79.41 us |
| Delayed short match | 32,768 | 2 / 32,768 | 1.43 | 1.70 | 21.85 / 18.38 ms |
| Unicode | 34,816 | 4 / 6,144 | 378.37 | 398.46 | 87.75 / 83.33 us |

The exact generators live in `added_vocabulary/benchmark.rs`: chat tokens are
`<|special_0|>` through `<|special_255|>`; the absent-prefix set is `a^n b`
for `n=32..63`; nested matches use `a^n` for `n=1..63`; the delayed case
uses `a` and `a^255 b`; both prefix inputs contain 32,768 `a` bytes. Unicode
patterns are `é`, `é🦀`, `東京`, and `京`. The fixture contains the exact prose
and repetition counts. Final canonical construction times for the six rows
were 223, 69, 25, 26, 7, and 7 microseconds; baseline times were 295, 130,
35, 43, 20, and 14 microseconds.

The absent-prefix case remains slower by about 34 microseconds per 32 KiB;
the other measured cases improved. The delayed-short case is deliberately
adverse: leftmost-longest selection must examine the long competitor before
committing each short match, then revisit that lookahead. Both implementations
therefore do work proportional to input length times competing spelling length
on this case. No linear-time claim is made for arbitrary overlapping match
populations. The same fixed search storage applies even in this case.

```sh
CARGO_INCREMENTAL=0 cargo test --manifest-path third-party/tokenizers-0.23.2/Cargo.toml \
  --release --lib added_matcher --no-default-features --features fancy-regex \
  -- --ignored --nocapture
```

To repeat the DAAC comparison, use a temporary checkout of `77e08045`, copy
the maintained `added_vocabulary/benchmark.rs` into the corresponding module,
declare `#[cfg(test)] mod benchmark;` in `added_vocabulary.rs`, and run the
same command with filter `added_matcher_throughput`. That benchmark uses APIs
present in both revisions; it does not retain the old implementation in the
current production build.

## Packed-model checkpoint validation

At this checkpoint, the full fork library suite passed 273 tests (289 with
`parity-aware-bpe`), with the two timing/storage measurements separately passing
above. Coverage includes added/special tokens,
overlaps, normalization and normalized decoding, Unicode word and whitespace
flags, source compilation, all five reserve failures, cache policy/clone/
deserialization behavior, BPE construction/training, and ordinary versus
source-compiled encoding. An additional deterministic 250-case test compares
the matcher directly with the independent literal oracle. `cargo check -p
eredu-text --no-default-features` passes. The retained fixture tests exercise
construction routes and supported semantics, not removed representation tags.

## Template and regex adapter consolidation

The follow-up audit removed the second executable template representation.
Builders, ordinary serde and source compilation now retain the same immutable
`TemplateProcessing` tables. The `CompiledTemplate` wrapper variant and
`Texts::Legacy/Packed` dispatch are gone, along with mutable setters and mixed
representation equality. Temporary builder/serde inputs are converted once;
execution, comparison, serialization and bounded ID projection borrow the same
tables. Undefined special references in ordinary JSON still allow encoding with
special tokens disabled, while checked source compilation rejects that invalid
profile. All 289 parity-feature library tests pass, including source reserve
failures, single/pair template encodings, offsets and serde round trips.

The initial regex audit also found `CompiledByteLevel`/`CompiledRegexSplit` source
wrappers alongside ordinary ByteLevel/Split deserialization. Their closed source
custody and checked scratch preparation are necessary semantics; whether those
semantics required separate executable wrapper variants was the audit question. Oniguruma
versus fancy-regex is a genuine selected regex mechanism and must remain
supported. Untagged decoder/normalizer JSON is an input format, not grounds for
removing model/tokenizer protocol support.

That inventory identified two redundant production wrapper
variants and their duplicate serde/ordinary pretokenization adapters:
`CompiledRegexSplit` beside `Split`, and `CompiledByteLevel` beside `ByteLevel`.
Their real semantic addition is a closed compiled regex owner and a checked
workspace plan. The planned canonical destination at that checkpoint was one
Split/ByteLevel representation whose regex source supports that ownership
directly. Ordinary construction of accepted exact patterns can use the same
checked compiler and workspace; arbitrary
regex syntax and a selected Oniguruma engine retain their distinct general
mechanisms. The admitted constructor must install its source eagerly, while
ordinary ByteLevel decoder/postprocessor objects should not compile an unused
regex. A lazy source cell in the canonical ByteLevel value can express that
initialization distinction without a second executable variant. Serde and
equality remain projections of settings, independent of source initialization.

That audit also identified an observable edge requiring correction: the previous
ordinary fancy `SysRegex` iterator stopped silently on execution errors,
whereas the compiled workspace adapter propagates them. Successful split spans
already use one coverage worker. Consolidation must preserve the original
encoding refusal contract and document any correction to ordinary error behavior;
it must not silently turn a failed regex into successful partial tokenization.

The regex wrappers are now consolidated. `Split` and `ByteLevel` own the same
closed checked source used by ordinary accepted-pattern construction, and each
operation reuses one workspace across its normalized segments. The old
`CompiledRegexSplit`/`CompiledByteLevel` variants and executable adapters are
removed. General arbitrary syntax and Oniguruma remain explicit selected engine
mechanisms. ByteLevel serde and equality ignore its lazy source initialization;
admitted construction installs its own source without touching the ordinary
shared default. Ordinary fancy-regex execution failures now propagate instead
of silently producing partial successful tokenization. The full parity-feature
fork suite passes 289 tests, including an independent general-engine oracle for
every checked pattern, actual backtrack-limit failure, and all source/workspace
reserve failures. Oniguruma-only compilation and seven text source tests pass.

The release measurement `canonical_regex_performance` compares complete split
operations against the independent general engine. Default-pattern chat input
(20,992 bytes), Unicode input (17,408 bytes), and whitespace input (32,769 bytes)
take 1,505/1,544/1,265 microseconds, versus 1,095/1,097/529 microseconds. The third
checked pattern takes 1,671/1,630/1,855 microseconds, versus 905/758/536. Its fixed
direct PikeVM workspace replaces the general engine's adaptive execution caches;
this is an ordinary-path cost of the consolidation. Admitted encoding keeps its
existing engine. The two source heap sizes are 48,919 and 165,250 bytes; complete
workspace requirements are 56,050,935 and 56,145,881 bytes, primarily bounded by
the dependency's one-million-backtrack limit. No smaller unproved stack capacity
is substituted. Reproduce with the ignored release test; temporary output is
`/tmp/tokenizer-regex-performance.log`.

## Checked grammar tokenizer derivative

The initial checked derivative covered only configurations whose prefix removal
was an identity transformation. It has been replaced by source-native projection:
recursive Prepend removal and Metaspace prefix suppression borrow the retained
model/component root, rebuild only changed normalized added-token data, and
retain both original and derived custody. The serialized-JSON derivative and
grammar-startup fallback are removed. Equal vocabulary or independently compiled
equal configurations cannot grant trie authority. See
[prefix sources](bounded-followup-prefix-sources.md) for the transformation matrix
and [tokenizer composition](bounded-followup-tokenizer-composition.md) for the
remaining unqualified component and decoder profiles.

At the initial identity-only checkpoint, the checked constructor and direct
trie-root authentication passed all 16 runtime tokenizer tests, including
independent same-configuration source rejection, complete configuration mismatch, an actual
decode reserve failure, one-byte-short admission and shared final retirement.
The tests verify both original and derivative source charges remain held through
terminal failure or the final trie alias. Reproduction uses the real compiler
path described in `bounded-followup-parameters.md` and `cargo test -p
eredu-runtime --lib original_tokenizer::tests`.

The first search optimization now avoids propagating capture slots when a
checked delegate only needs the match end. It calls the existing PikeVM worker
with an empty slot slice, preserving its leftmost, greedy and UTF-8 behavior.
Nine focused workspace tests and all 289 tokenizer library tests pass. Full
split latencies improve to 1,358/1,388/1,060 microseconds for the first pattern
and 1,460/1,440/1,575 for the third on the inputs above. The independent general
engine takes 1,059/1,063/517 and 873/734/533 microseconds in the same run. Source
and heap geometry remain unchanged; the new fixed half-match return carrier is
included in control storage. Workspace preparation is only 3–10 microseconds,
so workspace reuse does not resolve the remaining search cost. A prototype that
kept separate consuming-state lists was removed because it increased storage
without a meaningful throughput improvement.

The capture-free replacement is now implemented. Both ordinary accepted-pattern
construction and enforced construction use immutable anchored DFA delegates;
capture extraction and Unicode word assertions retain the NFA mechanism they
require. The private emitted recipes validate exact default source bytes and
complete table structure without allocating. Construction declares five actual
vector copies and the boxed DFA owner before it allocates. Failures retain every
partial table. No dynamic parser, NFA builder or DFA determinizer runs during
admitted construction. The old unconsumed NFA recipe constructor, capture-table
copy adapter and finalization scratch are removed.

At the recorded fixed-DFA checkpoint, all 18 distinct regular delegates have
independent compiler parity over Unicode, random concrete text and byte-span starts, plus refusal tests for every actual
table destination. The emitter produces both byte orders; only the target byte
order's 4,453,360-byte static image was linked in that build. Later retained-source
and Onig recipe additions expand this inventory; see
[tokenizer composition](bounded-followup-tokenizer-composition.md). Static image
residency is separate from per-source heap admission, as it was for the former NFA recipe image.
The copied source tables occupy 433,903 bytes for the first full pattern and
2,561,898 bytes for the third. Their mutable workspace is 56,002,345 bytes,
including named input/result carriers. The three nonempty buffers are saves,
backtrack branches and undo records; there are no nested mutable DFA caches.
The existing one-million-backtrack limit remains unchanged.

Final full-split measurements resolve the intermediate regression:

| Pattern | Input | Bytes | Independent general, µs | Shared DFA, µs |
| --- | --- | ---: | ---: | ---: |
| First | Chat | 20,992 | 1,069.27 | 855.04 |
| First | Unicode | 17,408 | 1,088.78 | 868.62 |
| First | Long whitespace then letter | 32,769 | 529.08 | 432.78 |
| Third | Chat | 20,992 | 884.81 | 641.05 |
| Third | Unicode | 17,408 | 746.18 | 518.36 |
| Third | Long whitespace then letter | 32,769 | 550.34 | 444.46 |

The shared path is 18–31% faster in these cases. Each timed case first checks
complete split output against the independent general engine; each timing runs
for at least 300 milliseconds. Run `canonical_regex_performance` as an ignored
release test with the fancy-regex feature. Temporary results are
`/private/tmp/tokenizer-dfa-performance.log`. These measurements include normal
split storage/work; they are not isolated DFA microbenchmarks.

Final validation passes 298 tokenizer library tests with compiler test support
and parity-aware BPE (three ignored measurements), 20 fancy workspace/emitter
tests, and two DFA source suites covering all 18 emitted delegates. All 36
byte-order-specific binary images and both private recipe inventories reproduce
exactly from the pinned compiler. Logs are
`/private/tmp/tokenizer-dfa-tests-3.log`,
`/private/tmp/fancy-dfa-final-tests.log`, and
`/private/tmp/dfa-source-final-tests.log`. These are the fixed-DFA checkpoint results. Later general-engine construction,
retained census and whole-engine oracle results are in
[regex engine funding](bounded-followup-regex-engine-funding.md); later source
coverage passed 60 runtime tokenizer tests at the
[retained configuration source checkpoint](bounded-followup-tokenizer-source.md).
The subsequent 67-test checkpoint and current admitted-profile limits are in
[tokenizer composition](bounded-followup-tokenizer-composition.md).
