# Bounded tokenizer and text processing

## Retained configuration and model workers

The tokenizer source compiler retains the exact model, normalizer, pre-tokenizer,
postprocessor and decoder configuration. Ordinary and admitted execution use
the same model equations. Unsupported profiles return typed refusals before
execution. BPE, WordLevel and deterministic Unigram produce IDs from their
retained models rather than reconstructed vocabularies.

Unigram preserves score comparisons, tie ordering, unknown fusion and byte
fallback through a shared Viterbi worker. Sampling with `alpha`/`nbest_size` is
not part of the deterministic admitted source. `ModelCachePolicy { capacity }`
sets an entry count; zero disables the cache. It is neither a byte bound nor
native allocator-cache policy.

Template processing retains one immutable set of tables. Builder, serde,
comparison, serialization and bounded ID projection borrow the same ownership.
Source compilation rejects undefined special references even when ordinary
encoding with special tokens disabled can avoid them.

## Added vocabulary

One packed matcher supplies ordinary and admitted added-token extraction.
Duplicate spellings preserve first ID, final flags and sticky special membership.
Normalization, Unicode word and whitespace flags and special-token skipping are
profile semantics, not choices of matcher representation.

For `N` raw declarations and `B` total spelling bytes, including duplicates,
construction reserves five targets: `N` entries, `B` bytes, two `N` indexes and
`B + 2` nodes. Layout and arithmetic are checked before reservation; partial
failure retains allocated buffers. Nodes contain the construction queue, so
failure links need no auxiliary heap or recursive stack.

On arm64, the retained buffer bound is `56*N + 29*B + 56` bytes, with 11,844
bytes of fixed construction controls. Search uses a 56-byte iterator and fixed
locals, without heap scratch. These are requested storage bounds, not RSS.
Overlapping patterns can require input-length times competitor-length work;
there is no universal linear-time claim.

## Composition and regular expressions

Ordered normalizers and pre-tokenizers preserve their configured order,
including Unicode transforms, literal replacement, ByteLevel and Metaspace.
The grammar tokenizer derivative removes `Prepend` normalizers and uses
`Metaspace` prepend scheme `Never` through the same configuration workers.
Refreshing a normalizer updates normalized added-token spellings and matching.

`Split` and `ByteLevel` own closed checked regex sources and reuse one workspace
across normalized segments. Admitted construction installs its source eagerly;
ordinary ByteLevel initialization is lazy and does not affect serde or equality.
General regex syntax and Oniguruma remain selected engine mechanisms. Regex
execution errors propagate rather than producing a partial successful encoding.

Checked patterns preserve the selected default-Oniguruma behavior, including
contextual Unicode boundaries and span ordering. Independent coverage includes
10,008,576 contextual comparisons. Unknown custom patterns, callbacks and
unqualified normalization profiles require explicit support and bounds.

## Storage and search cost

The arm64 release measurement uses Rust 1.98.0, repeats hot-input searches for at
least 350 ms and checks matches with an independent `str::find` leftmost/longest
oracle. DAAC is a separate reference implementation, not a production option.

| Input | Patterns / matches | DAAC reference MiB/s | Packed matcher MiB/s |
| --- | ---: | ---: | ---: |
| 33,792-byte chat with delimiters | 256 / 512 | 828.32 | 1,303.57 |
| 30,208-byte prose without delimiters | 256 / 0 | 1,146.90 | 94,446.39 |
| 32,768-byte shared prefixes without matches | 32 / 0 | 620.61 | 370.26 |
| 32,768-byte nested matching prefixes | 63 / 521 | 327.99 | 393.51 |
| 32,768-byte delayed short match | 2 / 32,768 | 1.43 | 1.70 |
| 34,816-byte Unicode | 4 / 6,144 | 378.37 | 398.46 |

The unmatched shared-prefix case costs about 34 additional microseconds per
32 KiB. The delayed-match case scans a long competitor before committing each
short match. These workload-specific measurements do not imply universal speedup
or generation throughput. Exact generators and independent expected matches are
in `third-party/tokenizers-0.23.2/src/tokenizer/added_vocabulary/benchmark.rs`.

```sh
CARGO_INCREMENTAL=0 cargo test --manifest-path third-party/tokenizers-0.23.2/Cargo.toml \
  --release --lib added_matcher --no-default-features --features fancy-regex \
  -- --ignored --nocapture
```

The reference build uses the tokenizer sources identified by revision `77e08045`
and the same benchmark module. Dependency archive and local-source provenance
are in `third-party/tokenizers-upstream.json` and `third-party/parser-upstream.json`.
