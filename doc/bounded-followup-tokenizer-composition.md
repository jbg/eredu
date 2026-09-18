# Tokenizer composition preservation

The former prepared-text front called the loaded tokenizer's ordinary `encode`
worker for rendered input. A literal chat without a selected protocol did not
require successful grammar construction. Its canonical replacement first
compiles the retained tokenizer configuration and then uses the original ID
operation. Existing restrictions in that compiler therefore gate some formerly
reachable public preprocessing and decoding behavior.

The admitted component profiles below close those gaps through shared workers.
Other custom-profile limits remain explicit in the final section. This work
does not restore an allocating public fallback or reinterpret ordered transforms
as a different normalization policy. Ordinary/source token IDs, normalized added
tokens, prefix derivatives, decoder frontiers and exact source identity must
remain consistent.

The implemented profiles include Lowercase, ordered literal Replace with
NFC/Prepend, repeated and ordered supported pre-tokenizers, and Metaspace decoding.
The source compiler, normalized added-token preparation and ID encoder must agree
on every admitted composition. Unigram stochastic sampling and training use a
separate existing lattice; their accounting is not established by deterministic
Unigram tests, and no concrete published-config use is claimed here.

## Metaspace decoder checkpoint

Metaspace source construction now uses the same paid replacement-string producer
as its pre-tokenizer component. Direct and singleton Sequence decoder forms keep
the actual replacement scalar, prepend policy and split field. Configuration
authentication compares the complete component, including fields that do not
alter decoding.

Ordinary decoding and the fixed-destination decoder share one allocation-free
per-token character iterator. It removes every replacement scalar from the first
retained token when prepend policy is First or Always; subsequent tokens map
replacement scalars to spaces. Unknown IDs and skipped special tokens do not
consume the first-token position; a retained empty token does. The immutable
decoder source retains original UTF-8 spellings, and execution writes into its
source-derived fixed destinations.

The source compiler prices the real replacement String before construction.
Its injected reserve failure and all three decoder-buffer reserve failures retain
their complete actual account until error retirement. Exact-capacity source
construction succeeds; one-byte-short admission refuses before construction.

Validation at this checkpoint:

- Text decoder stream/frontier parity and all decoder reserve prefixes pass.
- Runtime cold admission, source identity, partial failure custody and final-alias
  retirement pass.
- The public portable plain-source test passes all three prepend policies,
  comparing ordinary encoding and decoded output and retaining the original
  payer through escaped output after the model and source handles retire.
- The independent pristine tokenizers 0.23.2 runner passes 36,864 comparisons over
  three replacement scalars, all prepend policies, split settings, direct and
  singleton Sequence forms, empty/unknown/special tokens and ordered triples.
  Source: `validation/metaspace-decoder-source-compare.rs`.

Logs: `/private/tmp/contracts-metaspace-decoder-text-tests2.log`,
`/private/tmp/contracts-metaspace-decoder-runtime-tests.log`, and
`/private/tmp/contracts-metaspace-decoder-pristine.log`, and
`/private/tmp/contracts-metaspace-decoder-public-tests2.log`.

## Ordered normalization checkpoint

The source compiler now retains the flattened order of Lowercase and mixed NFC,
Prepend and literal Replace components. Existing pure NFC/Prepend canonical forms
continue through their original worker. The mixed form applies the same Unicode
NFC decomposition/recomposition workers, ordinary lowercase scalar/change
iterator and literal coverage worker, carrying the first-original-scalar frontier
through every stage. Raw added-token boundaries, normalized added spellings and
input-prefix removal consume this same ordered projection.

Two text destinations are priced before construction. Each NFC stage borrows its
actual intermediate UTF-8 source into the original NFC plan and checks that plan's
real destinations against the reserved envelope before construction. Temporary
NFC storage retires before the next stage; the two text buffers remain with the
operation. The envelope derives its expansion limits from the selected Unicode
canonical table, Hangul decomposition and Rust lowercase iterator, without a
caller-selected capacity. Construction, failure and escaping output retain the
same source/account custody.

Validation:

- Runtime original-tokenizer suite: **64 passed**, including three new ordered
  normalization tests for ordinary parity, exact/one-short C and E admission,
  every reached text/NFC reserve refusal, changed added-token prefix derivatives
  and final source/error/output retirement.
- Public portable retained-source test: **1 passed**, covering standalone
  Lowercase and a nested ordered Prepend/Lowercase/NFC/literal Replace sequence
  through canonical plain generation with the real backend pool.
- Pristine tokenizers 0.23.2 oracle: **11,340 comparisons passed** across 252
  ordered triples, Unicode/case expansion, raw and normalized added tokens and
  all three Metaspace policies. The runner records an existing upstream alignment
  panic for empty Replace followed by a transforming normalizer; its successful
  reference comparisons therefore place empty Replace last. This is not an
  additional source admission restriction.

Reproducible runner: `validation/ordered-normalization-source-compare.rs` with
local path and pristine `=0.23.2` aliases as used by the Metaspace runner. Logs:
`/private/tmp/contracts-ordered-normalization-runtime-all.log`,
`/private/tmp/contracts-ordered-normalization-public-tests.log`, and
`/private/tmp/contracts-ordered-normalization-pristine-final.log`.

## Ordered pre-tokenizer checkpoint

The source compiler now retains arbitrary ordered and repeated admitted
pre-tokenizer leaves, including nested Sequence declarations. ByteLevel supports
its actual prefix-space and regex settings; Digits supports individual and
coalesced digit runs; Metaspace preserves every replacement/prefix/split setting.
Whitespace and the existing exact non-inverted Isolated Split expressions retain
one independently constructed regex source per stage. There is no new arbitrary
regex construction claim or pattern substitution.

The ID worker prepares each real regex workspace from its retained source. It
prices the stage table, two text destinations, two split tables and any ByteLevel
prefix scratch before construction. Text/split capacities follow checked
component expansion from the actual input bound, including earlier normalization;
no caller supplies the capacity. Every stage consumes the prior stage's ordered
nonempty splits using the existing numeric coverage, Metaspace, byte mapping and
regex search workers. The first-original-scalar frontier crosses both transforms
and split boundaries. Prefix projection removes Prepend and Metaspace prefixes
while preserving ByteLevel's prefix setting, exactly as the ordinary policy does.

Validation:

- Runtime original-tokenizer suite: **67 passed**, including three new tests for
  ordered parity, exact/one-short C and E, all six new destination reserve cuts,
  every regex stage's workspace cuts and early/late constructor cuts, full
  configuration authentication and prefix/source/error/output custody.
- The pristine tokenizers 0.23.2 oracle passes **30,618 comparisons**: every
  ordered triple of nine pre-tokenizer configurations, three normalization
  profiles and fourteen Unicode/raw-added-token inputs. BPE merges make split
  boundaries observable in IDs. The configurations include repeated regex and
  Metaspace stages, ByteLevel prefix insertion, nested sequences, Unicode digits,
  combining marks and a non-ASCII replacement scalar.
- A canonical portable plain-source regression uses retained configuration,
  Lowercase, nested repeated Whitespace, grouped Digits and Metaspace, with actual
  backend-pool custody through the escaped output. It passes in the complete
  portable suite: **37 passed, 2 external fixtures ignored**, alongside
  **126 backend conformance tests passed**. Logs are
  `/private/tmp/tokenizer-final-portable-facade.log` and
  `/private/tmp/tokenizer-final-backend-conformance.log`.

Runner: `validation/ordered-pretokenizer-source-compare.rs`, using the same local
and pristine dependency aliases as the normalization oracle. Runtime logs:
`/private/tmp/contracts-ordered-pre-runtime-tests2.log` and
`/private/tmp/contracts-ordered-pre-runtime-all.log`. Independent log:
`/private/tmp/contracts-ordered-pre-pristine-final.log`.

## Default Oniguruma profile closure

The default `onig` feature previously made ordinary ByteLevel/Whitespace/Split
construction select an Onig source while the enforced compiler selected its
fixed workspace source. Exact configuration authentication therefore refused
otherwise supported public requests. Recognized expressions now use the same
source constructor for ordinary and enforced execution under either feature
profile; general unlisted expressions keep their existing ordinary engine.

The selected Onig word expression preserves its actual language rather than
assuming that the two engines have identical Unicode classes. A complete scalar
census of the pinned Onig positive and negative classes found exactly six extra
Latin-1 scalars in positive `\w`: U+00B2, U+00B3, U+00B9, U+00BC, U+00BD and
U+00BE. The negated class uses the Unicode property table and has no such extras.
Onig also excludes join controls from the word class. The exact closed program
is emitted by the same pinned compiler from those measured class facts, while
serialization and source authentication retain the caller's original expression.
No runtime Onig call, arbitrary regex fallback or caller-supplied capacity enters
the enforced worker.
The internal lowered program is excluded from caller declaration selection in
both constructors, so it does not accidentally expand ordinary Onig syntax
acceptance. After that final visibility tightening, three focused fork tests pass:
all original declarations against the selected general engine and ordinary serde,
shared source/repeated workspace behavior, and rejection of the internal program
as an ordinary Onig declaration. Log:
`/private/tmp/contracts-onig-fork-regex-closed4.log`.

Independent evidence:

- Both positive and negative classes were compared over all **1,112,064 Unicode
  scalars**, collecting every mismatch before checking the resulting class facts.
- All **nine original selected expressions** pass contextual split-text and
  byte-offset comparison against pristine tokenizers 0.23.2 with Onig across the
  same full scalar census: **10,008,576 scalar/expression comparisons**, zero
  differing chunks.
- The public retained-source regression covers ByteLevel regex, Split followed
  by ByteLevel, and Whitespace. Its golden IDs distinguish superscripts,
  fractions, negative-class behavior and join controls, and canonical generation
  consumes the same prompt through the original source worker.
  It passes under `eredu --no-default-features --features onig`; the complete
  runtime original-tokenizer suite also passes **67/67** under `eredu-text/onig`,
  including original C/E short-budget and reached producer-refusal custody cases.
- The no-Onig pristine ordered pre-tokenizer oracle was rerun after the feature
  change and still passes **30,618 comparisons**.

Reproducible runners are `validation/onig-word-class-compare.rs` and
`validation/onig-fixed-source-compare.rs`. The former uses `onig = 6.5.3` with
its pinned `onig_sys = 69.9.3` and the local fancy-regex fork. The latter uses
local and pristine `tokenizers = 0.23.2` aliases, enabling `local/onig` and
`reference/onig`. Logs: `/private/tmp/contracts-onig-full-class-census.log`,
`/private/tmp/contracts-onig-fixed-pattern-spans-final3.log`, and the actual
recipe emitter `/private/tmp/contracts-onig-recipe-emitter3.log`. Public/runtime
logs are `/private/tmp/contracts-onig-public-source-closed.log` and
`/private/tmp/contracts-onig-runtime-closed.log`; the no-Onig oracle log is
`/private/tmp/contracts-ordered-pre-pristine-after-onig.log`.

```sh
cargo test --offline -p eredu --no-default-features --features onig --test portable_facade regex_feature_selected_sources_preserve_public_plain_configuration_and_word_semantics
cargo test --offline -p eredu-runtime --features eredu-text/onig --lib original_tokenizer
cargo test --offline --manifest-path third-party/tokenizers-0.23.2/Cargo.toml --no-default-features --features fancy-regex,onig --lib utils::regex::tests
```

## Remaining profile audit scope

The old prepared-text front's rendered-input branch called `LoadedModel::encode`
on the selected ordinary tokenizer; its incremental `TextDecoder` also called
the ordinary decoder. Canonical startup now requires the retained configuration
source and original ID/decoder workers. Consequently, unsupported source profiles
are real public gates for custom tokenizer configurations; calling them old
compiler restrictions does not establish capability preservation.

The bounded local released-profile census inspected these actual cached files:

| Published tokenizer | Immutable revision | Observed profile |
| --- | --- | --- |
| Inkling-Small | `8cc5877b44d343f88b92086aa1fb72897950f06a` | BPE, no normalizer, exact GPT-4o-style Split then ByteLevel, ByteLevel decoder |
| Muse-Glimmer-30B | `97c77dff50b2797bcc558fa2d909761dbc575c59` | Same admitted component form and exact selected expression |
| LFM2.5-1.2B-Instruct | `3d2b14bdb14c120e685895d23320542f0c80c1f6`, `0f604ada3f766f9f257460c4c9f0b5d6f69d431b` | BPE, no normalizer, admitted Split then ByteLevel, singleton ByteLevel decoder |
| Qwen3.5 | `2fc06364715b967f1860aea9cf38778875588b17` | BPE, NFC, admitted Split then ByteLevel, ByteLevel decoder |

Those observed component forms are admitted. This is a source-profile inspection,
not a new end-to-end validation of each complete checkpoint. The current GGUF
reconstruction workers also select admitted ByteLevel or Metaspace forms and the
supported literal Replace/ByteFallback/Fuse/optional Strip decoder orderings;
existing deterministic Gemma reconstruction tests provide separate evidence.
No unseen released tokenizer configuration is assumed to pass this census.

The supported architecture family registry contains no BERT/WordPiece family.
The repository's explicit WordPiece examples are generic tokenizer/decoder tests,
including refusal cases; the audit found no supported-family published caller
requiring BertNormalizer, BertPreTokenizer, StripAccents or a WordPiece model.
This does not claim those generic configurations work through canonical startup.
Other normalization primitives (NFD/NFKD/NFKC, Strip, StripAccents, BertNormalizer,
Precompiled), general regex normalization, unlisted regex Split expressions or
other Split delimiter modes, and other decoder programs remain outside the
current source/operation profile. Nested arbitrary decoder sequences are also
not covered by the fixed sequence classifier. These are implementation limits,
not inherent model restrictions. A concrete supported-family configuration using
one remains a capability-preservation gap to close rather than a fallback to an
allocating encoder. Stochastic Unigram/training accounting remains separate;
no concrete public inference caller selecting that lattice was established.

