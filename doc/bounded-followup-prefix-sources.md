# Grammar tokenizer input-prefix sources

Inventory before implementation:

- The ordinary facade removes `Prepend` normalizers recursively, removes empty
  normalizer sequences, and sets every `Metaspace` pre-tokenizer's prepend scheme
  to `Never`, including nested sequences. Setting the normalizer refreshes
  normalized added-token spellings and their matcher. It currently obtains this
  result by cloning/reconstructing a complete tokenizer.
- The original aggregate constructor accepts only absent/NFC normalization and
  flat ByteLevel/Digits/Split pre-tokenization. Its ID-only operation has the same
  narrower profile. `input_prefix_normalized_source` therefore proves an identity
  transformation only; its refusal is an implementation gap for other ordinary
  profiles, not an inherent tokenizer limitation.
- `compile_input_prefix_derivative` constructs another aggregate from serialized
  JSON and authenticates full equality on the identity-only accepted profile.
  It does not implement genuine nonidentity normalization and is scheduled for
  removal with the frozen-JSON grammar startup fallback.
- The immutable model and decoder are currently inside one owned
  `PreparedTokenizer`. A genuine derived source must retain its original root
  while borrowing the unchanged model/component storage. Normalized added-token
  spellings and matcher state need their actual changed-source producer; token-ID
  equality alone does not authenticate tokenizer semantics.

Canonical direction: express the original and prefix-normalized source as closed
borrowed component views consumed by the same source construction, vocabulary,
trie and ID-encoding workers. Preserve the ordinary component transformations,
pay changed components/added-token matching state and source controls before
construction, and retain the exact root through derived source, operation and
failure retirement. Do not serialize the tokenizer, clone the whole model, or
introduce a second encoding algorithm. Extend original source and operation
support for the affected components together. An identity view remains an
allocation-free fast path through that same contract.

The source-native `ConstraintCompiler` constructor passes sparse vocabulary,
special metadata, EOS parity, empty configured EOS, refusal and source/payer
retirement tests. Its prefix selection now consumes the genuine derived source
described below.

The original source constructor now accepts recursive NFC/Prepend sequences and
uses one canonical normalizer producer. Its fixed-stack borrowed traversal pays
prefix strings, leaf staging, final component rows and controls before creating
those destinations. The ID-only operation uses that same canonical source,
applies prefixes to each unmatched raw-token span, and runs normalized added-token
matching after normalization. Added spelling construction pays actual transformed
pattern bytes, matcher nodes and peak NFC workspace before construction; the same
normalized spelling is exposed to decoding. Borrowed configuration comparison
checks this canonical form against the nested declaration without reconstructing
it. The shared Unicode NFC worker now accepts the actual borrowed prefix and
preserves composition across the prefix/input boundary.

Validation: the tokenizers library with `fancy-regex,tokenizer-compiler-test-support`
passes 284 tests (3 ignored benchmarks). This includes nested Prepend/NFC parity,
Unicode prefix composition, normalized added-token matching and decoding, existing
NFC/regex/template cases, and real reserve failures. Command uses the pinned
`rustc`, `CARGO_INCREMENTAL=0`, and `/private/tmp/eredu-grammar-check` target; output
is `/private/tmp/contracts-tokenizers-prefix-full-tests.log`.

The original immutable tokenizer and decoder now have one closed shared root.
The source-native prefix producer retains that exact root, constructs only changed
normalized added-token tables and their decoder, and uses the same runtime source
admission/publication worker as original construction. The derivative retains the
original payer and rebuilds its canonical token-domain mask, because changed
normalized spellings can change domain membership. Identity removal remains an
alias with no new reservation. A trie retains both original and derived custody
and authenticates the exact derived tokenizer, while its semantic association
continues to identify the original root.

Metaspace construction consumes prospectively paid replacement storage. Recursive
pre-tokenizer sequences are flattened without allocation during inspection. The
ID worker reads the original component through a borrowed input view, selecting
`Never` for the derivative while leaving the root unchanged. NFC reports the
initial original-coordinate frontier through its existing alignment stream;
`First` therefore behaves correctly after normalized added-token boundaries and
Unicode decomposition. This does not reconstruct tokenizer JSON or clone a model.

Validation: the Metaspace matrix checks 3,960 original/derived encoding comparisons
against ordinary tokenizers, including every prepend scheme, three replacement
scalars, split selection, nested sequences, NFC/Prepend combinations, and raw and
normalized added-token boundaries. Runtime source tests pass 20 cases, including
changed domain membership, one-byte-short admission, concurrent final aliases,
and nested Metaspace with the existing fallback decoder and exact trie bytes.
Logs: `/private/tmp/contracts-metaspace-tests.log` and
`/private/tmp/contracts-prefix-runtime-complete-tests.log`.

The later [composition follow-up](bounded-followup-tokenizer-composition.md)
closes repeated and ordered admitted pre-tokenizer stages, standalone Lowercase,
and ordered literal Replace/NFC/Prepend/Lowercase mixtures. It preserves the same
first-origin projection and authentic source/operation custody. Prefix removal
continues to leave ByteLevel's prefix setting unchanged; only Prepend and
Metaspace prefix policy change. Runtime coverage now includes a derivative with
mixed normalization, repeated regex stages and ByteLevel prefix insertion.

Consumer migration and removal of the frozen-JSON fallback are complete; current
consumers retain the authentic original or derived source. The composition note
lists remaining primitive, regex and decoder profile limits and distinguishes
observed released-family profiles from generic library configurations. Those
limits do not imply a compatibility fallback or complete tokenizer coverage.
