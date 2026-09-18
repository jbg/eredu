# Unigram original source and ID operation

The source compiler and deterministic ID operation use the original Unigram
model's score comparison, tie ordering, unknown fusion and byte-fallback policy.
Ordinary token publication and original ID output share the same Viterbi worker
and winning spans. No vocabulary reconstruction or substitute tokenizer is used.

The admitted source has no model cache. Sampling (`alpha`/`nbest_size`) and
training retain the separate existing lattice; the deterministic source and
operation evidence below does not establish bounded lattice execution or
cache-enabled operations. Existing training tests check behavior, not admission.

## Current deterministic worker and evidence

The aggregate source compiler now selects the original Unigram model, including
scored vocabulary order, duplicate spelling overwrite behavior, unknown ID and
byte fallback. Its lookup table and indexed trie use exact hashbrown/vector
layout facts. Each failed actual reservation retains the previously constructed
vocabulary, trie, lookup table and pending spelling. Floating scores go through
serde's ordinary typed f64 deserializer with its declared scratch/error/control
requirements; arbitrary-precision Number transport is not selected for a f64.

The original optimized Viterbi traversal now writes caller-owned best-path rows
and winning spans. Ordinary output materializes token strings from those spans;
the ID operation visits them directly. Both outputs use the same whole-span
unknown/byte-fallback decision. Prefix traversal borrows lengths and the indexed
trie retires without recursively owned maps. These replace the previous
per-match prefix copies, backtracking string fragments and temporary fallback
Token vector.

The independent pristine 0.23.2 comparison in
`doc/validation/unigram-source-compare.rs` passed 18,000 cases: exact ordinary
and source-compiled IDs, spellings and offsets, plus the original ID operation.
The matrix includes missing unknown IDs, score ties, duplicate pieces, Unicode,
unknown fusion, complete/incomplete byte fallback and all three Metaspace prefix
modes. A standalone Cargo manifest must select `reference` from the pinned
registry archive and `local` from the repository fork, with `fancy-regex` and
without default features; `serde_json = "=1.0.151"` supplies fixture serialization.
Run the comparison with `cargo run --release --offline --bin unigram`.

The corresponding benchmark in `doc/validation/unigram-source-benchmark.rs`
used 100,007 scored pieces and 50 uncached runs of 12,800 UTF-8 bytes. One local
release run measured 33.78 ms pristine ordinary, 22.61 ms local ordinary and
9.27 ms original IDs. Source construction took 24.25 ms against a concrete
35,611,163-byte requirement (35,540,167 buffers and 70,996 controls). Numeric
parsing retires before the next score, so its scratch/control contribution is
the largest actual invocation, including the one possible escaping error.
These are synthetic performance observations, not released-model validation.
The library, reserve-failure and runtime custody results below followed this
initial benchmark checkpoint.

## Literal normalization and composition

The released GGUF reconstruction uses literal space-to-metaspace Replace before
Metaspace pre-tokenization. The ordinary Replace already owns literal strings and
an allocation-free `match_indices`/coverage matcher; regex patterns retain their
separate actual matcher. At the start of this follow-up, the original source
normalizer accepted NFC/Prepend compositions only. The source, ID and
normalized-added-token producers now share the original literal replacement
coverage traversal, account the two source strings and normalized destinations,
and preserve the original first-byte alignment used by Metaspace First. This
does not reinterpret general ordered mixed normalizer sequences as a commuting
NFC/prefix form.

The existing Unigram/trainer tests passed in the full library run (298 passing,
one new decoder-profile fixture corrected, three ignored). The five focused
Unigram source/ID tests then passed, including every actual reserve prefix and
floating-score parity. Metaspace decoder construction was a separate source
profile gap at that checkpoint; the released GGUF decoder uses the supported
Replace/ByteFallback/Fuse/Strip sequence. Runtime custody results appear in the
final source checkpoint below.

The shared literal coverage now feeds ordinary replacement, original normalized
added patterns and original ID text. Its source owns exactly the two literal
strings; operation and added-pattern text use prospective capacity bounds.
The first-origin boundary follows ordinary NormalizedString's alignment to the
last removed scalar, including empty-pattern insertion and deleting replacement.
`doc/validation/literal-normalizer-source-compare.rs` passed 54,000 pristine
comparisons of tokens, offsets and IDs, with normalized/raw added-token barriers
and all Metaspace prefix modes. Ordered mixtures were still a source/operation
gap at that checkpoint; the later composition follow-up below closes admitted
ordered mixtures.

`doc/validation/f64-source-compare.rs` passed 1,376 exact pristine serde 1.0.151
floating-bit and diagnostic comparisons in the default profile, and another
1,376 with both `float_roundtrip` and `arbitrary_precision`. The latter is a typed
f64 destination, so arbitrary-precision retained Number strings are not created.
Use separate `local` path and `reference` registry serde aliases; enable the same
feature list on both, and run the standalone probe in release mode. This checks
numeric semantics independently of the tokenizer/model comparison.

At the final deterministic-source checkpoint, the complete tokenizer library
suite passed 302 tests (three upstream ignored tests), the text source-storage
filter passed nine, and the released GGUF Gemma reconstruction fixture passed for
both supported Gemma families and prefix settings. The runtime original-tokenizer suite passed all
60 tests after retaining a supported decoder sequence in its derivative trie
fixture. This includes exact source/operation refusal and lifetime custody for
Unigram, WordLevel, normalized-added-token refresh and prefix projections.

The later [composition follow-up](bounded-followup-tokenizer-composition.md)
implements Metaspace decoder source construction and fixed-destination decoding
through the ordinary shared per-piece traversal, ordered normalization, and
repeated/ordered admitted pre-tokenizer stages. It records the independent
decoder oracle and actual source/refusal tests separately from this deterministic
Unigram checkpoint.

Before this implementation, the aggregate source compiler selected only BPE and
WordLevel. Unigram used recursively owned trie maps, copied prefix matches and
backtracking strings, and collected temporary fallback tokens. That inventory is
historical: the indexed trie and shared span worker above replace those paths.

The [released text/image tool runs](bounded-followup-released-tools.md) and
[native numerical matrix](bounded-followup-native.md) retain their own model,
tokenizer and execution qualifications. Their Qwen checkpoint does not establish
released Unigram-model numerical parity. The Gemma tokenizer reconstruction
fixture and independent pristine-tokenizer comparisons above are the evidence
for this source; pending CPU/public reruns add no results until recorded.
