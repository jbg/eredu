# Retained complete tokenizer source consolidation

The public `TokenizerSourceInput` and `ChatSourceInput` selectors accept a
consumed file or the loaded model's complete retained configuration. Both forms
converge on the original funded source compiler. GGUF's existing producer
supports embedded HF JSON, GPT BPE, Llama BPE including score-derived merges,
and Gemma Unigram; it does not require a tokenizer sidecar.

The initial source entry consumed only `tokenizer.json`. The replacement
serializes the complete retained configuration through the same tokenizer
`Serialize` worker used ordinarily, then consumes the existing JSON-root
compiler. It preserves model flags, merges, normalizer, pre-tokenizer,
postprocessor, decoder, added-token settings and exact selected template
identity. File and retained-configuration inputs remain distinct source forms;
no existing unfunded graph is relabeled as an admitted graph.

The serializer originally built BPE's rank-sorted merge vector followed by a
redundant vector of cloned token strings. Its shared replacement keeps the rank
sort and emits borrowed pairs through the same serializer. Ordinary and funded
serialization prepare the same scratch; enforced preparation reserves it before
construction. Added vocabulary, packed BPE vocabulary and compiled template
serializers emit borrowed rows. The admitted aggregate wrapper funds output
growth, concrete controls and the fixed JSON writer-error transport before use.
The sizing/preparation traversal is admitted before allocating.

Runtime owns the original input allowance and preserves partial serialized
bytes plus typed failures through retirement. The fresh aggregate compiler
derives its source bound from those bytes. The facade binds the source to its
complete loaded tokenizer and selected named/default template; the backend
supplies the runtime pool. The former file-only backend source method is removed;
file convenience uses `From<File>` into the closed input selectors.

Unigram construction and ID encoding, plus ordered/repeated admitted
pre-tokenizer compositions, now use their bounded original producers. See
[Unigram](bounded-followup-unigram.md) and
[tokenizer composition](bounded-followup-tokenizer-composition.md) for the
validated component profiles and the remaining explicit configuration limits.

The implemented public selectors are `TokenizerSourceInput` and `ChatSourceInput`,
each with a consumed `File` and a `RetainedConfiguration` form. The latter reads
only the corresponding loaded model's actual complete configuration. Tokenizer
inputs converge on `WorkingMemoryPool::compile_tokenizer_source_for_generation`;
selected templates use the already shared `ChatTemplatePlan::prepare_model`
worker and preserve exact model/name/source selection. Complete configuration
comparison also preserves the tokenizer's non-serde `encode_special_tokens`
flag explicitly before fresh construction. Mutable nondefault Unigram sampling
settings require the same explicit identity comparison; serialization cannot
silently reset them.

The original input allowance now extends through one shared ledger charge
worker before each reached serializer destination. Its active constructor count
is incremented once, every successful charge stays held through input/error
retirement, and C construction overlaps the still-retained serialized I buffer.
No source/compiler attempt or payer-dependent retry is introduced. A failed
writer retains its actual partial bytes and first original funding error. The
serializer stops nested component descent at the original fixed JSON compiler's
own depth boundary.

Initial public embedded-GGUF validation exposed an additional concrete mechanism
gap: the existing Qwen GGUF producer uses the exact single-digit `\\p{N}` split
expression, while the original regex recipe inventory admitted the related
`{1,3}` expression. The exact source is now
compiled into the same recipe inventory, together with the existing Han-aware
and hexadecimal-joiner GGUF sources. No producer expression changes. The nine
complete sources share nineteen unique delegates. All 42 development-emitted
files reproduce byte for byte; all 32 prior endian DFA images remain unchanged.
The fixed-source/ordinary/scoped workers pass 7,020 exact comparisons against
pristine fancy-regex 0.19.0 with untouched lower dependencies. The shared DFA
source suite and twenty fancy workspace tests pass (one timing test ignored).

The reproducible standalone probe is
`doc/validation/tokenizer-source-serialize-compare.rs`, compiled with distinct
local and pristine tokenizers 0.23.2 aliases and pristine lower dependencies for
the reference. All nine complete serialized byte comparisons pass, including
shuffled vocabulary insertion and actual merge/normalizer/pre/post/decoder
configuration. On the pinned compiler in release mode, minimum of five runs:

| Model / vocabulary | Output bytes | Serializations | Pristine ms | Local ordinary ms | Funded writer ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| BPE / 16 | 1,247 | 2,000 | 4.507 | 3.572 | 7.716 |
| BPE / 1,024 | 38,339 | 100 | 9.425 | 7.122 | 12.898 |
| BPE / 32,768 | 1,463,685 | 5 | 18.765 | 13.122 | 21.345 |
| WordLevel / 16 | 769 | 2,000 | 1.849 | 1.845 | 4.126 |
| WordLevel / 1,024 | 11,739 | 100 | 1.418 | 1.424 | 3.014 |
| WordLevel / 32,768 | 469,948 | 5 | 2.999 | 3.252 | 5.178 |
| Unigram / 16 | 895 | 2,000 | 1.994 | 1.957 | 4.520 |
| Unigram / 1,024 | 18,919 | 100 | 1.948 | 1.884 | 3.800 |
| Unigram / 32,768 | 699,335 | 5 | 2.778 | 2.804 | 6.306 |

These compare complete current storage/serialization implementations. The funded
column includes actual writer growth requests to a counting policy, not the
runtime pool's locks or fresh tokenizer compilation. It does not establish a
zero-overhead source path. Initial large ordinary BPE/WordLevel regressions
were traced to binary-searching dense IDs and sorting a dense reverse map.
The shared workers now verify the actual dense index before direct lookup, and
prove contiguous map IDs before ordered emission; sparse sources retain their
original checked lookup/sorted-row work without traversing holes. The table
records the corrected implementation. Runtime whole-source timing must be
reported separately from this byte producer microbenchmark.


The public retained-source fixtures verify named default/tool selection, template
overrides, cancellation, foreign-pool rejection and original input refusal before
buffer creation. An embedded scored-tokenizer fixture compares every actual f64
score bit after 266 nontrivial f32-to-f64 metadata conversions; the selected public
Serde profile uses its existing `float_roundtrip` feature. No score equality is
relaxed, and non-serialized mutable sampling options still reject on exact
configuration comparison instead of silently resetting them.

The unused JSON-plan prefix-derivative constructor is removed. Prefix projection
now derives only from the actual retained source and keeps that original owner;
equal independently compiled sources do not acquire semantic-root authority.
The shared constructor specializes its terminal error to reachable projection
failures. One exactly priced projection Box exists before refresh starts and is
retained unchanged by success or failure, including partial added-token tables
and the original source. On the pinned compiler this reduces the prefix error
from 6,840 bytes to 168 bytes, with 16-byte prepared/projection-failure carriers.
At that prefix-error checkpoint, all sixty original-tokenizer runtime tests passed,
covering exact short admission, partial failure, source authentication, alias retirement and unwind custody.
The later 67-test checkpoint and component-profile scope are recorded in
[tokenizer composition](bounded-followup-tokenizer-composition.md).

The ignored `pinned_gguf_retained_source_preserves_ids_template_and_retirement`
fixture is a source-only released-artifact check. It reads GGUF metadata through
the actual loader, then passes the complete retained configuration to the same
fresh compiler; no tensor payload is read or executed and no tokenizer sidecar
is substituted. It compares Unicode/special-token encodings and embedded chat
renders with the ordinary workers, and verifies complete original-account
retirement. Artifact provenance, commands, source fingerprints and measured
results are in `doc/validation/retained-gguf-source-2026-09-18.json`. This is not
released-model numerical inference evidence.
