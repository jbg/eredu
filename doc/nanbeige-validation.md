# Nanbeige 4.2 validation

The dense Nanbeige family uses the ordinary portable decoder and prepared
execution drivers. Its physical blocks are expanded into logical invocations
with exact checkpoint aliases, independent KV state and optional inter-pass
RMSNorm. No family-specific backend implementation is required.

Cold architecture discovery exposes the released configuration as a decoder
group with 22 physical layers, two passes and shared weights. Its 44 logical
execution nodes retain separate input/output capture paths, and explicit
parameter aliases include the normalization between passes. The portable
discovery example prints “22 layers × 2 passes · shared weights” and expands
the group into selectable executions. See [architecture discovery](architecture-discovery.md).

Descriptor coverage includes the pinned released configuration below, one/two/three
passes, both inter-pass normalization policies, and legacy JSON deserialization.
The nonzero scalar fixture checks distinct per-pass captures through prefill
and two cached decode steps, with exact agreement between selected and complete
captures. The existing GGUF numerical fixture also verifies that SafeTensors
and GGUF expose identical descriptors. Reproduce with:

```sh
cargo test -p eredu-architectures --lib discovery::tests
cargo test -p eredu-architectures --test reference_numeric discovery::
cargo test -p eredu-architectures --test reference_numeric nanbeige_official_gguf_layout
```

## Native tools and reasoning

Tagged-parameter activation occurs at `<tool_call>`. Grammar and parsing accept
XML whitespace between structural tags while raw parameter values preserve their
whitespace, including newlines and carriage returns. EOS is masked until an active
call is complete. Interrupted calls emit no `ToolCallEnd` and cannot execute. See
[the wire-format contract](tool-calling.md) for framing and replay rules.

Grammar vocabulary honors each tokenizer entry's `special` flag. Nanbeige's
`<think>`, `</think>`, `<tool_call>` and `</tool_call>` are non-special added tokens;
they participate in ordinary text spelling. Explicit structural token IDs also
work for non-special added tokens.

### Matched prompt comparison

The comparison request is:

```text
System: You are a concise assistant. Follow the user's instructions and use tools when requested.
User: Reply with just the word hello.
```

Thinking is enabled. The official template renders a 40-token prompt ending
in `<|im_start|>assistant\n<think>\n`. Python and Eredu render identical
UTF-8 text and produce identical prompt IDs, with no extra BOS token:

```text
[166100,8481,13,4321,392,261,38503,13886,152361,7916,269,2028,152402,
152350,10126,297,828,4651,724,16668,152361,166101,13,166100,2714,13,
37380,358,983,269,3670,32349,152361,166101,13,166100,66354,13,166103,13]
```

The recorded Eredu run uses MLX Metal with the checkpoint BF16 weights. The
independent reference uses the unchanged released `modeling_nanbeige.py`, PyTorch 2.14.0,
Transformers 4.48.3, CPU float32 and eager attention. Its model source SHA-256 is
`eab3554c228491b99ff518d6a27b3d3cc5320d541966be369069bd1c5f545fcf`.
The reference loads the released fast tokenizer directly through
`PreTrainedTokenizerFast`: the legacy Llama tokenizer class requests a slow
SentencePiece conversion in this environment. Prompt rendering and token IDs
match exactly before inference; the comparison uses unmodified model/template
sources.

| Run | Output tokens | Result |
| --- | ---: | --- |
| Publisher, greedy | 70 | `hello`; all output IDs match Eredu |
| Eredu, ordinary / constrained / controlled greedy, budget 2048 | 70 each | identical output IDs; `hello` |
| Eredu, ordinary / constrained / controlled temperature 0.6, budget 2048 | 68 each | identical output IDs; `hello` |
| Publisher, temperature 0.6, budget 2048 | 148 | `hello` |

At zero-based output position 65, greedy Eredu and the publisher choose
`166104` (`</think>`). The publisher ranks it first with cached logit `40.0951538`
and uncached logit `40.0951843`.

The sampled comparison explicitly sets temperature `0.6`, top-k `40`,
top-p `0.95`, min-p `0.05`, repetition penalty `1`, seed `0` and budget `2048`
in both implementations. These match the supplied checkpoint directory's
effective Eredu sampling defaults with the requested temperature override;
that directory has no `generation_config.json`. Different native RNGs and
numerical precision mean equal seeds do not imply identical cross-runtime
sampled outputs. Within Eredu, all three drivers produce exactly the same IDs.

### Tool execution and portable coverage

The native regression uses the complete checkpoint tokenizer and weights. It
dispatches schema-validated calls, inserts the actual result into conversation
history, and requires an answer containing only a receipt/secret word absent
from the original prompt. It covers both `lookup(value=7)` and multiline
`todo__todo_write(content=...)`, thinking on/off, and uninterrupted/controlled
generation. Execution requires `ToolCallEnd`. The existing native LFM2 basic
and optional Python keyword argument tests are also exercised because vocabulary
construction is shared.

Portable coverage uses the unmodified official template and a compact vocabulary
retaining the checkpoint's added-token IDs, flags, Metaspace processing and
byte-fallback decoding. It covers immediate activation, EOS rejection after a
bare marker, token boundaries, every parser byte split, no whitespace/two
spaces/LF/CRLF-tab structural separators, inline and framed parameter values,
Unicode and preserved boundary whitespace, string enums, nullable/JSON values,
schema rejection, every incomplete call prefix, parallel calls, and replay of
tool results. Replay rejects unescaped `</parameter>` in raw string arguments.

These requests establish matching driver behavior for the recorded fixtures.
They do not guarantee that arbitrary prompts finish within a particular token
budget or select an appropriate tool.

### Reproduce the chat comparison

The reference environment also needs the dependencies from the checkpoint
comparison below. Keep all generated reports outside the source tree:

```sh
export NANBEIGE_VALIDATION=/private/tmp/eredu-nanbeige
CARGO_INCREMENTAL=0 cargo run -p eredu --example chat_probe -- \
  "$NANBEIGE_VALIDATION/checkpoint" "$NANBEIGE_VALIDATION/final-greedy" 2048 0 metal
CARGO_INCREMENTAL=0 cargo run -p eredu --example chat_probe -- \
  "$NANBEIGE_VALIDATION/checkpoint" "$NANBEIGE_VALIDATION/final-sampled" 2048 0.6 metal

HF_HOME="$NANBEIGE_VALIDATION/hf-cache" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=16 \
  python validation/chat_reference.py \
  --probe "$NANBEIGE_VALIDATION/final-greedy-ordinary.json" \
  --output "$NANBEIGE_VALIDATION/publisher-greedy.json"
HF_HOME="$NANBEIGE_VALIDATION/hf-cache" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=16 \
  python validation/chat_reference.py \
  --probe "$NANBEIGE_VALIDATION/final-sampled-ordinary.json" \
  --output "$NANBEIGE_VALIDATION/publisher-sampled.json"

EREDU_NANBEIGE_CHECKPOINT="$NANBEIGE_VALIDATION/checkpoint" \
  CARGO_INCREMENTAL=0 cargo test -p eredu --test native_tool_checkpoints \
  nanbeige_real_checkpoint_executes_tool_and_answers_from_result -- --ignored --nocapture
cargo test -p eredu --no-default-features --lib --test portable_facade --test backend_conformance
```

`chat_probe` writes separate ordinary, semantic and explicitly stepped controlled
reports containing the prompt, token IDs, settings and semantic events.
`chat_reference.py` verifies rendering/tokenization equality, records the
unconstrained closing-token rank at every step and checks the closing prediction
again without cache.

## Released checkpoint

The reference artifact is the official
[Nanbeige/Nanbeige4.2-3B checkpoint](https://huggingface.co/Nanbeige/Nanbeige4.2-3B/tree/0e137298720f7241e83b8aabecc4263dcc7d84b3),
revision `0e137298720f7241e83b8aabecc4263dcc7d84b3`. The two SafeTensors shards match the publisher's LFS SHA-256 hashes:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `model-00001-of-00002.safetensors` | 4,973,547,960 | `09d265d5ec837bc64462796b7f8c110be9a135a55ed7a6eb5d07e0e90c976a94` |
| `model-00002-of-00002.safetensors` | 3,366,076,760 | `31019e7870a044f44bc3f7e981f8c5ecd42d341e5ca6cfdbfd07fb95d95be389` |

The index contains 201 tensors and 8,339,601,408 bytes of BF16 tensor data.
The 22 physical blocks execute twice. Query projection width is 6144 despite
hidden width 3072; each of the 44 logical invocations has its own KV cache.

The recorded native CPU run on an Apple M3 Ultra uses checkpoint BF16 weights. The
independent reference uses the publisher's `modeling_nanbeige.py` unchanged,
with Python 3.11.16, PyTorch 2.14.0, Transformers 4.48.3, float32 and eager
attention. It uses no compatibility patches. The raw prompt is
`The capital of France is`, token IDs `[363, 5463, 290, 7914, 322]`.

One prefill and four cached decode steps pass the standard parity thresholds:
relative L2 at most 0.02, cosine similarity at least 0.999, top-five overlap at
least four and matching argmax. All five predictions match, with all five
top-five candidates shared on every row.

| Logit row | Relative L2 | Cosine similarity | Matching argmax |
| --- | ---: | ---: | ---: |
| Prefill | 0.009683 | 0.999954 | 9965 |
| Decode 1 | 0.007474 | 0.999972 | 152361 |
| Decode 2 | 0.009182 | 0.999966 | 13 |
| Decode 3 | 0.017338 | 0.999853 | 166104 |
| Decode 4 | 0.013785 | 0.999920 | 13 |

The resident measurement reports 14,637,625,344 active allocator bytes after
loading. Resident materialization creates a separate module for each logical invocation,
including copies of repeated block weights. Storage schemas describe physical
weights; execution and residency estimates include the actual replicas.
The complete official checkpoint also passes with a one-block host window and
disk streaming. Both modes produce bit-for-bit identical logits to the resident
run for the prefill and all four decode steps. On this CPU setup the MLX allocator
counters include host staging allocations as well as active execution tensors;
they should not be interpreted as accelerator memory savings.
These short correctness runs do not measure model quality or establish a
performance benchmark.

## Reproduce the released-checkpoint comparison

Keep downloaded weights and generated artifacts outside the repository. The
validation environment additionally needs `accelerate`, `safetensors`, `numpy`
and `PyYAML`. For example:

```sh
export NANBEIGE_VALIDATION=/tmp/eredu-nanbeige
hf download Nanbeige/Nanbeige4.2-3B \
  --revision 0e137298720f7241e83b8aabecc4263dcc7d84b3 \
  --local-dir "$NANBEIGE_VALIDATION/checkpoint"
shasum -a 256 "$NANBEIGE_VALIDATION"/checkpoint/model-*.safetensors

CARGO_INCREMENTAL=0 cargo run -p eredu-backend-mlx --no-default-features \
  --example checkpoint_probe -- \
  --model "$NANBEIGE_VALIDATION/checkpoint" --device cpu \
  --prompt 'The capital of France is' --decode-steps 4 --warmup-runs 0 \
  --output "$NANBEIGE_VALIDATION/official-eredu"

HF_HUB_OFFLINE=1 OMP_NUM_THREADS=16 python validation/reference_runner.py \
  --probe "$NANBEIGE_VALIDATION/official-eredu.json" \
  --model "$NANBEIGE_VALIDATION/checkpoint" \
  --output "$NANBEIGE_VALIDATION/official-transformers" \
  --device cpu --dtype float32 --attn-implementation eager \
  --trust-remote-code --local-files-only --warmup-runs 0

CARGO_INCREMENTAL=0 cargo run -p eredu-evaluation --bin eredu-parity -- \
  --actual "$NANBEIGE_VALIDATION/official-eredu.json" \
  --reference "$NANBEIGE_VALIDATION/official-transformers.json" \
  --output "$NANBEIGE_VALIDATION/official-parity.json"
```

`checkpoint_probe --residency-plan PATH` accepts a serialized `ResidencyPlan`.
A one-block host window uses:

```json
{"mode":"layerwise_host","device_layer_window":1,"device_budget_bytes":4294967296,"host_budget_bytes":34359738368}
```

The disk-streamed plan uses:

```json
{"mode":"dense_disk_stream","device_budget_bytes":4294967296,"host_budget_bytes":34359738368,"host_lookahead":1,"background_queue":1}
```

The probe also accepts `--quantization-bits 4 --quantization-group-size 32`
for a selected affine transformation. These options use the ordinary neutral
execution plan and load policy.

## GGUF and execution coverage

GGUF follows the
[publisher's converter and model implementation](https://github.com/Nanbeige/llama.cpp/tree/nanbeige42):
`nanbeige.block_count` is physical depth, `nanbeige.num_loops` controls repeated
execution and `nanbeige.skip_loop_final_norm` controls intermediate RMSNorm.
Q/K rows are permuted by the converter and use adjacent-pair RoPE. They must
not be interpreted as the SafeTensors split-half layout.

| Coverage | Evidence |
| --- | --- |
| Dense equations and independent pass caches | Independent scalar oracle; one, two and three loops; both normalization policies; full and incremental forwards |
| Exact GGUF layout | Nonzero F32 GGUF compared with SafeTensors through prepared backend-neutral execution; packed Q8 source encoding/provenance contract |
| Packed GGUF execution | Native CPU Q4_0 and Q8_0, resident/host/disk, compared with independently dequantized SafeTensors; prefill and two decode steps |
| Tensor/pipeline execution | Nonzero actual payloads through neutral TP=2, PP=2, TP=2/PP=2 and PP=3, preserved and affine four-bit weights; cuts at and inside loop boundaries; local state and reset |
| Bounded residency | One-block host/disk windows; acquisition order, peak bound units, logits and state compared with resident execution |
| Selected transforms | Independent affine numerical oracle across repetitions; native controlled host/disk with preserved and affine four-bit weights |
| Controlled sessions | Native ordinary/controlled greedy and seeded sampling equality; snapshot, restore and fork of all logical caches |
| Discovery and text | Executable captures for logical layers; cache identity includes loop policy; official tokenizer template fixture |

Packed GGUF parity passes the same standard thresholds as above. Maximum
absolute logit error is below `3e-7` for Q8_0 and `9.23e-4` for Q4_0; all
argmax predictions match. Host and disk results match their resident GGUF
results exactly. The quantized fixture generator is independent of Eredu:

```sh
python eredu-architectures/tests/fixtures/nanbeige/gguf_reference.py \
  --output "$NANBEIGE_VALIDATION" --bits 8
# Repeat with --bits 4 for Q4_0.
target/debug/examples/checkpoint_probe \
  --model "$NANBEIGE_VALIDATION/tiny-q8.gguf" --device cpu \
  --input-ids 1,3,2 --teacher-forced-ids 4,5 --warmup-runs 0 \
  --output "$NANBEIGE_VALIDATION/tiny-q8-run"
target/debug/examples/checkpoint_probe \
  --model "$NANBEIGE_VALIDATION/tiny-q8-oracle" --device cpu \
  --input-ids 1,3,2 --teacher-forced-ids 4,5 --warmup-runs 0 \
  --output "$NANBEIGE_VALIDATION/tiny-q8-oracle"
target/debug/eredu-parity \
  --actual "$NANBEIGE_VALIDATION/tiny-q8-run.json" \
  --reference "$NANBEIGE_VALIDATION/tiny-q8-oracle.json" \
  --output "$NANBEIGE_VALIDATION/tiny-q8-parity.json"
```

This single-host CPU validation matrix contains no native distributed TP/PP
or accelerator measurements. Parallel support has numerical neutral
conformance coverage, rather than a claim of native distributed validation.

Run the checked-in behavioral suites with:

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --lib \
  --test reference_numeric --test reference_conformance
CARGO_INCREMENTAL=0 cargo test -p eredu --no-default-features \
  --test portable_facade --test backend_conformance
CARGO_INCREMENTAL=0 cargo test -p eredu --no-default-features --features mlx \
  --test native_execution_control nanbeige
CARGO_INCREMENTAL=0 cargo test -p eredu-text --no-default-features \
  --test nanbeige_template
```
