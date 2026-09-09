# Nanbeige 4.2 validation

The dense Nanbeige family uses the ordinary portable decoder and prepared
execution drivers. Its physical blocks are expanded into logical invocations
with exact checkpoint aliases, independent KV state and optional inter-pass
RMSNorm. No family-specific backend implementation is required.

## Released checkpoint

Validated on 2026-09-09 using the official
[Nanbeige/Nanbeige4.2-3B checkpoint](https://huggingface.co/Nanbeige/Nanbeige4.2-3B/tree/0e137298720f7241e83b8aabecc4263dcc7d84b3),
revision `0e137298720f7241e83b8aabecc4263dcc7d84b3`. The two downloaded
SafeTensors shards were checked against the publisher's LFS SHA-256 hashes:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `model-00001-of-00002.safetensors` | 4,973,547,960 | `09d265d5ec837bc64462796b7f8c110be9a135a55ed7a6eb5d07e0e90c976a94` |
| `model-00002-of-00002.safetensors` | 3,366,076,760 | `31019e7870a044f44bc3f7e981f8c5ecd42d341e5ca6cfdbfd07fb95d95be389` |

The index contains 201 tensors and 8,339,601,408 bytes of BF16 tensor data.
The 22 physical blocks execute twice. Query projection width is 6144 despite
hidden width 3072; each of the 44 logical invocations has its own KV cache.

The native CPU run on an Apple M3 Ultra used checkpoint BF16 weights. The
independent reference ran the publisher's `modeling_nanbeige.py` unchanged,
with Python 3.11.16, PyTorch 2.14.0, Transformers 4.48.3, float32 and eager
attention. It used no compatibility patches. The raw prompt was
`The capital of France is`, token IDs `[363, 5463, 290, 7914, 322]`.

One prefill and four cached decode steps passed the standard parity thresholds:
relative L2 at most 0.02, cosine similarity at least 0.999, top-five overlap at
least four and matching argmax. All five predictions matched, with all five
top-five candidates shared on every row.

| Logit row | Relative L2 | Cosine similarity | Matching argmax |
| --- | ---: | ---: | ---: |
| Prefill | 0.009683 | 0.999954 | 9965 |
| Decode 1 | 0.007474 | 0.999972 | 152361 |
| Decode 2 | 0.009182 | 0.999966 | 13 |
| Decode 3 | 0.017338 | 0.999853 | 166104 |
| Decode 4 | 0.013785 | 0.999920 | 13 |

The latest resident run reproduced the original native logits exactly. It
reported 14,637,625,344 active allocator bytes after loading. Resident
materialization currently creates a separate module for each logical invocation,
including copies of repeated block weights. Storage schemas describe physical
weights; execution and residency estimates include the actual replicas.
The complete official checkpoint also passed with a one-block host window and
disk streaming. Both modes produced bit-for-bit identical logits to the resident
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
A one-block host window used:

```json
{"mode":"layerwise_host","device_layer_window":1,"device_budget_bytes":4294967296,"host_budget_bytes":34359738368}
```

The disk-streamed plan used:

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

Packed GGUF parity passed the same standard thresholds as above. Maximum
absolute logit error was below `3e-7` for Q8_0 and `9.23e-4` for Q4_0; all
argmax predictions matched. Host and disk results matched their resident GGUF
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

Native distributed TP/PP and accelerator execution were not run on this
single-host CPU validation setup. Parallel support has numerical neutral
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
