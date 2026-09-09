# Gemma 2 support and validation

Gemma 2 uses `eredu-architectures::gemma2` and the shared dense decoder. Its
configuration, checkpoint names, normalization conventions, attention schedule,
parameter placement and cache identity are portable. Backend changes add only
general attention score soft-capping and propagation through paged attention.

The family admits Hugging Face `gemma2` SafeTensors directories and llama.cpp
`gemma2` GGUF. This includes the 2B, 9B and 27B geometry: query projection width
need not equal hidden size, and the query scaling denominator need not equal
head dimension. The published models alternate sliding attention in even layers
with full attention in odd layers. Explicit HF schedules are also validated.

Each block applies RMSNorm before and after attention, and before and after the
GELU-gated feed-forward branch. Post-sublayer normalization follows the complete
projection, including TP reduction, and precedes residual addition. Embeddings
are scaled once by the square root of hidden size. Attention score caps precede
masking and softmax; output caps follow vocabulary projection. HF normalization
weights are offsets from one; llama.cpp GGUF stores the resulting scales. GGUF
Q/K use split-half rotary order without a Llama row permutation.

The implementation follows [Transformers Gemma 2](https://github.com/huggingface/transformers/tree/v4.48.3/src/transformers/models/gemma2),
[llama.cpp's Gemma 2 equations](https://github.com/ggml-org/llama.cpp/blob/master/src/models/gemma2.cpp)
and its [checkpoint conversion](https://github.com/ggml-org/llama.cpp/blob/master/conversion/gemma.py).

## Execution coverage

The family uses ordinary resident, per-block host-layerwise and disk-streamed
execution, including selected matrix transformations. TP, PP and combined TP/PP
use the same four-normalizer blocks and local KV geometry. PP cuts may fall
between sliding and full layers. State retains absolute positions and the
appropriate per-layer history. Controlled and uninterrupted generation share
sampling, text termination, capture, restore and fork drivers.

SafeTensors and GGUF expose the same logical architecture and observation
points. Their cache identities retain their physical format and normalization
convention. Tokenizers, checkpoint chat templates, BOS and EOS policy use the
existing facade/text utilities, including GGUF Gemma Unigram reconstruction.

Nonzero neutral tests cover an independent scalar oracle, full and chunked
prefill, multiple cached decode steps across sliding-window eviction, snapshot
branching, per-block reconstruction, prepared payload loading, TP/PP and
combined TP/PP, affine transformation, GGUF numerical equivalence, and packed
GGUF source/encoding retention. The four-way PP fixture isolates every block.
Native CPU facade tests compare controlled/uninterrupted output for resident,
host-windowed and disk-streamed weights, both preserved and affine-transformed,
and exercise capture, snapshot restore and fork.

Reproduce:

```sh
python3 eredu-architectures/tests/fixtures/gemma2/reference.py
cargo test -p eredu-architectures --lib gemma2
cargo test -p eredu-architectures --test reference_numeric gemma2
cargo test -p eredu-architectures --test reference_conformance
cargo test -p eredu --no-default-features --features mlx \
  --test native_execution_control gemma2
cargo test -p eredu-backend-mlx --no-default-features --lib score_softcap -- --ignored
```

The scalar oracle uses separately written equations and deterministic named
weights. Comparisons use the reference suite's absolute tolerance of `2e-4`.
Native score tests use absolute/relative `1e-5`
and cover masks, learned sinks, and paged chunked prefill with both full and
sliding history. These fixtures supplement released-checkpoint validation.

## Independent Transformers fixture

`tests/fixtures/gemma2/transformers_reference.py` creates a separate four-layer
checkpoint using PyTorch 2.14.0, Transformers 4.48.3, float32, eager attention and
seed 20260909. It uses hidden size 32, four query heads, two KV heads, head size
8, sliding window 2 and deliberately active score/output caps. Prompt IDs are
`[1, 3, 2]`; four cached decode inputs are `[4, 5, 6, 7]`.

```sh
export GEMMA2_VALIDATION=/private/tmp/eredu-gemma2
HF_HOME="$GEMMA2_VALIDATION/hf-cache" python \
  eredu-architectures/tests/fixtures/gemma2/transformers_reference.py \
  "$GEMMA2_VALIDATION"
cargo run -p eredu-backend-mlx --no-default-features --example checkpoint_probe -- \
  --model "$GEMMA2_VALIDATION/synthetic-transformers" --device cpu \
  --input-ids 1,3,2 --teacher-forced-ids 4,5,6,7 --warmup-runs 0 \
  --output "$GEMMA2_VALIDATION/native"
HF_HOME="$GEMMA2_VALIDATION/hf-cache" python validation/reference_runner.py \
  --probe "$GEMMA2_VALIDATION/native.json" --device cpu --dtype float32 \
  --attn-implementation eager --local-files-only --warmup-runs 0 \
  --output "$GEMMA2_VALIDATION/reference"
cargo run -p eredu-evaluation --bin eredu-parity -- \
  --actual "$GEMMA2_VALIDATION/native.json" \
  --reference "$GEMMA2_VALIDATION/reference.json" \
  --output "$GEMMA2_VALIDATION/parity.json"
```

The synthetic comparison passed with all five greedy predictions and all top-five
sets equal. Maximum absolute error was `2.68221e-7` for prefill and `3.50177e-7`
for cached decode; relative L2 errors were below `3.6e-7`. The independently
computed cached and fresh full-prefix logits differ by at most `2.08617e-7`.

## Released-checkpoint validation

On 2026-09-09, the official `google/gemma-2-2b-it` checkpoint was downloaded to
`/private/tmp/eredu-gemma2/hf-cache` at revision
`299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8`. Every downloaded file's size was
verified. The SHA-256 of both weight shards and both tokenizer payloads matched
the publisher's Hugging Face LFS metadata:

| File | SHA-256 |
| --- | --- |
| `model-00001-of-00002.safetensors` | `532d792c9178805064170a3ec485b7dedbfccc6fd297b92c31a6091b6c7e41bf` |
| `model-00002-of-00002.safetensors` | `6d6d9ce84db398fb6e0191f91542e5da0a73da2cb695e172a24edc2146dc8d20` |
| `tokenizer.json` | `3f289bc05132635a8bc7aca7aa21255efd5e18f3710f43e3cdb96bcd41be4922` |
| `tokenizer.model` | `61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2` |

The original BF16 weights were run on the MLX CPU backend, without weight
transformation. The independent reference used PyTorch 2.14.0, Transformers
4.48.3, CPU BF16 and eager attention. A 16-token chat prefill was followed by
eight cached decode inputs along the native greedy path. The prompt's IDs also
matched `AutoTokenizer.apply_chat_template` for the official checkpoint.

| Observation | Maximum absolute error | Maximum row relative L2 | Minimum cosine similarity | Top-five overlap |
| --- | ---: | ---: | ---: | ---: |
| Prefill | 0.227229 | 0.005424 | 0.999988 | 5/5 |
| Eight cached decode rows | 0.285862 | 0.005528 | 0.999992 | 5/5 |

All nine greedy predictions matched. The decoded text was
`The capital of France is **Paris**. ` and prediction IDs were
`[651, 6037, 576, 6081, 603, 5231, 29437, 168428, 235248]`.
The comparison passed the existing default thresholds: relative L2 at most
`0.02`, cosine similarity at least `0.999`, top-five overlap at least four,
and matching unambiguous argmax. No tolerance was relaxed.

The reference runner initializes bounded hybrid caches with explicit absolute
positions and enough capacity for the full probe plus one spare slot.
Transformers 4.48 rotates its sliding storage on the last allocated slot;
reserving the spare slot prevents premature rotation in short probes. The final
cached reference logits were also verified bit-for-bit against a fresh full
24-token prefix without a cache.

Reproduce after accepting the publisher's access terms:

```sh
export GEMMA2_VALIDATION=/private/tmp/eredu-gemma2
export GEMMA2_REVISION=299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8
hf download google/gemma-2-2b-it --revision "$GEMMA2_REVISION" \
  --cache-dir "$GEMMA2_VALIDATION/hf-cache"
export GEMMA2_MODEL="$GEMMA2_VALIDATION/hf-cache/models--google--gemma-2-2b-it/snapshots/$GEMMA2_REVISION"
shasum -a 256 "$GEMMA2_MODEL"/model-*.safetensors \
  "$GEMMA2_MODEL"/tokenizer.json "$GEMMA2_MODEL"/tokenizer.model
cargo run -p eredu-backend-mlx --no-default-features --example checkpoint_probe -- \
  --model "$GEMMA2_MODEL" --device cpu --add-special-tokens \
  --prompt '<start_of_turn>user
What is the capital of France?<end_of_turn>
<start_of_turn>model
' --decode-steps 8 --warmup-runs 0 --output "$GEMMA2_VALIDATION/official-native"
HF_HOME="$GEMMA2_VALIDATION/hf-cache" OMP_NUM_THREADS=8 python validation/reference_runner.py \
  --probe "$GEMMA2_VALIDATION/official-native.json" --device cpu --dtype bfloat16 \
  --attn-implementation eager --local-files-only --warmup-runs 0 \
  --output "$GEMMA2_VALIDATION/official-reference"
cargo run -p eredu-evaluation --bin eredu-parity -- \
  --actual "$GEMMA2_VALIDATION/official-native.json" \
  --reference "$GEMMA2_VALIDATION/official-reference.json" \
  --output "$GEMMA2_VALIDATION/official-parity.json"
```

Validation artifacts and the complete hash manifest remain in that temporary
directory. No released weights are in the source tree. The 9B and 27B variants
have configuration/schema coverage but have not had released-checkpoint native
validation. Native multi-device TP/PP has not been validated; its numerical
coverage uses the neutral collective/transport backend. These are validation
gaps, not architectural restrictions or unsupported execution paths.
