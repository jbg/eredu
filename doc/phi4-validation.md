# Phi-4 support and validation

The original Microsoft Phi-4 14B model uses the existing Llama-compatible decoder with `model_type: "phi3"`. Its Q, K and V projections share one matrix; its gate and up projections share another. Configuration and checkpoint admission retain those fused matrices. The shared decoder describes their component boundaries for tensor parallelism and derives each rank's local head and intermediate dimensions. Native backend operators are unchanged.

This support targets [microsoft/phi-4](https://huggingface.co/microsoft/phi-4/tree/2db69c1c3e91a05d2c64a3185acfbaf36f744e25), revision `2db69c1c3e91a05d2c64a3185acfbaf36f744e25`. It does not establish support for Phi-4 Mini, multimodal Phi variants, partial rotary encoding or LongRoPE. The original model has 40 layers, 40 query heads, 10 KV heads, head width 128, hidden width 5120, intermediate width 17920 and vocabulary size 100352. It uses SiLU, biasless projections, full attention and unscaled full-head RoPE with base 250000.

SafeTensors admission uses the official `self_attn.qkv_proj` and `mlp.gate_up_proj` names. The published `phi3` GGUF layout uses `attn_qkv` and fused `ffn_up`, including their quantization companions. GGUF retains the HF half-head rotary ordering. A zero GGUF sliding window means full attention. These conventions follow the [llama.cpp Phi converter](https://github.com/ggml-org/llama.cpp/blob/master/conversion/phi.py) and [Transformers Phi3 implementation](https://github.com/huggingface/transformers/blob/v4.48.3/src/transformers/models/phi3/modeling_phi3.py).

The official tokenizer and chat template use the existing SafeTensors text utilities. GGUF reconstruction uses the publisher's three-digit pre-tokenization rule and strips whitespace around special tokens. Six comparisons against the pinned official tokenizer pass, covering ordinary text, numeric boundaries, combining marks, Chinese, emoji, chat formatting and whitespace around chat markers.

## Execution verification

Execution state, controlled advancement, cancellation, speculation, snapshot
isolation and consumed budgets use the shared conformance drivers described in
[family acceptance](family-acceptance.md). Numerical acceptance uses a complete
model and an independent pinned reference. Native distributed released-checkpoint
parity is unvalidated unless explicitly recorded below.

## Independent numerical reference

`eredu-architectures/tests/fixtures/phi4/transformers_reference.py` creates a four-layer, width-32 checkpoint with nonzero deterministic weights, seed 20261013, four query heads and two KV heads. The independent reference uses PyTorch 2.14.1, Transformers 4.48.3, float32 and eager attention. GGUF fixtures use `gguf` 0.19.0. The prompt is `[1, 3, 2]` and the four cached decode inputs are `[4, 5, 6, 7]`.

```sh
export PHI4_VALIDATION=/private/tmp/eredu-phi4-validation
python eredu-architectures/tests/fixtures/phi4/transformers_reference.py "$PHI4_VALIDATION"
for artifact in synthetic-transformers synthetic.gguf synthetic-q8.gguf; do
  prefix="$PHI4_VALIDATION/${artifact%.*}"
  cargo run -p eredu-backend-mlx --features accelerate --example checkpoint_probe -- \
    --model "$PHI4_VALIDATION/$artifact" --device cpu --input-ids 1,3,2 \
    --teacher-forced-ids 4,5,6,7 --warmup-runs 0 \
    --output "$prefix-native"
  reference_model="$PHI4_VALIDATION/synthetic-transformers"
  if [ "$artifact" = synthetic-q8.gguf ]; then
    reference_model="$PHI4_VALIDATION/synthetic-dequantized-transformers"
  fi
  python validation/reference_runner.py --probe "$prefix-native.json" \
    --model "$reference_model" --device cpu --dtype float32 \
    --attn-implementation eager --local-files-only --warmup-runs 0 \
    --output "$prefix-reference"
  cargo run -p eredu-evaluation --bin eredu-parity -- \
    --actual "$prefix-native.json" --reference "$prefix-reference.json" \
    --output "$prefix-parity.json"
done
```

Compare the probe's `prefill.logits` with the final row of `prefill_logits` in `synthetic-independent.safetensors`, and `decode.logits` with `decode_logits[0]`. Q8 uses `synthetic-q8-independent.safetensors`, computed by Transformers from the independently dequantized Q8 weights. The maximum absolute tolerance is `1e-5`.

| Source | Prefill maximum absolute error | Four decode rows maximum absolute error |
| --- | ---: | ---: |
| SafeTensors F32 | `3.57628e-7` | `3.57628e-7` |
| GGUF F32 | `3.57628e-7` | `3.57628e-7` |
| GGUF Q8_0 | `5.36442e-7` | `5.96046e-7` |

All five greedy predictions match: `[47, 31, 9, 9, 55]`. Relative L2 errors stay below `5.5e-7`. Independent cached logits also match fresh full-prefix evaluation within `1e-5`. Quantized and unquantized weights are separate numerical references.

## Released checkpoint

The pinned official configuration is included at `eredu-architectures/tests/fixtures/configs/phi-4-2db69c1c.json`. Its fused QKV shape is `[7680, 5120]` and gate/up shape is `[35840, 5120]`. All six official weight shards pass size and SHA-256 verification against the publisher's metadata at revision `2db69c1c3e91a05d2c64a3185acfbaf36f744e25`.

The full 14B BF16 checkpoint passes a 14-token chat prefill and eight cached decode steps on an Apple M3 Max with 64 GiB memory, using the optimized CPU build. The independent reference uses PyTorch 2.14.1, Transformers 4.48.3, BF16 and eager attention. The prompt asks “What is the capital of France?” with the official chat template; its token IDs match `AutoTokenizer.apply_chat_template` exactly.

| Phase | Maximum relative L2 error | Minimum cosine similarity | Maximum absolute error |
| --- | ---: | ---: | ---: |
| Prefill | `0.0111183` | `0.9999389` | `0.28125` |
| Eight cached decode rows | `0.0148109` | `0.9998937` | `0.265625` |

Every row has finite logits, identical top-five token sets and matching argmax. The unchanged parity thresholds are relative L2 at most `0.02`, cosine similarity at least `0.999`, top-five overlap at least four, and equal argmax whenever the reference's top-two logits differ. All nine predictions match: `[791, 6864, 315, 9822, 374, 12366, 13, 100257, 100264]`, decoding to `The capital of France is Paris.<|endoftext|><|im_start|>`. The low-level probe deliberately continues past the end token to exercise all eight cache updates.

Reproduce the pinned download and provenance checks outside the source tree:

```sh
export PHI4_VALIDATION=/private/tmp/eredu-phi4-validation
export PHI4_CHECKPOINT="$(python - <<'PY'
import hashlib, os
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
revision = '2db69c1c3e91a05d2c64a3185acfbaf36f744e25'
info = HfApi().model_info('microsoft/phi-4', revision=revision, files_metadata=True)
path = Path(snapshot_download(
    'microsoft/phi-4', revision=revision,
    cache_dir=Path(os.environ['PHI4_VALIDATION']) / 'hf-cache',
    allow_patterns=['config.json', 'generation_config.json', 'tokenizer*.json',
                    'special_tokens_map.json', 'model.safetensors.index.json',
                    'model-*.safetensors'],
))
for entry in info.siblings:
    if entry.rfilename.endswith('.safetensors'):
        file = path / entry.rfilename
        assert file.stat().st_size == entry.size
        with file.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == entry.lfs.sha256
print(path)
PY
)"
```

Run the native probe, then the reference and comparison sequentially to bound memory use:

```sh
cargo run --release -p eredu-backend-mlx --features accelerate --example checkpoint_probe -- \
  --model "$PHI4_CHECKPOINT" --device cpu --decode-steps 8 --warmup-runs 0 \
  --prompt '<|im_start|>user<|im_sep|>What is the capital of France?<|im_end|><|im_start|>assistant<|im_sep|>' \
  --output "$PHI4_VALIDATION/official-native"
OMP_NUM_THREADS=8 python validation/reference_runner.py \
  --probe "$PHI4_VALIDATION/official-native.json" --device cpu --dtype bfloat16 \
  --attn-implementation eager --local-files-only --warmup-runs 0 \
  --output "$PHI4_VALIDATION/official-reference"
cargo run -p eredu-evaluation --bin eredu-parity -- \
  --actual "$PHI4_VALIDATION/official-native.json" \
  --reference "$PHI4_VALIDATION/official-reference.json" \
  --output "$PHI4_VALIDATION/official-parity.json"
```
