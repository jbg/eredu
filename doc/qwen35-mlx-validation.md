# Qwen3.5 converted MLX checkpoint validation

Qwen hybrid admission supports the MLX-VLM SafeTensors layout used by
`mlx-community/Qwen3.5-0.8B-8bit`, alongside the existing official Hugging Face
and GGUF layouts. The converted layout is selected from the tensor catalog,
not the repository name or quantization bit width.

## Compatibility behavior

- `language_model.model.*` and `vision_tower.*` bind to canonical text and
  vision parameters, including quantization scales and biases.
- Converted recurrent convolution and vision patch kernels transpose back to
  canonical axes. Text normalization scales are converted back to offsets
  from one. Native subtraction preserves the stored dtype and the inferred
  materialization byte count.
- Embeddings retain their selected per-parameter quantization, including when
  the composite model has no global quantization default.
- A converted artifact that omits every MTP tensor has no embedded draft,
  even when its copied upstream configuration declares one. A partial draft
  remains invalid. CLI drafting selection uses the admitted architecture.

Official SafeTensors retain their original aliases, shapes, normalization
conventions, and declared-MTP requirements. Mixed official and converted
namespaces are rejected. Missing quantization companions still fail admission.
GGUF keeps its existing format-specific recipes; the MLX-VLM recipe path
explicitly excludes GGUF sources.

## Released checkpoint

Validation on 2026-09-12 used the existing Hugging Face cache outside the source
tree, pinned to revision `87e768fbfa03994095f3d14527c80c5ae70c5758` of
[mlx-community/Qwen3.5-0.8B-8bit](https://huggingface.co/mlx-community/Qwen3.5-0.8B-8bit/tree/87e768fbfa03994095f3d14527c80c5ae70c5758).
The `model.safetensors` SHA-256 matched the publisher's LFS metadata:
`9a887c5731520e33bbd324378ecb7b560c1f750af16a6dc28229d500304e656c`.
The offline fixture in `eredu-architectures/tests/fixtures/configs/` records the
published configuration and all 847 tensor names, shapes and dtypes; it contains
no weight payloads.

Native Metal ordinary and semantic generation produced the same 32 greedy
tokens. Independently loading the same checkpoint with MLX-LM and replaying the
exact 39-token prompt matched all 32 generated tokens, including cached decode.
The reference used `mlx-lm==0.31.3`, `mlx==0.32.2`, and
`transformers==5.17.0`. The comparison requires exact token equality, without a
numeric tolerance. It does not assert full-logit equality for this released
BF16/8-bit model.

The native CLI also completed 16 greedy tokens for `What is 2 + 2?` with
thinking disabled in resident, host-layerwise (one device layer), and
disk-streamed (1 GiB device budget, zero host cache/lookahead) modes. All three
outputs matched exactly, beginning `The answer is **4**.`. These runs retained
the CLI's default draft-token setting, verifying that stale MTP configuration
does not enable an absent draft.

Reproduce on macOS with Metal access:

```sh
export CARGO_BUILD_BUILD_DIR="$HOME/Library/Caches/cargo-build/eredu"
export QWEN35_CHECKPOINT="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.5-0.8B-8bit/snapshots/87e768fbfa03994095f3d14527c80c5ae70c5758"
shasum -a 256 "$QWEN35_CHECKPOINT/model.safetensors"
cargo run -p eredu --example chat_probe --features metal -- \
  "$QWEN35_CHECKPOINT" /private/tmp/eredu-qwen35 32 0 metal
python3 -m venv /private/tmp/eredu-qwen35-reference
/private/tmp/eredu-qwen35-reference/bin/pip install \
  mlx-lm==0.31.3 mlx==0.32.2 transformers==5.17.0
/private/tmp/eredu-qwen35-reference/bin/python \
  eredu-backend-mlx/validation/qwen35_mlx_reference.py \
  /private/tmp/eredu-qwen35-ordinary.json \
  --output /private/tmp/eredu-qwen35-reference.json
cargo run -p eredu-cli --features metal -- \
  --model mlx-community/Qwen3.5-0.8B-8bit \
  --revision 87e768fbfa03994095f3d14527c80c5ae70c5758 \
  --no-auto --temperature 0 --max-tokens 16 --thinking off 'What is 2 + 2?'
```

Add `--layerwise-host` for the host-windowed run, or
`--dense-disk-stream --device-budget-bytes 1073741824 --host-budget-bytes 0
--dense-host-lookahead 0 --dense-background-queue 0` for the disk-streamed run.

`chat_probe` writes successful ordinary and semantic reports, then exits with
the existing composite-executable controlled-session rejection:
`complete native state copying is unavailable for this executable`.
This checkpoint-layout change does not implement composite native state
copying. Controlled/snapshot/fork validation for this released model therefore
remains unavailable; ordinary generation and the reference comparison above
succeed. Released image/video and native distributed execution were not run.

## Regression coverage

Nonzero official-layout and converted-layout fixtures produce matching
prefill logits and two cached decode steps through production inspection,
preparation and parameter binding. Resident, host-layerwise and disk-streamed
execution use an absolute logit/parameter tolerance of `2e-4`; retained state
matches exactly. A conditional-model fixture verifies text and vision
transformations against the original payloads, including the patch kernel.
Admission tests check the full released catalog, reject partial MTP and missing
companions, and preserve the official schema. Native recipe tests verify
F16/BF16/F32 dtype, values and materialized byte counts.

The complete architecture library and numerical suites passed: 537 library
tests (one ignored) and 156 numerical tests. These include existing official
SafeTensors, GGUF, and parallel-execution coverage. The portable facade suites
passed 72 tests (one ignored); CLI tests passed 54 tests (one ignored). A fresh
released official upstream or GGUF
checkpoint comparison was not performed in this change; compatibility is
covered by preserved format selection and the regression fixtures.

```sh
export CARGO_BUILD_BUILD_DIR="$HOME/Library/Caches/cargo-build/eredu"
cargo test -p eredu-architectures --lib --test reference_numeric
cargo test -p eredu-backend-mlx --lib --features metal \
  subtract_one_preserves_inferred_dtype_and_residency_bytes
cargo test -p eredu --no-default-features \
  --test portable_facade --test backend_conformance
cargo test -p eredu-cli --features metal
```
