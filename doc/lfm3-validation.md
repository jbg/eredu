# LFM3 text execution and acceptance

The experimental LFM3 text route admits `lfm3` SafeTensors and GGUF checkpoints.
Its declaration owns width-seven gated convolutions, global and sliding attention,
per-head sigmoid gates, separate rotary policies, and sandwich normalization before
residual addition. Global rotary pairs retain full-head frequency spacing; only the
highest-frequency half rotates. Rotary tables, products and sums use FP32 before
the final input-dtype cast.

Resident, host-layerwise and disk-streamed execution use independently leased decoder
layers. TP/PP construction consumes the same declaration and local state geometry.
Ordinary and controlled text generation use the shared drivers. The existing tagged
tool dialect consumes the released template's function/parameter tags; the LFM2
Python-call dialect does not describe this template.

This is a text implementation, not complete LFM3-family integration. Vision/video,
audio, LiquidCodec, omni and d1 decision execution are not implemented. Published
NVFP4 GGUF encoding is not admitted by the current generic GGUF reader. No support
for these paths is inferred from the shared text backbone or source inventory.

## Released SafeTensors evidence

The source manifest is `validation/fixtures/lfm3/sources.json`. Instruction weights
from `LiquidAI/LFM3-1.7B` revision
`1da06bec6a819da10b54c67e5c90972b46d81871` have SHA-256
`135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330`.
Exact widening from BF16 to F32 produces SHA-256
`6f1823dc67f4f6367f61702379ac5505124e7e4719013753e2ff0ebb08230c98`.
This measures execution of the widened weights, not native BF16 numerical parity.

The independent publisher implementation uses PyTorch 2.15.0 and Transformers
5.22.2, eager attention and F32 execution. MLX CPU compares every vocabulary score
after prefill and after each of eight teacher-forced cached inputs
`[5,8,13,21,34,55,3,2]`, with maximum absolute tolerance `5e-4`.

| Prompt | Maximum absolute error across all nine rows | Argmax agreement |
| --- | ---: | --- |
| `[1,3,2,5,8,13,21,34,55]` | `6.318092346191406e-5` | 9/9 |
| 269 tokens, `(13*i + 1) % 128000` | `1.544952392578125e-4` | 9/9 |

A greedy CPU run also matches all nine predicted IDs and full score rows, with
maximum absolute error `6.318092346191406e-5`.

Metal F32 execution with `MLX_ENABLE_TF32=0` matches the same independent reference:
greedy maximum error is `6.67572021484375e-5`; the 269-token input has maximum error
`1.1399388313293457e-4`. All nine argmax IDs agree in both cases. Default TF32
execution exceeds the `5e-4` F32 tolerance (maximum decode error `0.0238952637`);
the precise setting is part of this acceptance configuration.

The longer input crosses the 256-token sliding window. Full tensors and diagnostic
outputs live outside the repository. The shared `chunked_prefill` conformance driver
passes on CPU and Metal with a compact convolution/sliding/global LFM3 fixture,
using the same
residency, affine loading and controlled/uninterrupted checks as existing fixtures.
Shared execution agreement does not substitute for released-checkpoint acceptance.

## Same-artifact GGUF evidence

`LiquidAI/LFM3-1.7B-GGUF` revision
`cd0c74c575b5c416f3011b5b4c298077f26ca8c2`, file
`LFM3-1.7B-Q4_K_M.gguf`, has SHA-256
`3793e1e6e2266919d6e015287221b41100175a98eddaba92e111aed3b05c93c4`.
Eredu Metal with `MLX_ENABLE_TF32=0` and llama.cpp Metal execute this identical
encoded artifact. The reference is llama.cpp `cd959bae99ff4bed22bef2604f45241c09bbb8c8`
(0.9.0, build 12110), with F32 KV and flash attention disabled. This reference pin
is the installed version used for measurement.

The short prompt and eight teacher-forced cached inputs compare every vocabulary
score. Maximum relative L2 is `0.01882853`, minimum cosine is `0.99983101`, minimum
top-five overlap is 4/5, and all nine argmax IDs agree. These meet the existing shared
logit profile (`relative_l2 <= 0.02`, `cosine >= 0.999`, top-five overlap at least four,
and matching unambiguous argmax). Maximum absolute difference is `0.24158096`;
quantized evidence does not use the F32 absolute tolerance.

llama.cpp CPU uses Q8_K activation dots for Q4_K weights. The CPU-to-CPU comparison
has relative L2 `0.052–0.088` and does not meet the shared score profile, despite
matching token IDs. It is not recorded as passing numerical acceptance. This result
is separate from the passing Metal-to-Metal comparison.

Compact measured rows, precision settings and token IDs are retained in
`validation/fixtures/lfm3/acceptance-results.json`. Other published GGUF encodings
have no recorded numerical acceptance.

## Reproduce

Download with the pinned shared helper, keeping `HF_HOME` and all outputs outside
the checkout. Custom Python code is fetched from the same immutable revision.

```sh
python eredu-evaluation/scripts/download_component_fixture.py /private/tmp/lfm3-source --model lfm3
python eredu-evaluation/scripts/prefill_reference.py /private/tmp/lfm3-source/provenance.json /private/tmp/lfm3-reference.json --f32-checkpoint /private/tmp/lfm3-f32 --trust-remote-code --decode 5,8,13,21,34,55,3,2
cargo run -p eredu-backend-mlx --no-default-features --example checkpoint_probe -- --model /private/tmp/lfm3-f32 --device cpu --input-ids 1,3,2,5,8,13,21,34,55 --teacher-forced-ids 5,8,13,21,34,55,3,2 --warmup-runs 0 --output /private/tmp/lfm3-actual
EREDU_PREPARED_CHECKPOINT_CACHE=/private/tmp/lfm3-prepared-cache cargo test -p eredu-backend-mlx --no-default-features --test chunked_prefill cpu_chunked_prefill_matches_full_observed_logits_and_controlled_generation -- --exact
```

`validation/reference_runner.py` also produces the ordinary shared comparison
artifact from a probe JSON using `--trust-remote-code --local-files-only --device cpu
--dtype float32`. A same-artifact GGUF reference uses
`validation/gguf_reference.cpp`, built against the exact llama.cpp revision recorded
with its result. The exporter emits full prefill and cached-decode vocabulary rows;
generated token agreement alone is insufficient.

Native BF16, the base checkpoint and physical multi-device numerical parity
have no recorded acceptance here. Vision, speech and decision results are absent.

For precise Metal acceptance, build the probe with `--features metal`, select
`--device gpu`, and set `MLX_ENABLE_TF32=0`. Metal execution needs device access.
The same probe accepts the GGUF file as `--model`; use
`--output /private/tmp/lfm3-gguf-actual` for the comparison command below. Build the independent exporter
against the pinned llama.cpp headers and library:

```sh
clang++ -std=c++17 validation/gguf_reference.cpp -I/opt/homebrew/include -L/opt/homebrew/lib -lllama -Wl,-rpath,/opt/homebrew/lib -o /private/tmp/gguf_reference
/private/tmp/gguf_reference /private/tmp/LFM3-1.7B-Q4_K_M.gguf /private/tmp/lfm3-gguf-reference.safetensors 1,3,2,5,8,13,21,34,55 5,8,13,21,34,55,3,2 gpu
cargo run -p eredu-evaluation --bin eredu-parity -- --actual /private/tmp/lfm3-gguf-actual.json --reference /private/tmp/lfm3-gguf-reference.json --output /private/tmp/lfm3-gguf-parity.json
EREDU_TOOL_CHECKPOINT=/private/tmp/lfm3-f32 cargo test -p eredu --no-default-features --test portable_facade lfm_checkpoint_native_tool_activation_with_portable_backend -- --exact --ignored
```

The portable tokenizer/template case uses scripted backend outputs. It verifies
native tool grammar activation and decoded semantic events, not model tool quality.
