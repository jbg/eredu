# LFM3 text execution and acceptance

The experimental LFM3 text route admits `lfm3` SafeTensors and GGUF checkpoints.
Its declaration owns width-seven gated convolutions, global and sliding attention,
per-head sigmoid gates, separate rotary policies, and sandwich normalization before
residual addition. Global rotary pairs retain full-head frequency spacing and
rotate the highest-frequency half. Rotary tables are computed in F32 and stored
in the activation dtype. The products and sums use F32 before the final cast.
The causal convolution uses four-lane FP32 tap accumulation and rounds its
affine output once. Carry stays in the activation dtype.

Resident, host-layerwise and disk-streamed execution use independently leased decoder
layers. TP/PP construction consumes the same declaration and local state geometry.
Ordinary and controlled text generation use the shared drivers. The existing tagged
tool dialect consumes the released template's function/parameter tags; the LFM2
Python-call dialect does not describe this template.

The public LFM3 generation routes cover text and single-device VL
SafeTensors. VL accepts images and video through the Rust facade and the
ordinary CLI. The Audio route admits SafeTensors and GGUF with retained
codec assets and exposes controlled text and speech events.

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

## Independent vision components

`lfm3::vision` contains the released encoder's linear patchifier, gated spatial
depthwise-convolution blocks, frame-local bidirectional GQA blocks with axial RoPE,
and LayerNorm/spatial-merge/GELU projector. The projector's two affine layers
accumulate the dot product and add bias before rounding to the activation dtype.
For BF16 input, projector LayerNorm runs in FP32 and rounds its affine output
once to BF16.
The BF16 spatial mixer accumulates convolution products in F32 and rounds the
convolution output to BF16 before applying its output gate.
BF16 frame attention rounds query-key products, scaled scores and softmax
probabilities to BF16, with softmax computed in F32.
Each encoder layer and each input/output unit is constructible independently.
The host patch-packing utility accepts already normalized frames. It duplicates
still images across temporal slots and repeats the last frame when a video ends
with an incomplete temporal patch.

`validation/fixtures/lfm3/vision-sources.json` pins the VL instruction checkpoint,
its complete weight hash and the publisher source-file hashes. The reference
entry point executes the pinned publisher encoder, projector and image/video
patchifiers on an exact F32 widening of the released component weights. Its cases
are a rectangular image, an odd-length three-frame clip, and two differently
shaped images. Inputs are already sized; resizing and frame sampling are outside
this comparison. It does not establish end-to-end conditional-generation parity.

The compact measured results in
`validation/fixtures/lfm3/vision-component-results.json` record exact host packing
agreement for all three cases. CPU encoder maximum absolute error is
`8.7738037109375e-5`; precise Metal encoder maximum is `1.3017654418945312e-4`.
Projector maxima are `6.705522537231445e-7` on CPU and
`1.259148120880127e-6` on Metal. Every encoder/projector comparison meets the preset
`5e-4` maximum absolute tolerance. These measurements use F32 widening; they do
not establish native BF16 component parity.

```sh
python -c 'from huggingface_hub import snapshot_download; snapshot_download("LiquidAI/LFM3-VL-2B", revision="23387beda43c60310348e83e25f78f33c3140d3a", local_dir="/private/tmp/lfm3-vl-source", allow_patterns=["*.py", "*.json", "*.jinja", "*.safetensors"])'
HF_HOME=/private/tmp/lfm3-hf HF_MODULES_CACHE=/private/tmp/lfm3-hf/modules python validation/lfm3_vision_reference.py /private/tmp/lfm3-vl-source /private/tmp/lfm3-vl-reference
cargo run -p eredu-backend-mlx --no-default-features --example lfm3_vision_reference -- /private/tmp/lfm3-vl-reference/f32-component /private/tmp/lfm3-vl-reference /private/tmp/lfm3-vl-reference/cpu-results.json --device cpu
MLX_ENABLE_TF32=0 cargo run -p eredu-backend-mlx --features metal --example lfm3_vision_reference -- /private/tmp/lfm3-vl-reference/f32-component /private/tmp/lfm3-vl-reference /private/tmp/lfm3-vl-reference/metal-results.json --device gpu
```

The reference runner needs PyTorch, Transformers, SafeTensors, torchvision and
Pillow, and verifies the immutable weights and source hashes before executing
publisher Python code. The native runner validates source geometry and checks the
derived F32 weight hash before comparing values. Payloads and detailed outputs
remain outside the checkout.

### Mixed image and video conditioning

The released VL SafeTensors artifact admits its nested language weights and
vision units together. The shared decoder replaces the expanded image and
video markers with projected rows. Media delimiters and the tokens between
video frame groups keep their ordinary embeddings.

The CPU reference case uses the released checkpoint widened to F32. It
combines the rectangular image and odd-length clip above in one 30-token
prompt. All 30 prefill rows and eight cached rows cover the full 128,000-token
vocabulary and pass the preset `5e-4` absolute tolerance. The eight greedy
IDs match exactly and are `[334, 34, 334, 34, 334, 34, 334, 34]`.

| Scores | Maximum absolute error | Relative L2 |
| --- | ---: | ---: |
| Prefill | `2.205e-4` | `6.140e-6` |
| Eight cached rows, worst per measure | `4.363e-5` | `2.462e-6` |

```sh
HF_HOME=/private/tmp/lfm3-hf python validation/lfm3_vision_reference.py \
  /private/tmp/lfm3-vl-source /private/tmp/lfm3-vl-conditional --conditional
cargo run -p eredu-backend-mlx --no-default-features --features image --example lfm3_vision_reference -- \
  /private/tmp/lfm3-vl-source /private/tmp/lfm3-vl-conditional \
  /private/tmp/lfm3-vl-conditional/eredu-cpu.json --released-checkpoint
```

The 30-token fixture starts with already sized RGB frames and explicitly
expanded token IDs. The same command also replays a raw image and a seven-frame
video at 6 fps. That fixture uses the shared native media preparation path for
checkpoint-default resizing, frame sampling, and timestamp tokenization. The
runner passes the prepared media through the ordinary opaque MLX prompt and
checks that the owned VL cursor retains its semantic cache identity. The native
decoder prefills the expanded prompt in 17-token chunks. The runner compares
the concatenated vocabulary rows against the publisher's full prefill, then
compares eight cached decode steps and their greedy IDs.

The raw fixture samples four frames, expands to 97 tokens, and matches the
publisher's patch grids, normalized patch values, and token IDs exactly.
All eight cached score rows pass, with maximum absolute error `5.627e-5` and
greedy ID `954` at each step. The full prefill comparison exceeds the preset
`5e-4` tolerance with maximum absolute error `2.816e-3` and relative L2
`3.080e-5`. The rows above tolerance are inside the image span. Substituting
the publisher's projected rows in a diagnostic run reduces the maximum prefill
error to `4.683e-4`, isolating the gap to the decoder's sensitivity to projection
differences. The raw fixture remains an open numerical acceptance case.

The raw fixture uses resident execution with SafeTensors widened to F32.
With `--component-graph --device gpu` and `MLX_ENABLE_TF32=0`, the same
raw fixture matches patches, grids and tokens on Metal. Its prefill error is
`2.238e-3` and its maximum cached-row error is `1.140e-4`. All eight greedy
IDs match. The pre-sized prefill in the same run passes at `3.765e-4`.

On CPU, the raw F32 fixture has maximum prefill error `2.820e-3` and maximum
cached-row error `3.731e-5`. All eight greedy IDs match. With publisher media
rows, the CPU decoder's maximum prefill error is `3.763e-4`. In the Metal
image tower, the first numerical difference is at the patch projection
(`5.722e-6` maximum, `6.954e-7` relative L2). The final tower output has
relative L2 error `4.659e-6`.

With publisher patchifier rows, the Metal F32 image projection has maximum
error `2.623e-6` and the raw decoder prefill has maximum error `4.035e-3`.
Substituting patchifier rows accumulated in FP64 and rounded to F32 gives
`4.292e-6` and `2.880e-3`, respectively. Both replays match all eight greedy IDs.

The pre-sized fixture also runs through the shared host-layerwise and
disk-streamed weight policies for both the vision tower and the decoder. All
three modes produce identical comparison results for vision components, all 30
prefill rows, eight cached rows, and the greedy IDs.

| Weight policy | Peak vision device parameter bytes | Peak decoder device parameter bytes |
| --- | ---: | ---: |
| Resident | 757,521,408 | 6,915,206,144 |
| Host-layerwise | 100,696,064 | 1,304,526,848 |
| Disk-streamed | 148,975,616 | 1,560,469,504 |

The disk run admits 16 vision units and 24 decoder blocks from the retained
checkpoint. Joint preparation binds those units through the portable VL
component graph and projects them to the two executors. Decoder peak bytes
include pinned parameters. The figures report
logical device parameter residency and exclude activation and cache allocations.
Cold selection reports the sum of the vision and decoder device parameter
requirements. On this fixture that sum matches the two policy peaks in each mode.
Adding `--controlled-vision` cancels the first tower pass after the patchifier
and first encoder block, restarts on the same weight policy, and steps through
each remaining tower invocation one unit at a time. The released pre-sized
fixture yields identical comparison results to uninterrupted execution.

```sh
for mode in resident host disk; do
  cargo run -p eredu-backend-mlx --no-default-features --example lfm3_vision_reference -- \
    /private/tmp/lfm3-vl-source /private/tmp/lfm3-vl-conditional \
    "/private/tmp/lfm3-vl-conditional/eredu-${mode}.json" \
    --released-checkpoint --pre-sized-only --residency "$mode"
done
```

Use `--component-graph` in place of `--pre-sized-only` to replay the same
checkpoint through one shared weight policy and the portable VL kernel. This
route projects the pre-sized image and video, then runs the conditioned
prefill and eight cached decoder steps through the same graph. On CPU, the
maximum absolute errors are `5.514e-7` for projected rows, `2.205e-4` for
prefill scores, and `4.363e-5` for cached scores. All eight greedy IDs match.
Resident, host-layerwise, and disk-streamed runs produce identical comparison
results. The combined policy peaks at 7,672,727,552, 1,304,526,848, and
1,560,469,504 device parameter bytes, respectively.
With the `image` feature and the raw fixture, `--component-graph` also
replays processor output through the retained VL cursor. Each media projection
and decoder prefix uses the same graph and request state as cached decode.
The combined route reproduces the raw fixture's patch, prompt, score, and
token comparisons, including the prefill tolerance gap.
The CPU replay also reproduces those comparisons after forks following the
first media step and first decoder prefix.
Pass `--pre-sized-only` to limit this route to the accepted pre-sized fixture.

Cold inspection of the same SafeTensors revision resolves all 424 physical
tensors (3,836,363,776 encoded bytes) to 40 execution units and their pinned
resources. The 24 decoder state slots exclude the vision tower. The tied
embedding supplies the vocabulary head. Shared processor selection accepts
raw video with frame sampling and rejects that request when sampling is absent.

The selected CPU MLX graph binds the original BF16 SafeTensors with
host-layerwise weight residency and device state. A binding smoke check
produces finite image projections of shape `[6, 2048]` and text logits of
shape `[1, 2, 128000]`. The image projection is identical for F32 processor
patches and explicitly BF16 patches after the patchifier's input cast.

An explicitly prepared MLX model session uses the original BF16 SafeTensors
on Metal with `MLX_ENABLE_TF32=0`, host-layerwise weights and device state.
The smoke request interleaves IDs `[124894, 5]`, one 64 × 64 RGB image, a
seven-frame 6 FPS RGB clip with all frames selected, and IDs `[25, 53]`.
Each clip frame is uniformly filled with the value `27 * (frame_index + 1)`,
and the image is the first frame. With decoder chunks bounded at 17 tokens,
ordinary and controlled generation both emit greedy IDs `[24, 22]`. The
controlled run reports eleven prefill advances. The smoke measures agreement
between the two Eredu entry points.

The Rust facade also selects the same released checkpoint through
`LoadedModel::load_execution_plan` with host-layerwise weights and the
`image` feature. The publisher's template renders a user turn containing
“Describe the color in this image and video.” followed by the same decoded
image and clip, with thinking disabled and the assistant suffix enabled.
Typed preparation reports 54 text tokens, 80 media positions and 134 total
decoder positions. On Metal, ordinary and controlled generation both emit
greedy IDs `[597, 4145]` with a 17-token prefill chunk bound. Controlled
generation reports 14 advances. A resident controlled run on Metal takes
budgeted snapshots before ingress, after the first media advance, and after
the first 17-position decoder prefix. It forks from the initial and decoder
boundaries and restores the media boundary. The restored and forked runs
both emit `[597, 4145]`. The fourth snapshot is rejected at the three-snapshot
limit without changing usage. Releasing all handles returns retained usage to
zero while cumulative copy usage stays at 41,535,745 bytes.

A resident `LoadedModel::load` run selects
raw image input explicitly and uses the first frame with “Describe the
image.” It reports 17 text tokens and 64 media positions, retains the
explicitly disabled parameter-conversion policy, and emits `[597, 4145]`.

The ordinary CLI accepts the same user turn through `--image`,
`--video-frames`, and `--video-fps 6`. The resident Metal smoke uses JPEG for
the still and PNG for the clip frames. Checkpoint-default frame sampling
produces a 110-position prompt. With thinking off, temperature zero, and a
two-token limit, the CLI emits `The image` with explicit resident placement
and with the default quick planner.

The pre-sized fixture also passes with a two-rank CPU Ring decoder and a
replicated vision tower. Both ranks produce the same comparison results and
greedy IDs. The maximum absolute errors against the publisher are
`2.069e-4` for prefill and `5.162e-5` over the eight cached steps. Each rank
reports 3,458,006,016 decoder and 757,521,408 vision device parameter bytes.

```sh
cargo build -p eredu-backend-mlx --no-default-features --example lfm3_vision_reference
printf '%s\n' '[["127.0.0.1:53101"],["127.0.0.1:53102"]]' > /private/tmp/lfm3-vl-ring.json
for rank in 0 1; do
  MLX_HOSTFILE=/private/tmp/lfm3-vl-ring.json MLX_RANK="$rank" \
    target/debug/examples/lfm3_vision_reference /private/tmp/lfm3-vl-source \
      /private/tmp/lfm3-vl-conditional "/private/tmp/lfm3-vl-tp-rank-${rank}.json" \
      --released-checkpoint --pre-sized-only --tensor-cohort &
done
wait
```

The same pre-sized fixture passes with two CPU Ring pipeline stages and either
one or two tensor ranks per stage. Three consecutive combined TP/PP replays
pass on all four ranks with fresh loopback ports. Every rank matches the
reference's eight greedy token IDs, and the full-vocabulary prefill and eight
cached score comparisons pass at an absolute tolerance of `5e-4`. The
pipeline replay limits vocabulary publication to 256 positions per pass and
replicates the vision tower on each rank.

| Decoder topology | Maximum prefill error | Maximum cached error | Device decoder parameter bytes by stage per rank |
| --- | ---: | ---: | --- |
| TP=1, PP=2 | `2.205e-4` | `4.363e-5` | 3,994,432,512 / 3,969,349,632 |
| TP=2, PP=2 | `2.069e-4` | `5.162e-5` | 1,997,415,424 / 1,984,878,592 |

To replay PP=2, add `--pipeline-stages 2` to the Ring command. Use two
hostfile entries for TP=1 and four distinct loopback entries for TP=2.
Each rank writes its own result file.

The released Ring runs exercise the low-level decoder route with replicated
vision. Public VL loading selects single-device SafeTensors execution.

### Native BF16 vision-language scores

The conditional reference exporter uses the pinned publisher implementation
with the original BF16 weights. The Eredu runner selects the same precision
from the reference manifest, binds the combined VL graph, and converts score
values to F32 for comparison. The F32 replay remains a separate measurement.

On Metal with `MLX_ENABLE_TF32=0`, all eight greedy IDs match in both fixtures.
The raw fixture also matches grids, patches, and expanded prompt IDs exactly.
The score differences exceed the fixture's unchanged `5e-4` threshold.

| BF16 fixture | Maximum prefill error | Relative L2, prefill | Maximum cached-row error |
| --- | ---: | ---: | ---: |
| Pre-sized, 30 positions | 0.8125 | 0.0239272 | 0.375 |
| Raw, 97 positions | 7.3125 | 0.0951709 | 0.296875 |

The pre-sized image and video projected rows each differ by at most
`1.953e-3`, with relative L2 of `0.0081800`. Substituting publisher BF16
projections in the pre-sized prompt reduces the decoder prefill maximum
to `0.28125` and the cached maximum to `0.1875`. The remaining difference
is downstream of projection. The publisher reference uses PyTorch 2.15.0
and Transformers 5.22.2 on CPU.
In the raw image tower, the patchifier, first mixer input projection and gated
convolution input match exactly. The first spatial convolution differs in two
of 286,720 outputs, with maximum error `5.960e-8`. The first four complete
blocks match exactly. The fifth block's Q, K and V projections match exactly,
with one mismatch each after Q and K RMS normalization. The rotary outputs
have five Q mismatches and three K mismatches. With Eredu's rotated Q and K
and projected V, the publisher attention reproduces Eredu's value mix exactly.
The tower output has relative L2 error `0.0127496`. The diagnostic uses the
publisher's `from_pretrained` BF16 loader, which preserves the axial rotary
frequency buffer in F32.

On the raw image, the Metal rotary tables differ from the publisher tables
by at most `1.192e-7`.

The component-graph replay also compares projected rows for the raw image and
video, including the admitted frame-pair boundaries. The publisher export
records these rows directly from the conditional model's output. BF16 projected
rows have maximum error `0.0126953` and relative L2 `0.0246280` for the image.
The BF16 video projection matches exactly.
F32 projected rows meet tolerance, with maximum error `6.109e-6` for the image
and `5.923e-7` for the video.

```sh
HF_HOME=/private/tmp/lfm3-hf python validation/lfm3_vision_reference.py \
  /private/tmp/lfm3-vl-source /private/tmp/lfm3-vl-bf16 --conditional --dtype bfloat16
cargo build -p eredu-backend-mlx --no-default-features --features image,metal \
  --example lfm3_vision_reference
MLX_ENABLE_TF32=0 target/debug/examples/lfm3_vision_reference \
  /private/tmp/lfm3-vl-source /private/tmp/lfm3-vl-bf16 \
  /private/tmp/lfm3-vl-bf16/eredu-metal.json --released-checkpoint --component-graph --device gpu
```

### Matched raw prefill schedules

The raw fixture can also use 17-position cached prefixes in both the publisher
and Eredu. The publisher replay reuses its own projected rows and placeholder
admission, then runs each prefix through its decoder with the preceding cache.
The fixture records the chunk size, which the native runner uses for the same
prefix boundaries.

| Precision | Publisher full vs. chunked, max prefill / cached error | Eredu Metal vs. publisher, matched chunks, max prefill / cached error |
| --- | ---: | ---: |
| F32 | 0 / 0 | `2.238e-3` / `1.140e-4` |
| BF16 | `0.2555542` / `0.15625` | `7.375` / `0.3203125` |

All eight greedy IDs match in each comparison. The BF16 matched-chunk
prefill relative L2 error is `0.0945140`. The score gap persists when the
prefill schedules match.

The `--publisher-raw-projections` option runs the raw decoder with publisher
image rows, publisher video rows, and both. Each replay uses the same native
prefix boundaries as the ordinary replay. They compare the full vocabulary and
eight cached steps using the publisher's preceding token IDs. Eredu's greedy
sequence advances on its own state after a divergence. The report locates each
nonzero maximum error in tensor coordinates. Score coordinates are batch,
prompt position and vocabulary ID.
All eight greedy IDs match for both precisions.

The replay reports the shared checkpoint logit profile for every prefill and
cached score row. It requires relative L2 at most `0.02`, cosine at least
`0.999`, at least four shared top-five IDs, and agreement when the reference
argmax is unambiguous. The overall acceptance result still uses the fixture's
`5e-4` absolute threshold and greedy sequence agreement.

| Precise Metal replay | Prefill rows meeting logit profile | Cached rows meeting logit profile | Worst prefill relative L2 |
| --- | ---: | ---: | ---: |
| F32 raw | 97/97 | 8/8 | `2.031e-4` |
| BF16 pre-sized | 11/30 | 2/8 | `0.047498` |
| BF16 raw, matched chunks | 12/97 | 5/8 | `0.456681` |
| BF16 raw with publisher media, matched chunks | 82/97 | 5/8 | `0.034560` |

All F32 raw rows agree on the argmax and the full top-five set. In the BF16 raw
replay, the prefill argmax agrees on 80 of 97 rows with Eredu media and 96 of 97
rows with publisher media.

| Precision | Eredu media, max prefill / cached error | Publisher media, max prefill / cached error |
| --- | ---: | ---: |
| F32 | `2.238e-3` / `1.140e-4` | `4.520e-4` / `3.219e-5` |
| BF16, matched chunks | `7.375` / `0.3203125` | `0.3046875` / `0.25390625` |

With publisher media rows, F32 decoder scores meet the `5e-4` threshold.
Publisher image rows alone meet it with maximum prefill and cached errors
`4.520e-4` and `3.654e-5`. With publisher video rows alone, the prefill maximum
is `2.238e-3` at image position 8 and vocabulary ID 3118.
In BF16, substituting the video leaves every reported score and greedy ID
unchanged. Substituting the image alone yields the same result as substituting
both modalities. The decoder still exceeds tolerance, with its maximum at
image-token position 69 and vocabulary ID 1931.

For the BF16 decoder, the four-lane FP32 convolution matches all 3,514,368
convolution outputs in the publisher's matched-chunk trace. This includes the
first output of the final chunk at position 85, where sequential FP32
accumulation differs by one BF16 step. The convolution carry matches exactly.

With publisher media rows, the first ten BF16 decoder layers match exactly on
Metal. The first mismatch is one normalized query element at position 68 in
the eleventh layer. Cache keys, values and mask visibility match for that chunk.
The publisher's attention reproduces Eredu's value mix exactly when given the
Eredu query. The next convolution propagates the difference to position 69.

An isolated eight-lane FP32 reduction reproduces all 272 publisher query-head
variances in the affected chunk. Using that reduction for BF16 query/key RMS
on Metal gives decoder-only prefill relative L2 `0.0104584` and maximum error
`0.375`. The ordinary Metal replay measures `0.0111874` and `0.3046875`,
respectively.

```sh
HF_HOME=/private/tmp/lfm3-hf python validation/lfm3_vision_reference.py \
  /private/tmp/lfm3-vl-source /private/tmp/lfm3-vl-bf16-chunk17 \
  --conditional --dtype bfloat16 --raw-prefill-chunk 17
MLX_ENABLE_TF32=0 target/debug/examples/lfm3_vision_reference \
  /private/tmp/lfm3-vl-source /private/tmp/lfm3-vl-bf16-chunk17 \
  /private/tmp/lfm3-vl-bf16-chunk17/eredu-metal.json \
  --released-checkpoint --component-graph --device gpu --publisher-raw-projections
```

Use `--dtype float32` and a separate output directory for the F32 replay.

### VL GGUF media projector

The [pinned GGUF revision](https://huggingface.co/LiquidAI/LFM3-VL-2B-GGUF/tree/1add221a5ea6c047fcf1eaaf2d247939d7cf8849)
publishes BF16, F16 and Q8_0 media projectors. The portable catalog check
resolves all 139 descriptors in each projector. The two temporal patch
kernels use F16 in every variant. Depthwise kernels, normalization weights and
biases use F32. The other matrices use the named encoding.

The source manifest records the exact descriptor prefix, declared file size
and publisher payload hashes. The check verifies the prefix and declared
tensor ranges. Public VL generation currently selects SafeTensors.

The standalone vision graph accepts a retained GGUF projector through
`PreparedVisionExecution::new_gguf`. Its 16 vision units bind 138 canonical
weights from 139 physical tensors. The patchifier stacks the two F16 kernels
along time inside each channel before flattening. The remaining bindings use
the declared tensor geometry and requested floating dtype. Q8_0 matrices
decode through bounded portable leases.

```sh
manifest=validation/fixtures/lfm3/vision-gguf-sources.json
headers=/private/tmp/lfm3-vl-projector-headers
mkdir -p "$headers"
repo=$(jq -r .repository "$manifest")
revision=$(jq -r .revision "$manifest")
jq -r '(.projectors[], .backbones[]) | [.file, .header_bytes] | @tsv' "$manifest" |
  while IFS="$(printf '\t')" read -r member bytes; do
    curl -fsSL --range "0-$((bytes - 1))" --max-filesize "$bytes" \
      "https://huggingface.co/$repo/resolve/$revision/$member?header=$bytes" \
      -o "$headers/$member.header"
  done
cargo run -p eredu-architectures --example inspect_lfm3_vl_gguf_headers -- \
  "$manifest" "$headers" /private/tmp/lfm3-vl-source/config.json
```

Supply the downloaded projector as a fourth argument to verify its full
publisher hash, canonical binding shapes and all bounded payload reads.
All three released payloads pass these checks.

All eight published backbone headers resolve 424 VL graph bindings with
the F16 projector. QAD-Q4_0 has its own file and publisher digest.
The NVFP4 header contains 431 physical tensors, including 72 FFN matrices
in GGML type 40 and 144 F32 scale companions.

The Metal component replay accepts `--gguf-projector` with the same F32
SafeTensors reference fixture shown above. With F32 execution, BF16 and F16
GGUF projectors meet the component tolerance for all three cases. Their
maximum projected-row errors are `1.114e-6` and `1.138e-6` respectively.
Q8_0 resident, host-layerwise and disk-streamed runs produce identical
comparison metrics, with maximum projected-row error `3.909e-3` against
the widened SafeTensors reference. Controlled disk-streamed execution, including
cancellation after the first encoder block, produces the same metrics. Q8_0
same-artifact acceptance remains open.

```sh
MLX_ENABLE_TF32=0 target/debug/examples/lfm3_vision_reference \
  /private/tmp/lfm3-vl-reference/f32-component /private/tmp/lfm3-vl-reference \
  /private/tmp/lfm3-vl-reference/gguf-projector-metal.json --device gpu \
  --gguf-projector "$headers/mmproj-LFM3-VL-2B-BF16.gguf"
```

### VL GGUF paired execution

`PreparedVisionLanguage::open_gguf` admits the GGUF decoder and media projector
with the publisher's VL configuration and processor assets. It checks the
backbone geometry against the supplied configuration and validates the rotary
companion before retaining the sources. Both sources feed the shared VL
component graph. The graph decodes GGUF weights into F32 for this replay.

On precise Metal, the pinned Q4_K_M backbone and Q8_0 projector run the
pre-sized and raw image/video fixtures through prefill and eight cached steps.
The raw grids, patches and expanded prompt IDs match the publisher fixture.
The resident graph peaks at `7,672,727,552` device parameter bytes. Its raw
greedy continuation is `[334, 34, 334, 34, 334, 34, 334, 34]`.
Host-layerwise replay with `--residency host --pre-sized-only` peaks at
`1,304,526,848` device parameter bytes across two simultaneously resident units.
Its pre-sized prefill and cached score metrics and greedy IDs match the
resident report.
The comparison fixture uses widened SafeTensors, so the score comparison
measures differences across encodings. The pre-sized same-artifact comparison
below measures the numerical gap against llama.cpp.

```sh
MLX_ENABLE_TF32=0 target/debug/examples/lfm3_vision_reference \
  /private/tmp/lfm3-vl-source /private/tmp/lfm3-vl-raw \
  /private/tmp/lfm3-vl-raw/gguf-pair-metal.json --device gpu --component-graph \
  --gguf-backbone "$headers/LFM3-VL-2B-Q4_K_M.gguf" \
  --gguf-projector "$headers/mmproj-LFM3-VL-2B-Q8_0.gguf"
```

The first directory can contain just the pinned `config.json`,
`processor_config.json` and `tokenizer.json` files. The replay verifies both
GGUF payload hashes against the manifest.

The public loader recognizes the VL GGUF backbone and requires its projector
and the publisher's `config.json` and `processor_config.json`. Place those
sidecars beside the backbone. The CLI discovers a sibling `mmproj-*.gguf`.
The Rust facade also accepts an explicit projector through
`ArtifactSource::with_projector`. Both select the same VL graph and media
processor as SafeTensors. Text policy follows the shared GGUF loader, including
the checkpoint generation defaults from `generation_config.json` when present.

```sh
eredu --no-auto --model "$pair/LFM3-VL-2B-Q4_K_M.gguf" --device gpu:0 \
  --image "$image" --video-frames "$frames" --video-fps 2 \
  --layerwise-host --thinking off --temperature 0 --max-tokens 8 \
  'Describe the image and video.'
```

The Q4_K_M and Q8_0 pair completes public CLI generation on Metal with
`--layerwise-host --thinking off`. A 96×64 image and three 96×64 frames at
2 FPS expand to 117 model positions and generate eight tokens. The runtime
reports `1,304,526,848` peak device parameter bytes.
Resident facade replay of that prompt reports 82 media positions. Explicit
advancement takes 11 steps and matches uninterrupted generation's eight
greedy IDs `[597, 4145, 3926, 267, 46814, 657, 23266, 6424]` and token-limit
finish.

The same CLI prompt with `--dense-disk-stream` completes with eight generated
tokens and the same decoded text as the host-layerwise run. The disk policy
plans `6,620,997,632` parameter bytes from disk and peaks at `4,279,793,664`
device parameter bytes under its default 4 GiB device budget.

#### Same-artifact VL GGUF comparison

The pre-sized replay compares the Q4_K_M backbone and Q8_0 projector against
llama.cpp revision `7c2653acc749258dc62067da62d07024d971ccd2` on Metal.
The reference takes a 96×64 still and a three-frame 96×64 clip. It lowers the
image token minimum to preserve the pre-sized geometry, then captures the six
image and twelve video projection rows. Both runners use the same 30-position
prompt and compare all 128,000 vocabulary scores at each prefill position and
eight cached steps. The fixture uses fixed prompt IDs and deliberately bypasses
raw resizing and chat-template differences.

| Compared output | Maximum absolute error | Relative L2 |
| --- | ---: | ---: |
| Image projection | 0.00120327 | 0.00352780 |
| Video projection | 0.00120334 | 0.00352803 |
| Full prefill scores | 0.599468 | 0.0177588 |
| Worst cached score row | 0.443803 | 0.0218290 |

Both implementations generate `[334, 4160, 302, 8426, 395, 9283, 278, 788]`.
The score comparison exceeds the fixture's `5e-4` absolute tolerance. The
shared logit profile passes seven of eight cached rows and fails the full
prefill comparison. This fixture uses the same periodic RGB pattern in each
video frame, so it provides limited coverage of temporal ordering.
Single-rank Eredu CPU replay against the same llama.cpp Metal fixture gives
maximum errors of `0.599621` in prefill and `0.443882` in cached scores. The
third prefill row, before any projected media enters the decoder, has maximum
error `0.404173`.

For the three-token prefix `[124894, 5, 125009]`, the embedding rows match
exactly. The batched llama.cpp Metal convolution input projection at layer
zero differs from its own single-row result by up to `0.003703`. Disabling
only the Q4_K and Q6_K MMA dispatch reproduces that single-row projection
exactly at the original microbatch size. Against Eredu CPU, the maximum
prefix logit error is `0.404140` with default batched Metal dispatch,
`0.000092268` with single-row dispatch, and `0.000079334` with the diagnostic
MMA bypass.

| llama.cpp Metal comparison against Eredu CPU | Maximum prefill error | Maximum cached error |
| --- | ---: | ---: |
| Single-row decoder, independent projections | 0.212204 | 0.125914 |
| Single-row decoder, identical projected inputs | 0.000186443 | 0.0000491142 |
| Single-row decoder, Q8_0 batched projector kernels bypassed | 0.0232742 | 0.0117574 |

With identical projected inputs, all 38 vocabulary rows meet the
`5e-4` tolerance and have matching argmax IDs. Bypassing the Q8_0 batched
projector kernels reduces the largest independent projection error to
`0.000164144`. The unmodified single-row reference with independent
projections matches all eight greedy continuation IDs. One intermediate
prefill argmax differs by a reference margin of `0.005462`, so full
same-artifact acceptance remains open.

```sh
# llama_dir contains the pinned source and a Metal mtmd build.
pair=/private/tmp/eredu-vl-gguf-projector
c++ -std=c++17 -O2 -I "$llama_dir/include" -I "$llama_dir/ggml/include" \
  -I "$llama_dir/tools/mtmd" validation/lfm3_vision_gguf_reference.cpp \
  -L "$llama_dir/build-metal/bin" -Wl,-rpath,"$llama_dir/build-metal/bin" \
  -lmtmd -lllama -o "$pair/vision-reference"
python validation/lfm3_vision_gguf_fixture.py "$pair/vision-reference" "$pair" \
  /private/tmp/lfm3-vl-raw "$pair/llama-fixture"
MLX_ENABLE_TF32=0 target/debug/examples/lfm3_vision_reference \
  "$pair/assets" "$pair/llama-fixture" "$pair/same-pair-metal.json" \
  --device gpu --component-graph --pre-sized-only \
  --gguf-backbone "$pair/LFM3-VL-2B-Q4_K_M.gguf" \
  --gguf-projector "$pair/mmproj-LFM3-VL-2B-Q8_0.gguf"
```

Pass `--single-row` to the fixture builder and choose a separate output
directory to compare the unmodified single-row reference.

#### Released F16 and BF16 GGUF pairs

Eredu widens the released dense GGUF weights to F32 for the CPU replay.
The pinned llama.cpp decoder uses Metal with a single-row microbatch.
Each reference independently computes the still and video projections
from the matching published projector.

| Pair | llama.cpp projector | Maximum projection error | Maximum prefill error | Maximum cached error | Prefill logit profile |
| --- | --- | ---: | ---: | ---: | --- |
| F16 / F16 | Metal | 0.000892714 | 0.239307 | 0.0313820 | Pass |
| F16 / F16 | CPU | 0.000306603 | 0.102447 | 0.0115485 | Pass |
| BF16 / BF16 | Metal | 0.000939123 | 0.337467 | 0.0398035 | Outside tolerance |
| BF16 / BF16 | CPU | 0.00205039 | 0.274311 | 0.0394001 | Outside tolerance |

All four comparisons match the eight greedy IDs
`[334, 34, 334, 34, 334, 34, 334, 34]` and pass the shared logit profile
on each cached row. All four exceed the fixture's `5e-4` absolute score
tolerance.

```sh
for encoding in F16 BF16; do
  python validation/lfm3_vision_gguf_fixture.py "$pair/vision-reference" "$pair" \
    /private/tmp/lfm3-vl-raw "$pair/llama-$encoding" \
    --backbone "$encoding" --projector "$encoding" --single-row
  target/debug/examples/lfm3_vision_reference \
    "$pair/assets" "$pair/llama-$encoding" "$pair/$encoding-cpu.json" \
    --device cpu --component-graph --pre-sized-only \
    --gguf-backbone "$pair/LFM3-VL-2B-$encoding.gguf" \
    --gguf-projector "$pair/mmproj-LFM3-VL-2B-$encoding.gguf"
done
```

Pass `--projector-cpu` to the fixture builder for the CPU projector
comparison and use a separate output directory.

#### Released quantized backbones with the F16 projector

The comparison pairs each backbone with the published F16 projector.
Eredu runs widened F32 weights on CPU. The pinned llama.cpp reference uses
a single-row Metal decoder and a CPU projector. Both
implementations project the pre-sized media independently. The fixture
builder verifies each backbone and projector against its publisher digest.

| Backbone | Maximum prefill error | Maximum cached error | Worst prefill row relative L2 | Prefill logit profile |
| --- | ---: | ---: | ---: | --- |
| Q4_K_M | 0.0733948 | 0.00945330 | 0.00373467 | Pass |
| Q5_K_M | 0.0547077 | 0.00873470 | 0.00431260 | Outside tolerance |
| Q6_K | 0.111796 | 0.0123227 | 0.00546469 | Pass |
| Q8_0 | 0.101429 | 0.0114520 | 0.00561313 | Pass |
| QAD-Q4_0 | 0.113141 | 0.00687826 | 0.00745316 | Pass |

All five pass the shared logit profile on all eight cached rows and
exceed the fixture's `5e-4` absolute score tolerance. The maximum
projection error is `0.000306603`. Q5_K_M differs on one intermediate
prefill argmax at zero-based position 16. The reference margin is
`0.00183439` and its top-five overlap is 5/5.

Both runtimes generate `[334, 4160, 302, 8426, 395, 9283, 278, 788]`
for Q4_K_M and `[334, 34, 334, 34, 334, 34, 334, 34]` for each of the
other four backbones.

```sh
for encoding in Q4_K_M Q5_K_M Q6_K Q8_0 QAD-Q4_0; do
  python validation/lfm3_vision_gguf_fixture.py "$pair/vision-reference" "$pair" \
    /private/tmp/lfm3-vl-raw "$pair/llama-$encoding" \
    --backbone "$encoding" --projector F16 --single-row --projector-cpu
  target/debug/examples/lfm3_vision_reference \
    "$pair/assets" "$pair/llama-$encoding" "$pair/$encoding-same-pair-cpu.json" \
    --device cpu --component-graph --pre-sized-only \
    --gguf-backbone "$pair/LFM3-VL-2B-$encoding.gguf" \
    --gguf-projector "$pair/mmproj-LFM3-VL-2B-F16.gguf"
done
```

#### Released NVFP4 backbone

GGML type 40 stores four unsigned E4M3 block scales and 64 E2M1 weights
per 36-byte block. The VL declaration applies the F32 matrix multiplier
to the widened FFN weights and retains the input scale as checkpoint
metadata. The pinned llama.cpp graph requests BF16 accumulation for the
NVFP4 projections. Eredu realizes the decoder with F32 weights and
arithmetic.

The released NVFP4 backbone and F16 projector run the pre-sized fixture on
Eredu CPU. The independent reference uses the same artifacts with
single-row Metal decoding and a CPU projector. Both generate
`[440, 1008, 4145, 3926, 267, 4309, 302, 3553]`. The shared logit profile
passes all eight cached rows and fails prefill. Maximum absolute score
errors are `1.318984` in prefill and `0.373202` over the cached rows. The
largest prefill error is at the third position, before projected media.
The public CLI also completes the 117-position mixed image/video prompt on
CPU with `--layerwise-host --thinking off --temperature 0 --max-tokens 8`.
Its peak parameter residency is `1,304,526,848` bytes in the execution
window and `6,251,446,272` bytes in the host cache.

```sh
python validation/lfm3_vision_gguf_fixture.py "$pair/vision-reference" "$pair" \
  /private/tmp/lfm3-vl-raw "$pair/llama-nvfp4" \
  --backbone NVFP4 --projector F16 --single-row --projector-cpu
target/debug/examples/lfm3_vision_reference \
  "$pair/assets" "$pair/llama-nvfp4" "$pair/nvfp4-same-pair-cpu.json" \
  --device cpu --component-graph --pre-sized-only \
  --gguf-backbone "$pair/LFM3-VL-2B-NVFP4.gguf" \
  --gguf-projector "$pair/mmproj-LFM3-VL-2B-F16.gguf"
```

#### Released VL GGUF CPU Ring comparison

The native Ring replay uses the portable VL rank selection to retain a
vision replica, the local decoder partition and both residency policies. A
single-rank CPU replay records the pre-sized projection and score rows for
the same encoded pair. Each Ring rank compares its published output against
those rows at the fixture's `5e-4` absolute tolerance. This measures the
effect of placement and collectives on Eredu's outputs.

| Decoder topology | Maximum prefill error | Maximum cached error | Decoder parameter bytes per rank by stage |
| --- | ---: | ---: | ---: |
| TP=2, PP=1 | 0.000179768 | 0.0000349283 | 3,458,006,016 |
| TP=1, PP=2 | 0 | 0 | 3,994,432,512 / 3,969,349,632 |
| TP=2, PP=2 | 0.000179768 | 0.0000349283 | 1,997,415,424 / 1,984,878,592 |

Every rank generates the single-rank baseline's eight greedy IDs. The
replicated vision tower uses `757,521,408` parameter bytes per rank.

Retained rank selection reports `4,215,527,424` device parameter bytes per
rank for resident TP=2. With `--residency host` and combined TP=2/PP=2, it
reports `752,971,776` bytes on stage zero and `752,979,968` on stage one.
For each rank the selected total equals the sum of its observed vision
and decoder parameter peaks. Both replays preserve the score errors and
greedy IDs in the table. Resident TP=2 uses the same all-rank commit
and output publication as the PP placements, with no inter-stage transfer.

For the Q4_K_M and Q8_0 pair, cold TP=2/PP=2 selection exposes a manifest
for each rank before source preparation. Binding the ordinary loader's
selected recipes preserves each manifest exactly. With resident weights
those recipes account for `754,375,680` bytes of replicated vision weights,
with totals of `2,751,791,104` bytes per rank on stage zero and
`2,739,254,272` on stage one. The pre-sized replay above uses F32 recipes
for its projector comparison.

The same Q4_K_M backbone and Q8_0 projector complete controlled raw image and
video prefill on all four TP=2/PP=2 CPU Ring ranks with host residency. The
17-position prefixes cover all 97 expanded tokens. Both media grids, packed
patches and the expanded prompt match the raw publisher fixture. All four ranks
report the same score metrics and generate `[334, 34, 334, 34, 334, 34, 334, 34]`.
Against the publisher's unquantized raw fixture, the maximum absolute score
errors are `11.1346264` for prefill and `2.5685104` across eight cached steps.
The single-rank host replay generates the same IDs and reports errors of
`11.1345644` and `2.5685090`. The raw numerical comparison fails its tolerance.
Separate TP=2/PP=1 and TP=1/PP=2 replays set the prefix size to 17 on one
rank and 19 on the other. Both ranks in each replay reject the request at
prefill admission before returning a cursor.

The shared native generation session also binds the Q4_K_M and Q8_0 pair at
PP=2 using the ordinary loader's selected weight recipes. A four-token text
prompt with a two-position invocation limit gives identical cached-decode
scores after ordinary prefill and after stepped prefill with a reset between
the runs. Only the selected output rank exposes scores. Both ranks receive
token `334` when the sampling filter permits that token alone.

The ordinary inspected-model loader binds the same released pair with
resident weights on PP=2 and combined TP=2/PP=2 CPU Ring. Every rank
completes a four-token text prefill with a two-position invocation limit
and one cached decode. Only the selected rank exposes logits, and all
exposed decode scores are finite.

The same loader processes a raw RGB image and five-frame video through the
checkpoint processor with host-layerwise TP=2/PP=2 and direct disk-streamed
PP=2 weights. All ranks complete prefill and one cached decode, and the
output rank exposes finite scores. With a 2 GiB device budget per rank,
the sum of the two component peak ledgers is `752,971,776` and
`752,979,968` bytes by stage for host-layerwise execution, and
`1,709,436,928` and `1,684,354,048` bytes for disk streaming. The static
device-residency observation equals the sum of the current vision and
decoder ledgers on every rank.

For the raw replay, build with `--features image`, use the raw publisher
fixture in place of the pre-sized baseline, omit `--pre-sized-only`, and set
`--residency host`. To isolate the raw case, set `cases` to an empty array
and remove `conditional` in the fixture metadata. Use rank-local copies of
`reference.json` with
`raw_processor.prefill_chunk` set to the tested size for the disagreement case.
Both copies use the same reference tensors.

```sh
cargo build -p eredu-backend-mlx --no-default-features --example lfm3_vision_reference
pair=/private/tmp/eredu-vl-gguf-projector
runner=target/debug/examples/lfm3_vision_reference
backbone="$pair/LFM3-VL-2B-Q4_K_M.gguf"
projector="$pair/mmproj-LFM3-VL-2B-Q8_0.gguf"
"$runner" "$pair/assets" "$pair/llama-fixture" "$pair/ring-single.json" \
  --pre-sized-only --record-conditional "$pair/ring-baseline" \
  --gguf-backbone "$backbone" --gguf-projector "$projector"
printf '%s\n' '[["127.0.0.1:53101"],["127.0.0.1:53102"],["127.0.0.1:53103"],["127.0.0.1:53104"]]' > "$pair/ring-hosts.json"
pids=
for rank in 0 1 2 3; do
  MLX_HOSTFILE="$pair/ring-hosts.json" MLX_RANK="$rank" \
    "$runner" "$pair/assets" "$pair/ring-baseline" "$pair/ring-rank-$rank.json" \
    --pre-sized-only --tensor-cohort --pipeline-stages 2 \
    --gguf-backbone "$backbone" --gguf-projector "$projector" &
  pids="$pids $!"
done
for pid in $pids; do wait "$pid" || exit 1; done
```

The speech input path admits the released Audio SafeTensors encoder and projector
geometry. Its host processor accepts 16-kHz mono PCM and computes 128-bin Slaney
power-mel features with an uncentered 400-sample window and 160-sample hop.
Right padding completes the final window. Three convolutional stages reduce
time by a factor of eight before the causal LFM2 backbone, whose attention
uses a 32-token window. The encoder can advance over complete stride groups
and emit the final partial group on flush. Finite batches contain equal-length
unpadded clips. The standalone encoder and projector do not establish full
audio language-model acceptance.

The standalone speech output path admits the released depth and feedback
SafeTensors scopes. The three depth blocks score all twelve codebook positions
bidirectionally. The checkpoint schedule reveals four codebooks per pass in a
request-supplied permutation. Codebook tables are shared between input lookup
and output scoring. A frame containing any EOS code terminates the speech
segment. The completed frame exposes either ordinary codec codes or EOS. The
language feedback table embeds ordinary frames and canonical all-EOS frames.
The resident driver uses these components for controlled speech output.
The [native BF16 conditional comparison](#native-bf16-conditional-comparison)
measures them in the complete Audio model.

Standalone LiquidCodec neural encoding and finite/cached decoding, CPU/Metal
F32 reference results and integration limits are documented in
[LiquidCodec validation](liquid-codec-validation.md). Resident codec acceptance
does not establish bounded/distributed codec or speech-model acceptance.

## Mixed speech input and feedback

`PreparedAudio` admits the full released Audio SafeTensors checkpoint and
bundled LiquidCodec. It retains restricted language, speech encoder, depth
and codec sources, as well as the tokenizer, template, generation defaults
and processor policy. `AudioPrompt` checks the number of projected rows
against speech placeholders and uses the shared LFM3 decoder traversal to
replace those positions. Its readout returns normalized language states for
depth scoring or the tied text head. A caller can also use the same input
hook with an already configured layerwise language runtime.

The independent publisher case uses the Audio revision
`2d6c1f99c129b3ef07a8d168bc0ff08e2bdb93f1`. It compares 128,000-entry
prefill score rows from a prompt with both input speech and replayed speech,
eight cached language steps including EOS feedback, all three depth score
passes, one synthesized frame and the subsequent feedback step. The fixture
uses a fixed reveal order and greedily selects from the codec vocabulary to
guarantee a speech frame. Source BF16 values widen exactly to F32 when bound.

Both CPU and precise Metal pass the preset `5e-4` maximum absolute tolerance
and match all 17 greedy text decisions and all twelve sampled speech codes.
The synthetic speech frame has 1,920 PCM samples and a reference peak of
`1.272e-6`. Artifact hashes, inputs, package versions and per-phase maxima
are retained in
[`audio-acceptance-results.json`](../validation/fixtures/lfm3/audio-acceptance-results.json).

| Measurement | CPU maximum absolute error | Metal maximum absolute error |
| --- | ---: | ---: |
| Prefill scores | `1.149e-4` | `2.399e-4` |
| Eight cached score rows | `2.694e-5` | `5.007e-5` |
| Depth scores | `2.289e-5` | `2.575e-5` |
| PCM samples | `2.786e-12` | `4.406e-12` |
| Scores after generated speech feedback | `4.054e-6` | `6.170e-6` |

Keep checkpoint payloads and generated traces outside the checkout. The Python
runner verifies source and weight hashes before loading the publisher code.

```sh
python -c 'from huggingface_hub import snapshot_download; snapshot_download("LiquidAI/LFM3-Audio-2B", revision="2d6c1f99c129b3ef07a8d168bc0ff08e2bdb93f1", local_dir="/private/tmp/lfm3-audio", allow_patterns=["*.py", "*.json", "*.jinja", "model.safetensors", "audio_tokenizer/*"])'
HF_HOME=/private/tmp/lfm3-hf HF_MODULES_CACHE=/private/tmp/lfm3-hf/modules python validation/lfm3_audio_reference.py /private/tmp/lfm3-audio /private/tmp/lfm3-audio-reference
cargo run -p eredu-backend-mlx --features audio --example lfm3_audio_reference -- /private/tmp/lfm3-audio /private/tmp/lfm3-audio-reference /private/tmp/lfm3-audio-reference/cpu.json --device cpu
MLX_ENABLE_TF32=0 cargo run -p eredu-backend-mlx --features audio,metal --example lfm3_audio_reference -- /private/tmp/lfm3-audio /private/tmp/lfm3-audio-reference /private/tmp/lfm3-audio-reference/metal.json --device gpu
```

The conditional case measures resident execution with fixed-order greedy
sampling. LiveASR, bounded execution and distributed replay have separate cases
below.

### Native BF16 conditional comparison

The same pinned SafeTensors case can bind the encoder, projector, backbone,
feedback and depth decoder as BF16, keeping the mel frontend and LiquidCodec
in F32. This is a reference-runner precision mode. The selected Audio runtime
uses F32 widening. Depth score comparisons feed the publisher's preceding
reveals to both implementations. The PCM and subsequent feedback comparisons
also use the publisher's codes.

The score profile requires relative L2 at most 0.02, cosine at least 0.999,
at least four of the top five IDs in common and matching unambiguous argmax.
The rotary operation generates its tables in F32, rounds them to the
activation dtype, and applies the rotation in F32. The downsampler and
projector keep their bias and normalization arithmetic in F32 before each
corresponding BF16 output cast. Speech feedback sums codebook embeddings
in F32 before restoring the activation dtype.

CPU and precise Metal pass all 17 vocabulary rows and 36 depth rows.
Both match all 17 text argmax IDs, all 12 greedy code IDs and all language
hidden states exactly. The runner uses F32 codec arithmetic.

| Measurement | CPU maximum relative L2 | Precise Metal maximum relative L2 |
| --- | ---: | ---: |
| Mixed prefill scores | 0 | 0.00003742 |
| Eight cached vocabulary rows | 0 | 0.00002415 |
| Depth scores | 0 | 0.00000571 |
| Speech feedback scores | 0 | 0.00002259 |

Maximum absolute PCM error is `2.02e-12` on CPU and `2.90e-12` on
precise Metal.

```sh
HF_HOME=/private/tmp/lfm3-hf python validation/lfm3_audio_reference.py \
  /private/tmp/lfm3-audio /private/tmp/lfm3-audio-bf16 --bfloat16
cargo run -p eredu-backend-mlx --features audio --example lfm3_audio_reference -- \
  /private/tmp/lfm3-audio /private/tmp/lfm3-audio-bf16 /private/tmp/lfm3-audio-bf16/cpu.json
MLX_ENABLE_TF32=0 cargo run -p eredu-backend-mlx --features audio,metal --example lfm3_audio_reference -- \
  /private/tmp/lfm3-audio /private/tmp/lfm3-audio-bf16 /private/tmp/lfm3-audio-bf16/metal.json --device gpu
```


## Resident Audio sessions

`eredu::api::realtime::prepare_realtime_model` discovers `lfm3_audio` from
the model assets and admits the specified SafeTensors file or directory, or
the explicit backbone/projector pair in `RealtimeArtifactRequest::audio_gguf`.
GGUF preparation uses a local publisher asset directory with tokenizer,
generation and processor metadata and `audio_tokenizer/config.json`.
The local directory can omit both SafeTensors weight files. The embedded
GGUF vocabulary, BPE merges, task templates and sampling settings determine
the admitted pair's text policy alongside the publisher's EOS set.
The Rust realtime facade supplies the selected graph to the turn-based
`eredu audio` command. The CLI exposes transcription, translation to text,
text-to-speech and spoken responses. The separate `eredu audio realtime` command
drives full-duplex VoiceChat conversations.
CPU smoke runs on the released SafeTensors checkpoint produce an English
transcript, a French translation, and finalized mono 24-kHz WAV files for
speech synthesis and spoken response. The released Q4_K_M pair also produces
a finalized 24-kHz WAV through the CLI synthesis task. CLI device I/O is
build-checked.
`prepare_mlx_realtime` with `RealtimeLoadOptions::Lfm3Audio` selects the
resident model and reports the logical F32 parameter bytes by component.
Cold preparation produces 75 parameter units for both the pinned SafeTensors
checkpoint and the Q4_K_M backbone/projector pair. The largest unit is the
tied language embedding at 1,048,576,000 F32 bytes. The sum of the unit
sizes matches the resident parameter report for each layout. These are
logical parameter sizes and exclude session state and temporary allocations.
Load consumes the selection with explicit weight and execution contexts.
The bound model's `start` accepts a batch-one `AudioRequest` and sampling seed. The request supplies
token IDs with expanded speech placeholders, finite mel features in prompt
order, optional replayed speech codes, a mode, and a slot budget. The retained
processor transforms mono 16-kHz PCM to the input features.

The session emits text token IDs, completed codebook frames with 24-kHz
mono PCM, and speech EOS. Both `advance` and `events` use the same controlled
driver. Idle snapshots and restore use explicit copy limits and preserve
unfinished refinement and codec carry. Live forks resume the settled state
on the same selected graph, with independent restore origins and a shared
copy budget. The audio feature also exposes controlled mono 16-kHz PCM
ingress for one streamed utterance. The caller supplies token spans around
the utterance and finalizes it before generating a response. The low-level
`AudioParameterResidency` selects independent resident, host-layerwise or
disk-streamed policies for language blocks, speech input, depth and feedback,
and LiquidCodec synthesis. The Rust facade retains it through
`RealtimeLoadOptions::Lfm3AudioWithResidency` or
`PreparedAudioTask::with_parameter_residency`. The default facade and CLI
options select resident weights. The native tensor-cohort adapter described
below shards the language backbone. The native pipeline adapter also supports
pipeline-only and combined tensor/pipeline placement with replicated speech
components. The facade and CLI select single-rank Audio sessions.

The selected language, encoder, depth/feedback and codec synthesis bindings
total 6,915,206,144, 461,112,576, 533,873,408 and 145,484,296 F32 bytes
respectively. A CPU loader check supplies the pinned Audio weight file and
asset directory separately. After the configuration, processor, generation
defaults and tokenizer paths are removed from the temporary asset directory,
the selected model loads and commits token 13 for prompt IDs
`[124894,5,13,21,34]` with seed 41 and a one-slot text budget.

The released-checkpoint session check uses seed 41, a three-slot speech
budget, and prompt IDs `[124894,5,125031,125031,13,21,34]` before marker
insertion. The input is 2,151 samples of 413-Hz and 863-Hz sine waves
with amplitudes 0.12 and 0.05 at 16 kHz. Controlled advancement and iterator
draining produce identical events within each device on CPU and precise Metal.
Each run
commits three speech frames after six intermediate depth advances and emits
5,760 samples, including the final codec tail. Cancelling a separate session
after its first depth pass leaves zero committed slots.

The snapshot replay check uses the same fixture. It saves the first speech
frame after one depth pass, then saves the second after one pass with codec
carry from the first committed frame. On CPU and precise Metal, restoring
either boundary reproduces all three codebook frames and all 5,760 PCM
samples exactly within each device, including the final flush. A third
snapshot is rejected at the two-snapshot limit without charging a copy.
Dropping both handles releases all snapshot retention while preserving
cumulative copy usage. Explicit cancellation rejects restore.

The fork check uses the same fixture and forks after the first depth pass.
Interleaved parent and child advances reproduce all three codebook frames
and all 5,760 PCM samples exactly within each device on CPU and precise
Metal. After the first speech commit, child retention grows from 8,914,005
to 9,505,358 logical bytes. The branch count limit and cross-origin restore
reject before charging. A CPU run capped at the initial retention rejects
the child's first speech commit on both attempts, leaves its slot count at
zero, and lets the source publish that frame. Dropping the child releases
its retention and keeps cumulative copying charged.

The ingress check sends the same 2,151 samples as three chunks ending at
samples 399, 1,520, and 2,151. One projected speech position commits before
finish and a second enters the prompt on flush. The streamed and finite
prompts produce identical codebooks and all 5,760 PCM samples within each
device on CPU and precise Metal. Invalid sample rates and non-finite samples
leave progress unchanged. A separate CPU check reserves one audio position,
accepts its exactly aligned 1,520 samples, rejects an additional sample,
and finalizes with zero new mel frames before emitting a text token.

The native runner's `--ingress-replay` mode saves at 1,055 samples, before
the first audio position, and at 1,520 samples, after that position enters
the language cache. It restores each boundary and resubmits only the later
samples. On CPU and precise Metal, the restored response matches all three
speech frames and all 5,760 PCM samples from finite prompting within each
device. The response keeps the ingress copy budget, and dropping saved
ingress states releases their retention while preserving cumulative copy work.
Run it with the same checkpoint and reference directory as the conditional
reference case.

The same runner forks at 1,055 samples and advances the source and child
independently through finalization and speech generation. Both match all
three finite speech frames and all 5,760 PCM samples on CPU and precise
Metal. The check covers branch
limits and restore origins, retained growth during input, and the transfer
of the same branch slot and cumulative copy charges into the child's
response. Dropping the child releases its retention.

## LiveASR

The resident LiveASR route takes mono 16-kHz PCM or unpadded mel blocks and
publishes greedy English transcription tokens before the stream finishes.
A context restart keeps the encoder continuation and begins a fresh language prompt. The
publisher defines the cadence and closing-token policy. [Liquid AI LiveASR](https://huggingface.co/LiquidAI/LFM3-Audio-2B#usage-examples)

The LiveASR reference runner takes mono 16-kHz signed 16-bit WAV with more
than two full blocks of audio. It keeps the PCM and token traces outside the
checkout. The native runner applies Eredu's portable processor to the same
PCM and compares projected audio-position counts, closing status, and
generated IDs for every block.

CPU and precise Metal match all seven blocks and all 20 generated IDs from
the pinned Python reference. The fixture begins with seven mel frames that
complete no audio position, restarts the language context in the third
block, and flushes a 16-frame final block. Its mono WAV is generated by the
macOS Samantha voice reading “The quick brown fox jumps over the lazy dog.
The bright red boat sailed across the lake.” The WAV SHA-256 is
`9ad5d0e647cbbc25d9008c26a006805c4f79dcd136e7e5c28e5398978f9eacee`.
The LiveASR comparison measures block and token parity. The conditional
reference case above provides the full-vocabulary score comparisons.

The native runner's `--live-asr-pcm` option feeds 100 ms PCM chunks and lets
the session assemble its own mel blocks. It uses the same Python trace and
compares blocks that reach the transcript driver. CPU and precise Metal match
all six transcript blocks and all 20 generated IDs. The runner also checks
sample backpressure, finish, and rejection of further input.

Both native input paths compare replay after restore. The mel check saves
after the seven-frame block that completes no audio position. The PCM check
saves after the first published block with queued mel frames and a partial
PCM window, restores before replaying the next block, and repeats the final
flush from a second snapshot. It also checks the snapshot count limit,
release of retained storage, and cumulative copying across restore.

The mel comparison also forks after the initial seven frames and compares
the first transcript block in both continuations. It checks that the
branch limit and foreign snapshots are rejected before charging, and that
growth charges retention while preserving cumulative copy usage. The PCM
comparison forks before the final flush and closes the parent and child
independently with the same transcript.

```sh
python validation/lfm3_audio_reference.py /private/tmp/lfm3-audio /private/tmp/lfm3-live --live-asr-wav /private/tmp/speech.wav
cargo run -p eredu-backend-mlx --features audio --example lfm3_audio_reference -- /private/tmp/lfm3-audio /private/tmp/lfm3-live /private/tmp/lfm3-live/cpu.json --live-asr --device cpu
MLX_ENABLE_TF32=0 cargo run -p eredu-backend-mlx --features audio,metal --example lfm3_audio_reference -- /private/tmp/lfm3-audio /private/tmp/lfm3-live /private/tmp/lfm3-live/metal.json --live-asr --device gpu
```

Add `--live-asr-pcm` to the native command to compare the PCM path.

## Audio GGUF admission

`eredu::api::realtime::AdmittedAudioGguf` separately admits the published
backbone and projector headers. The backbone retains tokenizer and template
metadata. The projector contains the input encoder, depth decoder, feedback,
and LiquidCodec decoder. A complete LiquidCodec encoder requires an additional
source. The resident Audio selector uses the same F32 binding mechanism for
SafeTensors and GGUF pairs.

The [pinned source manifest](../validation/fixtures/lfm3/audio-gguf-sources.json)
records publisher payload hashes, downloaded header hashes, and per-type tensor
counts for all ten files. Retrieve and validate the descriptor prefixes without
downloading the weights.

The portable runner admits all five same-encoding pairs and the published
Q4_K_M backbone with the Q8_0 projector. Each backbone has 286 parameter
tensors and one rotary companion. Each projector has 319 tensors. A range
read of the Q4_K_M backbone's rotary companion confirms 32 active pairs
and 32 pairs suppressed with the published large divisor.

```sh
python3 - <<'PY'
import json
from pathlib import Path
from urllib.request import Request, urlopen

manifest = json.loads(Path("validation/fixtures/lfm3/audio-gguf-sources.json").read_text())
output = Path("/private/tmp/eredu-audio-gguf")
output.mkdir(exist_ok=True)
base = f"https://huggingface.co/{manifest['repository']}/resolve/{manifest['revision']}"
for variant in manifest["variants"].values():
    for source in variant.values():
        size = source["header_bytes"]
        request = Request(f"{base}/{source['file']}", headers={"Range": f"bytes=0-{size - 1}"})
        with urlopen(request) as response:
            (output / f"{source['file']}.header").write_bytes(response.read(size))
PY
cargo run -p eredu-architectures --example inspect_lfm3_audio_gguf_headers -- validation/fixtures/lfm3/audio-gguf-sources.json /private/tmp/eredu-audio-gguf
```

Supplying the pinned Audio `config.json` as a third argument also checks
canonical binding. All five projector encodings resolve 176 encoder and
projector parameters and 65 depth and feedback parameters. Supplying the
released Q4_K_M projector as a fourth argument verifies its publisher hash
and decodes the twelfth codebook as a bounded read. The lease reads
1,161,792 encoded bytes for the `[2017, 1024]` codebook, and the input scope
rejects access to its packed source. A fifth argument supplies the pinned
LiquidCodec config and resolves 100 canonical decoder parameters for each of
the five encodings.
The released Q4_K_M payload also supplies the complete 8,064-value F32 scalar
code table for exact comparison against the codec's mixed-radix lookup. The
retained decoder scope includes 1,280 generated Hann samples using the
[pinned llama.cpp window construction](https://github.com/ggml-org/llama.cpp/blob/7c2653acc749258dc62067da62d07024d971ccd2/tools/mtmd/mtmd-audio.cpp#L27).
This runner covers source admission and binding.

```sh
cargo run -p eredu-architectures --example inspect_lfm3_audio_gguf_headers -- \
  validation/fixtures/lfm3/audio-gguf-sources.json /private/tmp/eredu-audio-gguf \
  /private/tmp/lfm3-audio/config.json \
  /private/tmp/eredu-audio-gguf/mmproj-LFM3-Audio-2B-Q4_K_M.gguf \
  /private/tmp/lfm3-audio/audio_tokenizer/config.json
```

### Resident Q4_K_M Audio replay

The native Audio runner accepts the released Q4_K_M pair with both publisher
hashes checked. The same speech-only request runs from a finite mel prompt
and from partitioned PCM with snapshots, restore and fork. This compares
exact emitted codebooks and PCM between the two execution paths. It exercises
the loaded speech input, backbone, depth, feedback and codec graph, with
controlled budgets carried from prompt ingestion into the response. The F32
CPU run matched all three speech frames and 5,760 PCM samples exactly, including
the restored and forked prompt continuations. This end-to-end session replay
uses Q4_K_M. The runner's `--language-residency host` and
`--language-residency disk` options compare streamed prompt continuations
using bounded language blocks against the finite resident baseline. Both
profiles set a 2 GiB language device budget. The host profile sets a 12 GiB
language host budget and the disk profile sets a 2 GiB language host budget.
Both bounded profiles match three codebook frames and 5,760 PCM samples
exactly against the resident baseline, including restore and fork. Speech
modules are resident in the language-only comparison. The runner's
`--encoder-residency host` and `--encoder-residency disk` options independently
select speech input placement with a 128 MiB device budget. The host profile
sets a 1 GiB speech input host budget, and the disk profile sets a 128 MiB
speech input host budget. Both speech-input profiles match the resident
baseline's three codebook frames and 5,760 PCM samples exactly through restore
and fork on CPU, with resident language blocks.
The `--depth-residency` and `--codec-residency` options extend this replay to
speech output. They use 256 MiB for depth and feedback and 64 MiB for codec
synthesis as device parameter budgets. Both profiles combine bounded speech
input and output with resident language. Each matches all three codebook
frames and 5,760 PCM samples exactly on CPU, including restore and fork.
Run the command below with `--encoder-residency host --depth-residency host
--codec-residency host` for the host profile or use `disk` for all three
speech components for the disk profile.


```sh
audio_dir=/private/tmp/eredu-audio-gguf
cargo run -p eredu-backend-mlx --features audio --example lfm3_audio_reference -- \
  /private/tmp/lfm3-audio validation/fixtures/lfm3/audio-gguf-sources.json \
  "$audio_dir/ingress-cpu.json" --device cpu --ingress-replay --language-residency resident \
  --gguf-backbone "$audio_dir/LFM3-Audio-2B-Q4_K_M.gguf" \
  --gguf-projector "$audio_dir/mmproj-LFM3-Audio-2B-Q4_K_M.gguf"
```

### Q4_K_M mixed Audio scores

The independent runner uses the same pinned Audio backbone and projector as
resident replay. llama.cpp runs on Metal with F32 KV and flash attention
disabled. Both runtimes encode a 2,151-sample 16-kHz prompt into two audio rows
between token spans `[124894, 5]` and `[13, 21, 34]`. The first cached position
uses speech feedback from codebook IDs `149 * book_index`. The seven remaining
cached inputs are teacher-forced IDs `[5, 8, 13, 21, 34, 55, 3]`.
Eredu executes F32 on CPU from the same encoded weights. The comparison
passes the shared quantized logit tolerance across all 128,000 scores in each
of the nine rows. The maximum relative L2 is `0.00297663`, minimum cosine is
`0.99999704`, and maximum absolute error is `0.05620670`. Every row has 5/5
top-five overlap, with matching argmax IDs `[13, 39, 13, 13, 13, 21, 34, 7, 13]`.
The speech-feedback row has relative L2 `0.00297663`. These results cover the
speech encoder, backbone and fixed-code feedback under teacher forcing.

Use the pinned Metal llama.cpp build described below for the PCM reference.

```sh
llama_dir=/private/tmp/eredu-llama-audio-reference
audio_dir=/private/tmp/eredu-audio-gguf
c++ -std=c++17 -O2 -I "$llama_dir/include" -I "$llama_dir/ggml/include" \
  -I "$llama_dir/tools/mtmd" validation/lfm3_audio_gguf_reference.cpp \
  -L "$llama_dir/build-metal/bin" -Wl,-rpath,"$llama_dir/build-metal/bin" \
  -lmtmd -lllama -o "$llama_dir/audio-reference"
"$llama_dir/audio-reference" "$audio_dir/LFM3-Audio-2B-Q4_K_M.gguf" \
  "$audio_dir/mmproj-LFM3-Audio-2B-Q4_K_M.gguf" "$audio_dir/audio-scores.f32"
cargo run -p eredu-backend-mlx --features audio --example lfm3_audio_reference -- \
  /private/tmp/lfm3-audio validation/fixtures/lfm3/audio-gguf-sources.json \
  "$audio_dir/audio-score-parity.json" --device cpu \
  --gguf-backbone "$audio_dir/LFM3-Audio-2B-Q4_K_M.gguf" \
  --gguf-projector "$audio_dir/mmproj-LFM3-Audio-2B-Q4_K_M.gguf" \
  --gguf-scores "$audio_dir/audio-scores.f32"
```

### Q4_K_M depth decoder

Both references score all 2,017 entries in each codebook across the three
masked refinement passes. They use the same fixed 2,048-value conditioning
vector `(index % 67 - 33) / 32` and reveal four codebooks per pass in order
`[0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]`. Each pass feeds the
reference's preceding greedy choices to both models. Greedy selection excludes
EOS, which remains in the score comparison.

The two references use the released Q4_K_M projector. The pinned llama.cpp
Metal runner executes the encoded weights, while the publisher's pinned Python
depth decoder uses F32 values decoded by the pinned llama.cpp `gguf-py` package.
Eredu executes F32 from the same encoded weights on CPU.

| Reference | Rows within tolerance | Maximum relative L2 | Maximum absolute error | Greedy reveals |
| --- | ---: | ---: | ---: | ---: |
| llama.cpp Metal | 35/36 | 0.02820427 | 0.14743495 | 12/12 |
| Publisher Python, decoded F32 | 36/36 | 0.00000299 | 0.00002456 | 12/12 |

Both references agree on all 36 argmax IDs and each row's top-five set. The
llama.cpp row for codebook 7 in the second pass exceeds the shared relative
L2 limit of 0.02. Disabling media flash attention in the same pinned Metal
runner produces identical score bytes. The decoded-F32 result passes the additional maximum
absolute limit of `5e-4`. Both references and Eredu reveal
`[296, 820, 1404, 55, 0, 622, 24, 616, 2, 620, 1302, 323]` in codebook order.
The fixture exercises greedy reveals with a fixed order and conditioning vector.

```sh
llama_dir=/private/tmp/eredu-llama-audio-reference
audio_dir=/private/tmp/eredu-audio-gguf
backbone="$audio_dir/LFM3-Audio-2B-Q4_K_M.gguf"
projector="$audio_dir/mmproj-LFM3-Audio-2B-Q4_K_M.gguf"
"$llama_dir/audio-reference" "$backbone" "$projector" "$audio_dir/depth-scores.f32" --depth
cargo run -p eredu-backend-mlx --features audio --example lfm3_audio_reference -- \
  /private/tmp/lfm3-audio validation/fixtures/lfm3/audio-gguf-sources.json \
  "$audio_dir/depth-score-parity.json" --gguf-backbone "$backbone" \
  --gguf-projector "$projector" --gguf-depth-scores "$audio_dir/depth-scores.f32"
PYTHONPATH="$llama_dir/gguf-py" python validation/lfm3_audio_reference.py \
  /private/tmp/lfm3-audio "$audio_dir/publisher-depth" --gguf-depth "$projector"
cargo run -p eredu-backend-mlx --features audio --example lfm3_audio_reference -- \
  /private/tmp/lfm3-audio validation/fixtures/lfm3/audio-gguf-sources.json \
  "$audio_dir/publisher-depth-parity.json" --gguf-backbone "$backbone" \
  --gguf-projector "$projector" --gguf-depth-scores "$audio_dir/publisher-depth/depth-scores.f32" \
  --gguf-depth-f32
```

### Q4_K_M LiquidCodec PCM

The decoder-only reference supplies ten deterministic twelve-codebook frames
in chunks of 1, 3 and 6 frames, followed by an empty final flush. This crosses
the 30-position decoder window and compares all 19,200 output samples from the
same released Q4_K_M projector. Eredu widens its GGUF weights for F32 CPU
execution. The pinned llama.cpp revision is
`7c2653acc749258dc62067da62d07024d971ccd2`.

| llama.cpp execution | Maximum absolute PCM error | Relative L2 | Result |
| --- | ---: | ---: | --- |
| Metal | 0.02080601 | 0.02159048 | Pass |
| CPU | 0.16601064 | 0.16216959 | Outside tolerance |

The tolerance is 0.05 maximum absolute error and 0.08 relative L2. Both runs
agree on the per-chunk sample counts `[1440, 5760, 11520, 480]`. The llama.cpp
CPU quantized matrix path also quantizes activations for its block dot products.
This comparison covers standalone PCM synthesis. It does not exercise GGUF
backbone, encoder, depth sampling or speech feedback execution.

Build the pinned llama.cpp checkout with Metal and the `mtmd` target enabled.
The text GGUF used by the C++ runner is the pinned Q4_K_M artifact from
[same-artifact text acceptance](#same-artifact-gguf-evidence), loaded in
vocabulary-only mode to initialize the projector interface. The Audio backbone
header is used by Eredu to validate the projector pairing.

```sh
llama_dir=/private/tmp/eredu-llama-audio-reference
audio_dir=/private/tmp/eredu-audio-gguf
cmake -S "$llama_dir" -B "$llama_dir/build-metal" -DGGML_METAL=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$llama_dir/build-metal" --target mtmd -j 8
c++ -std=c++17 -O2 -I "$llama_dir/include" -I "$llama_dir/ggml/include" \
  -I "$llama_dir/tools/mtmd" validation/lfm3_liquid_gguf_reference.cpp \
  -L "$llama_dir/build-metal/bin" -Wl,-rpath,"$llama_dir/build-metal/bin" \
  -lmtmd -lllama -o "$llama_dir/liquid-reference"
"$llama_dir/liquid-reference" /private/tmp/eredu-lfm3-gguf/LFM3-1.7B-Q4_K_M.gguf \
  "$audio_dir/mmproj-LFM3-Audio-2B-Q4_K_M.gguf" "$llama_dir/codec.f32" --metal
cargo run -p eredu-backend-mlx --no-default-features \
  --example lfm3_liquid_gguf_reference -- \
  validation/fixtures/lfm3/audio-gguf-sources.json \
  "$audio_dir/LFM3-Audio-2B-Q4_K_M.gguf.header" \
  "$audio_dir/mmproj-LFM3-Audio-2B-Q4_K_M.gguf" \
  /private/tmp/lfm3-audio/config.json /private/tmp/lfm3-audio/audio_tokenizer/config.json \
  "$llama_dir/codec.f32"
```

### BF16, F16, Q4_0 and Q8_0 Audio parity

Each row uses the published backbone/projector pair whose payload SHA-256
digests match the [source manifest](../validation/fixtures/lfm3/audio-gguf-sources.json).
The pinned llama.cpp Metal runners and Eredu's CPU F32 realization of the
same encoded weights run the mixed prefill and eight cached positions, the
three depth refinement passes, and streaming PCM with the final flush. The
BF16 and F16 rows measure widening from their respective published layouts.
The Q4_0 and Q8_0 rows measure same-artifact quantized parity.

| Layout | Mixed logits max relative L2 | Mixed logits max abs | Depth max relative L2 | Depth max abs | PCM max abs | PCM relative L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 0.00011841 | 0.00329494 | 0.00052404 | 0.00378788 | 0.00232878 | 0.00278818 |
| F16 | 0.00012602 | 0.00357914 | 0.00051487 | 0.00381649 | 0.00174028 | 0.00251233 |
| Q4_0 | 0.00011548 | 0.00247192 | 0.00060427 | 0.00377560 | 0.00097012 | 0.00215241 |
| Q8_0 | 0.00012062 | 0.00338364 | 0.00064514 | 0.00520205 | 0.00173993 | 0.00305636 |

All nine 128,000-entry language rows and all 36 2,017-entry depth rows pass
the shared logit tolerance in each layout, with matching argmax IDs and
5/5 top-five overlap. All twelve greedy codebook reveals match. Each PCM
comparison passes the 0.05 maximum absolute and 0.08 relative L2 limits for
19,200 samples with chunk counts `[1440, 5760, 11520, 480]`.

Build the pinned Metal runners as above. Use the same pair for each
comparison, with the corresponding verified Audio backbone header for the
decoder-only Eredu runner.

```sh
assets=/private/tmp/lfm3-audio
audio_dir=/private/tmp/eredu-audio-gguf
llama_dir=/private/tmp/eredu-llama-audio-reference
manifest=validation/fixtures/lfm3/audio-gguf-sources.json
for variant in BF16 F16 Q4_0 Q8_0; do
  backbone="$audio_dir/LFM3-Audio-2B-$variant.gguf"
  projector="$audio_dir/mmproj-LFM3-Audio-2B-$variant.gguf"
  "$llama_dir/audio-reference" "$backbone" "$projector" "$audio_dir/$variant-scores.f32"
  "$llama_dir/audio-reference" "$backbone" "$projector" "$audio_dir/$variant-depth.f32" --depth
  "$llama_dir/liquid-reference" /private/tmp/eredu-lfm3-gguf/LFM3-1.7B-Q4_K_M.gguf \
    "$projector" "$audio_dir/$variant-pcm.f32" --metal
  cargo run -p eredu-backend-mlx --features audio --example lfm3_audio_reference -- \
    "$assets" "$manifest" "$audio_dir/$variant-scores.json" --device cpu \
    --gguf-backbone "$backbone" --gguf-projector "$projector" \
    --gguf-scores "$audio_dir/$variant-scores.f32"
  cargo run -p eredu-backend-mlx --features audio --example lfm3_audio_reference -- \
    "$assets" "$manifest" "$audio_dir/$variant-depth.json" --device cpu \
    --gguf-backbone "$backbone" --gguf-projector "$projector" \
    --gguf-depth-scores "$audio_dir/$variant-depth.f32"
  cargo run -p eredu-backend-mlx --features audio --example lfm3_liquid_gguf_reference -- \
    "$manifest" "$backbone.header" "$projector" "$assets/config.json" \
    "$assets/audio_tokenizer/config.json" "$audio_dir/$variant-pcm.f32"
done
```

### Q4_K_M tensor-cohort replay

The native `MlxTensorParallelAudioModel` and `MlxPipelineAudioModel` accept
externally initialized MLX Ring cohorts. On CPU the pinned Q4_K_M pair
runs the same finite speech-conditioned prompt and seed on every rank,
with rank-local language weights and caches and replicated speech input and
output components. Each pipeline run carries the residual stream between
stages and commits the candidate state and event across all ranks.

| TP | PP | Ranks | Matching codebook IDs | Matching PCM samples | Max absolute PCM error | Relative L2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1 | 2 | 36 | 5,760 | 0 | 0 |
| 1 | 2 | 2 | 36 | 5,760 | 0 | 0 |
| 2 | 2 | 4 | 36 | 5,760 | 0 | 0 |

The combined TP=2/PP=2 run also matches all 36 codebook IDs and 5,760
PCM samples exactly in both bounded replays. Each selects host-layerwise
or disk-streamed residency for language, speech input, depth/feedback and
codec synthesis together. All four workers exit successfully in both cases.

The PCM comparison includes the final codec flush. The launcher requires
exact codebook and peer agreement, a maximum absolute PCM error of 0.0005,
and relative L2 of at most 0.0001. The combined topology also matches
serial text readout with generated IDs `[13, 13, 13]` on every rank.
In a PP=2 admission check with text mode on one rank and speech mode on
the other, both ranks reject the request and neither publishes output.
In a separate TP=2/PP=2 check, all ranks complete prefill before sampling
their first depth pass with distinct seeds. All four reject the divergent
decision and report zero committed slots. The shared runtime tests
`partitioned_phase_agreement_propagates_execution_observation_and_completion_failures`
and `failed_completion_rolls_back_every_component` cover injected local
phase failures and rollback after failed completion.

The TP-only adapter uses caller-managed agreement. The pipeline adapter
uses the shared manifest, transport, and all-rank candidate commit. The
serial side also checks cold
language partition construction for PP=2 and combined TP=2/PP=2 from
the same retained sources. Each pipeline stage owns twelve layers and
the first and last stages own input and output respectively.

| TP | PP | First-stage bytes per rank | Last-stage bytes per rank |
| ---: | ---: | ---: | ---: |
| 1 | 2 | 3,994,432,512 | 3,969,349,632 |
| 2 | 2 | 1,997,415,424 | 1,984,878,592 |

These are logical F32 parameter bytes for the language partition. They
include the tied vocabulary storage at both ends of the pipeline and
exclude mutable state, activations, and speech components. The PP and
combined checks construct stage-local modules and bindings without
materializing tensor payloads. The execution replay materializes and runs
these same bindings.


```sh
audio_dir=/private/tmp/eredu-audio-gguf
cargo build -p eredu-backend-mlx --features audio --example lfm3_audio_reference
python3 validation/lfm3_audio_tensor_replay.py \
  target/debug/examples/lfm3_audio_reference /private/tmp/lfm3-audio \
  validation/fixtures/lfm3/audio-gguf-sources.json \
  "$audio_dir/LFM3-Audio-2B-Q4_K_M.gguf" \
  "$audio_dir/mmproj-LFM3-Audio-2B-Q4_K_M.gguf" \
  "$audio_dir/pipeline-replay" --topology pp2 --topology tp2-pp2
```

Run the same command with a separate output directory and
`--topology tp2-pp2 --text` in place of the speech topology options to
check the vocabulary readout. Use `--topology tp2-pp2 --residency host`
or `--topology tp2-pp2 --residency disk` to replay with all components
bounded. Use `--reject-divergent-seeds` to check agreement during the first depth
advance. Use a fresh output directory for each case.

## LFM3 Omni SafeTensors admission

The architecture admission uses [LFM3-Omni-2B revision
`21d75123f86c1f7ffb4bef1ea202710c4e7ebe67`](https://huggingface.co/LiquidAI/LFM3-Omni-2B/tree/21d75123f86c1f7ffb4bef1ea202710c4e7ebe67).
It binds the language model, audio encoder and projector, speech depth and
feedback, and vision tower and projector as restricted views of one validated
model source. The bundled LiquidCodec has its own source. The Omni processor
supplies image and video framing IDs through the retained tokenizer and keeps
the nested 16 kHz feature extractor policy.

| File | SHA-256 of length prefix and SafeTensors header |
| --- | --- |
| `model.safetensors` | `4243cc9337ca2d03465b5d44d11f33caf7fb4a3e6cff9697dc726b239554e101` |
| `audio_tokenizer/model.safetensors` | `b82ba4739376cf48ed9117f03a2f085987970488521b9683cffcc76a664c24d5` |

The metadata replay calls `PreparedOmni::open` and
`prepare_vision(RecipeDtype::BF16)`. It admits both published headers and
prepares all 16 vision units. Its inputs retain the exact headers and file
extents with sparse payloads. The replay establishes admission and recipe coverage.

The released model and codec payload SHA-256 hashes are
`76c0ec2c3e27832fb19f886d5eb0afe37f03c5c32c4a796a7ce08cef119d2763` and
`45984a872fdcf0b94cd0ec68822bbf26f0c6fcb580cfaba0f7fa3ee60355137f`.
A resident CPU run with F32 materialization projects one image from a
`(1, 2, 2)` patch grid and encodes eight mel frames. The mixed prompt starts
the shared speech session with seed 42 and a five-slot budget. It commits
five twelve-codebook frames and returns 9,600 finite PCM samples at 24 kHz,
including the codec tail. This run starts from prepared patches and features.
A separate resident CPU run uses `api::audio::PreparedMultimodalAudioTask`
with a 96 × 96 RGB image, 1,600 mono samples at 16 kHz and a text instruction.
With seed 42, speech mode and a five-slot budget, the shared response session
returns 9,600 finite PCM samples at 24 kHz. The run verifies the public raw
image and PCM path. Publisher numerical parity remains unverified.

The synchronized video replay pairs four decoded 64 × 64 frames at 4 FPS
with 16,000 mono PCM samples at 16 kHz. The pinned Python prompt expansion
places six audio positions after the first frame group and seven after the
second for a 13-position soundtrack. Eredu reproduces that split and the outer
video framing. The released resident CPU session runs the paired request through
`PreparedMultimodalAudioTask::new_with_soundtracks` and returns 9,600 finite PCM
samples at 24 kHz with seed 42 and a five-slot speech budget. This checks
prompt ordering and execution. Score and PCM parity remain unverified.

## LFM3 Omni GGUF descriptor admission

The [published Omni GGUF revision
`d508b88f08a5b4286f2d2805a6a8d69a0ff44c72`](https://huggingface.co/LiquidAI/LFM3-Omni-2B-GGUF/tree/d508b88f08a5b4286f2d2805a6a8d69a0ff44c72)
provides BF16, F16, Q4_0, Q4_K_M, and Q8_0 backbones and projectors. Each
backbone has 287 tensor descriptors. Each combined projector has 458, of
which 139 are vision tensors. The combined catalog is checked as one
physical schema. The architecture checks canonical recipes for vision, speech
input, speech depth and feedback, and LiquidCodec decoder against that catalog.

The manifest records publisher file sizes and SHA-256 values, exact header
extents and SHA-256 values, and tensor type counts. The header replay verifies
the ten descriptor prefixes and the component recipe geometry using the model
and codec configs from the pinned SafeTensors revision.

`api::audio::PreparedOmni::open_gguf(backbone, projector, assets, max_cached_readers)`
takes the explicit GGUF pair and the matching Omni sidecar directory. The
sidecars supply the model, processor, tokenizer, generation and LiquidCodec
configs. The backbone supplies the chat template and vocabulary metadata.
The combined projector supplies both the vision and speech paths, including
LiquidCodec synthesis. `PreparedMultimodalAudioTask` accepts this artifact
through the same finite-turn interface as SafeTensors.

The released Q4_K_M pair matches the full publisher SHA-256 values in the
manifest. The resident CPU replay opens that pair with the pinned SafeTensors
sidecars and calls `PreparedMultimodalAudioTask::new_with_soundtracks` in speech
mode. The input contains four constant RGB frames (64 × 64, all channels 90) at
4 FPS with `VideoSampling::All`, 16,000 mono samples at 16 kHz
(`(i as f32 * 0.1).sin() * 0.1`), and the instruction “Describe what you see
and hear.” With seed 42 and five response slots, draining
`AudioTaskSession::next_output` returns 9,600 finite F32 PCM samples at 24 kHz,
including the codec tail. The little-endian PCM bytes hash to
`aed453d50e086f025e0c4cdf63983c7d6054d38c40bee9bd5cf4b1d81c27e479`.
The same turn returns byte-identical PCM with direct disk streaming for
all components. The `load_with_residency` device budgets are 160 MiB for
vision, 1,600 MiB for language, 100 MiB for the audio encoder, 320 MiB for
depth and feedback, and 96 MiB for the codec. The host budget, lookahead,
and background queue capacity are zero for each component.
The other encodings have descriptor evidence. Independent score and PCM
parity remain unverified.

```sh
manifest=validation/fixtures/lfm3/omni-gguf-sources.json
headers=$(mktemp -d)
repo=$(jq -r .repository "$manifest")
revision=$(jq -r .revision "$manifest")
jq -r '.variants[] | (.backbone, .projector) | [.file, .header_bytes] | @tsv' "$manifest" |
  while IFS="$(printf '\t')" read -r member bytes; do
    curl -fsSL --range "0-$((bytes - 1))" --max-filesize "$bytes" \
      "https://huggingface.co/$repo/resolve/$revision/$member?header=$bytes" \
      -o "$headers/$member.header"
  done
cargo run -p eredu-architectures --example inspect_lfm3_omni_gguf_headers -- \
  "$manifest" "$headers" /private/tmp/lfm3-omni-source
```

### Omni distributed replay

The released Q4_K_M video-and-soundtrack fixture above also exercises the
native `MlxPipelineAudioModel::start_omni` adapter on CPU. Each rank projects
the decoded video with `MlxVlVision::project_prompt`, expands its soundtrack
positions with `PreparedOmni::video_prompt`, and submits the resulting
`OmniAudioRequest` with seed 42. Vision and speech components are replicated
on each rank. The language layers and their state follow the selected
pipeline and tensor placement.

The replay uses the MLX Ring backend on loopback. Set `MLX_HOSTFILE` to a
JSON array with one single-address array per rank and `MLX_RANK` to the
process's zero-based rank. Select the rank-local graph with
`ParallelTopology::new(tensor_ranks, pipeline_stages, 1, 1)` and drain
`MlxPipelineAudioSession::events`. The resident comparison submits the same
request through `MlxAudioModel::start_omni`.

TP=2, PP=2, and combined TP=2/PP=2 each report five committed slots on every
rank. All 60 committed codebook IDs and the 9,600 PCM samples match the
resident run on every rank. The PCM SHA-256 matches the resident and
disk-streamed result above.

In the TP=2 failure replay, prepending token 5 on rank 1 causes both ranks
to reject prompt admission. With matching prompts and seeds 42 and 43,
both ranks reject the first depth advance and report zero committed slots.
Independent numerical parity remains unverified.

The public replay uses the same decoded video, soundtrack, instruction, seed,
and slot budget through `PreparedMultimodalAudioTask::new_with_soundtracks`.
In each Ring rank process, use the same `ParallelTopology` as the native replay
and call:

```rust,ignore
let mut session = task.load_distributed(
    &weights,
    &execution,
    topology,
    Default::default(),
    Default::default(),
)?;
while let Some(output) = session.next_output()? {
    // Consume committed text or PCM on every rank.
}
```

The two residency arguments select vision and speech parameter placement
independently. The replay uses resident weights on each CPU rank.

TP=2, PP=2, and combined TP=2/PP=2 each return 9,600 samples on every
rank with the same PCM SHA-256 as the resident baseline. In a TP=2 failure
replay, rank 1 includes an 8 kHz mono PCM segment where the selected
processor requires 16 kHz. Rank 1 reports its media error and rank 0 reports
collective rejection. Neither rank returns a response session.
