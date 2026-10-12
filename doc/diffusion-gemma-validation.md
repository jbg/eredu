# DiffusionGemma numerical validation

`eredu-architectures::diffusion_gemma` declares DiffusionGemma checkpoint schemas,
shared text layers, independent encoder residual scales, soft-embedding feedback,
and context/refinement invocations. Context appends authoritative projections;
refinement attends to retained context and the complete transient canvas without
changing the cache. One parameter unit contains one shared layer. Its matrices
serve both invocations, without a second encoder weight materialization.

Validation uses the neutral layered-runtime driver and public model admission
with MLX CPU tensors. The public-loader fixture constructs the complete official
image/text architecture and invokes context and refinement through the retained
session, without reconstructing a separate family model in the test.

## Execution contract

The released SafeTensors checkpoint admits text, images and video frames. Its
published GGUF contains the text stack and tokenizer, and declares text-only
input capabilities. Vision layers remain individual residency and pipeline units;
the shared text layers and their mutable context are distinct from refinement
invocations. Each expert bank is independently addressable, including stored
quantization companions.

The released encoder is causal between text positions and bidirectional inside
each image interval. Text chunk boundaries preserve that equation; image spans
remain atomic. A configuration with fully bidirectional encoder attention retains
a complete context invocation because an earlier position depends on later
positions. Cold chunking capabilities reflect that visibility policy.
Refinement attends to the complete canvas and retained context, so its scores
are position scores rather than causal next-token conditionals. Causal sampling
overrides and speculative verification receive typed rejections. The trained
refinement policy controls proposal noise and stopping; evolving token constraints
apply once per authoritative finalization position.

Ordinary and controlled generation share the operation driver. Context and
refinement are explicit pause boundaries; snapshots preserve active canvases,
feedback, RNG and partial finalization. Opt-in draft events have independent
branch/epoch identities and clear before commitment or reconciliation. Capture
and resource budgets remain monotone through restore and fork.

## Released artifacts

The official BF16 artifact is `google/diffusiongemma-26B-A4B-it`, revision
`f7f5b7f5fa82ffc52addd066915886d497f5517b`. All eleven SafeTensors shards are
verified against their pinned Hugging Face LFS SHA-256 and size. Strict schema
admission accepts all 1,047 tensors, including the published vision parameters.
The direct text comparison materializes the shared text stack and its ingress
and readout modules; the public-loader comparison constructs the complete family. The independent expert catalog declares 3,840 bounded units;
their total admitted projection storage is 45,675,970,560 bytes, exactly equal
to the official shared expert tensors. Encoding and refinement do not duplicate
that storage.

The published text GGUF is
`unsloth/diffusiongemma-26B-A4B-it-GGUF`, revision
`f4183a2c7a354128d02545752303c4354d165bf0`, file
`diffusiongemma-26B-A4B-it-Q4_K_M.gguf`. Its size is 16,806,810,208 bytes and SHA-256
is `24523b6c833c9ce9f5f34f9b333ab1517d73d6f1e76a103645353114c8028bc5`.
Strict admission accepts all 692 tensors. The artifact contains text parameters,
self-conditioning projections, encoder scales and Gemma4 tokenizer metadata.
Tokenizer reconstruction consumes the published BPE merges. Token IDs and
decoded text match the official tokenizer JSON for spaces, repeated whitespace,
newlines, tabs, multilingual text, emoji and chat markers.

## Independent reference

The reference is Transformers source revision
`55c3cc9f03cd4fbf2d27d382dcc8fcc544631f8e`, with PyTorch 2.14.1 on CPU.
The official SafeTensors comparison uses BF16.
Attention and experts both explicitly select `eager`. Expert accumulation follows
ascending expert identities with BF16 rounding of each contribution and addition;
it differs numerically from the reference library's separate batched expert
implementation. GELU-tanh widens BF16 input to FP32, uses the vector tanh tail
saturation and rounds its output once.

The fixture executes three successive authoritative context appends and canvas
invocations, retaining KV and soft feedback between them:

| Context tokens | Canvas tokens | Authoritative frontier |
| --- | --- | --- |
| 2, 100, 200 | 101, 211 | 3 |
| 17 | 421, 653 | 4 |
| 31, 29 | 1019, 2047 | 6 |

Feedback is the previous FP32 output divided by 0.8 and cast to BF16. Encoder
outputs and all 30 refinement-layer outputs match exactly. Final softcapped FP32
scores have maximum absolute difference 0.00000191 across all three invocations.
The CPU validation accepts maximum absolute error 0.00002 and rejects nonfinite
outputs. Every refinement preserves all authoritative cache frontiers.

The GGUF reference independently dequantizes all 691 executable parameters with
`gguf` 0.19.0 and runs the reference equations in FP32, including retained feedback.
The remaining tensor is validated rotary-frequency metadata. Native execution
retains packed projections with the selected affine conversion's FP16-rounded
scales and biases; embedding lookup also has a rounded dequantization boundary.
These differ from the reference's original GGML dequantization into FP32 weights.
Encoder comparisons allow 0.001 absolute error. Refinement comparisons allow
0.01 maximum absolute error and 0.0002 relative RMS error, require identical
argmax tokens at every position, and reject nonfinite outputs. The relative RMS
limit is below one half of FP16 unit roundoff. A separate second refinement uses
exact reference feedback to distinguish representation effects from drift in
independently retained feedback.
With exact reference feedback, the second refinement has maximum absolute error
0.004367 and RMS error 0.0006921, with identical argmax tokens. This isolates the
representation and execution difference from previously generated feedback.

The public-loader GGUF measurements with independently retained feedback are:

| Invocation | Maximum absolute error | RMS error |
| --- | ---: | ---: |
| Context 0 | 0.0003800 | 0.00004901 |
| Refinement 0 | 0.0003481 | 0.00007171 |
| Context 1 | 0.0002569 | 0.00004438 |
| Refinement 1 | 0.004511 | 0.0007162 |
| Context 2 | 0.0002890 | 0.00004282 |
| Refinement 2 | 0.006602 | 0.001105 |

The maximum relative score RMS error is 0.0001448 (0.01448%). All comparisons
pass their bounds. Argmax tokens match at every position, and every refinement
preserves the authoritative frontiers 3, 4 and 6.

Released image fixtures use BF16 patches of width 768 on 3×3 and 3×6 grids.
The rectangular fixture follows the processor's column/row position ordering
and produces two spatial pooling cells.
Patch embeddings, all 27 vision blocks, the spatially pooled and standardized
features, and their projection into text width match exactly. Vision normalization
and pooling use FP32 intermediates with explicit BF16 rounding boundaries. The
image modules bind the official encoder parameter root and materialize individual
vision layers independently.

## Reproduction

Keep artifacts outside the tracked source tree. The following reference setup
uses Python 3.12:

```sh
python3.12 -m venv /tmp/diffusion-gemma-reference
/tmp/diffusion-gemma-reference/bin/pip install torch==2.14.1 safetensors huggingface_hub gguf==0.19.0
/tmp/diffusion-gemma-reference/bin/pip install 'https://github.com/huggingface/transformers/archive/55c3cc9f03cd4fbf2d27d382dcc8fcc544631f8e.zip'
```

Download the official pinned release and verify each shard's LFS digest. The
reference script requires the resulting `source_manifest.json`:

```python
import dataclasses, hashlib, json, pathlib
from huggingface_hub import HfApi, snapshot_download
repo = "google/diffusiongemma-26B-A4B-it"
revision = "f7f5b7f5fa82ffc52addd066915886d497f5517b"
info = HfApi().model_info(repo, revision=revision, files_metadata=True)
path = pathlib.Path(snapshot_download(repo, revision=revision,
    local_dir="/tmp/diffusion-gemma-official"))
for item in info.siblings:
    if item.rfilename.endswith(".safetensors"):
        with (path / item.rfilename).open("rb") as source:
            assert hashlib.file_digest(source, "sha256").hexdigest() == item.lfs.sha256
        assert (path / item.rfilename).stat().st_size == item.size
(path / "source_manifest.json").write_text(json.dumps(dataclasses.asdict(info)))
```

Run fixture generation and native comparison from the repository root:

```sh
/tmp/diffusion-gemma-reference/bin/python eredu-evaluation/scripts/diffusion_gemma_reference.py /tmp/diffusion-gemma-official /tmp/diffusion-gemma-reference.safetensors
cargo run -p eredu-backend-mlx --no-default-features --example diffusion_gemma_reference -- /tmp/diffusion-gemma-official /tmp/diffusion-gemma-reference.safetensors
/tmp/diffusion-gemma-reference/bin/python eredu-evaluation/scripts/diffusion_gemma_reference.py /tmp/diffusion-gemma-official /tmp/diffusion-gemma-vision-reference.safetensors --vision-only --vision-grid 3 6
cargo run -p eredu-backend-mlx --no-default-features --example diffusion_gemma_reference -- /tmp/diffusion-gemma-official /tmp/diffusion-gemma-vision-reference.safetensors --vision-only
EREDU_DIFFUSION_CHECKPOINT=/tmp/diffusion-gemma-official EREDU_DIFFUSION_REFERENCE=/tmp/diffusion-gemma-reference.safetensors cargo test -p eredu-backend-mlx --no-default-features --lib released_diffusion_public_loader_matches_cached_reference -- --ignored --nocapture
```

The public-loader BF16 comparison matches all three encoder outputs exactly and
has maximum score difference 0.00000191, with the same 0.00002 tolerance and
authoritative frontiers 3, 4 and 6.

Download the published text GGUF and produce its independent reference:

```python
import hashlib, pathlib
from huggingface_hub import hf_hub_download
path = pathlib.Path(hf_hub_download(
    "unsloth/diffusiongemma-26B-A4B-it-GGUF",
    "diffusiongemma-26B-A4B-it-Q4_K_M.gguf",
    revision="f4183a2c7a354128d02545752303c4354d165bf0",
    local_dir="/tmp/diffusion-gemma-gguf"))
assert path.stat().st_size == 16806810208
with path.open("rb") as source:
    assert hashlib.file_digest(source, "sha256").hexdigest() == "24523b6c833c9ce9f5f34f9b333ab1517d73d6f1e76a103645353114c8028bc5"
```

```sh
/tmp/diffusion-gemma-reference/bin/python eredu-evaluation/scripts/diffusion_gemma_reference.py /tmp/diffusion-gemma-official /tmp/diffusion-gemma-gguf-reference.safetensors --gguf /tmp/diffusion-gemma-gguf/diffusiongemma-26B-A4B-it-Q4_K_M.gguf
EREDU_DIFFUSION_CHECKPOINT=/tmp/diffusion-gemma-official EREDU_DIFFUSION_GGUF=/tmp/diffusion-gemma-gguf/diffusiongemma-26B-A4B-it-Q4_K_M.gguf EREDU_DIFFUSION_REFERENCE=/tmp/diffusion-gemma-gguf-reference.safetensors cargo test -p eredu-backend-mlx --no-default-features --lib released_diffusion_gguf_public_loader_matches_cached_reference -- --ignored --nocapture
EREDU_DIFFUSION_CHECKPOINT=/tmp/diffusion-gemma-official EREDU_DIFFUSION_GGUF=/tmp/diffusion-gemma-gguf/diffusiongemma-26B-A4B-it-Q4_K_M.gguf cargo test -p eredu-text --no-default-features --lib released_diffusion_gemma_bpe_matches_official_tokenizer -- --ignored
```

Portable nonzero fixtures compare resident and rebuilt bounded units through
context, feedback, append and restored context. A two-rank tensor-parallel fixture
compares complete canvas scores and local authoritative projections with ordinary
execution. Pipeline fixtures compare TP1/PP2 and TP2/PP2 through the ordinary
partition driver, transporting invocation metadata and checking retained local
state. These fixtures use neutral backend contracts and host collectives.
The prepared composite fixture compares dense and sparse image/text execution
through TP1/PP4, TP2/PP2, TP1/PP2/EP2 and TP2/PP2/EP2 (expert axes apply to the
sparse stack). It covers all three weight residencies, unequal padded image grids,
vision-layer pipeline cuts, subsequent context appends and multiple complete
refinement canvases. Every rank retains its authoritative frontier through
refinement. Native resident, host-layerwise and disk-streamed fixtures also cover
ordered image/video inputs, text chunk boundaries, snapshots and constrained
finalization.

The focused conformance commands are:

```sh
cargo test -p eredu-architectures --test reference_numeric diffusion_composite_prepared_partitions_preserve_padded_media_context_across_layer_cuts
cargo test -p eredu-backend-mlx --no-default-features --lib replicated_text::tests::diffusion_gemma::
cargo test -p eredu --no-default-features --test backend_conformance iterative
```

Native Metal and native distributed execution have no released-checkpoint
numerical result in this document.
