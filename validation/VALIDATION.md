# Checkpoint validation tools

The text-checkpoint path uses three artifacts:

1. `checkpoint_probe` runs SafeMLX and writes JSON metadata plus F32 logits in
   SafeTensors.
2. `reference_runner.py` runs Transformers with the probe's exact prompt and
   cache-feed token IDs.
3. `compare_checkpoints.py` checks input identity, finiteness, relative L2,
   cosine similarity, top-k overlap, and unambiguous argmax agreement.

Install the reference dependencies in a CUDA-enabled Python environment:

```bash
python -m pip install -r validation/requirements.txt
```

## Pinned CUDA image

The published `linux/amd64` image contains the CUDA-enabled
`checkpoint_probe`, an MXFP4 JIT smoke binary, PyTorch, Transformers, and the
validation scripts. Its build compiles the installed CUDA, CuTe, and CUTLASS
headers with NVRTC. At runtime, `MLX_CUDA_JIT_INCLUDE_DIRS` gives MLX explicit
header roots instead of relying on the executable's installed location.

```text
ghcr.io/jbg/safemlx-validation:cuda12.9.1-rust1.89.0-torch2.8.0-v1
```

Run it on a host with the NVIDIA Container Toolkit, mounting checkpoints and
results separately so downloaded weights do not become container layers:

```bash
docker run --rm --gpus all \
  -v "$PWD/models:/models:ro" \
  -v "$PWD/validation/results:/data/results" \
  ghcr.io/jbg/safemlx-validation:cuda12.9.1-rust1.89.0-torch2.8.0-v1 \
  checkpoint_probe --help
```

Use the `-git-<12-character commit>` tag emitted by the publish workflow, or
the registry digest reported by it, when a run must remain tied to one source
revision. The version tag is intentionally not named `latest`.

Qualify each newly published image once on an NVIDIA host by running its real
MXFP4 JIT path:

```bash
docker run --rm --gpus all \
  ghcr.io/jbg/safemlx-validation:<immutable-tag-or-digest> \
  cuda_mxfp4_smoke
```

This is an image/runtime qualification check, so individual checkpoint pilots
do not repeat it.

Run SafeMLX:

```bash
cargo run -p safemlx-lm --features cuda --example checkpoint_probe -- \
  --model /models/tinyllama \
  --input-ids 1,2,3,4 \
  --teacher-forced-ids 5,6,7,8 \
  --output validation/results/tinyllama_safemlx
```

Run the reference along the exact same token path:

```bash
python validation/reference_runner.py \
  --probe validation/results/tinyllama_safemlx.json \
  --output validation/results/tinyllama_transformers \
  --device cuda:0 --dtype bfloat16
```

Compare using the case's tolerance profile from the manifest:

```bash
python validation/compare_checkpoints.py \
  --actual validation/results/tinyllama_safemlx.json \
  --reference validation/results/tinyllama_transformers.json \
  --manifest validation/models.yaml \
  --case llama_dense_untied \
  --output validation/results/tinyllama_comparison.json
```

The comparator exits `0` on success, `1` for a correctness failure, and `2`
for an invalid invocation or artifact. The reference runner selects
`AutoModelForImageTextToText` for Qwen3-VL and `AutoModelForCausalLM` for
text-only families. A Qwen3-VL run through this interface validates only the
text decoder; image and video inputs still need a modality-specific probe and
reference path. Other multimodal and realtime families likewise need their
modality-specific reference runners.

## Hugging Face CUDA pilot

`run_pilot_case.py` executes one enabled manifest case end to end:

1. download the checkpoint at its immutable revision;
2. run SafeMLX and Transformers correctness on the same 16-token decode path;
3. compare the logits with the manifest tolerance profile; and
4. run a SafeMLX 128-token prefill plus 128-token decode performance smoke.

The pilot uses this immutable image reference:

```text
ghcr.io/jbg/safemlx-validation@sha256:3a2cd1dff93aa3dac986b1b21392fe4518f0cc6d674aa8d57a934b71097cdff0
```

Mount this directory at `/opt/pilot` and a persistent artifact destination at
`/artifacts`, then invoke a case such as:

```bash
python /opt/pilot/run_pilot_case.py \
  --case qwen3_dense \
  --manifest /opt/pilot/models.yaml \
  --output-root /artifacts/safemlx-cuda-pilot/<pilot-id>
```

The case directory always receives `pilot-summary.json`, including partial
phase evidence if a download, load, correctness, or performance phase fails.
Use `--prompt-id <id>` to select any deterministic correctness prompt from the
manifest, `--skip-performance` for correctness-only diagnostics, and
`--refresh-model` only when the pinned checkpoint cache must be downloaded
again. `--overwrite` replaces result artifacts without discarding that cache.

For numerical localization, `run_prompt_matrix.py` runs one case across every
manifest prompt in a single GPU allocation while reusing the checkpoint:

```bash
python /opt/safemlx/validation/run_prompt_matrix.py \
  --case nemotron_h_dense \
  --manifest /opt/safemlx/validation/models.yaml \
  --output-root /artifacts/safemlx-cuda-pilot/<matrix-id>
```

Pass `--prompt-id <id>` repeatedly to select a subset. The matrix writes an
incremental `prompt-matrix-summary.json` and exits `0` when every comparison
passes, `1` for threshold failures, or `2` for execution errors.
