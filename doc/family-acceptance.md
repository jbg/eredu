# Model-family acceptance

All families use the verification policy in [AGENTS.md](../AGENTS.md).
Durable numerical evidence comes from complete-model execution against an
independent pinned reference. A scalar backend that repeats family equations,
a component fixture, agreement between Eredu execution modes, or successful
checkpoint admission is not released-checkpoint numerical acceptance.

## Reference routes and evidence

The table indexes the retained routes. A route is not a passing result. The
linked manifests and family documents specify artifact revisions, reference
versions, inputs, tolerances and measured results where available. Unlisted
formats, encodings and modalities have no established numerical acceptance.

| Family | Retained reference route | Evidence and limits |
| --- | --- | --- |
| Llama, Mistral | `checkpoint_probe`, `reference_runner.py`, `eredu-parity`; cases in `validation/models.yaml` | Pinned SafeTensors inputs and tolerance profiles. The manifest schedules validation; it does not record a pass. Published GGUF needs an independent same-artifact reference. |
| Phi-4 | Same shared text route; independent Transformers checkpoint generator | [Phi-4](phi4-validation.md) records released BF16 prefill/eight decode rows and synthetic F32/Q8 GGUF results. Released GGUF and native distributed numerical acceptance are unvalidated. |
| Gemma 2 | Shared text route; independent Transformers checkpoint generator | [Gemma 2](gemma2-validation.md) records released BF16 and independent synthetic execution. Released quantized GGUF acceptance is unvalidated. |
| Nanbeige | Shared text route; independent GGUF-to-Transformers export | [Nanbeige](nanbeige-validation.md) records released SafeTensors and synthetic GGUF results. The chat comparison also covers the official assistant prompt. |
| Qwen 2/2.5, Qwen 3 dense/MoE | Shared text route; pinned manifest cases | SafeTensors reference route is available. Released GGUF encoded-artifact acceptance has no result in the manifest. |
| Qwen 3 Next, Qwen 3.5 dense/MoE | Shared text route; `qwen35_mlx_reference.py` for the pinned MLX artifact | [Qwen 3.5 MLX](qwen35-mlx-validation.md) records its exact scope. Text results do not establish image/video parity. |
| Qwen 3 VL dense/MoE | Shared text route selects Transformers image-text models | Manifest pins the releases. This route tests the text decoder only; complete image/video reference acceptance is unvalidated. |
| Qwen3.8-Flash-Next (`Qwen4Exp`) | Shared text probe/comparator; independent source pins in [Flash-Next](qwen4-exp-validation.md) | Released text, image/video, encoded GGUF and embedded-prediction numerical acceptance are unvalidated. A compatible pinned independent runner is required for each path. |
| DeepSeek V3/R1 | Shared text route; pinned publisher remote-code manifest case | Same encoded FP8 artifact is required. Manifest input/threshold declarations do not establish a passing result. |
| DeepSeek V4 | Shared text probe/comparator | A pinned complete-model independent runner/result is not recorded. Text, encoded GGUF and embedded prediction remain numerical acceptance gaps. |
| GPT-OSS | Shared text route; native MXFP4 manifest profile | Compare the same MXFP4 artifact; a BF16 comparison is a separate measurement. Manifest is a route, not a passing result. |
| Gemma 4 and assistant | Shared text route; `speculative_reference` and pinned assistant manifest | Text alone does not validate image/audio/video or assistant proposal scores. Released modality/assistant numerical results require their exact independent references. |
| K2 Horizon dense/MoE/MoVA | `k2_checkpoint_parity`, publisher `k2_horizon_reference.py` | [K2 Horizon](k2-horizon-validation.md) and compact released-result manifests record exact SafeTensors/GGUF evidence and remaining variants. Reduced synthetic FP8 matrices are not released parity. |
| Kimi Linear | Shared text route; pinned publisher remote-code manifest case | Manifest records the source and workload. Released encoded GGUF numerical acceptance is unvalidated. |
| LFM2/LFM2.5 dense/MoE | Shared text route; released `chunked_prefill` acceptance and `prefill_reference.py` | [LFM2](lfm2-prefill-validation.md) records a released 350M checkpoint widened exactly to F32, full prefill scores and three cached steps. This does not validate released BF16 or quantized execution. |
| Nemotron H | Shared text route; pinned dense/MoE manifest cases | Independent reference follows the exact decode path. A manifest case does not establish a result for every H variant or GGUF encoding. |
| Nemotron Labs Diffusion | `nemotron_diffusion_reference.py`, `nemotron_diffusion_reference`, `nemotron_generation_modes`; Pixtral reference runner | [Nemotron Diffusion](nemotron-labs-diffusion-validation.md) records released causal/diffusion, adapter and image-context results, including independently decoded GGUF artifacts. Native distributed released parity remains unvalidated. |
| DiffusionGemma | `diffusion_gemma_reference.py`, native reference example and `reference_acceptance` tests | [DiffusionGemma](diffusion-gemma-validation.md) pins official/GGUF artifacts and independent implementation. Context, refinement feedback and output scores are the applicable complete-model contract. Native distributed released parity is unvalidated. |
| Inkling | Shared text probe/comparator; pinned publisher manifest case | Publisher custom architecture/reference compatibility is required. Complete media and encoded GGUF numerical acceptance are unvalidated. |
| Muse Glimmer | Shared text probe/comparator; pinned manifest | Complete image/video and encoded GGUF numerical acceptance are unvalidated. Text-only execution is insufficient media evidence. |
| Muse Spark Open and assistants | `validation/muse_spark_*reference*.py`, native target/assistant reference examples | [Compact acceptance results](../validation/fixtures/muse-spark/acceptance-results.json) retain the measured scopes and source pins. Those scopes do not establish every artifact/layout. |
| Moshi, PersonaPlex | `moshi_mlx_token_fixture.py`, `personaplex_torch_fixture.py`, released realtime acceptance; `personaplex_pytorch_backend_reference.py` | Compare frame score rows and cached text/audio decisions, with exact prompt conditioning. MLX-token agreement alone is not an independent numerical reference. PersonaPlex quantization quality is separate from same-artifact execution acceptance. |
| Mimi | Complete codec path in the PersonaPlex PyTorch/native reference and evaluation tools | Codec acceptance compares the same released weights, code IDs and decoded PCM over streaming frames. Synthetic codec unit equations do not establish released codec parity. |
| EAR encoder/decoder and PRVQ; VoiceChat | Released VoiceChat CLI route and pinned NVIDIA [source identities](../eredu-evaluation/fixtures/voicechat/sources.json) | [VoiceChat](voicechat-validation.md) records execution evidence. Independent complete-model speech/codec numerical parity remains unvalidated for official, packed SafeTensors and GGUF artifacts. No extracted-component fixture substitutes for this gap. |

## Reproduce one acceptance case

For manifest-backed text cases, use the existing one-case driver in
[validation/VALIDATION.md](../validation/VALIDATION.md). Select one layout and
workload that resolves the uncertainty under investigation. Keep its checkpoint,
derived weights, full score tensors and detailed traces outside the repository.

For an already pinned local text checkpoint, the common route is:

```sh
export ACCEPTANCE_ROOT=/private/tmp/eredu-family-acceptance
mkdir -p "$ACCEPTANCE_ROOT"
cargo run --release -p eredu-backend-mlx --features accelerate --example checkpoint_probe -- \
  --model "$PINNED_CHECKPOINT" --device cpu --input-ids 1,3,2 \
  --decode-steps 8 --warmup-runs 0 --output "$ACCEPTANCE_ROOT/native"
python validation/reference_runner.py --probe "$ACCEPTANCE_ROOT/native.json" \
  --device cpu --dtype bfloat16 --attn-implementation eager --local-files-only \
  --warmup-runs 0 --output "$ACCEPTANCE_ROOT/reference"
cargo run -p eredu-evaluation --bin eredu-parity -- \
  --actual "$ACCEPTANCE_ROOT/native.json" --reference "$ACCEPTANCE_ROOT/reference.json" \
  --relative-l2-max 0.02 --cosine-similarity-min 0.999 \
  --top-k 5 --top-k-overlap-min 4 --output "$ACCEPTANCE_ROOT/comparison.json"
```

Use the family's pinned reference environment and tolerance profile. The common
BF16 thresholds above are not a quantized acceptance profile. For GGUF or packed
SafeTensors, the reference executes the same encoded artifact or independently
decodes those exact bytes. Comparing with a different BF16 checkpoint measures
quantization loss instead. Media and assistants use the family-specific route
and preserve exact prepared inputs and proposal/verification histories.

Retain only compact provenance and result summaries: source hashes/revisions,
reference implementation/version, reproducible invocation, workload identity,
unchanged thresholds, per-phase maxima and token agreement, and the precise
validation scope. A missing checkpoint, reference runtime or native device is a
validation gap, not a passing synthetic substitute.

## Shared execution conformance

`eredu-runtime/tests/backend_independence.rs` exercises portable execution,
bounded residency and partition/collective contracts with neutral mechanisms.
`eredu-evaluation/src/execution_control` provides shared execution-control
comparison drivers. The facade's `backend_conformance` and
`portable_realtime_facade` targets exercise common generation, controlled
advancement, speculative commit/rollback, snapshots, cancellation and budget
preservation. `native_execution_control` supplies a representative native
integration of the shared public lifecycle, without repeating it for each family.
Native rank agreement uses the shared partition, expert-exchange, Cartesian and
realtime Ring drivers.

Run the smallest relevant shared case. These contracts establish state and
execution behavior; independent family acceptance establishes numerical fidelity.
Compiler checks, visibility, manifests and feature selections establish ownership
and API boundaries. Narrow numerical localization runs in temporary files outside
the tracked tree and is removed when the discrepancy is understood.
