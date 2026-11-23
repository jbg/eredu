# Candle CPU and Metal acceptance

## SafeTensors

Candle CPU loads the pinned SafeTensors release of
`HuggingFaceTB/SmolLM2-135M-Instruct` at revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`. The source
`model.safetensors` has SHA-256
`5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`.

The released chat template renders the user message “What is the capital of
France?” with the assistant generation prompt. Greedy generation emits
`[504, 3575, 282, 4649]`, which decodes to “The capital of France”. The
ordinary run and a controlled run that steps once and resumes produce the same
four IDs and complete on both the original BF16 weights and the weights widened
exactly to F32.

The independent reference uses Transformers 5.22.2 and PyTorch 2.15.0
(`d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`) with CPU eager attention on Apple
M3 Ultra. Each comparison uses the same checkpoint and activation dtype.
The comparison covers all 49,152 vocabulary scores at all 37 prefill positions
and after three cached inputs `[504, 3575, 282]`. F32 passes the existing
absolute tolerance of `5e-4`. BF16 passes the existing limits of relative L2
at most `0.02`, cosine similarity at least `0.999`, and top-five overlap of at
least four. It matches every score bit for bit and has 5/5 top-five overlap
on every row.

| Precision | Prefill maximum absolute error | Cached decode maximum absolute error | Maximum per-row relative L2 | Greedy decisions matching |
| --- | ---: | ---: | ---: | ---: |
| Exactly widened F32 | 0.000247955322 | 0.0000514984131 | 0.0000114444856 | 4/4 |
| Original BF16 | 0 | 0 | 0 | 4/4 |

### Seeded sampling

With the same pinned BF16 artifact, the user prompt “Write a short story about
a cat who finds a key.” exercises both shared host sampling strategies. Each
request uses temperature `0.85`, repetition penalty `1.12`, frequency penalty
`0.06`, presence penalty `0.04`, an eight-token penalty window, and a 12-token
budget. Standard sampling uses top-k `32`, top-p `0.91`, and min-p `0.05`.
Mirostat V2 uses tau `4.0` and eta `0.1`.

For seeds `12` and `949`, ordinary and controlled generation match all 12
tokens under each strategy. The controlled run advances two tokens, pauses,
and resumes. A separate seed-`12` run cancels after two tokens and retains
exactly that prefix. A fresh request with the same seed reproduces the
original sequence.

### Reproduce

Fetch the pinned checkpoint and run the independent prefill and cached-decode
reference. Keep checkpoints and full scores outside the repository.

```sh
ACCEPTANCE_ROOT=/private/tmp/eredu-smollm2-acceptance
PROMPT=1,9690,198,2683,359,253,5356,5646,11173,3365,3511,308,34519,28,7018,411,407,19712,8182,2,198,1,4093,198,1780,314,260,3575,282,4649,47,2,198,1,520,9531,198
python eredu-evaluation/scripts/download_component_fixture.py "$ACCEPTANCE_ROOT" --model smollm2
python eredu-evaluation/scripts/prefill_reference.py \
  "$ACCEPTANCE_ROOT/provenance.json" "$ACCEPTANCE_ROOT/reference-bf16.json" \
  --dtype bfloat16 --all-prefill-positions --tokens "$PROMPT" --decode 504,3575,282
python eredu-evaluation/scripts/prefill_reference.py \
  "$ACCEPTANCE_ROOT/provenance.json" "$ACCEPTANCE_ROOT/reference.json" \
  --f32-checkpoint "$ACCEPTANCE_ROOT/f32" --all-prefill-positions \
  --tokens "$PROMPT" --decode 504,3575,282
```

Load the original or widened checkpoint as shown in the
[backend usage example](../eredu-backend-candle/README.md). Supply the prompt IDs
above and request four greedy tokens. For score comparison, bind the same
artifact with `prepare_inspected_model` and `ModelLoadFacts`, install it
into `ModelRuntime` using `PreparedTextDriver`, and observe the completed `prefill` and `decode`
outputs at `MODEL_LOGITS_OBSERVATION_PATH`. Compare the prefill tensor to
`all_prefill_logits` and the decode rows to `logits[1:]` in the reference output.

## Metal placement

Use the [Candle backend example](../eredu-backend-candle/README.md) with
the `metal` feature and `metal:0`. MLX uses the same pinned BF16 checkpoint,
prompt, and independent CPU reference from the SafeTensors case, with an
MLX `metal:0` execution plan. Compare all 37 prefill rows and three cached
decode rows using the procedure above. All placements in the table produce
`[504, 3575, 282, 4649]` in ordinary and controlled generation.

| BF16 placement | Maximum prefill relative L2 | Maximum cached-decode relative L2 |
| --- | ---: | ---: |
| Candle CPU resident or paged KV, 7 or 16 tokens per page | 0 | 0 |
| MLX Metal resident or paged KV, 7 or 16 tokens per page | 0.00006477 | 0.00001437 |
| Candle Metal resident KV | 0 | 0 |
| Candle Metal paged KV, 7 or 16 tokens per page | 0 | 0 |

Resident and paged placements produce identical short-prompt scores on each
backend. Select paged KV through `NormalizedLoadRequest` with
`PagedCacheOptions` and full-attention enablement.

The same weights widened exactly to F32 meet the `5e-4` absolute tolerance
on Candle Metal, with maximum absolute errors of `8.89e-5` in prefill and
`4.48e-5` in cached decode. All 40 FP32 rows have the same argmax as the
reference.

For packed Q4_K_M acceptance, see [SmolLM2 Q4_K_M](#q4_k_m-gguf).

### 64-token BF16 cached decode

The same published checkpoint with a 148-token lighthouse story prompt
exercises one unchunked prefill and 63 cached decode steps. The CPU eager
reference supplies the same input prefix to both Metal backends for every
score comparison. All 49,152 scores per row are compared at a relative L2
tolerance of `0.02`.

| Resident KV placement | Final prefill relative L2 | Maximum cached-decode relative L2 | Ordinary greedy IDs | Controlled greedy IDs |
| --- | ---: | ---: | ---: | ---: |
| Candle Metal | 0 | 0 | 64/64 | 64/64 |
| MLX Metal | 0 | 0.00006666 | 64/64 | 64/64 |

The controlled sessions pause after token 58, advance token 59, and resume
to the 64-token limit. Both backends match the reference tie at token 59,
selecting ID `894` over `5701` at a shared score of `36.25`. With 16-token
paged KV, both backends reach the same prefill and cached-decode error bounds
and match all 64 score-row argmax IDs.

Use the pinned checkpoint and the same score readback procedure as above.
Pass the prompt IDs to the prepared driver, request 64 greedy tokens, and
use the following inputs to reproduce the independent score rows.

```sh
PROMPT=1,9690,198,2683,359,253,5356,5646,11173,3365,3511,308,34519,28,7018,411,407,19712,8182,2,198,1,4093,198,49,33059,553,9318,335,253,6431,5432,1675,216,33,40,38,32,30,533,4339,28,19998,20587,690,260,7739,284,29876,260,8725,13768,30,330,1619,936,1007,2644,3365,21404,2397,351,34099,28,260,33059,40061,30,1963,3163,28,915,990,25232,28,253,20933,1646,39678,32900,30,4781,4077,359,7957,702,5073,28,284,253,24501,4109,11812,288,357,351,2382,6566,3834,253,3589,338,1072,441,1972,335,750,4141,282,260,33059,30,9517,260,6021,282,253,986,6903,563,21404,284,34099,4212,260,3589,30,7734,281,260,3891,282,260,2353,284,2475,260,1977,1355,33748,357,30,2,198,1,520,9531,198
DECODE=1653,260,1596,4879,851,282,19998,26253,483,28,260,5432,506,47805,33059,9318,2362,25092,28,624,47805,5780,253,21651,288,260,7429,282,655,30,378,2375,14654,282,260,44588,5044,354,41389,1420,335,260,19284,26060,28,284,260,935,894,282,260,10091,19834,6745,288,13689,260,935,894,282,260,33059,2581
python eredu-evaluation/scripts/prefill_reference.py \
  "$ACCEPTANCE_ROOT/provenance.json" "$ACCEPTANCE_ROOT/reference-bf16-long.json" \
  --dtype bfloat16 --tokens "$PROMPT" --decode "$DECODE"
```

## Prompt-cache and residency parity

On the short SmolLM2 fixture, Candle CPU with 16-token paged KV matches
resident scores for BF16 and Q4_K_M. For Q4_K_M, host-layerwise and
disk-streamed weights also match resident scores bit for bit. At budgets of
1,280 KiB device and 20 MiB staging, the paged run peaks at 1,310,208 device
bytes and 589,824 staging bytes. The disk-streamed run peaks at 94,090,752
device bytes and 22,560,768 host bytes with budgets of 90 MiB and 22 MiB and
one unit of lookahead.

Memory and directory prompt caches reproduce the suffix scores on Candle
CPU, Candle Metal, and MLX Metal with resident or 16-token paged BF16 KV.
The Qwen3.5-0.8B hybrid case restores six KV blocks and 36 fixed tensors. A
corrupt resident shard is rejected before installation.

Use `NormalizedLoadRequest::with_state_residency` for paged KV,
`with_weight_residency` for bounded weights, and
`with_prompt_cache_persistence(true)` for prefix export. Automatic reuse with
an eight-token bounded prefill chunk reports a 32-token hit on a repeated
SmolLM2 prompt through `GenerationStats::prompt_cache`.

## Q8_0 GGUF

The [published SmolLM2-135M-Instruct Q8_0 GGUF](https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/blob/9a7c2ed64ce7d8d00f566e812975521e17b2909d/SmolLM2-135M-Instruct-Q8_0.gguf)
has SHA-256 `5a1395716f7913741cc51d98581b9b1228d80987a9f7d3664106742eb06bba83`.
Candle CPU and the independent llama.cpp CPU runner load those same bytes
on Apple M3 Ultra. The reference uses the llama.cpp source packaged with
`llama-cpp-sys-2` `0.1.146`, GGML `0.9.11`, default CPU weight repacking,
F32 KV storage, and flash attention disabled.

With the same chat fixture as SafeTensors, ordinary generation and controlled
generation that steps once and resumes both complete with
`[504, 3575, 282, 4649]`. The independent reference selects those same greedy
tokens from its final prefill and three cached-decode score rows. Each row
compares all 49,152 vocabulary entries.

| User message | Greedy tokens in both Candle modes and llama.cpp | Final prefill relative L2 | Three cached rows, maximum relative L2 |
| --- | --- | ---: | ---: |
| What is the capital of France? | `[504, 3575, 282, 4649]` | 0 | 0 |
| Write one short sentence about the ocean. | `[504, 5065, 314, 253]` | 0 | 0 |

All compared scores match exactly, satisfying the existing relative L2 limit
of `0.02`. Every row has cosine similarity 1 and top-five overlap 5/5.

### Reproduce

Use the checked-in `validation/gguf_reference.cpp` with the llama.cpp source
directory from the pinned crates.io package and set `PROMPT` to the IDs above.

```sh
GGUF_ROOT=/private/tmp/eredu-candle-gguf
mkdir -p "$GGUF_ROOT"
artifact="$GGUF_ROOT/SmolLM2-135M-Instruct-Q8_0.gguf"
curl --fail --location -o "$artifact" \
  https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/9a7c2ed64ce7d8d00f566e812975521e17b2909d/SmolLM2-135M-Instruct-Q8_0.gguf
shasum -a 256 "$artifact"
cmake -S "$llama_dir" -B "$GGUF_ROOT/build" -DCMAKE_BUILD_TYPE=Release \
  -DGGML_BLAS=OFF -DGGML_METAL=OFF -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF \
  -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=OFF -DBUILD_SHARED_LIBS=ON
cmake --build "$GGUF_ROOT/build" --target llama
clang++ -std=c++17 -O2 -I"$llama_dir/include" -I"$llama_dir/ggml/include" \
  validation/gguf_reference.cpp -L"$GGUF_ROOT/build/bin" \
  -Wl,-rpath,"$GGUF_ROOT/build/bin" -lllama -lggml -o "$GGUF_ROOT/gguf_reference"
"$GGUF_ROOT/gguf_reference" "$artifact" "$GGUF_ROOT/reference.safetensors" \
  "$PROMPT" 504,3575,282 cpu
OCEAN_PROMPT=1,9690,198,2683,359,253,5356,5646,11173,3365,3511,308,34519,28,7018,411,407,19712,8182,2,198,1,4093,198,19161,582,1890,6330,563,260,5065,30,2,198,1,520,9531,198
"$GGUF_ROOT/gguf_reference" "$artifact" "$GGUF_ROOT/ocean.safetensors" \
  "$OCEAN_PROMPT" 504,5065,314 cpu
```

Use the same facade and score-observation procedure as for SafeTensors with
the single GGUF file as the artifact. Compare the final prefill row with
`prefill.logits` and the cached rows with `decode.logits`. For controlled
generation, prepare the same chat with `prepare_controlled_chat`, start it
with `start_controlled`, then call `step` once and `run` to completion.

## Q4_K_M GGUF

The [published SmolLM2-135M-Instruct Q4_K_M GGUF](https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/blob/c6f2d396a5faa22c9722908d47e8da81a5d8cebc/SmolLM2-135M-Instruct-Q4_K_M.gguf)
has SHA-256 `2e8040ceae7815abe0dcb3540b9995eaa1fa0d2ca9e797d0a635ae4433c68c2d`.
It contains Q5_0, Q8_0, Q4_K, and Q6_K matrices alongside F32 tensors.
Both engines load the same file on Apple M3 Ultra with the same reference
configuration and prompts as Q8_0.

Ordinary generation and controlled generation that steps once and resumes
both match llama.cpp. The France prompt yields `[504, 3575, 282, 4649]`,
and the ocean prompt yields `[504, 5065, 314, 253]`. All 49,152 scores at
the final prefill position and at each of three cached decode steps match
exactly on both prompts. All eight rows pass the shared `eredu-parity`
profile with relative L2 of 0 against the `0.02` limit.

### Candle Metal

On Candle Metal, all 37 France prefill rows, 38 ocean prefill rows, and
three cached decode rows per prompt match llama.cpp bit for bit. Ordinary
and controlled generation emit the same four greedy IDs for each prompt.

### MLX CPU and Metal

MLX CPU and Metal match the same full-prefill and cached-decode rows bit for
bit. Both generation modes emit the same greedy IDs on each prompt.

### Reproduce

```sh
artifact="$GGUF_ROOT/SmolLM2-135M-Instruct-Q4_K_M.gguf"
curl --fail --location -o "$artifact" \
  https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/c6f2d396a5faa22c9722908d47e8da81a5d8cebc/SmolLM2-135M-Instruct-Q4_K_M.gguf
shasum -a 256 "$artifact"
"$GGUF_ROOT/gguf_reference" "$artifact" "$GGUF_ROOT/q4-france.safetensors" \
  "$PROMPT" 504,3575,282 cpu
"$GGUF_ROOT/gguf_reference" "$artifact" "$GGUF_ROOT/q4-ocean.safetensors" \
  "$OCEAN_PROMPT" 504,5065,314 cpu
```

For all-position acceptance, add `--all-prefill` after `cpu` in each
reference invocation and compare the full prefill score tensor from the
shared driver.

## Qwen3-0.6B Q4_K_M GGUF

The [published Qwen3-0.6B Q4_K_M GGUF](https://huggingface.co/bartowski/Qwen_Qwen3-0.6B-GGUF/blob/7bcae0bc7b0606f1e948f8cdb31b98a2c10635db/Qwen_Qwen3-0.6B-Q4_K_M.gguf)
has SHA-256 `9acfc1e001311f34b4252001b626f2e466d592a42065f66571bff3790d4e1b14`.
The shared resident driver loads its F32, Q4_K, and Q6_K tensors on Candle CPU
through `ArtifactSource::new`. The independent llama.cpp runner loads that
exact Qwen3 file on Apple M3 Ultra with F32 KV storage and flash attention
disabled, using the same CPU build as the SmolLM2 comparison.

Both Candle generation modes match all four greedy decisions from the
reference for each prompt. Controlled generation steps once and then resumes
to completion. Each comparison covers all 151,936 vocabulary scores at
the final prefill position and after each of three cached inputs.

| User message | Thinking | Greedy tokens in both Candle modes and llama.cpp | Final prefill relative L2 | Three cached rows, maximum relative L2 |
| --- | --- | --- | ---: | ---: |
| What is the capital of France? | Template default | `[151667, 198, 32313, 11]` | 0 | 0 |
| Write one short sentence about the ocean. | Disabled | `[785, 17951, 374, 264]` | `4.42e-8` | 0 |

All eight rows have 5/5 top-five overlap.

### Reproduce

Use the same llama.cpp build, Candle facade, and score-observation procedure
as SmolLM2. For the ocean prompt, set `ChatTemplateRequest::enable_thinking`
to `Some(false)`. Leave it unset for the France prompt.

```sh
artifact="$GGUF_ROOT/Qwen_Qwen3-0.6B-Q4_K_M.gguf"
curl --fail --location -o "$artifact" \
  https://huggingface.co/bartowski/Qwen_Qwen3-0.6B-GGUF/resolve/7bcae0bc7b0606f1e948f8cdb31b98a2c10635db/Qwen_Qwen3-0.6B-Q4_K_M.gguf
shasum -a 256 "$artifact"
QWEN_FRANCE=151644,872,198,3838,374,279,6722,315,9625,30,151645,198,151644,77091,198
QWEN_OCEAN=151644,872,198,7985,825,2805,11652,911,279,17951,13,151645,198,151644,77091,198,151667,271,151668,271
"$GGUF_ROOT/gguf_reference" "$artifact" "$GGUF_ROOT/qwen-france.safetensors" \
  "$QWEN_FRANCE" 151667,198,32313 cpu
"$GGUF_ROOT/gguf_reference" "$artifact" "$GGUF_ROOT/qwen-ocean.safetensors" \
  "$QWEN_OCEAN" 785,17951,374 cpu
```

## Qwen3.5-0.8B Q4_K_M GGUF

The [published 24-layer Qwen3.5-0.8B text GGUF](https://huggingface.co/bartowski/Qwen_Qwen3.5-0.8B-GGUF/blob/e0e7548404a0735ae4d78b07305ff8a595cdf60b/Qwen_Qwen3.5-0.8B-Q4_K_M.gguf)
has SHA-256 `9d8472987aed5b36a0d167543a695bcbf349939445ca5382a4245219829f4581`.
The shared resident driver loads its F32, Q8_0, Q4_K, and Q6_K tensors through
`ArtifactSource::new`. The llama.cpp CPU runner uses the same file and build
as above on Apple M3 Ultra. Its context uses F32 KV storage, disables flash
attention, and selects the fused gated-delta kernel for prefill and decode.

Both prompts use the released chat template with thinking disabled. Ordinary
generation and controlled generation that steps once and resumes match all
four reference greedy decisions per prompt. Each comparison covers all
248,320 vocabulary scores at the final prefill position and after each of
three cached inputs.

| User message | Greedy tokens in both Candle modes and llama.cpp | Final prefill relative L2 | Three cached rows, maximum relative L2 |
| --- | --- | ---: | ---: |
| What is the capital of France? | `[760, 6511, 314, 9338]` | 0 | 0 |
| Write one short sentence about the ocean. | `[760, 17415, 369, 264]` | `5.30e-8` | 0 |

All eight rows pass the shared `eredu-parity` profile with 5/5 top-five
overlap against the `0.02` relative L2 limit. Validation covers single-device
resident text generation on this revision's backbone. The measured artifact
contains no prediction layers or vision projector.

### Reproduce

Use the same llama.cpp build, Candle facade, and score-observation procedure
as above. Set `ChatTemplateRequest::enable_thinking` to `Some(false)` for both
prompts. Compare with the shared `eredu-parity` GGUF profile.

```sh
artifact="$GGUF_ROOT/Qwen_Qwen3.5-0.8B-Q4_K_M.gguf"
curl --fail --location -o "$artifact" \
  https://huggingface.co/bartowski/Qwen_Qwen3.5-0.8B-GGUF/resolve/e0e7548404a0735ae4d78b07305ff8a595cdf60b/Qwen_Qwen3.5-0.8B-Q4_K_M.gguf
shasum -a 256 "$artifact"
HYBRID_FRANCE=248045,846,198,3710,369,279,6511,314,9338,30,248046,198,248045,74455,198,248068,271,248069,271
HYBRID_OCEAN=248045,846,198,7734,799,2716,11316,883,279,17415,13,248046,198,248045,74455,198,248068,271,248069,271
"$GGUF_ROOT/gguf_reference" "$artifact" "$GGUF_ROOT/hybrid-france.safetensors" \
  "$HYBRID_FRANCE" 760,6511,314 cpu
"$GGUF_ROOT/gguf_reference" "$artifact" "$GGUF_ROOT/hybrid-ocean.safetensors" \
  "$HYBRID_OCEAN" 760,17415,369 cpu
```

## Qwen3-30B-A3B Q4_K_M GGUF

The [published Qwen3-30B-A3B Q4_K_M GGUF](https://huggingface.co/bartowski/Qwen_Qwen3-30B-A3B-GGUF/blob/46f17e079cba70b04390bef39b57d2783e9fd015/Qwen_Qwen3-30B-A3B-Q4_K_M.gguf)
has SHA-256 `a015794bfb1d69cb03dbb86b185fb2b9b339f757df5f8f9dd9ebdab8f6ed5d32`.
The 18,632,184,480-byte file contains F32, Q8_0, Q4_K, Q5_K, and Q6_K
tensors. The shared driver loads it through `ArtifactSource::new` and binds
the packed expert banks with their layers. The model selects eight of
128 experts per token. Candle CPU and llama.cpp use the same file on Apple
M3 Ultra, with the pinned llama.cpp build and context settings above.

Ordinary and controlled generation match all four reference greedy decisions
for each prompt. The controlled run steps once and resumes to completion.
Each comparison covers all 151,936 vocabulary scores at the final prefill
position and after each of three cached inputs.

| User message | Thinking | Greedy tokens in both Candle modes and llama.cpp | Final prefill relative L2 | Three cached rows, maximum relative L2 |
| --- | --- | --- | ---: | ---: |
| What is the capital of France? | Template default | `[151667, 198, 32313, 11]` | `7.62e-8` | 0 |
| Write one short sentence about the ocean. | Disabled | `[785, 17951, 49599, 78737]` | `1.10e-7` | 0 |

All eight rows pass the shared `eredu-parity` profile with 5/5 top-five
overlap against the `0.02` relative L2 limit. This comparison uses resident
expert banks on one CPU device.
With every France prompt position requested from llama.cpp, all 15 prefill
rows and the three cached decode rows match bit for bit.

### Independent expert cache

On the same France prompt, the shared driver with ordinary weights resident
and an independent expert cache produces `[151667, 198, 32313, 11]` in both
ordinary and controlled generation on Candle CPU, Candle Metal, and MLX Metal.
Each starts with zero resident experts and owns 6,144 addressable experts
totaling 17,553,162,240 bytes. With a 192 MiB device budget and a 256 MiB
host budget, Candle's peak occupancies are 201,277,440 and 268,406,784 bytes,
respectively. MLX Metal peaks at 201,277,440 and 268,386,304 bytes.
On both Candle devices, the final prefill row has relative L2 error
`7.62e-8` against llama.cpp. All three cached decode rows match bit for bit,
and top-five overlap is 5/5 on all four rows.

Repeat the same prompt and four-token generation through the planned facade
with this expert-cache policy. Use each backend's local device and the exact
prompt IDs below.

```rust,ignore
let plan = ExecutionPlan::fully_resident(device).with_expert_cache(Some(
    ExpertCachePlan::new(
        Some(192 << 20),
        Some(256 << 20),
        160 << 20,
        64 << 20,
        CacheEvictionPolicy::LeastRecentlyUsed,
    ),
));
```

### Reproduce

Use the same llama.cpp build, Candle facade, and score-observation procedure
as for Qwen3-0.6B. The prompts use the same template and token IDs. Feed the
first three greedy IDs to the reference and compare its four argmaxes and
score rows with the shared GGUF parity profile.

```sh
artifact="$GGUF_ROOT/Qwen_Qwen3-30B-A3B-Q4_K_M.gguf"
curl --fail --location -o "$artifact" \
  https://huggingface.co/bartowski/Qwen_Qwen3-30B-A3B-GGUF/resolve/46f17e079cba70b04390bef39b57d2783e9fd015/Qwen_Qwen3-30B-A3B-Q4_K_M.gguf
shasum -a 256 "$artifact"
"$GGUF_ROOT/gguf_reference" "$artifact" "$GGUF_ROOT/moe-france.safetensors" \
  "$QWEN_FRANCE" 151667,198,32313 cpu
"$GGUF_ROOT/gguf_reference" "$artifact" "$GGUF_ROOT/moe-ocean.safetensors" \
  "$QWEN_OCEAN" 785,17951,49599 cpu
```

## Qwen3-VL-2B-Instruct Q4_K_M GGUF

Candle CPU and llama.cpp load the same [published decoder and F32 vision
projector](https://huggingface.co/unsloth/Qwen3-VL-2B-Instruct-GGUF/tree/8dcb98e52a1d1d02dce9249e5ab15bae8121c666)
from `unsloth/Qwen3-VL-2B-Instruct-GGUF` at revision
`8dcb98e52a1d1d02dce9249e5ab15bae8121c666`.

| File | SHA-256 |
| --- | --- |
| `Qwen3-VL-2B-Instruct-Q4_K_M.gguf` | `858fcf2a39dc73b26dd86592cb0a5f949b59d1edb365d1dea98e46b02e955e56` |
| `mmproj-F32.gguf` | `74748fdfacfd922234e90cc1f79b1beca1746d52730bb98cb1ea0580b5eeeec8` |

The released chat template renders a user turn with a 256×256 RGB image of a
blue square on a red background and the request “Describe the colors and shapes
in the image in one short sentence.” The image contributes 64 positions and
the framed prompt contributes 23 text tokens. The independent CPU reference
uses llama.cpp and mtmd from `llama-cpp-sys-2` `0.1.146` (GGML `0.9.11`), F32
KV storage, and flash attention disabled. The comparison runs on Apple M3 Ultra.

Ordinary generation and controlled generation that steps once and resumes
both match all eight llama.cpp greedy decisions. The IDs are
`[785, 2168, 4419, 264, 6437, 2518, 4004, 448]`, which decode to
“The image features a solid red background with”.

The final prefill row and seven cached decode rows each compare all 151,936
vocabulary entries. All scores match bit for bit. Relative L2 is `0` for
each row against the `0.02` limit, cosine similarity is `1`, and top-five
overlap is 5/5 throughout.

### Reproduce

Use the llama.cpp CPU build from the Q8_0 case, then compile its matching mtmd
library and the image reference runner. The commands below use macOS linker
flags.

```sh
vision="$GGUF_ROOT/qwen3-vl"
mkdir -p "$vision"
repo=https://huggingface.co/unsloth/Qwen3-VL-2B-Instruct-GGUF/resolve/8dcb98e52a1d1d02dce9249e5ab15bae8121c666
for file in Qwen3-VL-2B-Instruct-Q4_K_M.gguf mmproj-F32.gguf; do
  curl --fail --location -o "$vision/$file" "$repo/$file"
done
shasum -a 256 "$vision"/*.gguf
python3 - "$vision" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
root.joinpath("image.rgb").write_bytes(bytes(
    channel
    for y in range(256)
    for x in range(256)
    for channel in ((16, 70, 235) if 64 <= x < 192 and 64 <= y < 192 else (235, 35, 28))
))
root.joinpath("prompt.txt").write_text(
    "<|im_start|>user\n<__media__>Describe the colors and shapes in the image in one short sentence."
    "<|im_end|>\n<|im_start|>assistant\n",
    encoding="utf-8",
)
PY

lib="$GGUF_ROOT/build/bin"
mtmd="$llama_dir/tools/mtmd"
clang++ -dynamiclib -std=c++17 -O2 -I"$mtmd" -I"$llama_dir/include" \
  -I"$llama_dir/ggml/include" -I"$llama_dir/vendor" \
  "$mtmd/clip.cpp" "$mtmd/mtmd.cpp" "$mtmd/mtmd-audio.cpp" "$mtmd/mtmd-image.cpp" \
  "$mtmd/mtmd-helper.cpp" "$mtmd"/models/*.cpp -L"$lib" -Wl,-rpath,"$lib" \
  -Wl,-install_name,"$vision/libmtmd.dylib" -lllama -lggml -lggml-base -lggml-cpu \
  -o "$vision/libmtmd.dylib"
clang++ -std=c++17 -O2 -I"$mtmd" -I"$llama_dir/include" -I"$llama_dir/ggml/include" \
  validation/gguf_image_reference.cpp -L"$vision" -L"$lib" -Wl,-rpath,"$lib" \
  -lmtmd -lllama -lggml -o "$vision/gguf_image_reference"
"$vision/gguf_image_reference" "$vision/Qwen3-VL-2B-Instruct-Q4_K_M.gguf" \
  "$vision/mmproj-F32.gguf" "$vision/image.rgb" 256 256 "$vision/prompt.txt" \
  "$vision/reference.safetensors" 8
```

Enable `image` in `eredu` and `eredu-backend-candle` and disable `eredu`'s
default features. Inspect the decoder with `ArtifactSource::with_projector`
and an explicit companion path, then load using `PreparedTextModel::load_on`
with `ModelLoadFacts` and default load options. Pass the user content
as an image item followed by a text item and enable the assistant generation
prompt. Bind the RGB image to the complete rendered marker
`<|vision_start|><|image_pad|><|vision_end|>` in
`prepare_chat_multimodal_input`. The processor inserts the vision framing as
it prepares the image.

Generate eight greedy tokens with `session().generate` using that prepared
input. Reset, pass the same prompt to `prepare_controlled_input`, then call
`start_controlled`, step once, and resume to completion. For score comparison,
clone the tokenizer-bound inspection from `PreparedTextModel::artifact()`,
bind and install it with the same `ModelLoadFacts` and
`PreparedTextDriver`, and submit the prepared image prompt. Export the last
vocabulary row from completed `prefill` as `prefill.logits` and the seven
rows from `decode` with the first seven greedy IDs as `decode.logits`. Compare
the rows with the standard `eredu-parity` profile. Retain the 23 text IDs in
`input.token_ids` and the seven cached inputs in `output.fed_token_ids`.

## Qwen3-VL-2B-Instruct Q4_K_M video

The video case uses the same [decoder GGUF and F32 projector](#qwen3-vl-2b-instruct-q4_k_m-gguf)
as the image case. The input is a 1 fps clip of four 128×128 RGB frames. A
blue rectangle moves right across a red background. The decoded RGB bytes have
SHA-256 `8c786839990535d2268f8fe4de592cfce2b5e3998568037d98cd410dee3233c0`.
The prompt asks “Describe the motion in the video in one short sentence.”
Both engines form two temporal groups of two distinct frames at `0.5` and
`2.5` seconds, for 32 media positions and 35 text positions.

Candle CPU on Apple M3 Ultra matches all 16 stock llama.cpp greedy tokens.
Ordinary generation and controlled generation that steps once and resumes both
produce
`[32, 6303, 22756, 7952, 389, 264, 2518, 4004, 11, 1221, 10797, 311, 279, 1290, 323, 60270]`.
The continuation includes “then moves to the right”.

The independent reference pins [llama.cpp `71102a73f2042d9aacae2d6dc70874a8611b8361`](https://github.com/ggml-org/llama.cpp/commit/71102a73f2042d9aacae2d6dc70874a8611b8361)
(GGML `0.17.0`). CPU reference runs use FP32 KV and disable flash attention.

| Score rows | Maximum relative L2 | Minimum cosine similarity | Top-five overlap |
| --- | ---: | ---: | ---: |
| Final prefill | 0 | 1 | 5/5 |
| 15 cached decode rows | 0 | 1 | 5/5 |

All 151,936 scores per row match the reference bit for bit.

### Reproduce video

Use the `vision` artifact directory from the image case and create the decoded
clip and chat text in that directory.

```sh
python3 - "$vision" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
root.joinpath("clip.rgb").write_bytes(bytes(
    channel
    for frame in range(4)
    for y in range(128)
    for x in range(128)
    for channel in ((16, 70, 235) if 12 + 20 * frame <= x < 48 + 20 * frame
                    and 46 <= y < 82 else (235, 35, 28))
))
root.joinpath("video-prompt.txt").write_text(
    "<|im_start|>user\n<__media__>Describe the motion in the video in one short sentence."
    "<|im_end|>\n<|im_start|>assistant\n",
    encoding="utf-8",
)
PY

video_llama="$vision/llama-video"
mkdir -p "$video_llama"
curl --fail --location \
  https://codeload.github.com/ggml-org/llama.cpp/tar.gz/71102a73f2042d9aacae2d6dc70874a8611b8361 \
  | tar -xz --strip-components=1 -C "$video_llama"
cmake -S "$video_llama" -B "$video_llama/build" -DCMAKE_BUILD_TYPE=Release \
  -DGGML_BLAS=OFF -DGGML_METAL=OFF -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF \
  -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_MTMD=ON -DBUILD_SHARED_LIBS=ON
cmake --build "$video_llama/build" --target mtmd
lib="$video_llama/build/bin"
clang++ -std=c++17 -DEREDU_MTMD_VIDEO_API -O2 -I"$video_llama/tools/mtmd" \
  -I"$video_llama/include" -I"$video_llama/ggml/include" \
  validation/gguf_image_reference.cpp -L"$lib" -Wl,-rpath,"$lib" \
  -lmtmd -lllama -lggml -lggml-base -o "$vision/gguf_video_reference"
"$vision/gguf_video_reference" "$vision/Qwen3-VL-2B-Instruct-Q4_K_M.gguf" \
  "$vision/mmproj-F32.gguf" "$vision/clip.rgb" 128 128 "$vision/video-prompt.txt" \
  "$vision/video-stock.safetensors" 16 1
```

Use the same Candle load and image features as the image case. Construct a
`Video` from the four `RgbImage` frames with source FPS `1.0` and
`VideoSampling::All`. Render one video item followed by the user text and
bind the `Media::Video` to
`<|vision_start|><|video_pad|><|vision_end|>`. Request 16 greedy tokens in both
generation modes. Export the final prefill row and 15 cached decode rows as
in the image case. Use the controlled request's ordered prompt IDs with only
the video placeholders removed for `input.token_ids`, retaining timestamp
and framing IDs. Compare the exported report with `video-stock.json` using
the standard `eredu-parity` profile.

## Gemma 4 E2B audio SafeTensors

The audio-conditioned text case loads [Gemma 4 E2B Instruct](https://huggingface.co/google/gemma-4-E2B-it/tree/ebfa5cfaf9ca4e695f227d136e847f814841961c)
at revision `ebfa5cfaf9ca4e695f227d136e847f814841961c`. Both engines read
the same `model.safetensors` (SHA-256
`2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550`).
The request supplies 0.7 seconds of mono 16 kHz PCM with two successive tones.
Its little-endian F32 PCM has SHA-256
`f842918a10916dbca41302128d9425f224ad6e28dc3c177fd8ff5469733a2c81`.
The prompt asks “Describe this sound in one short sentence.” Both processors
expand the prompt to 19 text positions and 18 audio positions.

On Apple M3 Ultra, ordinary generation and controlled generation that steps
once, pauses, and resumes match all eight greedy token IDs from Transformers.
The continuation is “The sound is a low, resonant hum”. The IDs are
`[818, 5057, 563, 496, 2708, 236764, 50777, 2346]`.

| Score rows | Maximum relative L2 | Minimum cosine similarity | Minimum top-five overlap |
| --- | ---: | ---: | ---: |
| Final prefill | 0.0192703 | 0.9998985 | 5/5 |
| Seven cached decode rows | 0.0197086 | 0.9999520 | 4/5 |

Each row compares all 262,144 vocabulary scores against the `0.02` relative L2 limit.

### Reproduce audio

Place the pinned checkpoint, tokenizer, chat template, and processor
configuration in `$audio/safetensors`. Generate the clip and run the
independent Apple CPU reference with Transformers 5.22.2 and PyTorch 2.15.0.

```sh
audio="$GGUF_ROOT/gemma4"
mkdir -p "$audio"
python3 - "$audio/clip.f32" <<'PY'
import math
import struct
import sys
from pathlib import Path

samples = []
for sample in range(11200):
    frequency = 330 if sample < 5600 else 660
    position = sample % 5600
    fade = min(position / 256, (5599 - position) / 256, 1)
    samples.append(0.2 * math.sin(2 * math.pi * frequency * sample / 16000) * fade)
Path(sys.argv[1]).write_bytes(struct.pack("<11200f", *samples))
PY
python validation/gemma4_audio_reference.py "$audio/safetensors" "$audio/clip.f32" \
  "$audio/reference.safetensors"
```

Enable `audio` and `image` in `eredu` and `eredu-backend-candle` for the
checkpoint's default multimodal selection. Disable `eredu`'s default
features. Load the checkpoint directory using `PreparedTextModel::load_on`
with `ModelLoadFacts`. Use the checkpoint's chat template with one
audio item followed by the user text and an assistant generation prompt.
Read the clip into `Audio::new(samples, 16000)` and bind it as
`Media::Audio` to `<|audio|>` in `prepare_chat_multimodal_input`.

Generate eight greedy tokens using `session().generate`. Reset, prepare the
same prompt with `prepare_controlled_input`, step once, pause, and resume to
completion. For score comparison, install the tokenizer-bound checkpoint
through the same `ModelLoadFacts` and `PreparedTextDriver`. Export
the final prefill vocabulary row and the seven cached decode rows conditioned
on the reference's emitted tokens. Keep the 19 text IDs, including the audio
framing IDs, in `input.token_ids`. Compare with the shared `eredu-parity`
profile.

## Gemma 4 E2B audio GGUF

The GGUF case loads the ggml-org release
[Gemma 4 E2B Instruct](https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF/tree/2a8538c2e2d4687953a86c422c7c623cfd022880)
at revision `2a8538c2e2d4687953a86c422c7c623cfd022880`. Both engines use
`gemma-4-E2B-it-Q8_0.gguf` (SHA-256
`c36d5e6be0d91124d8b995f5ad4e1ab0d307f6d95cdd08308820350dbd96ad15`) and
`mmproj-gemma-4-E2B-it-BF16.gguf` (SHA-256
`81a9cc75a12f1cd9fa91d69896794e1791489c423a59fbf0d50cf5ad633493c0`).
The PCM clip and user message are the same as in the SafeTensors case.
The published projector declares `gemma4a` audio and `gemma4v` vision.

On macOS arm64, the shared driver and stock llama.cpp both process 19
text positions and 18 audio positions. Ordinary Candle generation and
controlled generation with a step, pause, and resume both match all eight
greedy token IDs from llama.cpp. The continuation is “Please provide the
sound you would like me” and the IDs are
`[9366, 2847, 506, 5057, 611, 1093, 1133, 786]`.

| Score rows | Maximum relative L2 | Minimum cosine similarity | Minimum top-five overlap |
| --- | ---: | ---: | ---: |
| Final prefill | 0.0187973 | 0.9998570 | 5/5 |
| Seven cached decode rows | 0.0150112 | 0.9999506 | 5/5 |

All rows compare 262,144 vocabulary scores with FP32 KV and flash attention disabled.

### Reproduce GGUF audio

Use the stock llama.cpp revision and CPU build from the video case. Save
the two GGUF files and the PCM clip under `$audio`. For the llama.cpp
runner, substitute its media marker for the Gemma chat template's audio
placeholder.

```sh
printf '<bos><|turn>user\n<__media__>Describe this sound in one short sentence.<turn|>\n<|turn>model\n' \
  > "$audio/gguf-prompt.txt"
clang++ -std=c++17 -DEREDU_MTMD_VIDEO_API -DEREDU_MTMD_AUDIO_API -O2 \
  -I"$video_llama/tools/mtmd" -I"$video_llama/include" -I"$video_llama/ggml/include" \
  validation/gguf_image_reference.cpp -L"$lib" -Wl,-rpath,"$lib" \
  -lmtmd -lllama -lggml -lggml-base -o "$audio/gguf_audio_reference"
"$audio/gguf_audio_reference" "$audio/gemma-4-E2B-it-Q8_0.gguf" \
  "$audio/mmproj-gemma-4-E2B-it-BF16.gguf" "$audio/clip.f32" --pcm-f32 16000 \
  "$audio/gguf-prompt.txt" "$audio/gguf-stock.safetensors" 8
```

Use the same Candle features and shared-driver steps as in the SafeTensors
case. Point `ArtifactSource` at the primary GGUF and select the BF16
projector with `CompanionSelection::Explicit`. Leave the primary GGUF tensor
representation at its default so the shared loader selects packed Q8_0 for
Candle. Compare the score export with `gguf-stock.json` using the shared
`eredu-parity` profile.

```sh
eredu-parity --actual "$audio/gguf-candle.json" --reference "$audio/gguf-stock.json" \
  --output "$audio/gguf-parity.json"
```

## Gemma 4 E2B vision GGUF

This still-image case uses the same Q8_0 decoder, BF16 projector, and stock
llama.cpp CPU build as the [GGUF audio case](#gemma-4-e2b-audio-gguf).
Both engines consume the same 256×256 RGB8 image. The published image
processor resizes it to 336×336 and emits 49 pooled image positions.
The reference and the shared driver agree on the 23 decoder text IDs,
including the image framing.

| Score rows | Maximum relative L2 | Minimum cosine similarity | Minimum top-five overlap |
| --- | ---: | ---: | ---: |
| Final prefill | 0.0087912 | 0.9999698 | 4/5 |
| Seven cached decode rows | 0.0069623 | 0.9999837 | 4/5 |

All 262,144 scores per row are compared on macOS arm64 and pass the
standard `eredu-parity` profile with the `0.02` relative L2 limit. Ordinary
and controlled generation with a step, pause, and resume both match the
eight stock greedy token IDs `[818, 2471, 4065, 496, 5139, 2604, 1695, 607]`.
The decoded continuation is “The image features a solid red background with”.

### Reproduce GGUF vision

Use the audio case artifact directory and stock reference executable. The
fixture has SHA-256
`cc18dbe16a41cf4c85eeace38cace110b83dc042c0163e082be4ef02e5066422`.

```sh
python3 - "$audio" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
root.joinpath("gemma-vision.rgb").write_bytes(bytes(
    channel
    for y in range(256)
    for x in range(256)
    for channel in ((16, 70, 235) if 64 <= x < 192 and 64 <= y < 192 else (235, 35, 28))
))
root.joinpath("gemma-vision-prompt.txt").write_text(
    "<bos><|turn>user\n<__media__>Describe the colors and shapes in the image in one short sentence."
    "<turn|>\n<|turn>model\n",
    encoding="utf-8",
)
PY
"$audio/gguf_audio_reference" "$audio/gemma-4-E2B-it-Q8_0.gguf" \
  "$audio/mmproj-gemma-4-E2B-it-BF16.gguf" "$audio/gemma-vision.rgb" 256 256 \
  "$audio/gemma-vision-prompt.txt" "$audio/gemma-vision-stock.safetensors" 8
```

Use the shared-driver steps from the GGUF audio case with the image feature.
Submit one image item followed by the same user text. Bind the RGB bytes as
a 256×256 `RgbImage` to the rendered `<|image|>` marker. Omit that single
placeholder from the `input.token_ids` comparison and retain the opening
and closing image framing IDs. Feed the seven reference token IDs when
exporting cached decode scores.

```sh
eredu-parity --actual "$audio/gemma-vision-candle.json" \
  --reference "$audio/gemma-vision-stock.json" --output "$audio/gemma-vision-parity.json"
```

## SmolLM2 independent drafting and control

The BF16 SafeTensors target and Q4_K_M draft pinned above have identical
tokenizer vocabularies. On Candle CPU, ten greedy tokens match target-only
generation. The draft proposes nine tokens and the target accepts six.
Ordinary and controlled drafting emit identical tokens for greedy decoding
and stochastic seeds 12 and 949. The same controlled run verifies replay
after restore, isolated branch decisions, and nonrefundable copy and capture
charges. The capture plan reads raw top candidates and selected token scores
for both roles.

The [MLX acceptance driver](../validation/mlx_speculation.rs) uses the same
checkpoint pair and prompt with up to three proposals per round. On MLX
CPU and Metal, target-only, ordinary speculative, and controlled speculative
generation emit `[6403, 1980, 253, 655, 28, 281, 253, 18840, 1838, 2291]`.
The target accepts six of nine proposals.

Run the [acceptance driver](../validation/candle_speculation.rs) using the two
pinned local artifacts. The probe and all build outputs can live outside the
checkout.

```sh
repo="$PWD"
probe=$(mktemp -d)
mkdir "$probe/src"
ln -s "$repo/validation/candle_speculation.rs" "$probe/src/main.rs"
cat > "$probe/Cargo.toml" <<MANIFEST
[package]
name = "candle-speculation-acceptance"
version = "0.1.0"
edition = "2021"

[dependencies]
eredu = { path = "$repo/eredu", default-features = false, features = ["tensor-capture"] }
eredu-backend-candle = { path = "$repo/eredu-backend-candle" }
eredu-core = { path = "$repo/eredu-core" }
serde_json = "1"

[profile.dev.package.eredu-backend-candle]
opt-level = 3

[profile.dev.package.candle-core]
opt-level = 3
MANIFEST
cargo run --manifest-path "$probe/Cargo.toml" -- "$BF16_DIR" "$GGUF_FILE"
```

For split Candle placement, enable the `metal` feature on
`eredu-backend-candle` in that manifest. Positions three and four select the
target and explicit draft contexts.

```sh
cargo run --manifest-path "$probe/Cargo.toml" -- "$BF16_DIR" "$GGUF_FILE" metal:0 cpu:0
cargo run --manifest-path "$probe/Cargo.toml" -- "$BF16_DIR" "$GGUF_FILE" cpu:0 metal:0
cargo run --manifest-path "$probe/Cargo.toml" -- "$BF16_DIR" "$GGUF_FILE" cpu:0 cpu:0
```

The two cross-device directions and separate CPU contexts each match
target-only greedy generation, with six accepted tokens out of nine
proposals. Ordinary and controlled generation match for seeds 12 and 949.
The driver checks the reported split topology, cancellation, snapshot
replay, and isolation of a fork with a forced token.

For MLX, create a separate external Cargo project with `validation/mlx_speculation.rs`
as its `src/main.rs`. Set `mlx_probe` to that project directory and use these
dependencies.

```toml
[dependencies]
eredu = { path = "/path/to/eredu/eredu", default-features = false, features = ["mlx", "metal", "tensor-capture"] }
eredu-backend-mlx = { path = "/path/to/eredu/eredu-backend-mlx", default-features = false }
eredu-core = { path = "/path/to/eredu/eredu-core" }
serde_json = "1"
```

```sh
cargo run --manifest-path "$mlx_probe/Cargo.toml" -- "$BF16_DIR" "$GGUF_FILE" metal:0
cargo run --manifest-path "$mlx_probe/Cargo.toml" -- "$BF16_DIR" "$GGUF_FILE" cpu:0
```

## Embedded prediction and conditioned assistants

The embedded case uses [Qwen3.5-0.8B SafeTensors](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17).
The model shard SHA-256 is
`04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`.
The conditioned case uses the Gemma 4 E2B Q8_0 GGUF pinned above and the
[released E2B Instruct assistant](https://huggingface.co/google/gemma-4-E2B-it-assistant/tree/2d874ef7d29f9a30599a1e4b3c1cbc9595f005df).
The assistant weight SHA-256 is
`93682eb1c97639d18f007704dc880bd74cbe530adaf7b1bb561213863fdad2a6`.

For the conditioned assistant, Candle also passes ordinary and controlled
generation with CPU and Metal in both directions and with separate Metal
contexts on the same device. All three placements accept four of four
proposals in each sampling mode. Greedy output matches the target-only
sequence, and snapshots replay the controlled output.

Both use the rendered prompt “What is the capital of France?” and compare
ordinary and controlled speculation. Greedy output also matches target-only
generation. Seeded sampling uses seed `12`, temperature `0.85`, top-k `32`,
top-p `0.95`, and min-p `0.05`.

| Realization | Greedy proposals accepted | Seeded proposals accepted | Greedy committed IDs |
| --- | ---: | ---: | --- |
| Qwen3.5 embedded | 2/2 | 2/3 | `[760, 6511, 314, 9338, 369]` |
| Gemma 4 E2B conditioned | 4/4 | 4/4 | `[818, 5279, 529, 7001, 563, 5213]` |

The embedded case verifies phase-scoped capture and an editable BF16
predictor readout. Both cases check snapshot replay. The embedded case also
checks cumulative budgets across restore and fork isolation.

Use the external acceptance project above. Add
`eredu-runtime = { path = "$repo/eredu-runtime" }` to its `[dependencies]`,
enable the `image` feature on both `eredu` and `eredu-backend-candle`, and
replace the `src/main.rs` symlink with the
[advanced acceptance driver](../validation/candle_advanced_speculation.rs).
Run with local copies of the pinned artifacts.

```sh
cargo run --manifest-path "$probe/Cargo.toml" -- \
  "$QWEN35_BF16_DIR" "$GEMMA4_E2B_Q8_FILE" "$GEMMA4_ASSISTANT_DIR"
```

For Metal, enable the backend's `metal` feature and append `metal:0`.
The embedded and conditioned cases match ordinary and controlled generation
on this placement and replay their snapshots. To run only the conditioned
case with split placement, supply `-` and both devices.

```sh
cargo run --manifest-path "$probe/Cargo.toml" -- \
  - "$GEMMA4_E2B_Q8_FILE" "$GEMMA4_ASSISTANT_DIR" metal:0 cpu:0
```

## Metal media and routed weights

The released-checkpoint probes use the same artifact, projector when applicable,
prompt, and feed tokens as the corresponding CPU cases above. Candle Metal on
Apple M3 Ultra returns the llama.cpp greedy continuation in ordinary and
controlled generation for all four cases. Each score comparison includes the
whole vocabulary for the final prefill row and the indicated cached decode rows.

| Fixture | Prefill relative L2 | Maximum cached relative L2 | Matching greedy tokens |
| --- | ---: | ---: | ---: |
| Qwen3-30B-A3B Q4_K_M, France prompt | 7.62e-8 | 0 across 3 rows | 4/4 |
| Gemma 4 E2B Q8_0, image | 0.00652079 | 0.00742659 across 7 rows | 8/8 |
| Gemma 4 E2B Q8_0, audio | 0.01879730 | 0.01501118 across 7 rows | 8/8 |
| Qwen3-VL-2B Q4_K_M, video | 0 | 0 across 15 rows | 16/16 |

For media score comparison, use the same reference commands and fixtures as
above and open `CandleContext` with `DevicePlan::new("candle", "metal:0")` in
the prepared driver. All uses of a prepared media prompt share the same opened
context. For routed weights, the CLI accepts `--backend candle --device metal:0`
and uses the same retained checkpoint representation.
