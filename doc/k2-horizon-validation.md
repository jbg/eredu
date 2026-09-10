# K2 Horizon implementation and validation

The dense, MoE and MoVA implementations are integrated with SafeTensors/GGUF
loading, bounded expert residency, partitioned execution, tokenization and shared
generation/control drivers. Portable and native conformance passes as described
below. The released dense and MoVA SafeTensors BF16 comparisons now match the
pinned publisher implementation exactly for prefill and four cached decode steps.
Their official BF16 GGUFs, short natural-text prompts, and complete tool-call
prompts also pass the unchanged numerical criteria, with bit-exact CPU and Metal logits.
Extended speculative checks are recorded below.

## Sources and artifacts

| Artifact | Immutable revision |
| --- | --- |
| IFM/K2-Horizon-0.9B | `02d0da0fefe5a2f8dc3db091cad29b15c9d8e4fa` |
| IFM/K2-Horizon-MoVA-36B-A4B | `5d624db156710b3dc30fcca0a5cc25cb2898e110` |
| IFM/K2-Horizon-0.9B-GGUF | `8496259ac62d33192d0fffe708303ed2ca29a384` |
| IFM/K2-Horizon-MoVA-36B-A4B-GGUF | `c8dde8bc6afe5b28e75c7d78db4fbd65d5fcf679` |

Reference files and checkpoint payloads reside outside the tracked tree at
`/private/tmp/eredu-k2-horizon`. Downloads compare payload sizes and SHA-256
against the immutable Hugging Face manifest before renaming partial files.
The publisher GGUF converter is pinned to llama.cpp revision
`35999d101cf2233fc54f09c3c8d599da7303ce02`.
Sibling configuration fixtures retain their own revision and source hashes in
`eredu-architectures/tests/fixtures/k2_horizon/siblings`.

Validation host: Apple M3 Ultra, 256 GiB RAM, macOS 26.6.2, arm64.
The isolated reference environment uses Python 3.14.7, PyTorch 2.14.0,
Transformers 5.16.1 and NumPy 2.5.3. Publisher models run with eager attention,
CPU execution, eight PyTorch threads and local-only artifact resolution.
The dense publisher modeling source SHA-256 is
`fb09e010956bd51cfa7d4055b4381cff34c9e06164066b49e3546f38b2e6242f`.

## Equation and format checks

The independent NumPy generator does not import Eredu. Nonzero fixtures cover
dense GQA, grouped RMS normalization, distinct per-head Q/K scales, routed
SwiGLU, always-on shared experts, routed activated values, selection-only router
bias, distinct value/feed-forward coefficient normalization, partial rotary
pairing, YaRN amplitude and the softplus attention-output gate.

Portable numerical tests compare complete prefill, chunked prefill and four
cached teacher-forced decode steps using `atol=1e-5`, `rtol=1e-4`. Separate checks
compare routed expert IDs, original scores and mixture coefficients. Two-rank
TP preserves full value-projection inputs and partitions output rows by KV head.
Pipeline tests cut immediately before and after the dense-to-MoVA transition.
The registered checkpoint-backed matrix covers dense, grouped-normalization,
MoE and MoVA configurations. MoE/MoVA run TP2, PP2, TP2/PP2, EP2,
TP2/EP2, PP3/EP2 and TP2/PP3/EP2, each with resident, host-layerwise and
disk-streamed ordinary weights. Both bank placements are covered: parameters
loaded with each block and independently addressable experts in one shared
1,152-byte cache. The latter reads exact individual SafeTensors recipes,
applies the neutral rank-local binding projection, and verifies cross-bank
occupancy, completion, eviction and reacquisition. Every run compares prefill
and four cached decode steps to the independent oracle and verifies reset.
The fixture serializes the oracle's packed matrices into individual experts,
including the same selection-only router biases.

Native CPU Ring validation now passes 14 resident/ordinary-weight distributed
cases for dense K2 and MoVA: TP2, PP2, TP2/PP2, host-layerwise TP2,
disk-streamed PP2, and MoVA EP2, TP2/EP2, PP2/EP2 and TP2/PP2/EP2.
Eight further MoVA cases use independently addressable banks, including host
and disk ordinary weights under TP2/PP2/EP2. These compare resident reference
logits, distributed prompt-cache restore and continued decoding, and verify
rank-local ownership of both unequal expert banks within a shared 1,152-byte
cache. Local loopback sockets require running the native test outside the sandbox.

The scalar partition adapter now uses the shared bounded runtime constructor;
its evidence verifies one loaded block at a time for host/disk execution.
Collective-wave tests additionally check different bank cardinalities and value
output widths, bank order within a block, and cuts around the dense prefix.
Ordered expert exchange has an adversarial cancellation fixture and a native
BF16 regression for per-expert accumulation after reverse exchange.

The native MoVA test passes full prefill, four cached steps, paged KV storage,
checkpoint rollback and prompt-cache persistence/reload under resident and
addressable execution with resident, host-layerwise and disk-streamed ordinary
weights. The test limits paged device KV storage to 768 bytes, below the full
seven-token state, and verifies host demotion and promotion. Both expert banks
share one residency budget. Per-bank identities and counters
remain distinct, aggregate occupancy stays within that budget, and a generic
native regression forces cross-bank eviction/reacquisition with exact companion
recovery. Rollback testing found and fixed a shared paged-cache defect: restoring
a saved partial tail previously also left a sealed copy of that same token range.
The architecture conformance rerun after the shared-pool/exchange changes passes
154 numerical and 46 structural tests. Seventeen native bank-residency tests and
sixteen neutral expert-exchange tests also pass. The full AGENTS.md portable
build matrix and portable facade/backend conformance rerun also passes.
The final rerun includes 533 architecture unit tests, 312 runtime unit tests,
64 neutral session conformance tests, 623 native unit tests, and the 34 native
distributed cases.
Commands, toolchain versions, exit codes and log hashes are recorded in
[`k2_horizon_checks.json`](../eredu-backend-mlx/validation/k2_horizon_checks.json).

Nonzero FP8 execution compares both banks and explicitly excluded parameters
against the pinned publisher model with independently expanded F32 weights and
E4M3 activation quantization hooks. A weight-only oracle omitted the declared
dynamic activation quantization; the corrected CPU fallback and independent
oracle now agree with the GPU equation. The fixture records the exact input
artifact hash, publisher hash and per-projection quantization policy in
`validation/k2_fp8_reference.json`. Resident, host-layerwise and
disk-streamed runs pass prefill and four cached steps. The independently
addressable banks use a 393,312-byte cache and force eviction/reacquisition.
Nonzero Q4_0, MXFP4 and IQ4_NL GGUF fixtures likewise pass all five steps
against explicit block-arithmetic expansion under those residency modes with
a 16,384-byte expert cache. All nine quantized TP2/PP2/EP2 cases and all
three FP8 TP2/PP2/EP2 cases also pass under resident/host/disk ordinary weights.
These tests found and fixed missing packed companion recipes, logical-width
retention for derived native byte blocks, contiguous IQ4_NL byte-offset mapping,
and fused FP8 scale partitioning in block-row coordinates. A neutral bounded-read
regression also covers Q8_0, whose byte-block size exceeds its logical block width.

All four released artifacts are downloaded and hash-verified. Their SafeTensors
and BF16 GGUF headers pass strict family name/shape/catalog checks. The released
dense SafeTensors weights are BF16 despite the config's float32 declaration.
The complete pinned 7B FP8 header also passes companion/dtype validation;
compressed-tensors uses `weight_scale`, while MoVA block-FP8 uses
`weight_scale_inv`. Both are multiplicative dequantization scales. File sizes,
revisions and payload hashes are retained in
[`k2_horizon_artifacts.json`](../eredu-backend-mlx/validation/k2_horizon_artifacts.json).

## Released dense numerical results

The input IDs are `[1, 3, 2]` for prefill, followed by teacher-forced IDs
`[4]`, `[5]`, `[6]`, `[7]`. Full-checkpoint FP32 comparison passes all five steps;
the decoded-step relative L2 error is below 0.0002%, cosine similarity exceeds
0.99999999999 and all ten leading tokens overlap.

The complete official dense BF16 GGUF also passes the FP32 comparison after
strict admission and canonical tensor mapping: maximum relative L2 is
`0.0000030729697779130448`, minimum cosine `0.9999999999971694`, and all ten leading
tokens overlap at every step. This tests the released BF16 payload widened to
FP32 for execution. The complete official MoVA GGUF also passes all five steps:
maximum relative L2 `0.000011440136332692942`, minimum cosine
`0.9999999999386868`, with ten overlapping leading tokens in every row.

The CPU and Metal BF16 comparisons pass without changing the required tolerances.
Dense and MoVA SafeTensors and their official BF16 GGUFs produce bit-exact logits
for prefill and all four cached decode steps: relative L2 zero, cosine one,
top-ten overlap ten and matching argmax. The full MoVA capture also matches every
layer output and routing record.

The tolerance failures came from shared numerical mechanisms. CPU matrix
products now accumulate complete FP32 dots and preserve row-projection strides
when selecting a bank member. Dropping the selected bank axis uses a squeeze;
reshaping a transposed view can copy it and change the reduction order. Grouped
normalization uses a cascade sum, division by the group width and the reference
reciprocal-square-root rounding. Bounded resident attention tiles query rows
while preserving complete key rows. Explicit rotary products retain the inverse
frequencies instead of reconstructing their denominators.

FP32 softmax and sigmoid retain the pinned exponential polynomial and arithmetic
order. The previous symmetric sigmoid formula differed by one FP32 ULP in some
scores. In the six-token natural prompt, that rounded one layer-20 routing
coefficient differently and later changed a greedy choice. The corrected trace
matches that layer's scores, coefficients, expert IDs and output exactly.

Exact router cutoff ties use a value-only descending partition. In the original
MoVA layer-10 trace, experts 52 and 83 had equal selection scores; choosing the
other expert changed the feed-forward output despite exact normalization.
Independent PyTorch fixtures exercise dense and selected row projections,
adversarial cancellation, grouped normalization, default and YaRN rotary
products, masked softmax, bounded attention, sigmoid coefficients, and tied
top-k selection. Their generators, tensor hashes and dependency versions are
retained under `eredu-backend-mlx/validation`.

The final released matrix passes all sixteen runs: eight each on CPU and Metal.
Each device covers both SafeTensors checkpoints, both official BF16 GGUFs, both
natural-text prompts and both complete tool prompts. Every compared logit is
bit-exact. Metal uses generic BF16 row projections and batched products with
complete FP32 accumulation, cascade normalization and the same FP32
softmax/sigmoid arithmetic. Exact ties spanning a router cutoff use the
value-only CPU partition of routing scores; expert parameters remain on device.
The 86 native neural operator tests also pass, including independent attention
fixtures with short, masked and long rows.

The host has one physical GPU. Native multi-GPU, CUDA and NCCL distributed runs
were unavailable; the neutral numerical matrix and native multi-process CPU
Ring matrix cover the supported partition semantics.

The acceptance policy remains relative L2 ≤ 2%, cosine ≥ 0.999, top-ten overlap
≥ 8, and matching argmax when the reference margin exceeds 0.5. Short natural-text
and tool prompts use the same policy and also require every greedy choice to
match the publisher.

Reproducible commands, with `$K2` pointing to the external artifact directory:

```sh
HF_MODULES_CACHE="$K2/modules" /private/tmp/eredu-qwen-reference-env/bin/python \
  eredu-backend-mlx/validation/k2_horizon_reference.py \
  "$K2/artifacts/dense" "$K2/dense-reference.safetensors"
cargo run --release -p eredu-backend-mlx --features metal \
  --example k2_checkpoint_parity --offline -- \
  "$K2/artifacts/dense" "$K2/dense-reference.safetensors"
EREDU_VALIDATION_DEVICE=gpu target/release/examples/k2_checkpoint_parity \
  "$K2/artifacts/dense-gguf/K2-Horizon-1B-BF16.gguf" \
  "$K2/dense-reference-fp32.safetensors" float32
EREDU_VALIDATION_DEVICE=gpu target/release/examples/k2_checkpoint_parity \
  "$K2/artifacts/mova-gguf/K2-Horizon-36B-BF16.gguf" \
  "$K2/mova-reference-fp32.safetensors" float32
EREDU_VALIDATION_DEVICE=gpu target/release/examples/k2_checkpoint_parity \
  "$K2/artifacts/dense" "$K2/dense-reference.safetensors"
EREDU_VALIDATION_DEVICE=gpu target/release/examples/k2_checkpoint_parity \
  "$K2/artifacts/mova" "$K2/mova-reference.safetensors"
EREDU_VALIDATION_DEVICE=gpu target/release/examples/k2_checkpoint_parity \
  "$K2/artifacts/dense-gguf/K2-Horizon-1B-BF16.gguf" \
  "$K2/dense-reference.safetensors"
EREDU_VALIDATION_DEVICE=gpu target/release/examples/k2_checkpoint_parity \
  "$K2/artifacts/mova-gguf/K2-Horizon-36B-BF16.gguf" \
  "$K2/mova-reference.safetensors"
cargo test -p eredu-architectures --test reference_numeric k2_horizon --offline
EREDU_K2_ARTIFACTS="$K2/artifacts" EREDU_K2_VALIDATE=dense \
  cargo test -p eredu-architectures --test k2_horizon --offline -- --include-ignored
```

For CPU, omit `EREDU_VALIDATION_DEVICE=gpu`. Reference generation uses the
same script for dense and MoVA. Natural-text and tool cases retain exact prompt
bytes, token IDs, reference settings and per-step metrics in the released-results
manifest; its commands identify the retained validation binaries and their hashes.

## Tokenizer, templates and IFM protocol

All 64 exact rendered-prompt and token-ID cases pass for the four released
artifacts. SafeTensors token IDs come from each artifact's own Transformers
tokenizer; GGUF token IDs come from the pinned publisher `llama-tokenize`, built
with CPU operators and vocabulary-only loading. SafeTensors retains NFC
normalization. Published GGUF preserves decomposed combining marks. Dense and
MoVA use different vocabularies and BOS/EOS spellings. These are format-specific
behaviors, not interchangeable tokenizer configurations.

The Python/Jinja fixtures cover high/medium/low effort, assistant reasoning
history, tool results, JSON/XML/typed-XML calls and all three tool-presentation
formats. Shared rendering now matches Transformers' whitespace settings and
Python dictionary construction from iterable pairs. IFM parsing and constraints
use declarative envelopes, recognize rendered behavior without architecture
names, preserve literal argument newlines, validate type annotations, and pass
every byte split and parser fork across parallel calls. Incomplete calls never
produce completion events. The published GGUF template supports disabling
reasoning; the two SafeTensors templates do not expose that control and return
a typed preparation error for that request.

Fixtures and reference versions/hashes are in
`eredu-text/tests/fixtures/k2_horizon`. The pinned llama.cpp archive SHA-256 is
`7420149d96e3aeeb3be0ee050d9b9d0371fedce7fef002caf999247c6a41cf61`.
The parser's partial type annotations participate in snapshot storage accounting.

Both pinned publisher checkpoints also produced a complete XML call to
`get_weather` with `{"city":"Paris"}` from the same tool request. Exact rendered
prompts, generated token IDs and reference hashes are recorded in
`eredu-text/tests/fixtures/k2_horizon/tool-generation.json`. On both CPU and Metal, the dense
run matches all 22 greedy choices and the MoVA run matches all 21. Both models
produce bit-exact logits for the complete tool prefill and every cached step.
The short natural-text runs likewise match all five choices and all logits.
Per-step metrics, commands, reference hashes and log hashes are retained in
[`k2_horizon_tool_results.json`](../eredu-backend-mlx/validation/k2_horizon_tool_results.json)
and [`k2_horizon_released_results.json`](../eredu-backend-mlx/validation/k2_horizon_released_results.json).
Every byte split of both publisher-generated calls passes the shared IFM parser.
The generated-call check does not replace numerical conformance.

```sh
HF_MODULES_CACHE="$K2/modules" /private/tmp/eredu-qwen-reference-env/bin/python \
  eredu-backend-mlx/validation/k2_horizon_reference.py \
  "$K2/artifacts/mova" "$K2/mova-tool-reference.safetensors" \
  --chat-request "$K2/tool-request.json" --greedy --decode-steps 64 --stop-at-eos
EREDU_VALIDATION_PREFILL_TOKENS=181 EREDU_VALIDATION_GREEDY=1 \
  target/release/examples/k2_checkpoint_parity \
  "$K2/artifacts/mova" "$K2/mova-tool-reference.safetensors"
cargo test -p eredu-backend-mlx --no-default-features --lib ring_k2_ --offline \
  -- --ignored --nocapture --test-threads=1
```


```sh
cmake -S "$K2/llama.cpp-35999d101cf2233fc54f09c3c8d599da7303ce02" \
  -B "$K2/llama-build" -DGGML_METAL=OFF -DGGML_ACCELERATE=OFF \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON
cmake --build "$K2/llama-build" --target llama-tokenize -j 12
/private/tmp/eredu-qwen-reference-env/bin/python \
  eredu-text/tests/fixtures/k2_horizon/generate_reference.py "$K2"
EREDU_K2_ARTIFACT_ROOT="$K2/artifacts" cargo test -p eredu-text \
  --test k2_horizon --offline -- --include-ignored
cargo test -p eredu --no-default-features --lib ifm_ --offline
```

## Controlled execution

Native MoVA control tests now pass for resident and independently addressable
banks with device and paged KV state. They capture a prefix, advance four steps,
restore and fork it, intervene on value routing and feed-forward routing
independently, and compare the restored continuation exactly. Restoring state
preserves cumulative bank requests. Paged copies use independent namespaces in
one shared finite pool, and snapshot estimates include sealed blocks. The test
found and fixed copying of strided paged tails; separate full/sliding cache
conformance compares interleaved branches with a scalar attention oracle.

Distributed state-copy preparation uses the retained bounded failure-agreement
mechanism before publishing a copy or exchanging branches. Native dense/MoVA
and quantized GGUF branch schedules compare four continued decode steps,
mid-continuation snapshots and reuse of immutable snapshots after descendants
advance. The expanded FP8 branch schedule requires more host capacity than its
ordinary run: a 32 KiB host limit produced coordinated copy failure, and the
branch matrix reserves finite 1 MiB host and 128 KiB device tiers. The 34
distributed ordinary/bounded/quantized/FP8 cases all pass their branch schedules.
Neutral tests verify remote copy/exchange failure fences retries and leave
installed state unchanged. Commit epochs stay monotonic across branch exchange,
manual rollback and prompt-cache restore. All 64 neutral backend-independence
tests pass.

Three public-facade CPU tests cover dense K2 and MoVA: controlled/uninterrupted
sampling with resident, host-layerwise and disk-streamed weights; restoration and
forking during partial Unicode output after deleting the loaded artifacts; and
bounded captures with an unchanged and a modified continuation. These pass
through the existing public generation and control drivers. The rebuilt full
facade suite passes all fifteen tests, including existing family and assistant
regressions and the K2 speculative matrix below.

The public distributed facade matrix also passes six cases: dense TP2/PP2 and
MoVA TP2/PP2/EP2 with resident, host-layerwise and disk-streamed ordinary weights.
MoVA uses independently acquired banks. Each case compares greedy and seeded
sampled uninterrupted/control runs, restores a snapshot, exchanges a fork and
verifies that copying continues to consume the cumulative budget. The test lives
above the facade in `eredu-cli`; production backends do not depend on the facade.
Distributed facade cases cover continuation and state control. Bounded activation
capture remains subject to the current shared single-rank capture capability.

```sh
cargo test -p eredu-backend-mlx --no-default-features --lib k2_mova_native_control --offline
cargo test -p eredu-backend-mlx --no-default-features --lib isolated_ --offline
cargo test -p eredu --no-default-features --features mlx --test native_execution_control k2_ --offline
cargo test -p eredu-cli --no-default-features --features mlx --test k2_distributed_control --offline \
  -- --exact k2_distributed_facade_control_matrix --ignored --nocapture
```

## Cold resources and speculative execution

Automatic schema version 7 retains the unsharded artifact profile and a separate
selected-rank profile. Cold tests cover TP/PP/EP combinations, repeated/static
parameter owners, replicated routers, shared experts, each bank's unequal expert
cardinality, physical quantization companions and rank-local KV heads/layers.
The tensor-free parameter description used for these reports is compared with
the module's actual parameter description.

Materialization workspace reports distinguish ordinary recipes, individual
expert recipes, compact-bank assembly, native source/index/copy buffers and
load-time conversion. Ordinary queue concurrency includes foreground work;
converted outputs and independently prepared expert buffers remain explicit.
The report conservatively counts tensor buffers even if a native allocator can
reuse them. It does not claim to bound opaque driver bookkeeping, allocator
caches or request-sized attention activations. Missing native estimates remain
unknown instead of being reported as zero.

K2 checkpoints publish no embedded prediction head. They can now pair with an
ordinary tokenizer-compatible decoder through the existing external drafting
plan. Architecture preparation retains the exact ordinary sources, lowering,
placement and tokenizer proof. The runtime's independent-draft transaction uses
private draft branches, complete target verification logits, accepted-prefix
replay and joint target/draft rollback. Native binding supplies ordinary decoder
and isolated-state operations. Feature-conditioned Gemma4 and Muse-Glimmer
assistants retain their own architecture compatibility requirements.

The scalar dense/MoVA test verifies every accepted prefix of a four-token target
block against an independent oracle, private branch isolation and restoration of
both caches. The public native facade matrix passes fourteen cases: dense and
MoVA with resident, host-layerwise and disk-streamed weights under greedy and
seeded sampling, plus edited branches for both families. MoVA uses independently
addressable banks in a shared 1,152-byte cache. Resident cases use separate CPU
streams and adaptive lookahead. Controlled output matches uninterrupted output;
target/draft logit captures survive snapshot/replay, and restoring state continues
to charge cumulative copy budgets. Edited branches force a token, modify target
and draft logits independently, reject proposals, exchange a fork and restore the
parent's sampling state. The greedy MoVA fixture terminates at its first EOS;
seeded cases exercise complete speculative rounds and replay.
