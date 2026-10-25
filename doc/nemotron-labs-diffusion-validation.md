# Nemotron-Labs-Diffusion numerical evidence

`eredu-architectures::nemotron_labs_diffusion` owns the explicit text geometry,
SafeTensors source names, shared text invocation equations, Pixtral image layers,
spatial projector, and image preparation policy. The neutral architecture registry
admits SafeTensors text and VLM configurations and retains their prepared construction
and parameter catalogs. These modules remain available without selecting a concrete backend.

Text invocations carry immutable attention visibility, cache-write policy, adapter
scope, and score semantics. Causal calls append to the canonical cache. Diffusion
calls borrow a read-only cache and attend to the prefix plus the complete supplied
block. Both use the same static modules and independently addressable layers.
Parameter storage is separate from the transient canvas and cache state.

Low-level text construction uses `TextKernel::from_definition`,
`from_parallel_definition` or `from_partition` with an explicit retained text
declaration. `definition::text` and `definition::text_partition` declare these
resources without a native context. Executable format selection uses
`definition::project_text_parameters` and retains the same operation and state
identities.

Low-level composite construction uses `CompositeKernel::from_definition`,
`from_parallel_definition` or `SelectedCompositeKernel::from_partition` with an
explicit parent declaration. `definition::project_composite_parameters` selects
executable formats while retaining logical consumers and observation seams.
Complete, parallel and partitioned construction validate the parent's equation
paths, state references, transport and parameter topology before materialization.
Transform sources retain source-format placement independently of the target.

Composite text kernels obtain their child declarations from the parent through
`ComponentDefinition::project_groups`. Parameter ownership and observation seams
remain shared semantic facts; local context dimensions use the ordinary validated
state projection.

The dense attention equation uses an explicit independent `head_dim`, YaRN
with input-precision rotary products and eager input-score/probability rounding,
and the published position-dependent query gain
`1 + beta * log(1 + floor(position / original_max_position_embeddings))`.
NVIDIA's `encoder.*` and `diffusion_head.*` names map to one canonical parameter
catalog through source-only recipes. Recipes preserve physical tensor ancestry.

## Loaded execution evidence

The normal loader also passes the pinned IntellAgents 3B Q4_K_M GGUF through
causal prefill and three cached steps, with independent decoded-artifact logit
maximum errors 0.00006962, 0.00004315, 0.00003433 and 0.00003195. The disk-streamed
physical driver then settles AR → ordinary 32-position diffusion → AR at generated
position 34. Resumed token 11 agrees with fresh causal execution over the exact
committed prefix. The causal comparisons use the independent reference; the resumed
token comparison checks settlement against ordinary execution.

The pinned spiritbuun 14B Q8_0 artifact passes the same normal disk-streamed
loader sequence with optimized conversion. Prefill and three cached decode
maximum logit errors are 0.00010586, 0.00007629, 0.00007820 and 0.00007486 against
the independently decoded quantized artifact. AR → diffusion → AR settles at
generated position 34, and resumed token 11 agrees with fresh causal execution.
Four greedy shared self-speculative tokens also agree with AR. This is CPU
execution with decoded F32 parameters; it does not claim direct packed Q8 kernels.

The pinned suzukimain 14B Q4_K_M artifact also passes this disk-streamed normal
loader sequence. Its four causal maximum errors are 0.00008106, 0.00007010,
0.00009060 and 0.00008488. The position-34 handoff, resumed causal token and four
greedy self-speculative tokens agree with ordinary execution. The source retains
its published mixed K-quant encodings; the selected bounded neural execution
decodes projections to F32.

```sh
CARGO_INCREMENTAL=0 cargo build -p eredu-backend-mlx --release \
  --no-default-features --features metal --example nemotron_generation_modes
target/release/examples/nemotron_generation_modes \
  PINNED_SPIRITBUUN_14B_Q8_0_GGUF INDEPENDENT_REFERENCE_F32_SAFETENSORS disk
```

```sh
cargo run -p eredu-backend-mlx --no-default-features --features metal \
  --example nemotron_generation_modes -- \
  PINNED_INTELLAGENTS_3B_Q4_K_M_GGUF INDEPENDENT_REFERENCE_F32_SAFETENSORS disk
```

Execution state, controlled advancement, cancellation, speculation, snapshot
isolation and consumed budgets use the shared conformance drivers described in
[family acceptance](family-acceptance.md). Numerical acceptance uses a complete
model and an independent pinned reference. Native distributed released-checkpoint
parity is unvalidated unless explicitly recorded below.

## Pinned admission evidence

The fixture catalog in
`eredu-architectures/tests/fixtures/nemotron_labs_diffusion/releases.json` records
the configuration digests, published tensor header shapes, weight sizes and LFS
SHA256 digests for all seven official releases. Tokenizer, template and processor
assets also record their published SHA256 or Git blob digest. Independent
reference scripts verify those pins before importing checkpoint code.

| NVIDIA release | Revision |
| --- | --- |
| 3B | `0d51902da1f8869f83413ce642fab402fa5641e0` |
| 3B-Base | `4bd78816ba988ca8b71b8f432b00b682572b236c` |
| 8B | `16c67f0560b912e93e0cabb6e0c4f5c3086d95fc` |
| 8B-Base | `59ff0ffee284112fc6ccf37493ced41a11031434` |
| 14B | `f8c3e2c078e193599b8882d965b1001c456ba738` |
| 14B-Base | `ca30a6db6274c0b01d4eb21e6f1cb28222ea0c63` |
| VLM-8B | `adca93d16471c1e07d594ae444d23e1876f6b365` |

Strict header admission checks every required tensor, including the VLM vocabulary
of 131073, convolution-shaped patch kernel, 24 individually addressed vision
layers, and learned spatial merger. The 3B and 14B query projection widths are
4096, independently of their residual widths of 3072 and 5120.

## Released 3B comparison

The complete NVIDIA 3B artifact is outside tracked source at
`/private/tmp/eredu-nemotron-reference/3B`. Its `model.safetensors` is
7,663,347,000 bytes with SHA256
`8c4beff3a8a2ec76708f33e67d18c8baecf099ebd5aa5a152ea1cce494b2b51c`.
The independent reference executes the Python implementation from the pinned
checkpoint using PyTorch 2.10.0 and Transformers 5.0.0. The published BF16
parameters are decoded to FP32 in both implementations. Native execution uses
MLX CPU with FP32 parameters; this evidence does not establish BF16 execution
tolerances or native distributed correctness.

```sh
python eredu-evaluation/scripts/nemotron_diffusion_reference.py \
  /private/tmp/eredu-nemotron-reference/3B \
  /private/tmp/eredu-nemotron-reference/3B/reference-f32.safetensors
cargo run -p eredu-backend-mlx --no-default-features \
  --example nemotron_diffusion_reference -- \
  /private/tmp/eredu-nemotron-reference/3B \
  /private/tmp/eredu-nemotron-reference/3B/reference-f32.safetensors
```

The reference prefills `[1, 100, 200]`, then causally appends `[17]`, `[31]`, and
`[421]`. After each causal invocation it executes a full 32-position diffusion
canvas, with IDs 101 and 211 fixed at the first and last positions and the
published mask ID elsewhere. Every layer and every logit row is compared with an
absolute FP32 threshold of 0.002 for logits. Layer activations use absolute
tolerance 0.002 plus relative tolerance 0.00002 to account for FP32 accumulation
at large residual magnitudes. All comparisons pass. Maximum causal logit
error is 0.00006223; maximum diffusion logit error is 0.00009656. Maximum layer
error is 0.00048828, on a large-magnitude residual activation. The four canonical
cache frontiers are 3, 4, 5, and 6; transient calls preserve them.

### Normal loader with BF16 and the published draft adapter

The ordinary loader binds the pinned 3B BF16 base and separate FP32
`linear_spec_lora` parameters under host-layerwise residency. Four causal
invocations each alternate with a complete 32-position ordinary diffusion and
adapted draft invocation. The causal frontiers remain 3, 4, 5 and 6; transient
invocations preserve them. No merged base checkpoint is constructed.

The independent pinned NVIDIA eager implementation uses BF16 arithmetic and the
published PEFT additive update. Native MLX Metal execution retains BF16 through rotary
products, attention, residuals and readout; adapter projection arithmetic is
FP32 and the complete additive update restores the base output precision.
All 262 logit rows have matching greedy choices. The measured 3B row KL
divergence is at most 0.001 nats and each invocation has cosine similarity
at least 0.9995. BF16 conformance uses limits of 0.002 nats/0.9995 cosine for
base invocations and 0.005 nats/0.998 cosine for mixed FP32-adapter/BF16 draft
invocations. These criteria measure distribution agreement; the FP32
comparisons above retain their stricter absolute/relative criteria.

| Invocation | Maximum absolute logit difference | Maximum row KL divergence |
|---|---:|---:|
| Causal | 0.125 | 0.000548674 |
| Ordinary diffusion | 0.125 | 0.000038459 |
| Adapted draft | 3.1171875 | 0.000001458 |

The minimum invocation cosine similarity is 0.99973753. On initial causal
prefill, the first 23 text-layer outputs match bit for bit; later BF16 rounding
differences propagate through residual and adapter computation. The independent
precision checker evaluates first-layer base and adapter products in FP64,
rounds the FP32 adapter projections separately, and checks both implementations
against BF16 half-ULP intervals plus the FP32 absolute-product summation bound.
All twelve first-layer comparisons pass. Large raw differences in adapted
drafting therefore remain visible alongside the probability-level measurements.
This evidence covers Metal execution with CPU weight materialization and
host-layerwise residency. Released-checkpoint native distributed tolerances
remain a separate validation gap.

The same normal-loader sequence and independent FP64 first-layer checks pass
for the pinned 8B and 14B geometries with their published adapters. Each geometry
compares 262 rows, all with matching greedy choices. The 8B causal and ordinary
diffusion logits match bit for bit in all four cases.

| Geometry/invocation | Maximum absolute logit difference | Maximum row KL divergence |
|---|---:|---:|
| 8B causal and diffusion | 0 | 0 |
| 8B adapted draft | 0.39453125 | 0.003378270 |
| 14B causal | 0.25 | 0.001406738 |
| 14B ordinary diffusion | 0.3125 | 0.000187376 |
| 14B adapted draft | 0.8125 | 0.000903682 |

Minimum invocation cosine similarity is 0.99897828 for 8B and 0.99992731 for
14B. Both satisfy the mechanism-specific BF16 bounds. Select the corresponding
checkpoint directory in the commands below to reproduce those comparisons.

```sh
python eredu-evaluation/scripts/nemotron_diffusion_reference.py \
  /private/tmp/eredu-nemotron-reference/3B \
  /private/tmp/eredu-nemotron-reference/3B/reference-adapter-bf16-components.safetensors \
  --draft-adapter --compute-dtype bf16 --trace-components
EREDU_NEMOTRON_RELEASE_ROOT=/private/tmp/eredu-nemotron-reference/3B \
  CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx \
  --no-default-features --features metal --lib \
  normal_loader_released_adapter_matches_independent_causal_and_draft_rows \
  -- --ignored --nocapture --test-threads=1
python eredu-evaluation/scripts/nemotron_adapter_precision.py \
  /private/tmp/eredu-nemotron-reference/3B \
  /private/tmp/eredu-nemotron-reference/3B/reference-adapter-bf16-components.safetensors
```

The same released-weight reference covers the other two distinct text geometries.
The 8B artifact SHA256 is
`14de6f8fee56e551a397b692f962bd5acb108e9f86cf86413de15326f3359384`.
The 14B artifact SHA256 is
`272e78c39809711139deb08024b4fe8e6af83ab1316c8514bdfa35d7c880a320`.
The fixture catalog is authoritative for these artifact digests.

| Geometry | Maximum causal logit error | Maximum diffusion logit error |
| --- | ---: | ---: |
| 3B | 0.00006223 | 0.00009656 |
| 8B | 0.00016260 | 0.00014782 |
| 14B | 0.00010777 | 0.00019264 |

The 14B final residual reaches magnitude 1573.43; its maximum observed layer
difference is 0.00476074, while final logits retain the absolute threshold above.
These comparisons use MLX CPU resident execution and PyTorch CPU references.

The pinned optional 3B LoRA artifact is 95,427,600 bytes with SHA256
`897ef67dff8a69bd1a908fa390ef2164fdaa738e0e47bec502e2f0d86311ff74`.
Its strict adapter contract checks the caller’s expected base repository against
the PEFT declaration and checks every tensor against the loaded target geometry.
The pinned released artifact supplies independent official ancestry evidence.
The supported equations require ordinary
alpha/rank scaling, bias-free `o_proj` targets, and no DoRA, rank stabilization,
saved extra modules, or layer replication. Source tensors retain separate
`draft_adapter.*` parameter identities. The reference applies the independent
`4 * B(A(x))` equation only during draft invocations. Adding `--draft-adapter` to
both commands compares four additional draft blocks. Maximum draft logit error
is 0.00008583; causal and ordinary diffusion results are unchanged.

The reusable neutral `LowRankProjection` owns separate down/up matrices and has
no mutable enabled flag. TP reduces rank activations before the replicated up
projection. A mixed-precision additive update restores the base projection's
output dtype after the sum. Family text layers borrow the adapter only when the
immutable invocation scope is `Draft`. Cold parameter topology assigns both
adapter matrices to the corresponding base layer's residency owner.

Neutral text conformance covers resident versus rebuilt layers, repeated causal,
diffusion and adapted-draft alternation, and base-storage identity. TP2, PP2 and
combined TP2/PP2 comparisons cover independent head geometry, separate adapter
transforms, explicit visibility, and complete-canvas transport. Pipeline wire
metadata carries the mode and original causal frontier; each owner validates it
against its local canonical cache. These comparisons exercise the neutral
invocation and transport equations independently of native distributed hardware.

The portable shared-backbone speculative executor also passes a nonzero numerical
transaction fixture using the text model's actual masked attention. One canonical
cache matches ordinary causal replay after zero, partial and full proposal
acceptance. Drafting seeds the pending committed token without advancing that
cache, retains the full canvas with a shortened proposal budget, and preserves
selected tokens through target-matching evidence. Durable restore reinstalls both
the causal prefix and the independent proposal random stream. Base storage
identities remain stable across those transactions. Loaded physical generation
uses this executor with ordinary target sampling and controller commitment.

The released 3B comparison also passes through the existing MLX host-layerwise
and disk-streamed policies with the optional draft adapter. The host policy uses
one device layer, with a measured logical device peak of 3,690,500,096 bytes and
host peak of 12,204,867,584 bytes. It records 311 expected layer evictions over
the twelve invocations. Disk streaming uses two device layers and a 2 GiB host
cache; its forecast static bytes are 3,221,237,760 and maximum device-layer
window is 938,524,672 bytes. These are logical parameter residency facts,
separate from activations, cache, conversion workspaces and allocator capacity.
The generic policy owns native completion and retirement for both mechanisms.

## Pixtral equations

`nemotron_pixtral_fixture.py` uses Transformers 4.57.1 and the VLM release's
published projector implementation to generate the nonzero `pixtral-numeric.json`
fixture. Two differently shaped images exercise per-image rotary resets,
independent bidirectional attention, pre-merger RMS normalization, channel-major
spatial unfold, and exact GELU. Each image layer is independently constructible.
Attention executes independent image spans, equivalent to a packed block-diagonal
mask without allocating cross-image score visibility.

Pixtral head and intermediate-channel shards pass a two-rank comparison with the
published fixture. Q/K/V/O select matching complete-head ranges; gate/up/down
select matching channel ranges. Row reductions precede each residual addition.
An inconsistent same-width Q/K selection is rejected before layer construction.
Patch embedding and the projector have separate parameter owners and are replicated
within a tensor group; their replicas appear in the cold parameter catalog.

The VLM has three execution groups: independently addressable image layers, one
projection unit, and independently addressable text layers. A nonzero fixture
compares four pipeline owners, with and without TP2, against ordinary execution
across all six directed attention-mode transitions. Cuts separate two image layers,
the projector and text embedding, and the final text layer/readout. Stateless image
owners allocate no decoder cache. Every base parameter has one owner per actual TP
replica and retains its storage identity during alternation. This is typed local
partition conformance. The standard prepared VLM matrix above separately exercises
cold admission, source preparation and total pipeline construction.

The composite fixture independently enumerates a 13-position text/image prompt,
including a row break, and replaces four pad positions with published projected
features. Resident, individually rebuilt and TP2 traversal agree with its causal
decoder result. Native and already projected image parts agree. Subsequent causal,
diffusion and draft invocations retain the same image-conditioned canonical cache.
Conflicting host/grid positions and wrong projected-patch counts are rejected
before execution. These checks exercise invocation semantics, not public decoding
mode commands.

Image policy compares the published floating OpenCV cubic processing on three
nonzero RGB images with unaligned and downscaled sizes, including a dimension
exactly at a ties-to-even rounding boundary. Normalized patch values agree within
0.00002, and marker sequences agree exactly. The host resize retains at most four
resized source rows, preserves cubic overshoot, and uses the released CLIP
statistics. The released processor extent is 1400; the tower rotary extent is
1540. Marker IDs are start=18, pad=19, row-break=20, end=21.

The full pinned VLM-8B weights also pass the independent image reference on two
RGB images. The processor produces 24 unmerged patches and the projector returns
six text-width embeddings. Maximum normalized patch error is 0.00000393, maximum
image-layer error is 0.00047302, and maximum projected-feature error is 0.00001049.
MLX CPU compares every vision layer and the final projector against the published
VLM projector and Transformers 4.57.1 Pixtral implementation, with released BF16
values decoded to FP32.

The normal loader also passes BF16 Metal execution under host-layerwise
residency against the pinned independent BF16 implementation. Neural comparison
uses identical published prepared pixels: all 24 vision-layer outputs, projected
image features, initial 34 text-layer outputs and all 147 causal/diffusion logit
rows match bit for bit. Four causal frontiers are 16, 17, 18 and 19; each transient
32-position invocation preserves its frontier. The normalized patch embedding
differs by at most 0.000000239, without a difference after the first vision layer.

The normal-loader sampled test
`released_prepared_image_context_survives_all_six_sampled_handoffs` also passes
the official BF16 VLM in 59.93 seconds under host-layerwise Metal execution.
AR → self-speculation → AR → diffusion → self-speculation → diffusion → AR
acknowledges committed positions 1, 2, 3, 35, 36 and 68. Both diffusion blocks
use full 32-position geometry with an admitted one-step greedy unmasking
schedule. Greedy target verification before diffusion matches ordinary AR;
the resumed token 1278 matches a fresh causal prefill of both images and all
68 committed tokens.
The common request table passes the same released image-conditioned sequence in
70.21 seconds, with identical acknowledgement positions and resumed token 1278.
It retains both prepared images through two full 32-position diffusion blocks,
per-position finalization and causal append. Set
`EREDU_NEMOTRON_IMAGE_REQUEST_TABLE=1` to select that execution in the test below.

Processor agreement is measured separately before neural rounding. Maximum FP32
pixel error is 0.000003934 under the 0.00001 bound; five of 14,112 values round
to different BF16 values because scalar and OpenCV cubic accumulation orders
differ. Bitwise neural evidence therefore applies to identical prepared inputs,
not to end-to-end BF16 processing across those two resize implementations. With
eredu's prepared pixels, the measured causal maximum logit difference is
0.35351562 and row KL divergence is 0.005835124; diffusion maxima are 0.25 and
0.001542319. All 19 causal and 127 of 128 diffusion greedy rows agree. These
differences remain separate from the identical-input network comparison.

The independent patch-precision checker evaluates projection and RMS equations
in FP64 and bounds FP32 accumulation plus the published BF16 rounding before
gain and after gain. It checks native and published normalized embeddings;
this numerical envelope does not replace the final-logit comparison.

```sh
PYTHONPATH=/private/tmp/eredu-nemotron-reference/transformers-4.57.1 \
  python eredu-evaluation/scripts/nemotron_pixtral_reference.py \
  /private/tmp/eredu-nemotron-reference/VLM-8B \
  /private/tmp/eredu-nemotron-reference/VLM-8B/reference-bf16-context.safetensors \
  --text-context --compute-dtype bf16
EREDU_NEMOTRON_RELEASE_ROOT=/private/tmp/eredu-nemotron-reference/VLM-8B \
  CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx \
  --no-default-features --features metal --lib \
  normal_loader_released_pixtral_prepared_context_matches_independent_rows \
  -- --ignored --nocapture --test-threads=1
python eredu-evaluation/scripts/nemotron_pixtral_precision.py \
  /private/tmp/eredu-nemotron-reference/VLM-8B \
  /private/tmp/eredu-nemotron-reference/VLM-8B/reference-bf16-context.safetensors
```

```sh
python eredu-evaluation/scripts/nemotron_pixtral_reference.py \
  /private/tmp/eredu-nemotron-reference/VLM-8B \
  /private/tmp/eredu-nemotron-reference/VLM-8B/reference-vision-f32.safetensors
cargo run -p eredu-backend-mlx --no-default-features \
  --example nemotron_diffusion_reference -- \
  /private/tmp/eredu-nemotron-reference/VLM-8B \
  /private/tmp/eredu-nemotron-reference/VLM-8B/reference-vision-f32.safetensors \
  --vision-only
cargo run -p eredu-architectures --example nemotron_tokenizer_admission -- \
  /private/tmp/eredu-nemotron-reference
```

The complete released VLM also passes image-conditioned text prefill and three
cached decode steps, with a 32-position transient canvas after each causal call.
The published processor and tokenizer construct the full 16-position context;
eredu's marker IDs agree exactly. The canonical frontiers are 16, 17, 18 and 19.
Maximum causal logit error is 0.00008500 and maximum diffusion logit error is
0.00006104. Every image and text layer passes the same FP32 tolerance above;
maximum layer error is 0.00067139. Resident planned, current and peak parameter
storage are all 35,672,137,728 bytes, with no evictions during attention alternation.

The same image-conditioned reference passes host-layerwise and disk-streamed
execution. Host-layerwise peak logical device parameters are 5,169,876,992 bytes,
with 31,375,310,848 host bytes and 296 ordinary unit evictions. The disk policy
uses a 12 GiB device limit and 2 GiB host limit; measured peaks are
12,854,255,616 device bytes and 2,114,535,424 host bytes, with 583 ordinary
evictions. These are parameter/conversion telemetry, excluding the separately
retained KV cache and invocation activations. Source GGUF/SafeTensors payload
sizes are distinct from decoded logical parameter bytes.

```sh
PYTHONPATH=/private/tmp/eredu-nemotron-reference/transformers-4.57.1 \
  /private/tmp/eredu-nemotron-reference/python-text/bin/python \
  eredu-evaluation/scripts/nemotron_pixtral_reference.py \
  /private/tmp/eredu-nemotron-reference/VLM-8B \
  /private/tmp/eredu-nemotron-reference/VLM-8B/reference-context-f32.safetensors \
  --text-context
cargo run -p eredu-backend-mlx --features metal \
  --example nemotron_diffusion_reference -- \
  /private/tmp/eredu-nemotron-reference/VLM-8B \
  /private/tmp/eredu-nemotron-reference/VLM-8B/reference-context-f32.safetensors \
  --vision-context
```

Add `--residency host` or `--residency disk` to the native command for the
corresponding bounded reference comparison. All image layers, text layers and
logit rows pass the same tolerances in each policy.

All seven fast-tokenizer vocabularies pass canonical extent, BOS, EOS and reserved
mask checks through `eredu-text`. VLM additionally checks its image delimiters.
Encoding `Hello, world!` without added special tokens yields
`[22177, 1044, 4304, 1033]` for every release.

## MLX affine conversion

The pinned `mlx-community/Nemotron-Labs-Diffusion-3B-4bit` revision is
`5fd1ad4bcbd02a3fd489ec6e87059e072118548c`. The published conversion identifies
the official NVIDIA 3B ancestor, `mlx-vlm` version 0.6.3, affine mode, group size
64, and four-bit codes. Its 2,155,620,090-byte artifact has SHA256
`b575385053dc7230462f23c79643fa1500a4813252cd7f818b758a1a205a3b38`.
`conversions.json` retains the exact revision, settings, configuration digest,
and payload digest. Strict admission checks its `language_model.model.*` and
`language_model.diffusion_head.*` names, packed U32 matrix geometry, and every
scale and affine-bias companion.

The independent reference extracts little-endian UINT4 codes with NumPy, applies
each stored BF16 scale and bias in FP32, and executes the pinned NVIDIA model
over these decoded matrices. Native Metal execution retains packed weights and
FP32 copies of floating companions and norms. Across four cached causal steps and
four complete 32-position diffusion invocations, maximum logit error is
0.00028157. Reported planned, current and peak logical parameter residency are
all 2,395,336,704 bytes, with no parameter evictions during mode alternation.
This evidence covers native Metal execution; the full CPU quantized refinement
comparison and native distributed quantization comparisons are not established.

The normal loader passes native packed CPU execution under disk-streamed policy
with original BF16 companions and normalization parameters. Affine embedding
lookup rounds code/scale multiplication and bias addition separately in BF16,
then converts activations to F32. Projection kernels promote those companions
with F32 input activations. This selected arithmetic has a separate independent
reference; it is not the all-FP32 decoded-matrix reference above. Four causal
maximum errors are 0.00008583, 0.00006580, 0.00004959 and 0.00005245. The
AR → 32-position diffusion → AR sequence settles at position 34, resumed token
11 agrees with a fresh prefix, and four greedy self-speculative tokens agree
with AR. This establishes CPU refinement/verification execution and cache
handoff; a direct CPU comparison of every refinement position remains separate
from these cached causal comparisons.

```sh
python eredu-evaluation/scripts/nemotron_diffusion_reference.py \
  /private/tmp/eredu-nemotron-reference/3B \
  /private/tmp/eredu-nemotron-reference/mlx-3B-4bit/reference-affine-cpu-f32.safetensors \
  --mlx-affine /private/tmp/eredu-nemotron-reference/mlx-3B-4bit \
  --mlx-embedding-rounding bf16_cpu
target/release/examples/nemotron_generation_modes \
  /private/tmp/eredu-nemotron-reference/mlx-3B-4bit \
  /private/tmp/eredu-nemotron-reference/mlx-3B-4bit/reference-affine-cpu-f32.safetensors disk
```

Comparison with official BF16 weights is a separate quality measurement. On the
fixed prompts above, independently decoded quantized causal logits have RMS
differences 1.77417 (three-position prefill), 0.79464, 0.67312, and 0.81225;
diffusion-logit RMS differences are 0.71009, 0.69801, 0.77813, and 0.79702.
Greedy maxima match five of the six causal rows and all 128 diffusion rows.
These fixed numerical cases do not establish task-level quantization quality.
The smaller native-versus-decoded errors above establish execution of the
quantized artifact rather than equality to the official floating weights.

```sh
python eredu-evaluation/scripts/nemotron_diffusion_reference.py \
  /private/tmp/eredu-nemotron-reference/3B \
  /private/tmp/eredu-nemotron-reference/mlx-3B-4bit/reference-f32.safetensors \
  --mlx-affine /private/tmp/eredu-nemotron-reference/mlx-3B-4bit
cargo run -p eredu-backend-mlx --no-default-features --features metal \
  --example nemotron_diffusion_reference -- \
  /private/tmp/eredu-nemotron-reference/mlx-3B-4bit \
  /private/tmp/eredu-nemotron-reference/mlx-3B-4bit/reference-f32.safetensors \
  --device gpu
```

```sh
python eredu-evaluation/scripts/nemotron_pixtral_fixture.py \
  /private/tmp/eredu-nemotron-reference/VLM-8B \
  eredu-evaluation/fixtures/nemotron_labs_diffusion/pixtral-numeric.json
```

## Pinned GGUF execution

`TextGgufPlan` retains one exact source, configuration, tensor resolution and
content-backed provenance receipt. The three text-only payloads are pinned
in `conversions.json`; VLM language and companion payloads are pinned in
`vlm-gguf-conversion.json`. An unrecognized payload cannot gain family semantics
from `dream` or `mistral3` alone. Each source checks independent head widths,
YaRN metadata, complete tokenizer extent and reserved IDs, encoded tensor
geometry and required diffusion fields before construction. Missing legacy
query-scaling metadata is supplied only by an exact content-backed receipt.

| Conversion | Revision | Encoding | SHA256 |
| --- | --- | --- | --- |
| spiritbuun/14B | `7ec2bb277055ffbbcc8cb7e56e179216d3f4952d` | Q8_0 | `d25119a965e4781b5f1d4b5b2cf446e4102d949d9752d86144b94820368fa4d1` |
| suzukimain/14B | `f9fd26730ff276b02361c0f776085f60a122ec70` | Q4_K_M | `2473a97abba7effed8bd0baf49beed09d988c45c3834885eca12565cce77132e` |
| IntellAgents/3B | `b1367a57265c62f63e18f0faacaac202731ee0c4` | Q4_K_M | `f0a4db3dc37dc80de075eb36554f25ec80147d6f0e4bfc65ecfc42147b321224` |

The legacy Spiritbuun and Suzuki tensors retain NVIDIA's half-head Q/K ordering.
The IntellAgents conversion uses adjacent pairs, as its pinned llama.cpp
converter declares. Family configuration retains that proven rotary convention;
independent Python comparison reverses the IntellAgents permutation before
executing NVIDIA's equations. Quantized source tensors decode to FP32 through
bounded GGUF conversion. These results establish decoded execution, not packed
native K-quant execution. The 14B vocabulary matrix needs 2,684,354,560 decoded
bytes; cold reader limits admit a single matrix up to 3 GiB rather than an
unbounded complete stack.

The independently decoded quantized reference uses llama.cpp
`0f3a71be15af836d277c9f918adfafb45732677e` and the pinned `gguf-py/quants.py`
digest. Suzuki resident execution passes all four causal and four transient
32-position invocations with maximum causal logit error 0.00010467 and diffusion
error 0.00011444. Logical planned, current and peak FP32 parameter bytes are
54,024,294,400, with no evictions during alternation. IntellAgents host-layerwise
execution passes the same cases, with maximum causal error 0.00007057 and
diffusion error 0.00010014. Its device peak is 3,686,830,080 bytes, host peak is
12,105,613,312 bytes, and ordinary bounded loading records 207 layer evictions.
Each comparison uses the layer/logit tolerances stated above and checks that
transient calls preserve the causal frontier.

Spiritbuun Q8_0 disk-streamed execution also passes all eight invocations,
with maximum causal logit error 0.00015974 and diffusion error 0.00017357.
Its explicit device limit is 12 GiB and host limit is 4 GiB, including bounded
conversion workspaces. Measured logical device peak is 12,667,064,320 bytes and
host peak is 3,649,241,088 bytes. The policy records 631 ordinary parameter
evictions over the invocation sequence. Decoded planned parameter bytes are
54,024,294,400; the smaller encoded GGUF payload size is a separate storage fact.

Independent ancestry measurement is separate from execution of the quantized
artifact. `nemotron_gguf_ancestry.py` rebuilds source rows with the pinned primary
C quantizer. Spiritbuun's Q8_0 payload reproduces all 14,351,421,440 tensor bytes
exactly from the pinned official BF16 values decoded directly to FP32. Its named
older official revision has the same weight digest as the pinned 14B release.
Suzuki's Q4_K_M reconstruction with an FP16 intermediate differs in 3,933,464 of
8,231,178,240 compared bytes; this is not byte-identical producer reproduction.
The legacy publishers do not identify an exact original converter revision.
The pinned independent reconstruction and artifact comparisons do not imply that
unknown producer setting has been recovered. IntellAgents publishes its exact
converter revision and conversion settings in the retained provenance metadata.

GGUF Pixtral tokenization uses the official case-sensitive Unicode split,
single-digit grouping and ByteLevel policy without Unicode normalization. Five
independent released-tokenizer cases cover case changes, digits, contractions,
CRLF, combining marks, Greek, emoji and protocol tokens. Every official release
and all four pinned GGUF language tokenizers agree on these IDs.

```sh
PYTHONPATH=/private/tmp/eredu-nemotron-gguf/llama-cpp-reference/gguf-py \
  /private/tmp/eredu-nemotron-reference/python-text/bin/python \
  eredu-evaluation/scripts/nemotron_diffusion_reference.py \
  /private/tmp/eredu-nemotron-reference/14B REFERENCE.safetensors \
  --gguf PINNED.gguf
cargo run -p eredu-backend-mlx --features metal \
  --example nemotron_diffusion_reference -- \
  /private/tmp/eredu-nemotron-reference/14B REFERENCE.safetensors \
  --gguf PINNED.gguf
```

Use the corresponding 3B directory and `--residency host` for the IntellAgents
comparison. `nemotron_tokenizer_admission` accepts the seven-release root followed
by the pinned GGUF paths for the tokenizer comparison. Full payloads and numerical
outputs remain outside tracked source.

## VLM GGUF language and vision sources

`IntellAgents/Nemotron-Labs-Diffusion-VLM-8B-GGUF` is pinned at
`a436a745feda80389c99c7a15ba3cc46fbfee8e0`. `VLM8B-F16.gguf` has
16,987,579,808 bytes and SHA256
`4b5f15a545e5272b587a5150ad37cc6b5e338dc8352693a11973d1fa8f4862c7`;
`mmproj-F16.gguf` has 857,077,888 bytes and SHA256
`0705006bdaac2a3e330a9e0b80767653fce5a07fedd99995c869082b0278ff75`.
The pinned published converter uses the same primary llama.cpp revision above.
`nemotron_vlm_gguf_ancestry.py` reproduces all 17,836,752,896 tensor bytes exactly
from the pinned released weights, including the adjacent-pair Q/K transformations
and the separately validated image-break metadata vector.

Language admission checks the 131,073-entry VLM vocabulary and its distinct
masked-generation metadata. Image input requires the explicit Pixtral/projector
companion. Its strict schema includes every independently owned image layer and
projector parameter. Vision Q/K recipes restore half-head coordinates through
bounded FP32 cast/reshape/transpose operations; the marker vector is artifact
metadata and creates no duplicate embedding parameter.

The independent reference decodes F16 matrices and F32 vectors with the pinned
primary reader and executes NVIDIA's equations over FP32 values. Resident,
host-layerwise and disk-streamed image-conditioned comparisons pass every vision
and text layer, four causal calls and four complete diffusion canvases. Maximum
causal logit error is 0.00008726, diffusion error 0.00006104 and layer error
0.00097656. Canonical frontiers remain 16, 17, 18 and 19. Resident planned, current
and peak logical parameter storage are 35,672,137,728 bytes with no evictions.
Host execution peaks at 5,169,876,992 device and 31,375,310,848 host bytes, with
296 unit evictions. Disk execution uses 12 GiB device and 4 GiB host limits;
peaks are 12,854,255,616 device and 4,262,281,216 host bytes, with 581 unit evictions.

Separate quality comparisons against official weights retain all 19 causal and
128 diffusion greedy maxima. Causal RMS drift is 0.00000412–0.00001363 and
diffusion drift is 0.00000757–0.00000949 on these fixed image-conditioned cases.
The execution comparisons use the independently decoded artifact rather than
that official-weight quality baseline.

Normal model loading with the published vision companion also passes
`released_prepared_image_context_survives_all_six_sampled_handoffs` under
host-layerwise Metal execution in 92.32 seconds. The full six-transition schedule
acknowledges committed positions 1, 2, 3, 35, 36 and 68, including two complete
32-position diffusion blocks with one-step greedy unmasking. Initial greedy
verification matches ordinary image-conditioned AR. The resumed AR token 1278
equals a fresh prefill of the same two images and all 68 committed tokens.
The companion's declared image-break metadata remains a source admission
obligation and contributes no neural parameter or duplicate embedding allocation.

Normal loading retains dense F16 matrices, F32 normalization vectors and F32
vision Q/K coordinate-recipe outputs. Text Q/K stay in their admitted adjacent
pair representation and retain F16 storage. Resident parameter traversal totals
17,937,399,808 bytes, excluding conversions, cache and workspace. Patch inputs
and image placement follow their F16 parameter precision; queries and causal
cache arithmetic are F32 after promoted projections. Learned RMS normalization
rounds the normalized activation to its input precision before multiplying the
F32 gain. The independent `--compute-dtype mixed` reference follows these
published boundaries without changing the numerical tolerance.

`normal_loader_released_pixtral_prepared_context_matches_independent_rows`
passes that reference under host-layerwise Metal execution in 93.12 seconds.
Four causal maximum errors are 0.00100183, 0.00014305, 0.00013065 and 0.00010204;
four full diffusion-canvas maxima are 0.00013554, 0.00010324, 0.00010920 and
0.00008744. Every score passes `0.002 + 0.00002 * abs(reference)` and the
frontiers remain 16, 17, 18 and 19 during refinement. Normalized vision embedding,
final vision-layer and projector maxima are 0.00032425, 0.00360107 and
0.00005770. These mixed-precision results remain separate from the all-FP32
decoded comparison and its residency figures above.

```sh
PYTHONPATH=TRANSFORMERS_4_57_1:PINNED_LLAMA_CPP/gguf-py \
  python eredu-evaluation/scripts/nemotron_pixtral_reference.py \
  OFFICIAL_VLM_DIRECTORY REFERENCE.safetensors --text-context --gguf GGUF_DIRECTORY
cargo run -p eredu-backend-mlx --no-default-features --features metal \
  --example nemotron_diffusion_reference -- \
  OFFICIAL_VLM_DIRECTORY REFERENCE.safetensors \
  --gguf GGUF_DIRECTORY/VLM8B-F16.gguf \
  --vision-gguf GGUF_DIRECTORY/mmproj-F16.gguf --vision-context
```

The normal-loader mixed-precision comparison and sampled handoffs use:

```sh
PYTHONPATH=TRANSFORMERS_4_57_1:PINNED_LLAMA_CPP/gguf-py \
  python eredu-evaluation/scripts/nemotron_pixtral_reference.py \
  OFFICIAL_VLM_DIRECTORY GGUF_DIRECTORY/reference-context-mixed.safetensors \
  --text-context --gguf GGUF_DIRECTORY --compute-dtype mixed
EREDU_NEMOTRON_RELEASE_ROOT=OFFICIAL_VLM_DIRECTORY \
  EREDU_NEMOTRON_MODEL_SOURCE=GGUF_DIRECTORY/VLM8B-F16.gguf \
  EREDU_NEMOTRON_IMAGE_REFERENCE=GGUF_DIRECTORY/reference-context-mixed.safetensors \
  env CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx \
  --no-default-features --features metal --lib \
  normal_loader_released_pixtral_prepared_context_matches_independent_rows \
  -- --ignored --nocapture --test-threads=1
```

Add `--residency host` or `--residency disk` for the bounded comparisons. Source
sizes describe encoded artifacts; execution and residency reports describe FP32
normalized parameters and their actual conversion workspaces.

## Exact split GGUF sources

Language and vision admission accept exact split repacks through a complete
encoded-catalog SHA256. `catalog_sha256_v1` in the conversion pins covers every
effective metadata field, including tokenizer data, and every tensor's name,
encoding, dimensions, endianness and encoded bytes. Physical alignment, offsets
and split ordinals/counts do not contribute. Actual member SHA256 values remain
separate provenance records; a generic architecture tag supplies no authority.
The fingerprint pass retains one 8 MiB buffer and verifies current headers and
file identities without allocating decoded model parameters.

`split-conversions.json` pins repacks made with llama.cpp
`0f3a71be15af836d277c9f918adfafb45732677e`, including writer/script digests,
settings and every output member hash. The three-member 3B Q4_K_M repack passes
host-layerwise prefill, three cached decode steps and four diffusion canvases
against the independently decoded original. Maximum causal/diffusion logit
errors are 0.00007057/0.00010014; device/host peaks and the 207 unit evictions
match the single-file source. The five-member Pixtral/projector repack passes
the full VLM image-conditioned sequence with the same 0.00008726/0.00006104
logit bounds and host-layerwise residency measurements as its single-file source.

```sh
PYTHONPATH=PINNED_LLAMA_CPP/gguf-py \
  python eredu-evaluation/scripts/nemotron_gguf_repack.py \
  PINNED_ARTIFACT.gguf OUTPUT_DIRECTORY --maximum-bytes 1000000000
cargo run -p eredu-checkpoint --example gguf_catalog_fingerprint -- \
  OUTPUT_DIRECTORY/ARTIFACT-00001-of-00003.gguf
```

Pass the first member to `--gguf` or `--vision-gguf` in the native commands above.
These results establish exact source normalization and bounded execution through
split language and vision membership. Native distributed execution is not
measured by these artifact comparisons. The neutral controlled
and uninterrupted facade supports validated mode commands, settled acknowledgement
and independent snapshot/branch mailboxes. Its semantic backend fixture covers
all six directed transitions, repeated alternation, pending committed suffixes,
restore, fork, cancellation and failed settlement. This fixture is separate from
the released Nemotron numerical evidence.

## Weight-only tensor FP8

The scalar-scaled E4M3 source contract admits the published W8A16 metadata and
requires an F32 scalar companion for every compressed text projection. It rejects
activation quantization, conflicting exclusions and non-scalar scale geometry.
The neutral `DecodeTensorFp8` recipe describes E4M3 decoding followed by scalar
multiplication into FP32. Its selection contract pushes row/column restrictions
to the encoded weight while retaining the complete scalar scale. Source leases,
conversion intermediates and decoded output contribute to workspace forecasts.
Activations retain the dense operator's precision; the existing dynamic
block-FP8 activation quantizer is a distinct mechanism.

The pinned FP8 and AWQ candidate metadata and payload digests are recorded in
`compressed-conversions.json`. The pinned FP8 source is
`yury-kapoyko/nemotron-diff-vlm-8b-fp8` at
`f56d4af962fa95ee22ea43b22d91fb7c9d2652f8`. Its two shards have SHA256
`c157823fb451f0f66d1d90268226eea61270ca5376aef84266dd3226b7a7f012`
and `0b724124f6497ba612ff524a8c86804681773d85001c60687b6128c566d5d80a`.
The producer reports Model Optimizer 0.45.0 weight-only FP8, 128 domain
image/text calibration samples and dense embedding, readout, tower and projector
exclusions. NVIDIA's pinned Model Optimizer tag resolves to
`ec87a82927d003986d44fb7f4fa8b3d10c31b095`; its export equation confirms
E4M3 decoding followed by the scalar weight multiplier.

The independent PyTorch reference decodes each stored E4M3 weight to FP32 and
multiplies its exact scalar scale before executing the pinned NVIDIA VLM.
Native resident execution passes image preparation, complete causal prefill,
three cached decode steps and four 32-position diffusion canvases. Maximum causal
logit error is 0.00009537, diffusion error is 0.00007248 and layer error is
0.00085449. Every row passes the FP32 tolerances above. Logical planned, current
and peak decoded parameter bytes are 35,672,137,728, with no evictions during
attention alternation. This evidence establishes FP32 decoded W8A16-weight
execution; it does not establish native packed FP8/BF16 activation execution.

The normal loader also passes the pinned artifact under host-layerwise Metal
execution with its actual mixed parameter precision. Dense vision, projector,
embedding, normalization and readout parameters retain BF16 storage; compressed
text projections decode to FP32. Text query/cache and rotary arithmetic use FP32.
The independent reference preserves dense source precision and promotes linear
inputs with their weights; its text rotary tables use the selected FP32 query
precision. This is distinct from running every decoded parameter in FP32 or
retaining the stock BF16 rotary tables after replacing projections with FP32.
Processor comparisons precede neural comparisons on identical prepared pixels,
as in the official BF16 validation above. Vision and projector outputs are exact.
Four causal invocations, including three cached decode steps, have maximum logit
errors 0.00033283, 0.00007629, 0.00005245 and 0.00003338. Four full 32-position
diffusion canvases have maximum errors 0.00007439, 0.00006437, 0.00008488 and
0.00009918. Every score passes `0.002 + 0.00002 * abs(reference)`; causal
frontiers 16, 17, 18 and 19 remain unchanged by refinement. The native test
`normal_loader_released_pixtral_prepared_context_matches_independent_rows`
passes in 155.21 seconds with this root. This comparison covers prepared
image context and transient attention, rather than sampled mode handoffs.

The separate normal-loader test
`released_prepared_image_context_survives_all_six_sampled_handoffs` passes this
pinned FP8 VLM under host-layerwise Metal execution in 168.69 seconds. Its two
prepared images remain in the causal context throughout AR → self-speculation
→ AR → diffusion → self-speculation → diffusion → AR. The acknowledgements
occur at committed positions 1, 2, 3, 35, 36 and 68. Greedy verification before
diffusion agrees with ordinary AR. Both diffusion invocations retain the full
32-position geometry and use an explicitly admitted one-step greedy unmasking
schedule. The resumed AR token 1278 equals a fresh causal prefill of the same
images and all 68 committed text tokens. This validates sampled commitment and
handoff on the independently compared quantized execution; it does not claim
identical sampling output to the official BF16 weights.

```sh
PYTHONPATH=/private/tmp/eredu-nemotron-reference/transformers-4.57.1 \
  /private/tmp/eredu-nemotron-reference/python-text/bin/python \
  eredu-evaluation/scripts/nemotron_pixtral_reference.py \
  /private/tmp/eredu-nemotron-reference/VLM-8B \
  PINNED_FP8_DIRECTORY/reference-mixed-context.safetensors \
  --text-context --tensor-fp8 PINNED_FP8_DIRECTORY --compute-dtype mixed
EREDU_NEMOTRON_RELEASE_ROOT=PINNED_FP8_DIRECTORY \
  env CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx \
  --no-default-features --features metal --lib \
  normal_loader_released_pixtral_prepared_context_matches_independent_rows \
  -- --ignored --nocapture --test-threads=1
```

Host-layerwise execution passes the same image and text comparisons with the
projection in its own unit. Peak logical device parameters are 5,169,876,992 bytes,
host parameters 31,375,310,848 bytes, with 296 ordinary unit evictions. Parameter
storage is shared by causal and transient invocations under both policies.

Disk-streamed execution passes the same reference with 12 GiB device and 2 GiB
host budgets. Measured peaks are 12,854,255,616 device bytes and 2,114,535,424 host
bytes, with 583 ordinary unit evictions. These decoded parameter and conversion
measurements exclude separately retained KV cache and invocation activations.

```sh
PYTHONPATH=/private/tmp/eredu-nemotron-reference/transformers-4.57.1 \
  /private/tmp/eredu-nemotron-reference/python-text/bin/python \
  eredu-evaluation/scripts/nemotron_pixtral_reference.py \
  /private/tmp/eredu-nemotron-reference/VLM-8B REFERENCE.safetensors \
  --text-context --tensor-fp8 PINNED_FP8_DIRECTORY
cargo run -p eredu-backend-mlx --no-default-features --features metal \
  --example nemotron_diffusion_reference -- \
  PINNED_FP8_DIRECTORY REFERENCE.safetensors --vision-context
```

The AWQ candidates have output-row packed codes, group scales, a second weight
scale and input smoothing companions. W4A8 additionally declares an activation
quantizer; treating these artifacts as ordinary affine weights would discard
published semantics. These AWQ encodings are not admitted by the family.

Quantization quality is measured separately against the official-weight reference.
On the fixed cases above, all three independently decoded GGUF conversions retain
all six causal greedy maxima and all 128 diffusion maxima. Causal RMS drift ranges
are 0.283283–0.317275 for Q8_0 14B, 0.822419–0.962732 for Suzuki Q4_K_M 14B,
and 0.288483–0.888654 for IntellAgents Q4_K_M 3B. Corresponding diffusion RMS
ranges are 0.076277–0.194072, 0.516028–1.261666 and 0.283432–0.386365.
FP8 VLM retains all 19 causal and 128 diffusion greedy maxima in its image-conditioned
cases, with causal RMS drift 0.219131–0.263460 and diffusion drift
0.093945–0.112041. These cases do not establish task-level quantization quality.

```sh
python eredu-evaluation/scripts/nemotron_quantized_quality.py \
  QUANTIZED_REFERENCE.safetensors OFFICIAL_REFERENCE.safetensors \
  --output quality-vs-official.json
```

Scoped shared-layer capture passes the same native Ring text TP2, PP2 and
combined TP2/PP2 cases, and VLM TP2 and PP5 cases, under all three residency
policies. The VLM TP2/PP5 case passes in 112.86 seconds, and the draft-adapted
VLM TP2/PP5 case passes in 114.84 seconds; these two runs overlap. Every rank
uses the same full physical invocation geometry and mode schedule. Paid restore
reproduces all 28 committed tokens while activation invocation identities remain
monotone. The selected layer's prefill, ordinary refinement, masked draft and
verification records retain their distinct phases. The capture fixture reserves
64 MiB each for per-invocation retention and host work, with a 4 MiB encoded
ceiling; these logical bounds cover native transport and bounded receipt parsing
on all receivers, separately from model parameter residency.

The captured combined runs use the explicit native test binary built with
`cargo check -p eredu-backend-mlx --no-default-features --features metal --tests`:

```sh
EREDU_NEMOTRON_RING_REQUEST_TABLE_ONLY=1 EREDU_NEMOTRON_RING_ACTIVATIONS=1 EREDU_NEMOTRON_RING_COMBINED_ONLY=1 target/debug/deps/eredu_backend_mlx-1d5f204e48b47786 native_shared_decoding_ring_tp_pp_and_combined_match_single_owner --ignored --nocapture
EREDU_NEMOTRON_RING_REQUEST_TABLE_ONLY=1 EREDU_NEMOTRON_RING_ACTIVATIONS=1 target/debug/deps/eredu_backend_mlx-1d5f204e48b47786 native_adapted_shared_decoding_ring_combined_vlm_matches_single_owner --ignored --nocapture
```

Shared scoped activation discovery exposes ordinary text and Pixtral tower and
projector hooks. Public bounded captures of a vision residual and final projector
output pass combined TP2/PP5 under resident, host-layerwise and disk-streamed
execution. Records contain captured payloads and exact source shapes `[1,4,8]`
and `[1,1,6]`, independently of decoder delimiter positions. The ordinary and
draft-adapted runs pass in 120.99 and 120.97 seconds while overlapping. Their
three selected tensors reserve 192 MiB each for per-invocation retention and host
work, with a 4 MiB encoded ceiling. These runs use the captured combined commands
above and include one numerical refinement invocation per advancement.

The nonzero neutral Pixtral intervention fixture passes in 0.01 seconds. Zeroing
the MLP write preserves the independently captured attention residual; zeroing
the first image projection produces zero after GELU and the bias-free final
projection. Raw and spatially merged observation extents remain 16 and 4.
The neutral packed-media geometry test verifies that neither 13 decoder rows nor
21 context positions replaces those axes.

Controlled numerical snapshots pass in 3.92 seconds across all three residency
policies. Paid restore resumes the next numerical iteration, preserves cumulative
copy allowances and monotone capture IDs, and retains the selected output and
forced dependent suffix behavior. The common request table's paused numerical
state and all six handoffs pass in 4.24 seconds. The portable controlled
speculative conformance cases pass in 0.22 seconds; the focused diffusion table
selection, EOS, cancellation and failed-publication cases pass in 0.01 seconds.

```sh
env CARGO_INCREMENTAL=0 cargo test -p eredu-core --lib packed_media_extents -- --nocapture
target/debug/deps/eredu_backend_mlx-1d5f204e48b47786 controlled_shared_diffusion_forks_partial_canvases_and_refines_forced_suffixes --nocapture
target/debug/deps/eredu_backend_mlx-1d5f204e48b47786 shared_request_table_refines_initial_blocks_and_all_six_handoffs --nocapture
```

The shared neutral verification cursor passes the independently enumerable
stochastic target-distribution fixture and the accepted-EOS cache-retention
fixture. The nonzero incremental cache-prefix and fork comparison passes in
0.02 seconds. Native greedy equivalence and all six physical handoffs pass
resident and bounded execution in 5.51 seconds; native stochastic sampling with
evolving penalties passes in 1.98 seconds. The 16 portable controlled-session
conformance tests pass in 0.22 seconds. These checks exercise the common cursor
used by batched resolution and explicitly advanced shared-backbone verification.

Independent causal drafting with ordinary target diffusion passes all six mode
handoffs under resident, host-layerwise and disk-streamed execution in 4.88
seconds. Each acknowledgement checks that both cache frontiers equal prompt plus
committed output. The fixture executes complete likelihood-ratio draft
transactions between handoffs and reserves snapshot and restore allowances for
paused numerical target state. Its existing greedy target comparison remains
unchanged. The neutral dense/MoVA independent verification, rejection and durable
restore fixture passes in 0.12 seconds through the explicit target-refinement input
handoff.

```sh
target/debug/deps/eredu_backend_mlx-1d5f204e48b47786 independent_causal_source_handoffs_preserve_both_frontiers_and_target_output --nocapture
```

A single native storage survey compares canonical KV arrays with their settled
prefix checkpoint arrays. Every backing is available and corresponding indices
are equal under all three residency policies; this identity and rollback fixture
passes in 5.07 seconds. The
independent six-handoff fixture passes in 5.62 seconds and stochastic target
matching with evolving penalties passes in 5.44 seconds. These three runs overlap.
The public independent-drafter dense/MoVA conformance fixture passes greedy and
stochastic sampling under all three residency policies in 7.28 seconds with
controlled/uninterrupted output parity, forks, captures and snapshot budgets.

```sh
target/debug/deps/eredu_backend_mlx-1d5f204e48b47786 loaded_nemotron_shared_drafts_preserve_causal_frontier_across_residency_policies --nocapture
```

The common entropy/feedback driver matches DiffusionGemma's existing physical
policy through the prepared target-only handoff as well as direct executor
construction. The CPU fixture compares five unforced output IDs and numerical
iteration counts across resident, host-layerwise and disk-streamed execution,
including controlled and uninterrupted visitors. Its paid numerical snapshot
replays the same continuation. The extended fixture passes in 0.57 seconds:

```sh
```

The finalized entropy-score semantics remain `ConstrainedFinalization`.
The prepared entropy session's continuation forecast includes full refinement
geometry without an assistant and leaves output and budgets unchanged.
`shared_target_prepared_modes_preserve_output_and_pending_adapter_forecasts`
passes through the public CPU facade in 0.16 seconds. Its nonzero fixture uses
independent head geometry and a rank-two draft adapter. Default AR and controlled
AR/self/AR produce identical greedy target IDs; initial diffusion with a two-token
budget agrees under controlled and uninterrupted advancement. An adapter-enabled
pending command adds its retained workspace declaration to the forecast without
advancing output or spending snapshot credits.

```sh
env CARGO_INCREMENTAL=0 cargo test -p eredu --no-default-features --features mlx --test native_execution_control shared_target_prepared_modes_preserve_output_and_pending_adapter_forecasts -- --nocapture
```

The masked selected-token, forced dependent-suffix and numerical-fork regression
passes in 3.92 seconds through the shared refinement policy. The neutral single-model
suffix-settlement fixture passes in 0.03 seconds, and portable controlled
speculative conformance passes in 0.23 seconds. Public dense/MoVA greedy and
stochastic controlled/uninterrupted parity passes in 7.00 seconds with shared prefix checkpoints.

```sh
target/debug/deps/eredu_backend_mlx-1d5f204e48b47786 common_refinement_driver_preserves_entropy_feedback_outputs_iterations_and_snapshot_state --nocapture
```

## Common generation request driver

`GenerationRequestTable` owns ordinary, diffusion and speculative request
lifecycle, canonical IDs, commitment limits, cancellation and mode acknowledgement.
Opaque native samplers enter through a completed-decision source; score-backed
sources retain transactional verification. Native progress does not query a
controller, and sampled native tokens never become fabricated probability rows.
The ordinary source retains non-cloneable controllers and borrows its loaded
runtime through a scoped context. Canonical identity and exact completion precede
yielding the original native token handle.

Snapshots retain the real common-driver sequence together with native cache,
sampler, pending input and mode state. Sequence copies and child growth consume
snapshot reservations; restoring does not refund spent budgets. Host forecasts
include retained canonical IDs and the requested commitment horizon separately
from base parameters and native canvas/cache storage.

The native CPU six-handoff fixture passes across all three residency policies in
0.69 s. The complete text/VLM Ring TP, PP and combined TP/PP matrix passes in
119.53 s. The BF16-base/FP32-adapter combined VLM matrix passes in 87.04 s,
including isolated snapshot restore/fork and draft-only adapter scope. Native
shared-cache identity and rollback pass in 0.13 s. The prepared shared-target
shortened-diffusion and pending-adapter forecast fixture passes in 0.11 s.
Two public CPU snapshot/fork fixtures pass in 0.22 s. Independent dense/MoVA
external-drafter greedy and stochastic parity across all residency policies passes
in 12.82 s. DiffusionGemma entropy/feedback parity passes in 0.54 s.

Portable backend-control fixtures pass 21 cases, including a borrowed controller,
chunked prefill, cancellation and detached continuation parity. Ten existing
request-table lifecycle/verification fixtures pass, and 42 runtime forecast cases
pass. Portable facade conformance covers parser state, evolving constraints,
forced choices, mode commands, capture, snapshots/forks and invalid-token failures.
The native adapter, portable families, public MLX tests/examples and CLI compile.

```sh
env CARGO_INCREMENTAL=0 cargo test -p eredu-core --lib backend::tests
env CARGO_INCREMENTAL=0 cargo test -p eredu-runtime --lib memory_forecast::
env CARGO_INCREMENTAL=0 cargo test -p eredu --no-default-features --test backend_conformance
env CARGO_INCREMENTAL=0 cargo test -p eredu --no-default-features --features mlx --test native_execution_control native_cpu_ -- --nocapture
```

These Ring results validate local CPU participants. GPU collectives, multi-host
placement and released-checkpoint native distributed tolerances retain the
hardware-validation gaps described above.
