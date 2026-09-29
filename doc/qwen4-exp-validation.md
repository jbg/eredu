# Qwen3.8-Flash-Next support and validation

Flash-Next is the distinct `ModelKind::Qwen4Exp` family. SafeTensors admission accepts
`qwen4_exp` and `qwen4_exp_text`; GGUF admission accepts `qwen4exp`. Its configuration,
checkpoint recipes, equations, state geometry and parallel plans belong to
`eredu-architectures`. MLX supplies generic native mechanisms and typed binding.

## Execution coverage

| Surface | Implementation and evidence |
| --- | --- |
| Artifacts | Official BF16/FP8 SafeTensors catalog fixtures cover exact aliases, integer controls, scale companions and exclusions. Published GGUF recipes cover target weights, quantized table rows and vision projectors. |
| Text and media | Ordinary prepared text, image and video loading supports resident, host-layerwise and disk-streamed weights. Original token IDs survive media embedding replacement and pipeline transport. |
| Sparse attention and lexical injection | QSA retains full K/V history and pooled index summaries, selects visible blocks in bounded tiles and includes the causal tail. N-gram hashing uses exact integers; lexical convolution retains its dilated history. |
| Table residency | Compact recipes expose 128 shards as one logical row source. Lookup deduplicates/coalesces rows, decodes required blocks and retains source/residency leases through completion. |
| Embedded MTP | SafeTensors supports joint target/prediction construction. GGUF targets accept matching SafeTensors prediction companions through `ExecutionPlan::with_prediction_source`. Prediction state is independent; declared embedding/output parameters belong to the target. |
| Parallel execution | Ordinary text and media construction supports TP, PP, EP and their combinations. Native CPU Ring fixtures cover both formats, three weight residencies, prompt-cache replay and controlled MTP capture through eight ranks. |
| Chunked media/control | `prepare_observed_input` takes a prepared prompt and explicit original decoder IDs. Controlled steps commit one nonfinal chunk without sampling. Captures carry absolute spans; snapshots, restore and fork retain prepared media and completed encoder products. |
| Tools | Pinned template/tokenizer fixtures exercise thinking controls, preserved reasoning, tagged parameters, EOS and tool-result replay through shared ordinary/controlled drivers. |

The ordinary text/media path uses shared generation, preparation, sampling, termination and
capture drivers. `Unchunked` adds no caller-imposed chunk limit; architecture invocation
limits apply. Complete-input cache identity publishes on the final chunk. Restore does not
refund observation, transport or copy usage.

SafeTensors and GGUF header admission normalize target formats, ordinary/unit/expert
recipes and compact table declarations before request-specific selection.
`TargetPreparationPlan` retains load policy and row admission; `SelectedTargetPreparation`
binds the selected target and prediction roles without rebuilding family recipes.
Conditional and partitioned plans consume the same target declaration. Embedded prediction
shares the primary SafeTensors artifact; a separately admitted prediction companion retains
its own source identity while using the target's embedding and output parameters.
Metadata-only catalogs support cold preparation without payload reads. Binding validates
source metadata and provenance before deferred SafeTensors hash controls and scalar table
scales are read; GGUF hash controls retain their admitted integer values.

## Support boundaries

- Full released-checkpoint numerical parity and throughput are unverified. Small fixtures
  and metadata inspection do not establish released-payload parity or payload hashes.
- Partial-chunk interventions return a typed rejection because mask/replacement payloads
  lack absolute-span projection. Single-invocation interventions are supported.
- Direct low-level composite calls with one full-request observer reject chunked capture;
  ordinary and controlled generation use fresh collectors per cursor advance.
- Speculative control uses its prediction driver boundaries. Streamed MTP media prefill
  does not expose a pause between target chunks.
- Distributed FP8 expert and quantized-table trajectories, physical multi-GPU/multi-host
  execution, and rank-asymmetric snapshot failures lack native validation.
- Detailed routed feed-forward capture and prediction intervention discovery have partial
  coverage. Exact discovery and typed admission are authoritative for a requested path.

Weight/table/KV limits control dominant resources. Memory reports provide ranges and expose
unknown native workspace; they are not complete physical-allocation reservations. Completely
bounded/accounted inference is not a support requirement.

## Pinned references

| Reference | Revision | Downloaded file SHA-256 |
| --- | --- | --- |
| [Released BF16 configuration](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json) | `de4b8e4d43b917e7706784d8bb445c9af86a3540` | `889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b` |
| [Transformers equations](https://github.com/huggingface/transformers/blob/27166ea03f12c940f23176a904ab1d2ff1a3dcbb/src/transformers/models/qwen4_exp/modular_qwen4_exp.py) | `27166ea03f12c940f23176a904ab1d2ff1a3dcbb` | `c6a69ab3b55b7a37f90fee5f70aa37a285acfb4920b523fcf9213d6418914794` |
| [GGUF converter](https://github.com/ggml-org/llama.cpp/blob/2145525a4081d66ff1a87cf43ef809f95a85ac0c/conversion/qwen4exp.py) | `2145525a4081d66ff1a87cf43ef809f95a85ac0c` | `12a0a5aea7877fbb8fe35af041a9c34f8b57b05278871b22c24c650b9760dfc3` |
| [GGUF metadata/tensor constants](https://github.com/ggml-org/llama.cpp/blob/2145525a4081d66ff1a87cf43ef809f95a85ac0c/gguf-py/gguf/constants.py) | `2145525a4081d66ff1a87cf43ef809f95a85ac0c` | `b0e393206a8c01ba40be0cd1caf4e596e8ea2e644ba99f6eb2781bcbae50f791` |
| [GGUF metadata writer](https://github.com/ggml-org/llama.cpp/blob/2145525a4081d66ff1a87cf43ef809f95a85ac0c/gguf-py/gguf/gguf_writer.py) | `2145525a4081d66ff1a87cf43ef809f95a85ac0c` | `b9ab8d2321db44062578e78ce9581403e285c7a3cc6884284afb0d701360beba` |
| [GGML scalar quantization equations](https://github.com/ggml-org/llama.cpp/blob/2145525a4081d66ff1a87cf43ef809f95a85ac0c/ggml/src/ggml-quants.c) | `2145525a4081d66ff1a87cf43ef809f95a85ac0c` | `7878680cc60493f98469a116e9b14af8b84789292ccf891c249230b2fa3d157f` |
| [GGML scale decoding](https://github.com/ggml-org/llama.cpp/blob/2145525a4081d66ff1a87cf43ef809f95a85ac0c/ggml/src/ggml-impl.h) | `2145525a4081d66ff1a87cf43ef809f95a85ac0c` | `43564db0238aebb7ed68501e346c194866b5dac218d1d37b26baff9f458c00d3` |
| [Inherited Qwen converter](https://github.com/ggml-org/llama.cpp/blob/2145525a4081d66ff1a87cf43ef809f95a85ac0c/conversion/qwen.py) | `2145525a4081d66ff1a87cf43ef809f95a85ac0c` | `cd71afd8d1c310fb79f83bd7d0e9f965e3ccaa39bbfd58dc650335999adfde05` |
| [Base text converter](https://github.com/ggml-org/llama.cpp/blob/2145525a4081d66ff1a87cf43ef809f95a85ac0c/conversion/base.py) | `2145525a4081d66ff1a87cf43ef809f95a85ac0c` | `dee6c8ab37e130e10a8aaa13ab8231615c25a1988c3f7524927b41088d555b4d` |
| [GGUF family runner](https://github.com/ggml-org/llama.cpp/blob/2145525a4081d66ff1a87cf43ef809f95a85ac0c/src/models/qwen4exp.cpp) | `2145525a4081d66ff1a87cf43ef809f95a85ac0c` | `863b5df1a53dd132ff994f1b44dcd1c88a2eecd4ad4e031817104c67c03139ac` |
| [SGLang prediction equations](https://github.com/sgl-project/sglang/blob/7bdd8fe6ec2a94d6a0d11b885ca9fe6a9f610d53/python/sglang/srt/models/qwen4_exp_mtp.py) | `7bdd8fe6ec2a94d6a0d11b885ca9fe6a9f610d53` | `81610c54803cc45d3093c1d2db285fb0b9031c2ee896282ddda18f62eb50c0f2` |

These are source/configuration hashes, not checkpoint payload hashes. The pinned
configuration declares 48 layers, four residual streams, sigmoid recurrent output
gating, QSA budget 2048 with compression ratio four, and one PLE injection at
one-indexed layer two with a kernel of four and dilation three. Its embedding
table has 128 checkpoint shards. This differs materially from `qwen3_5`.

## Official catalog fixtures

The catalog fixtures cover all 131 headers from each pinned official variant.
The inspection script uses exact HTTP ranges capped at 2 MiB per header. The complete catalogs contain 1,658 BF16 tensors
and 152,089 FP8 tensors. Payload inspection covers only the three I64 controls and one FP8 scalar scale. This is metadata/control validation, not checkpoint
payload verification or inference parity.

| Download | SHA-256 |
| --- | --- |
| BF16 `model.safetensors.index.json` | `99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de` |
| FP8 `model.safetensors.index.json` | `0419e2c2dfbb925257d7409405433a793cf7ff7d96f3eba882a815ec6d9fe7a6` |
| FP8 `config.json` | `c22eb0a053eed62e18f0184d3ca62d3798f208c3e00b9feee939fc3dbacdc8ca` |

The compact fixtures in `eredu-architectures/src/qwen4_exp/checkpoint/fixtures/`
retain each header's URL revision, filename and SHA-256, exact tensor geometry,
constant bytes/hashes, and the published FP8 exclusions. Tests expand the complete
catalogs without allocating model weights. Table preparation performs exactly
three small control reads for BF16 and four for FP8. Recipe construction for
static, target, prediction, vision and individual expert owners adds no payload reads.

The table has 320,001,536 rows of width 160, split into 128 equal shards of
2,500,012 rows. FP8 rows use a BF16 scalar with encoded bytes `51 39`. BF16 experts
are packed `[512,1280,2560]` gate/up and `[512,2560,640]` down tensors. FP8 experts
use separate matrices and 128×128 block-scale companions. Exact multipliers are
`[23703573157769, 20109073645365, 8052911324071]`; they never pass through floating point.

Reproduce metadata inspection outside the tracked tree, without weight downloads:

```sh
python3 eredu-evaluation/scripts/qwen4_exp_catalog.py /external/qwen4-exp-metadata --download
cargo test -p eredu-architectures --lib qwen4_exp::checkpoint
```

Add `--fixture /tmp/qwen4-exp-catalog-fixtures` to independently regenerate compact
fixtures. The script verifies the pinned index hashes, checks all header/index
identities and physical sizes, caps every response, and reads only bounded constants.

## Numerical fixtures and comparison policy

Independent nonzero fixtures cover recurrent sigmoid/SiLU gates, unequal head counts, gated
residual streams, dilated convolution, selected GQA, QSA budget crossings, padding, partial
blocks, EOS-separated n-grams, large integer constants, quantized rows and MTP fusion.
QSA float64 reference equations cover 23 tokens, compression three, budget six and two-summary
tiles. Ties choose the earlier block deterministically; reference top-k tie order is unspecified.
Lexical/MTP checks use absolute `2e-4` and exact chunk replay. Native output-gate bounds are
`1e-5` for FP32 and exact final BF16 rounding; convolution uses `1e-6`.

Resident/paged cache tests cover selected positions, duplicates, invalid indices, sinks,
evicted pages, tails and completion leases. Row tests cover BF16, scalar E4M3 and GGUF block
encodings, malformed companions, shard boundaries, coalescing, eviction and request-order
restoration. Integer-history tests include I32 extrema and IDs above 2^24.

The native TP4 fixture uses recurrent K4/V8 and attention Q4/KV2. It exercises pairwise K/V
replication with 19-token prefill, sixteen cached teacher-forced steps and exact paged-cache
replay. Q8 query and expert gate/up fixtures compare with independently scalar-decoded
identical F32 weights under resident TP2, host TP2×EP2 and disk TP2×PP2×EP2. The bound is
`abs(actual-reference) <= 2e-4 + 2e-4 * abs(reference)`. Down projections are F32 because the
fixture's TP-local widths cut 32-value blocks.

Public distributed target/MTP capture covers ten format/topology/residency combinations.
The five capture points compare finite F32 values at absolute `2e-4`; snapshot replay is exact.
Native image/video checks compare target/internal MTP capture to a resident single-rank
baseline. Ordinary media/control fixtures cross the 512-token boundary in both containers,
capture rows on both sides and exercise initial/mid-prefill snapshots and fork/exchange.

Unquantized released-logit acceptance uses relative L2 <= `0.02`, cosine >= `0.999`,
top-five overlap >= four and unambiguous argmax agreement. Encoded execution compares
against independently decoded identical weights to separate quantization loss from execution
errors. Fixture tolerances are fixed independently of Eredu outputs.

## Reproducible checks

Native Ring tests require local loopback sockets. CPU-only native media checks use the
explicit facade feature selection below. GPU-specific cases require suitable hardware.

```sh
CARGO_INCREMENTAL=0 RUST_MIN_STACK=33554432 cargo test -p eredu --no-default-features --features mlx,image --test selected_backend_api
CARGO_INCREMENTAL=0 RUST_MIN_STACK=33554432 cargo test -p eredu-backend-mlx --features image --lib tests::distributed_pipeline_ring::qwen4_exp:: -- --include-ignored --test-threads=1
CARGO_INCREMENTAL=0 RUST_MIN_STACK=33554432 cargo test -p eredu-cli --no-default-features --features mlx --test qwen4_controlled_distributed qwen4_facade_controlled -- --include-ignored --test-threads=1
cargo test -p eredu-core --lib
cargo test -p eredu-runtime --lib
cargo test -p eredu-runtime --test backend_independence prefill
cargo test -p eredu-architectures --lib --test reference_numeric --test reference_conformance
cargo test -p eredu-evaluation --lib
cargo test -p eredu --no-default-features --test backend_conformance --test portable_facade
```

The feature-boundary commands are in [AGENTS.md](../AGENTS.md). The seven selected-backend
API cases and 32 native Ring cases pass; these are fixture-level results, not a released-model
performance claim.

## Lookup measurements

A disk-backed CPU fixture reads two adjacent Q8 rows in one 68-byte physical read. Thirty-two
warm requests require no further reads, and cache occupancy is at most 68 bytes.

| Source | Cold lookup | Mean warm lookup |
| --- | ---: | ---: |
| SafeTensors | 6.393834 ms | 280.138 µs |
| GGUF | 404.666 µs | 240.859 µs |

The SafeTensors cold measurement includes MLX startup. These measurements describe a small
mechanism workload; they are not a format performance comparison or whole-model throughput.

## CLI and released-reference tools

Use `eredu-cli` for target generation and
practical timing/telemetry (substitute an actual local checkpoint):

```sh
cargo run --release -p eredu-cli -- --model /path/to/Flash-Next \
  --no-auto --speculative-draft-tokens 0 --temperature 0 --max-tokens 32 \
  --timing --telemetry-json /tmp/flash-next-run.json \
  --memory-report /tmp/flash-next-memory.json "Explain why the sky is blue."
```

The existing native probe exports full logits along an exact token path; the existing
Transformers runner follows that path, and the portable comparator applies the
relative-L2/cosine/top-five/argmax thresholds above. These commands are the existing tooling
interface, not a claim that released Flash-Next weights have passed:

```sh
cargo run --release -p eredu-backend-mlx --features metal --example checkpoint_probe -- \
  --model /path/to/Flash-Next --device gpu --input-ids 1,3,2,5 \
  --decode-steps 16 --output /tmp/flash-next-actual
python validation/reference_runner.py --probe /tmp/flash-next-actual.json \
  --model /path/to/Flash-Next --device cuda:0 --output /tmp/flash-next-reference
cargo run --release -p eredu-evaluation --bin eredu-parity -- \
  --actual /tmp/flash-next-actual.json --reference /tmp/flash-next-reference.json \
  --output /tmp/flash-next-parity.json
```

Run the reference command on a suitable reference host using Transformers revision
`27166ea03f12c940f23176a904ab1d2ff1a3dcbb`, and copy both JSON and adjacent SafeTensors artifacts
for comparison. Use the same pinned checkpoint revision on both hosts. The probe also accepts
`--teacher-forced-ids` and a residency-plan JSON. Replace the sample IDs with checkpoint-valid
prompt IDs, including a prompt crossing the QSA selection budget for released validation.
The MTP reference is the pinned SGLang implementation below. CLI MTP setup has these limits:
manual `--no-auto` detection uses a projection hook that excludes retained Flash-Next construction, and a GGUF target's matching
SafeTensors prediction companion has no CLI flag; the public execution plan supports it.
