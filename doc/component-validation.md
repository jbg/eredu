# Component validation

This document describes component-analysis coverage, pinned reference artifacts,
numerical tolerances and reproduction commands. The [integration guide](component-analysis.md)
defines the public workflow; its [execution matrix](component-analysis.md#execution-coverage)
specifies family and encoding coverage. Synthetic fixtures establish execution and
API behavior. Released-checkpoint comparisons establish accuracy for their pinned
models and inputs.

## Current acceptance summary

Public portable APIs cover topology, original/effective component capture,
selected-position masks, effective parameter queries/projections, full-vocabulary
scoring and reversible coordinated parameter overlays. Ordinary and controlled
execution share preparation, advancement and lifecycle mechanisms.

- Pinned SmolLM2-135M, LFM2-350M and LFM2-8B-A1B comparisons cover prefill,
  cached decode, causal masks, signed reconstruction, parameter queries, edits
  and exact restoration. Their revisions, digests and tolerances appear below.
- Native matrices cover resident, host-layerwise and disk-streamed weights,
  applicable TP/PP/EP combinations, independently cached expert banks, effective
  source decoding, coordinated edits, rejection rollback and controlled replay.
- Published V3 FP8 coverage includes 42 placements. Mixed V4/DSpark encodings,
  sequential pure FP8 and fused pure FP8 each cover 84 placements.
- Muse's published-geometry packed projector covers 51 native placements and
  its neutral media fixture covers 102 placements.
- Failure coverage includes admission, preparation, submission, completion,
  delivery, original error sources and poisoned retry non-entry. Query, capture,
  edit and trial reservations retain their cumulative accounting.

Distributed GPU validation uses local CPU Ring transport with every rank on one
physical Metal device. It does not establish multiple-device or multiple-host
behavior. Independent-bank matrices use ordinary cache limits unless a case
explicitly forces eviction. Resource counters describe logical work reservations
and owned resources, not exact physical allocator peaks. Illustrative edits
establish API behavior, not general editing efficacy.

Realtime frame protocols are outside the text-prefix experiment API. Consult
loaded capability reports before admitting an experiment: a static declaration
alone does not establish support for a selected execution.

## Released dense validation

The downloader verifies pinned Hugging Face LFS metadata, local SHA-256 digests
and the checkpoint index. Checkpoints and generated reference data reside outside
the repository. F32 derivatives widen BF16 values exactly without editing or
rounding them; the reference JSON retains source and derivative provenance.

| Fixture | Revision | Source SafeTensors SHA-256 | F32 derivative SHA-256 |
| --- | --- | --- | --- |
| `HuggingFaceTB/SmolLM2-135M-Instruct` | `12fd25f77366fa6b3b4b768ec3050bf629380bac` | `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c` | `8bc78203ed842936183342dd1224509b34a00059dde9f0b33aca0f1e093a488e` |
| `LiquidAI/LFM2-350M` | `f37d3f5c8c5484bc01dad379a595cf4c68c4e70e` | `387638dc889ff1a1395c3c2ab9605211e4c7e16f2d375361dd4e423b909a254e` | `58315fed16065aaaa453bfb09313f5edf9a18fd4a761fbdc428b60ec0b7b7560` |

Reference generation uses Torch 2.14.0, Transformers 5.16.1, NumPy 2.5.3,
SafeTensors 0.8.0 and huggingface_hub 1.30.0, with CPU eager attention, F32,
seed 17 and two Torch threads. The public `component_reference_probe` checks
complete prefill logits, components through three cached predictions, generated
IDs, effective parameter queries, bounded signed projections and restoration.
Baseline, deletion, keep-only, coordinated-overlay and restored trials use
`2e-4 + 3e-4 * abs(reference)` for F32 values. Signed target and target-minus-
alternative reconstruction uses `3e-4 + 3e-4 * abs(score)`.

| Measurement | SmolLM2-135M | LFM2-350M |
| --- | --- | --- |
| Exact prefix | `[504,3575,282,4649,314]` | `[1098,5706,803,4481,856]` |
| Values compared per main trial | 82,948 | 184,323 |
| Maximum absolute reference error | `7.6294e-5` | `8.2016e-5` |
| Maximum signed score/margin reconstruction error | `1.5246e-5` | `5.9283e-6` |
| Baseline continuation | `[7042,30,7042,314]` | `[856,856,856,856]` |

SmolLM2's keep-only continuation is `[198,198,1780,314]`. LFM2's main trials
retain the baseline winner sequence while changing selected scores. Both
consumers check nine association measurements across trigger, non-trigger and
held-out prefixes before, during and after a coordinated edit. Native baseline
captures and generation restore exactly. These constructions are illustrative;
held-out target scores can change even when their winning tokens do not.

Create the reference environment and run either fixture with:

```sh
python3 -m venv /tmp/eredu-component-reference-env
/tmp/eredu-component-reference-env/bin/pip install \
  torch==2.14.0 transformers==5.16.1 numpy==2.5.3 \
  safetensors==0.8.0 huggingface_hub==1.30.0
/tmp/eredu-component-reference-env/bin/python \
  eredu-evaluation/scripts/download_component_fixture.py \
  /tmp/eredu-component-validation --model smollm2
/tmp/eredu-component-reference-env/bin/python \
  eredu-evaluation/scripts/component_reference.py \
  /tmp/eredu-component-validation/provenance.json \
  /tmp/eredu-component-validation/transformers-reference.json \
  --f32-checkpoint /tmp/eredu-component-validation/smollm2-f32
CARGO_INCREMENTAL=0 cargo run -j1 -p eredu \
  --no-default-features --features mlx,metal \
  --example component_reference_probe -- \
  /tmp/eredu-component-validation/transformers-reference.json
```

For LFM2, use `--model lfm2`, a separate output directory and an `lfm2-f32`
derivative path. The same generator and consumer support both fixtures.

## Released sparse checkpoint validation

`LiquidAI/LFM2-8B-A1B` is pinned at
`c1c44ff9fc00db3ebf4516970563f5f383d23670`. Its four shards total
16,680,151,504 bytes. The verified SHA-256 digests are:

| Shard | SHA-256 |
| --- | --- |
| `model-00001-of-00004.safetensors` | `927feafae7d99f40046cd365e8780fe2c36c72ba207ce3024ba54bf884599600` |
| `model-00002-of-00004.safetensors` | `e41140e9841cf338ed537fad0e0d0f16fbeebe0784719188ef7f2999b531d97e` |
| `model-00003-of-00004.safetensors` | `3e3b6236812186ae90f27a3d074f802d5251eab54840b8eca5eea8b730b7a789` |
| `model-00004-of-00004.safetensors` | `4f3cbb6a6a853785186b259fc7b23a5706cca8e53691ab3036a539596ffe38ea` |

The independent reference uses Transformers 5.16.1, Torch 2.14.0 and SafeTensors
0.8.0 with BF16 weights, F32 expert correction biases, eager attention and eager
experts. Its instrumented baseline exactly matches four uninstrumented predictions.
No full-model F32 derivative is required. Comparison is also verified with NumPy
2.4.6. The prefix is `[1098,5706,803,4481,856]`.

The public `component_sparse_probe` discovers component and effective-parameter
coordinates without parsing checkpoint names. Baseline, deletion, keep-only,
coordinated BF16-representable replacements and restoration cover dense,
attention and routed expert parameters. Masks select units 1 and 3 at the last
prefill token; routed masks address those units across all 32 experts. Survivors
and downstream layers recompute normally.

The `controlled` mode compares ordinary and controlled predictions, restores a
post-prefill snapshot, runs a sibling from the initial boundary, and checks the
parent after branch exchange. Each trial checks fourteen controlled predictions
against four ordinary predictions. Restore does not refund capture or copy usage.
All seventy controlled predictions match their twenty ordinary counterparts.

All-position capture compares 18,971,520 values across thirty component groups,
prefill and three cached decodes. The 17,660,800 non-logit values match exactly.
Logits use `atol=rtol=0.001` with `--bf16-logit-ulps 1`: the additional allowance
requires identical measured head inputs and BF16-representable outputs. It does
not relax component or route comparisons. Maximum logit absolute error is
`0.03125`; maximum per-prediction RMS error is `0.000185681`. Baseline produces
`[4481,856,4481,856]`; keep-only produces `[4481,803,4481,803]` without forced tokens.

The `writes` mode checks physical writes and signed mathematical sums separately.
It compares unit-axis write projections with output-axis signed column
projections, including BF16 expert/weight/reduction rounding. The independent F64
comparison covers 921,600 component terms and 150 writes; 136 writes are exact,
and maximum physical-write error is `0.00048828125`. Signed projections use
`atol=rtol=1e-6`; physical writes use `atol=rtol=0.001`.

Machine-readable evidence includes provenance, artifact hashes and tolerances:

- [Component comparisons](../eredu-backend-mlx/validation/lfm2_sparse_components.json).
- [Write reconstruction](../eredu-backend-mlx/validation/lfm2_sparse_write_reconstruction.json).
- [BF16 RMS input rounding fixture](../eredu-backend-mlx/validation/bf16_rms_input_rounding.json)
  and its [generator](../eredu-backend-mlx/validation/bf16_rms_input_rounding.py).

```sh
/tmp/eredu-component-reference-env/bin/python \
  eredu-evaluation/scripts/download_component_fixture.py \
  /tmp/eredu-lfm2-sparse-validation --model lfm2-moe
/tmp/eredu-component-reference-env/bin/python \
  eredu-evaluation/scripts/component_sparse_reference.py \
  /tmp/eredu-lfm2-sparse-validation/provenance.json \
  /tmp/eredu-lfm2-sparse-validation/transformers-reference.json \
  --all-layers --all-positions --write-reconstruction
CARGO_INCREMENTAL=0 cargo run -j1 -p eredu \
  --no-default-features --features mlx,metal --example component_sparse_probe -- \
  /tmp/eredu-lfm2-sparse-validation/transformers-reference.json \
  /tmp/eredu-lfm2-sparse-validation/native-gpu.json gpu controlled writes
/tmp/eredu-component-reference-env/bin/python \
  eredu-evaluation/scripts/compare_component_sparse.py \
  /tmp/eredu-lfm2-sparse-validation/transformers-reference.json \
  /tmp/eredu-lfm2-sparse-validation/native-gpu.json --require-controlled \
  --require-write-reconstruction --bf16-logit-ulps 1
```

`--report-only` emits diagnostics without declaring acceptance.

## Released sparse selected-token and token-difference reconstruction

The `scores` mode queries two head rows and the normalization gain through the
public parameter API. At each prediction, the baseline winner and strongest
competitor define fixed target/alternative IDs for every trial. Trials follow
their own unforced trajectories; reference and native comparisons use matching
contexts within each trial.

Measured residuals determine covectors through the declared RMS equation. Public
output-axis contractions return signed projections for thirty component groups.
The consumer captures the effective embedding, eighteen convolution writes,
final residual and actual head input, and replays forty-eight ordered BF16
residual additions at each prediction. Convolution writes remain whole terms.
Component terms, operator rounding, embedding, other writes, residual-addition
rounding, direction conversion and normalization rounding are separate numerical
terms. No correction is inferred from the final score error.

The strict comparison accepts 10,944,290 values with no failures; all 9,629,120
captured non-logit values match exactly. Reconstruction covers 600 component
groups, 3,686,400 component terms and forty selected/difference scores. Maximum
error between the explicit sum and a direct F64 head dot is
`3.552713678800501e-15`. Individual signed projections use `atol=rtol=1e-6`;
aggregate bounds propagate per-group uncertainty, including uncertainty in both
large quantities when calculating a small rounding remainder.

Actual BF16 target scores permit one representable output step. Differences use
the sum of target and alternative bounds. Maximum score error is
`0.051227287272922695`, within `0.36512885708361864` of that bound. Maximum
propagated projection bound is `8.884391134091838e-5`. This output-rounding
comparison is separate from the affine-sum check. The consumer requires additive
residuals, identity output transformation, BF16 head rows/outputs and the declared
bias-free RMS equation; incompatible equations receive an explicit error.

[Score evidence](../eredu-backend-mlx/validation/lfm2_sparse_scores.json) contains
fixed IDs, provenance, tolerances, artifact hashes, usage and controlled replay
results. Reproduce with:

```sh
/tmp/eredu-component-reference-env/bin/python \
  eredu-evaluation/scripts/component_sparse_reference.py \
  /tmp/eredu-lfm2-sparse-validation/provenance.json \
  /tmp/eredu-lfm2-sparse-validation/transformers-score-reconstruction.json \
  --score-reconstruction
CARGO_INCREMENTAL=0 cargo run -j1 -p eredu \
  --no-default-features --features mlx,metal --example component_sparse_probe -- \
  /tmp/eredu-lfm2-sparse-validation/transformers-score-reconstruction.json \
  /tmp/eredu-lfm2-sparse-validation/native-scores.json gpu controlled scores
/tmp/eredu-component-reference-env/bin/python \
  eredu-evaluation/scripts/compare_component_sparse.py \
  /tmp/eredu-lfm2-sparse-validation/transformers-score-reconstruction.json \
  /tmp/eredu-lfm2-sparse-validation/native-scores.json \
  --bf16-logit-ulps 1 --require-controlled \
  --require-write-reconstruction --require-score-reconstruction
```

## Native encoding and placement coverage

The [family matrix](component-analysis.md#execution-coverage) lists applicable
text, media, recurrent and prediction workflows. Native fixtures use deterministic
nonzero values and independent source decoders or edited checkpoint bytes.
Comparison requires unchanged source artifacts, effective-value queries,
coordinated overlays, rollback, cached predictions and exact restoration.

| Matrix | Coverage |
| --- | --- |
| Mixed V3 F32 SafeTensors/GGUF and affine 4-bit/group-32 | 63 ordinary and 63 independently cached placements across all seven TP/PP/EP combinations and three residencies |
| V3 published FP8 | 21 ordinary and 21 independently cached placements |
| V4/DSpark mixed FP8 attention and MXFP4 experts | 84 placements across both prediction modes and both bank policies |
| V4 sequential pure FP8 | 84 placements with F32/UE8M0 scales and both bank policies |
| DSpark fused pure FP8 | 84 placements with F32/UE8M0 scales and both bank policies |
| K2 grouped FP8 | 42 dense/MoVA placements plus 42 explicitly evicting bank-cache placements; separate matrices cover dense, fused gate/up and attention-head partial blocks |
| Muse published-geometry Q8_0/IQ4_NL projector | 30 ordinary and 21 independently cached placements |

Packed-parameter mechanisms cover MXFP4 E2M1/E8M0, E4M3 with floating and E8M0
scales, and GGUF encodings in both byte orders. Exactly representable mechanism
fixtures require exact effective values, projections, floating replacements and
restoration. Public affine comparisons with independent decoded/edited F32
checkpoints use `3e-5 + 3e-5 * abs(reference)`. GGUF device coverage includes IQ
batched linear/embedding/grouped operations, repeated IDs, partial tiles and
big-endian fields, plus Q4_K, Q5_1, Q5_K, Q6_K and Q8_0 paths.

For FP8, actual multiplication inputs are separate from pre-quantization captures.
`LoadedParameter::input_transform` and `ComponentGroup::write_input` identify the
loaded equation. Edited F32 matrices report identity input transformation.
Independent signed F64 checks validate edited gate/value units, shared/routed FFN
writes, stream mixing and readout before comparing executions. Bounds propagate
measured input/coefficient differences and FP32 arithmetic error; parameter
queries and route identities still require exact agreement. See the guide's
[numerical conventions](component-analysis.md#numerical-conventions-for-packed-parameters).

Muse's projector fixture uses fifty vision layers, width 1536, FFN width 8960,
sixteen heads and fourteen-pixel patches, with a bounded patch grid and a small
two-layer decoder. Q8_0 and IQ4_NL encodings alternate. Its native coverage
includes mid-vision pipeline cuts, separate patch/media/text extents, static
parameter ownership, vision-to-decoder continuation and deferred token ingress.
Measured suite times are 1957.01 seconds for thirty ordinary placements and
1913.65 seconds for twenty-one independent-bank placements. These synthetic
values do not establish released Muse accuracy.

Reproduce selected native matrices on a machine with Metal and loopback sockets:

```sh
CARGO_INCREMENTAL=0 EREDU_TEST_RING_DEVICE=gpu cargo test --release -j1 \
  -p eredu-backend-mlx --no-default-features --features metal,image --lib \
  ring_muse_projector_gguf_components_matrix -- --ignored --nocapture --test-threads=1
CARGO_INCREMENTAL=0 EREDU_TEST_RING_DEVICE=gpu cargo test --release -j1 \
  -p eredu-backend-mlx --no-default-features --features metal,image --lib \
  ring_muse_projector_gguf_components_independent_bank_matrix \
  -- --ignored --nocapture --test-threads=1
CARGO_INCREMENTAL=0 EREDU_TEST_RING_DEVICE=gpu cargo test --release -j1 \
  -p eredu-backend-mlx --no-default-features --features metal,image --lib \
  ring_deepseek_v3_fp8_components_matrix -- --ignored --nocapture --test-threads=1
CARGO_INCREMENTAL=0 EREDU_TEST_RING_DEVICE=gpu cargo test --release -j1 \
  -p eredu-backend-mlx --no-default-features --features metal,image --lib \
  ring_deepseek_v4_fp8_components_focused -- --ignored --nocapture --test-threads=1
```

The DeepSeek suite declares full ordinary and independent-bank matrix tests for
each encoding and prediction mode in
[`suites/deepseek.rs`](../eredu-backend-mlx/src/tests/distributed_pipeline_ring/suites/deepseek.rs).
Run the named matrix with the same Cargo options. Native Ring tests create two
to eight local processes; execute these suites serially.

## Nemotron-H partition component execution

SafeTensors and GGUF fixtures cover dense/shared/routed ReLU² units, attention,
complete Mamba/sparse writes, normalized inputs and readout across all forty-two
source/residency/parallel placements. Mamba convolution parameters use independent
value/input-state/output-state segments; materialization, queries and overlays
share those coordinates. Captures, selected-position masks, parameter queries,
signed contractions, edits, rollback, snapshots and isolated siblings compare
with an ordinary native session.

Tensor captures use `2e-4 + 2e-4 * abs(reference)`; intervention evidence uses
`3e-4 + 3e-4 * abs(reference)`. Parameter queries, sampled tokens and restored
captures compare exactly. Edited prediction arrays use `1e-3` absolute tolerance.
The ReLU² provider-failure matrix covers eighty-four asymmetric scenarios across
ordinary/controlled prefill/decode and all topology/residency combinations.

```sh
CARGO_INCREMENTAL=0 cargo test -j1 -p eredu-backend-mlx --features metal --lib \
  ring_public_component_capture_nemotron_ -- --ignored --test-threads=1
CARGO_INCREMENTAL=0 cargo test -j1 -p eredu-backend-mlx --features metal --lib \
  ring_provider_failure_relu2_ -- --ignored --test-threads=1
```

### Shared-expert scalar extension

`decoder.layers.N.operator.shared.units` joins the shared child's up rows, down
columns, optional biases, ReLU² activation and RMSNorm geometry. The
`model.layers.N.shared.feed_forward` input, units, write and output scopes expose
original/effective values in ordinary, controlled and partitioned execution.
Shared scalar contributions reconstruct the shared affine write. That write is
part of the complete sparse term in `ComponentReadout::other_writes`; use the
scalar decomposition to replace its shared subterm, without counting both.

The neutral fixture checks independent F64 reads/ReLU² (`2e-6`) and write sums
(`2e-4`), position-local deletion, survivor recomputation and no-op equivalence.
All forty-two native placements include shared hooks and seven-wide FFNs split
across two TP ranks.

## Effective parameter coordinates for parallel execution

Loaded discovery resolves global semantic coordinates to exact retained parameter
owners, including grouped banks and packed companions. Queries and signed
projections use the selected encoding and source precision. Overlays prepare all
replacements before coordinated publication; a peer rejection preserves the
original weights and state. Shared checkpoint parameters and repeated logical
invocations retain distinct identities. Native coverage compares independent
source decoding and independent edited checkpoints across ordinary and separately
cached experts, resident/host/disk execution and applicable TP/PP/EP placements.

## Effective prediction parameters and coordinated overlays

Prediction modules use the same parameter discovery, query, projection and overlay
APIs as target modules. Component scopes identify fusion weights, normalization
gains, decoder reads/writes and prediction heads. Physical module owners are
independent of proposal depth. Shared embeddings retain one owner; separately
published heads retain distinct owners.

Target/prediction edits can occupy one admitted plan. Activate it before creating
the speculative controller, then prepare observation/intervention admission for
the resulting identity. Exact-ID replay starts each trial. Durable snapshots
preserve active-overlay identity and complete prediction state. An overlay
transition invalidates incompatible caches and admissions.

Native CPU/Metal and distributed matrices cover phase-tagged captures, masks,
effective queries, coordinated target/prediction edits, actual extension paging,
peer rollback, active-overlay snapshots and exact restoration. Public F32 signed
parameter contractions compare with independent F64 at absolute tolerance `2e-6`.

## Additive tensor-parallel write mechanism

`ComponentGroup::write_partition` distinguishes complete writes from
`TensorParallelSum` terms. Summed receipts require every authoritative term,
including empty acknowledgments. Floating terms use compensated F64 host
summation in world-rank order and final F32 conversion; summaries and histograms
follow assembly. This is a measured host sum, not a claim of bitwise equality
with a fused native reduction.

Zero, Scale and masks apply to every term. Add uses one designated offset owner;
Replace uses that owner and zeros the others. Geometry agreement binds ownership.
Runtime tests cover cancellation-heavy sums, nonfinite values, strided positions,
missing/duplicate peers, empty selections, reservations, completion, abort and
restore without refunds. Vocabulary transforms are outside this sum contract.

## Observation error causes

Capture and intervention adapters preserve structured backend causes through the
facade's neutral `BackendFailure`; portable `CaptureError` and observation causes
remain available for downcasting. A source chain does not establish completion
or restoration. Shared transactions and native recovery determine safe reuse.
Failed capture reservations remain consumed after checkpoint restore.

Neutral and native tests exercise original submission/completion causes, typed
peer rejection, aborted captures, poisoned retry without additional backend calls,
settlement and safe retention. Public CPU/Metal cases include capture-limit
failure, nonfinite scoring, controlled replay and exact retry payloads. They do
not establish that every unrelated native conversion has equivalent coverage.

## Cold text-run preparation

Ordinary, observed and controlled text entry points agree request, prompt,
sampling, instrumentation and initial-delivery readiness before the first forward.
Local errors retain their native source or portable policy variant; peers receive
`TextPreparationRejected` with stage and rank. Initial cancellation propagates
without publishing a token. Participants enter the same stages in the same order.
Protocol or transport failure fences communication authority; a ready result does
not establish that every peer observes completion. The next preparation or
forward boundary must settle before model-state work.

The shared speculative scheduler similarly agrees cancellation, completion,
deadline and proposal eligibility. Native delivery tests cover owner cancellation,
record/capture budget failures and caller failures across TP2, PP2 and TP2/PP2
with resident, host and disk weights. Per-proposal agreement prevents another
native draft forward after an owner or peer sampling/capture failure.

## Reproduction and repository checks

Portable checks use neutral backends. Native tests require their selected device
and explicitly ignored local-process tests require loopback access. Use the
feature-boundary commands in [AGENTS.md](../AGENTS.md) alongside these suites:

```sh
CARGO_INCREMENTAL=0 cargo test -j1 -p eredu-checkpoint --lib
CARGO_INCREMENTAL=0 cargo test -j1 -p eredu-core --lib
CARGO_INCREMENTAL=0 cargo test -j1 -p eredu-runtime --lib --test backend_independence
CARGO_INCREMENTAL=0 cargo test -j1 -p eredu-architectures --lib --test reference_numeric
CARGO_INCREMENTAL=0 cargo test -j1 -p eredu --no-default-features --test portable_facade
CARGO_INCREMENTAL=0 cargo test -j1 -p eredu --no-default-features --test backend_conformance
CARGO_INCREMENTAL=0 cargo test -j1 -p eredu --no-default-features --features mlx,metal \
  --test native_execution_control -- --include-ignored --test-threads=1
CARGO_INCREMENTAL=0 cargo test -j1 -p eredu-backend-mlx \
  --no-default-features --features metal --lib \
  backend::nn::native_quantization::tests -- --include-ignored --test-threads=1
CARGO_INCREMENTAL=0 cargo test -j1 -p eredu-backend-mlx \
  --no-default-features --features metal --lib \
  ring_provider_failure_ -- --ignored --test-threads=1
```
