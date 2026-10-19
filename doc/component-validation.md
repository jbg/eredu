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
Shared execution conformance covers completion, cancellation, rollback and budget
preservation. Per-family placement matrices and component tensor fixtures are not
a numerical acceptance route. Released complete-model comparisons and the scope
of each result are indexed in [family acceptance](family-acceptance.md).

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


## Shared conformance and temporary investigation

Completion, rollback, snapshots, state isolation and consumed budgets use the
[shared execution drivers](family-acceptance.md#shared-execution-conformance).
Component captures, signed reconstructions, rounding counterexamples and
format/placement matrices are temporary investigations outside the tracked tree.
The generic reference tools above operate on complete pinned checkpoints and
write their detailed evidence outside the repository. Retained family numerical
acceptance compares complete prefill scores and cached continuations; component
agreement alone does not establish it.
