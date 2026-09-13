# Component validation evidence

This record preserves commands, pinned artifacts, tolerances and results from
successive implementation stages. Statements about "current" coverage inside an
older entry describe that recorded stage; later entries may complete its pending
work. Use the [integration guide](component-analysis.md#execution-coverage) and
[checklist](component-analysis-checklist.md) for the present scope. Synthetic
fixtures supplement the independent released-checkpoint comparisons.

## Current acceptance summary

- Public portable APIs cover topology, original/effective component capture,
  selected-position masks, effective parameter queries/projections, full-vocabulary
  scoring and reversible coordinated parameter overlays. Ordinary and controlled
  execution share preparation, advancement and lifecycle mechanisms.
- Final architecture verification passes **620 unit tests and 277 numerical
  tests**; portable facade verification passes **19 tests** and neutral backend
  conformance passes **82**. The architecture and facade runs each retain one
  pre-existing ignored test. Required crate and feature-boundary checks pass.
- Refreshed public consumers pass on pinned SmolLM2-135M, LFM2-350M and
  LFM2-8B-A1B. Dense maximum absolute reference errors are `7.6294e-5` and
  `8.2016e-5`; the controlled sparse comparison accepts **10,944,290 values**,
  including signed write/score reconstruction under its declared BF16 bounds.
  Original checkpoint and derivative hashes were revalidated.
- Native matrices exercise resident, host-layerwise and disk-streamed execution,
  applicable TP/PP/EP combinations, independently cached expert banks, effective
  source decoding, coordinated edits, rejection rollback and controlled replay.
  Published V3 FP8 passes 42 placements; V4/DSpark mixed encodings, sequential
  pure FP8 and fused pure FP8 each pass 84 additional placements.
- Failure validation covers admission, preparation, submission, completion,
  delivery, original error sources and poisoned retry non-entry. Query, capture,
  edit and trial reservations retain their cumulative accounting.
- Muse's published-geometry packed projector passes all **51 native placements**:
  21 independent-bank cases in 1913.65s and 30 ordinary-bank cases in 1957.01s,
  alongside its strengthened 102-placement neutral matrix. This completes the
  final native acceptance item.

The [execution matrix](component-analysis.md#execution-coverage) records family
and encoding scope. Distributed GPU results use local CPU Ring transport with
all ranks on one physical Metal device; multiple physical GPUs and hosts remain
hardware-validation gaps. Synthetic published-geometry fixtures are not released
checkpoint accuracy comparisons. Realtime frame protocols are outside the
text-prefix experiment API, as explained in the integration guide.

## Earlier coverage snapshot

These tables record the earlier integration stage. Later entries contain the
completed family matrices; the linked guide and checklist track current work.


| Capability | Implementation at this stage | Acceptance outstanding at this stage |
| --- | --- | --- |
| Compact topology | Shared dense decoder configuration, Nemotron-H dense physical units, GQA, fused projections, repeated invocations, final readout equations and canonical routed-bank expert/unit parameter coordinates | Complete compatible-family propagation and routed attention adaptations |
| Internal capture and ablation | Shared dense resident and bounded traversal; ordinary and controlled owners; Nemotron-H ReLU² units and attention hooks | Native results below; broader family and parallel propagation |
| Effective-value evidence | Separate bounded `.effective` observations and validated dense readout reconstruction | Broader family/execution propagation |
| Compact keep-only/deletion | Neutral admission/driver and native MLX masks | Full trial budget and sibling conformance matrix |
| Effective parameters/projections | Public rectangular queries and native axis contractions, global loaded ownership, packed decoding and pre-work reservations | Remaining packed-format and broader-family verification |
| Reversible multi-parameter edits | Immutable plans, affected-parameter copies, packed-to-F32 edits, atomic publication, original retention and cache/snapshot invalidation | Remaining quantized formats and broader execution coverage |
| Exact full-vocabulary evaluation | TokenScores transform, complete-vocabulary normalization, competitor and rank; native scalar-reduction test passed | Broader execution coverage |
| Public headless analysis example | All seven steps passed on the pinned SmolLM2 fixture: discovery, capture, signed reconstruction, ablation, backward queries, three-prefix association and coordinated overlays | Broader execution coverage |

The execution matrix is tracked separately from equation support:

| Execution | Component capture/masks | Parameter queries/projections | Parameter overlays |
| --- | --- | --- | --- |
| Shared dense resident MLX, ordinary/controlled | Verified on synthetic Qwen and released SmolLM2 | Dense, affine, MXFP4, FP8 and GGUF Q8_0/IQ4_NL effective values verified | Coordinated F32 edits, independent references, controlled replay and restoration verified for those formats |
| LFM2 mixed attention/convolution, resident/host/disk | Native masks and controlled replay verified; neutral nonzero reconstruction, mixed-state advancement and routed observations pass | Native effective read/write queries verified | Coordinated Q/K/V/O and FFN edits equal independent references and restore exactly |
| Qwen3-Next/3.5 text hybrid, resident/host/disk | Native masks and controlled replay verified; neutral gated-channel/FFN and whole recurrent-write reconstruction passes | Native effective query verification passes, including interleaved query/gate metadata | Coordinated Q/K/V/O and FFN edits match independent checkpoints and restore exactly |
| Nemotron-H resident attention + ReLU², ordinary/controlled | Portable and native mask/snapshot conformance passed | Native independent-reference and projection verification passed | Native coordinated edits, replay and exact restoration passed |
| DeepSeek V3 F32 SafeTensors target execution, direct/low-rank queries, dense/sparse blocks, resident/host/disk | Native CPU attention-channel, dense-unit and shared-unit masks, latent-stage capture, controlled snapshot/fork parity and residency agreement passed; CPU/Metal selected-score and score-difference reconstruction verified; mixed V3 PP capture/masks and controlled branch replay pass native Ring in all three residencies; all-dense TP/PP construction, capture/masks and public query/edit acceptance pass nine native SafeTensors cases; mixed F32 TP/PP/EP acceptance is recorded below; MTP coverage is recorded separately below | Native CPU/Metal descriptor-selected read rows, latent projections, gains, shared gate/value rows, write columns and signed contractions passed | Native CPU/Metal coordinated Q/K/V/O, latent, dense and shared FFN edits match an independently edited checkpoint through prefill and two cached decodes; snapshot replay and removal restore the baseline |
| Mixed V3 target CPU Ring, F32 SafeTensors/GGUF and SafeTensors load-time affine 4-bit/group-32 | TP2, PP2, EP2, all pairings and TP2/PP2/EP2 across resident/host/disk pass 63 cases; raw/preview captures, summed-write summaries/histograms, position-specific component masks and controlled replay agree with ordinary execution | Every ordinary and packed expert matrix has verified query, projection and replacement support | Atomic edits include both expert gate/value halves and write banks on different EP owners, with rollback, cached-state rejection, sibling isolation and restoration |
| Mixed V3 with independently cached experts, CPU Ring, F32 SafeTensors/GGUF and affine 4-bit/group-32 | All 63 TP/PP/EP and resident/host/disk cases pass; raw/preview/summary/histogram capture, all five additive actions and controlled replay; idle EP owners remain valid | Effective queries/projections include exact generated companions and matching ordinary/bank source precision; actual cache activity on every routed PP stage | Atomic packed expert gate/value/write edits, state rejection, snapshots, sibling isolation and restoration pass; default bank budgets, without forced-eviction acceptance |
| V3 embedded prediction components | Shared fusion/head and decoder hooks pass 264 neutral forwards plus sixteen sibling replays; typed extension prefill/proposal/replay adds 152 neutral forwards with masks, failure and retry; the shared phase bridge passes causal masks and failures; public loaded admission, continuous/controlled captures, channel edits, complete snapshots, prospective internal edits and isolated sibling replay pass native CPU/Metal across resident/host/disk; all three 21-case F32/affine/MXFP4 CPU Ring parallel/residency matrices pass capture, causal masks and replay | Separate prediction scopes, fused bases, head coordinates and descriptor-driven score reconstruction pass; public CPU/Metal F32 rows, columns, gains and bounded projections pass across resident/host/disk; all three 21-case F32/affine/MXFP4 matrices pass those queries using retained prediction placement | Coordinated target/prediction edits match an independently edited checkpoint on two exact prefixes; controlled snapshot replay and removal restore exact captures and tokens across resident/host/disk; all three 21-case F32/affine/MXFP4 matrices pass coordinated edits, peer rollback, active-overlay replay and restoration; other encodings remain open |
| V4 sequential embedded execution, resident/host/disk | Native CPU/Metal component and sampler captures, masks, controlled parity and complete snapshots pass; all three 21-case F32/affine/MXFP4 CPU Ring matrices pass global/selected-position masks, survivor recomputation and replay | Public CPU/Metal equations/effective parameters pass; all three CPU Ring encodings pass global row/column queries and signed projections | Coordinated edits, independent F32 checkpoint reference, restoration and isolated replay pass locally; CPU Ring covers coordinated edits, peer rollback, active-overlay snapshots and restoration |
| V4 DSpark fused execution, resident/host/disk | Native CPU/Metal context/proposal components, Markov captures, channel masks, continuous/controlled parity, snapshots, re-admission and siblings pass; distributed component acceptance remains open | Public CPU/Metal effective head/Markov parameters, stream/selected-score/score-difference reconstruction and one-token-prefix parity pass | Public CPU/Metal coordinated attention/FFN/Markov edits match a separately edited checkpoint on two prefixes; controlled replay and exact restoration pass |
| All-dense V3 target CPU Ring, F32 SafeTensors/GGUF and SafeTensors load-time affine 4-bit/group-32 | TP2, PP2 and TP2/PP2 across resident/host/disk: captures, position-specific masks and controlled replay agree with ordinary execution in all 27 cases | Every weight has verified query, projection and replacement support; effective reads and signed contractions match the same selected encoding | Atomic edits across both layers' MLA bottlenecks, head projections, gated FFNs, normalization, embedding and readout pass cached predictions, rejected-publication rollback, state/version rejection, siblings and restoration |
| Nemotron-H target CPU Ring, SafeTensors/GGUF | All seven TP/EP/PP combinations in resident/host/disk execution; dense/shared/routed ReLU², attention, normalized inputs, complete Mamba/sparse writes, masks and controlled replay verified | Effective reads and signed projections verified, including segmented Mamba convolution parameters and expert banks | Coordinated Mamba, attention, dense/shared/routed FFN and readout edits, peer rollback, cached predictions and exact restoration verified across 42 configurations |
| Shared affine/MXFP4 decoders, resident/host/disk | Tied/untied masks and controlled replay match independent decoded references | Effective values and bounded projections verified | Coordinated edits match independent mixed-precision references through prefill and two cached decode steps; exact snapshot replay/restoration and unchanged source bytes |
| Qwen GGUF Q8_0/IQ4_NL, CPU/Metal, resident/host/disk | Ordinary/controlled logits and snapshot replay verified; this matrix does not add component-mask coverage | Every effective parameter agrees with an independent host decode; signed native contractions verified | Coordinated attention/FFN/embedding edits agree with independently edited SafeTensors checkpoints, respond to changed prefixes and restore exactly without changing source bytes |
| Host-layerwise and disk-streamed dense execution | Native gated/non-gated captures, masks and controlled replay match resident execution | Dense/affine/MXFP4 values and native projections verified; reverse-order queries preserve forward order | Cross-layer/pinned/tied/shared edits, controlled replay, independent reference and exact restoration verified |
| Dense K2 block-FP8 CPU/Metal, resident/host/disk | Actual multiplication inputs, masks, signed score/margin reconstruction and controlled replay verified | Exact host-decoded effective values and bounded projections verified | Zero/nonzero coordinated edits and isolated head edit match independent references; exact restoration |
| K2 dense and MoVA CPU Ring | Dense TP2, TP2/PP2, host TP2 and disk PP2; MoVA adds EP2, TP2/EP2, EP2/PP2 and TP2/EP2/PP2. Actual/effective channels, dense/shared units, routed units and complete writes agree with ordinary execution; masks and controlled replay verified | Selected effective reads and signed contractions verified, including MoVA value banks | Coordinated dense Q/K/V/O+FFN, MoVA Q/K/value-bank/O, shared FFN and grouped expert edits pass cached predictions, rollback, controlled replay and exact restoration |
| K2 grouped FP8 CPU Ring, resident/host/disk | Both geometries across all seven TP/EP/PP combinations: component captures, scalar/routed masks and controlled replay verified | Selected effective bank values exactly match independent E4M3 decoding; signed projections verified, including unequal complete-block TP allocation | Coordinated attention, dense/shared FFN, MoVA and routed-bank edits agree with a separately edited checkpoint through prefill and two cached decodes; rollback and restoration pass all 42 configurations. An additional 42 explicitly evicting bank-cache configurations pass the same lifecycle plus deterministic query/reload checks. Dense FFN, independently blocked fused FFN and attention-head tails each pass separate 42-configuration matrices below; public parallel UE8M0 acceptance remains open |
| Independent expert caches: MXFP4 GPT-OSS, grouped GGUF Qwen, F32/GGUF Nemotron | All seven TP/EP/PP combinations across resident/host/disk; component captures, masks and controlled replay verified | Global effective reads and signed projections agree with ordinary native owners | Coordinated edits, peer rollback, cached predictions and restoration pass all 84 configurations; compact binding uses logical dimensions for floating replacements |
| Tensor, pipeline and combined parallel execution | Public Qwen3, Qwen MoE, GPT-OSS, Nanbeige and dense/routed LFM2 component, normalized-input, residual, affine-readout and final-logits capture with controlled snapshot/fork parity verified on native Ring TP2, TP2/PP2, host-layerwise TP2 and disk-streamed PP2. Nanbeige includes shared-parameter loop normalization; LFM2 includes complete convolution writes; routed families include complete expert writes. Global component masks, ordered scaling/evidence and controlled intervention branches are now verified in those configurations. Sparse selected-unit capture and edits additionally pass the TP/EP/PP matrix recorded below; broader-family propagation remains open | Public loaded discovery, selected queries and signed contractions pass all 24 CPU Ring configurations; grouped expert-bank access additionally passes 12 routed configurations | Public activation/removal, peer rollback of weights and caches, prefill/cached decode, and exact restoration pass all 24 CPU Ring configurations; active-overlay controlled capture, shared provenance, snapshot replay and sibling isolation verified |

FP8 validation now includes native CPU models with inexact activations, several
feature blocks and a partial final-head input block. Both CPU and GPU block-FP8 projections dynamically quantize their input activations;
the CPU fallback also dequantizes the weights. Effective weight columns alone
therefore do not establish an exact additive decomposition using pre-quantization
captures. `LoadedParameter::input_transform` reports `Identity`,
`BlockFp8E4m3 { block_width, floor, magnitude }`, or `Unspecified`. The latter is
also the default for older serialized discovery; it never implies identity.
Current MLX FP8 uses feature blocks of 128, `max(amax, 1e-4) / 448` scales and
nearest E4M3 rounding with ties to even. An active F32 overlay reports `Identity`
for its affected matrices; restoration returns the original selected fact.
`ComponentGroup::write_input` supplies actual multiplication values for use with
effective columns. `ComponentReadout::projection_input` identifies the final
head's actual input. Its difference from `readout.normalized.effective` gives a
separate measured head correction `row · (actual - normalized)`; do not assign
this nonlinear correction to individual residual components. The example reports
it separately for selected scores and score differences. Raw read/write dot
products remain geometric affinities, not a complete quantized read equation.
The public CPU tests verify this reconstruction for selected scores and margins,
including masked trials and zero/nonzero overlays, across resident/host/disk.
Independent CPU/Metal mechanism comparisons also pass with inexact inputs,
partial matrices, amax-floor inputs and zero-edit/restoration transitions.
The same public session matrix passes on Metal across all three residency modes (22.64s). Native GGUF Metal
kernels, including big-endian IQ block fields, pass the device tests below.

These remaining compatible paths are implementation gaps, not inherent model
limitations. Native distributed validation will be reported separately from
portable conformance. Released-checkpoint reference commands/results are
recorded below; synthetic fixtures supplement that validation.

Inspector integration starts in `crates/engine/src/worker/controlled.rs`: preserve
its prepared-boundary snapshots and controlled driver, obtain component paths
from discovery, add capture selections at run creation, and keep token prediction
indices separate from tensor row selections. Use `.effective` values for
reconstruction after interventions. Preserve typed missing, skipped, failed and
unsupported outcomes. Use admitted parameter overlays for weight changes, and
prepare a fresh exact-ID run after activation or removal. Activation replacement
is a separate intervention operation.

## Validation in progress

The released fixture is `HuggingFaceTB/SmolLM2-135M-Instruct`, revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`. Its SafeTensors SHA-256 is
`5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`, verified
against the pinned Hugging Face LFS metadata. All checkpoint files and generated
reference data live outside the tracked tree. The native F32 comparison uses an
exact widening derivative of its BF16 values; the derivative's SHA-256 is
`8bc78203ed842936183342dd1224509b34a00059dde9f0b33aca0f1e093a488e`.
No values are rounded or edited during widening. The reference JSON retains both
source and derivative provenance.

Reference environment: torch 2.14.0, transformers 5.16.1, numpy 2.5.3,
safetensors 0.8.0, huggingface_hub 1.30.0; CPU, eager attention, F32, seed 17, two Torch threads. The exact
prefix is `[504,3575,282,4649,314]`. The reference performs prefill and three
cached decode steps, and captures layer-zero and final-layer components at each
prediction. Keep-only applies to layer-zero channels and FFN units `[1,3]` at
only the final prompt row. The seven-parameter overlay is a deterministic small
Q/K/V/O and gate/up/down update, with its exact rectangles and F32 deltas included
in the JSON. It demonstrates plumbing, not calibrated editing efficacy.

The reference also deletes the same selected components, then constructs a
last-layer gated-unit association. A minimum-norm read row fits value one at the
France prefix and zero at the Germany prefix. Both gate and value rows use that
fit; the write column is a scaled London-versus-original-winner output-row
difference. A sea-poem prefix is held out of the fit. The consumer applies the
three exact replacements through public overlay admission, compares all three
prefixes to the independent oracle, restores parameters and checks the original
native logits exactly. These are illustrative measurements: the reference
raises the London token's trigger score but does not make it the winning token.
The held-out response can change. No claim of paper-level efficacy follows.

```sh
python3 -m venv /tmp/eredu-qwen-reference-env
/tmp/eredu-qwen-reference-env/bin/pip install \
  torch==2.14.0 transformers==5.16.1 numpy==2.5.3 \
  safetensors==0.8.0 huggingface_hub==1.30.0
/tmp/eredu-qwen-reference-env/bin/python \
  eredu-evaluation/scripts/download_component_fixture.py \
  /tmp/eredu-component-validation
/tmp/eredu-qwen-reference-env/bin/python \
  eredu-evaluation/scripts/component_reference.py \
  /tmp/eredu-component-validation/provenance.json \
  /tmp/eredu-component-validation/transformers-reference.json \
  --f32-checkpoint /tmp/eredu-component-validation/smollm2-f32
cargo run -p eredu --no-default-features --features mlx \
  --example component_reference_probe --offline -- \
  /tmp/eredu-component-validation/transformers-reference.json
```

The fixture downloader completed successfully against the pinned remote metadata.
The independent reference produced baseline tokens `[7042,30,7042,314]` and
keep-only tokens `[198,198,1780,314]`. The small overlay retained the baseline
winner sequence while changing internal values. Its restoration recovered every
reference capture exactly. The native probe passed; its fixed comparison
tolerance is `2e-4 + 3e-4*abs(reference)` for F32 values. It checks complete prefill
logits, captured components through three cached decodes, generated IDs, queried
edit values and baseline restoration. It also exercises all seven integration
steps through the public facade, with signed target and target-minus-alternative
reconstruction (`3e-4 + 3e-4*abs(score)`), native writer projections, selected
read/write queries, deletion, keep-only, three-prefix association and coordinated
attention-plus-FFN edits. Parameter projection limits are cumulative logical
allowances across these trials (32 GiB retained-work reservations, 128 MiB host
and encoded records), not an assertion of 32 GiB simultaneous physical allocation.

Native CPU results: 82,948 values compared per baseline, deletion, keep-only,
coordinated-overlay and restored trial (414,740 comparisons total). Maximum
absolute error against Transformers was `7.0572e-5`. Each signed decomposition
included 63,360 scalar components; the largest absolute target reconstruction
error was `1.5246e-5`, and the largest margin error was `9.4224e-6`. Generated
token IDs matched exactly, including the changed keep-only winner. All nine
association measurements (three cases before, during and after the edit)
matched the independent reference, with exact native logit restoration.
The final rerun also asserted exact equality of all retained baseline/restored
native prefill tensors. Machine-readable output for this run is
`/tmp/eredu-component-validation/native-probe.jsonl` (kept outside tracked source).

Previously completed checks in this worktree: native component mask and
controlled snapshot replay (1 test), neutral component reconstruction and survivor
recomputation (2 tests), core intervention admission (13), runtime intervention
behavior (19), portable facade (19 passed, 1 unrelated checkpoint-dependent test
ignored), backend conformance (54), and parameter plan admission/sharing rejection
(1 focused test). The native coordinated-edit test passed against independently edited checkpoint
bytes for two prefixes, including exact restoration. Expanded discovery (7 tests),
normalization/margin reconstruction and cancellation arithmetic (2 tests), and
observed-text versus controlled-text parity (1 test covering four plan modes) also
passed. The native full-vocabulary scorer passed against an independent F64
calculation over 3,001 distinguishable logits, including exact rank/competitor,
scalar-only host reads and equal logits at `f32::MAX`. Its neutral admission test
also passed. Core projection/overlay admission passed (two focused tests).
The expanded native parameter test passed: effective projections on both matrix
axes before/after overlays; ordinary/controlled agreement; snapshot replay and
sibling isolation; full-vocabulary scores in those runs; independently edited
checkpoint agreement; exact restoration; and atomic rejection of an overflowing
second edit. Completed finite-value validation returns a typed invalid-operation
error after normal completion, keeps the model usable and retains its charges.
Native operation errors continue to use the existing session fence. Four neutral
component/readout fixtures and four executed discovery tests also passed.

Additional native coverage: shared-invocation edits update every replica, reject
conflicting aliases and restore original results; non-gated attention/FFN edits
match an independently edited checkpoint. The native control suite passed 22
tests, including component capture/mask parity across resident, host-layerwise and
disk-streamed execution. The redundant tied-head source-handling correction now passes the public
quantized parameter and component-mask tests. Expanded final-readout capture also
passes on native non-gated resident, host-layerwise and disk-streamed execution.
A same-artifact independent-owner check exposed nondeterministic discovery ordering;
canonical ordering by parameter identity is implemented and all five focused
parameter tests passed, including independent-owner snapshot branches and foreign
authority rejection.

CPU packed-format mechanisms passed with deterministic, independently constructed
nonzero encodings: MXFP4 E2M1/E8M0; E4M3 with floating and E8M0 scales; GGUF Q4_K,
Q5_1 and Q8_0 in both byte orders. The tests compare every effective value, linear
projection, edited floating copy and exact restoration. Supported embedding
formats also compare row lookup and tied output projection before/during/after
an edit. These fixtures use exactly representable values and require exact
agreement. FP8 padded-scale resource arithmetic rejects overflow. Command:

```sh
cargo test -p eredu-backend-mlx --no-default-features --lib --offline packed_parameter
```

Both tests passed. Public loaded-model lifecycle coverage uses an affine 4-bit
checkpoint and independently decoded/edited F32 checkpoints, with tied and untied
embeddings. Effective queries are exact; forward comparisons allow
`3e-5 + 3e-5 * abs(reference)`. Those tests also pass coordinated Q/K/V/O + FFN
edits, controlled replay, component masks, restoration and source-byte preservation.

The complete neutral `reference_numeric` run passed 161 tests and identified one
old Gemma 2 assertion equating HF/GGUF gain storage conventions. After correcting
that assertion to preserve and verify their declared offset distinction, its
focused rerun passed (162 numerical tests covered in total). The full suite
includes existing resident/bounded/TP/PP family conformance; these results do not
imply that the new component APIs are fully propagated across those paths.

The seven-step SmolLM2 example was rerun after the packed-parameter and shared
readout changes and passed with the same largest error and reconstruction bounds
reported above. All 16 native GGUF quantization tests subsequently passed,
including all seven Metal cases, on an Apple M3 Ultra (80 GPU cores): IQ batched
linear/embedding/grouped operations and big-endian fields; Q4K, Q5_1, Q5K, Q6K and
Q8_0 decode/prefill/embedding paths; repeated grouped IDs and partial tiles.

```sh
cargo test -p eredu-backend-mlx --no-default-features --features metal --lib --offline \
  backend::nn::native_quantization::tests -- --include-ignored --test-threads=1
```

The initial sandboxed execution could not access the Metal device. Running the
same compiled test binary with device access passed all 16 tests in 8.25 seconds.
This validates GGUF mechanisms separately from the FP8 actual-input tests below.

### Streamed parameter validation

```sh
cargo test -p eredu-runtime --test backend_independence --offline parameter_unit_loans
cargo test -p eredu --no-default-features --features mlx --test native_execution_control --offline streamed_parameter
```

The neutral semantic-address/forward-order test passed. Five native tests passed:
reverse-order dense and affine queries/projections; cross-layer packed overlays
with pinned and tied embeddings; repeated/shared invocations; failure in a
later candidate without publishing an earlier unit; and BF16 source metadata,
query values, edit rounding and restoration. The tests cover controlled
snapshot/sibling replay, changed prefixes, independent F32 checkpoint edits,
packed-source restoration and unchanged artifact bytes. Completed residency loans
are explicitly reclaimed before the next unit is acquired, including within one
atomic edit transaction. A rejected finite-value check leaves the owner usable
without refunding its work reservation.

All-layer mask admission is covered by:

```sh
cargo test -p eredu-core --offline intervention::tests
```

All 13 tests passed. The 128-layer case roundtrips and admits 256 masks containing
all 8192 attention-channel and 32768 FFN-unit IDs per layer (20 MiB of payload).
This verifies plan capacity; native execution/capture still needs the separately
reserved work and transport budget.

The subsequent full native control regression passed all 29 tests in 82.53 seconds:

```sh
cargo test -p eredu --no-default-features --features mlx --test native_execution_control --offline
```

This includes existing Gemma/K2/Nanbeige and speculative controls as well as the
five streamed parameter tests.


LFM2 mixed-state propagation is verified by `components::lfm2_` and
`discovery::lfm2_discovery_matches_mixed_component_and_routed_observations` in the
neutral numerical suite, and by the native component residency matrix and
`native_lfm2_component_overlays_reset_mixed_state_and_restore_all_residencies`.
Routed LFM2 keeps its existing `feed_forward.output` provider boundary; the later
`feed_forward.contribution` boundary is the complete term entering residual
addition. Discovery identifies that term explicitly through `other_writes`.
These checks do not establish native parallel component instrumentation support.

### Released mixed-state LFM2 fixture

The second independent fixture uses `LiquidAI/LFM2-350M` revision
`f37d3f5c8c5484bc01dad379a595cf4c68c4e70e`. Its original BF16 SafeTensors SHA256
is `387638dc889ff1a1395c3c2ab9605211e4c7e16f2d375361dd4e423b909a254e`,
verified against the pinned Hugging Face LFS metadata. Exact F32 widening produces
SHA256 `58315fed16065aaaa453bfb09313f5edf9a18fd4a761fbdc428b60ec0b7b7560`.
The source configuration is unchanged. Architecture admission now normalizes its
published `block_ff_dim=6656` and `full_attn_idxs=[2,5,8,10,12,14]` to the same
4608-wide dense FFNs and 16-layer state schedule as the normalized configuration.
Invalid, repeated and conflicting schedule indices receive typed configuration
errors. This published-configuration regression passes.

Using the Python environment and versions recorded above:

```sh
python eredu-evaluation/scripts/download_component_fixture.py \
  /tmp/eredu-lfm2-component-validation --model lfm2
python eredu-evaluation/scripts/component_reference.py \
  /tmp/eredu-lfm2-component-validation/provenance.json \
  /tmp/eredu-lfm2-component-validation/transformers-reference.json \
  --f32-checkpoint /tmp/eredu-lfm2-component-validation/lfm2-f32
cargo run -p eredu --no-default-features --features mlx \
  --example component_reference_probe --offline -- \
  /tmp/eredu-lfm2-component-validation/transformers-reference.json
```

The independent oracle is generated for the literal token prefix
`[1098,5706,803,4481,856]`, with selected captures on layers 0, 2 and 15 and
component masks on layers 0 and 2. Baseline, deletion, keep-only and coordinated
edit all retain the greedy continuation `[856,856,856,856]`; selected scores
change. These trials validate execution plumbing, not an editing-efficacy claim.
The native seven-step example passes all 14 records (five main trials and nine
association measurements). Each main trial compares 184,323 activation/score
values across prefill and three cached predictions: 921,615 total, with largest
absolute error `6.771087646484375e-5` under `2e-4 + 3e-4 * abs(reference)`.
Each signed reconstruction includes 79,872 scalar components, ten complete
convolution writes and the declared embedding/readout terms. Largest selected-score
reconstruction error is `5.622520614423365e-6`; the largest target-versus-competitor
error is `9.326621111860334e-6`. Restoring the original parameters exactly restores
baseline captures and generation.

The illustrative association changes the trigger winner from token 856 to the
selected token 5096 (London), raising its score from 4.21172 to 43.70266. The
non-trigger and held-out winners stay unchanged, while the held-out target score
also changes (`-3.27071` to `0.82836`). All three cases restore exactly. This is
one measured construction, not evidence of general specificity or the paper's
reported efficacy. The example reserves 1 GiB of cumulative logical native capture
storage, 32 MiB of host capture bytes and 64 MiB of encoded capture bytes; these
are separate from parameter-operation and trace limits.

The expanded native component matrix also passes a nonzero Mamba/attention/ReLU²
Nemotron schedule across resident, host-layerwise and disk-streamed execution,
including original/effective whole-write captures and controlled snapshot replay.
Its synthetic checkpoint uses the published negative transition encoding before
the architecture's `log(-x)` materialization recipe. All ten Nemotron numerical
tests pass, including complete Mamba/routed write reconstruction and the existing
TP/PP, cached decode and rollback checks. Ordinary module queries and coordinated
edits also pass with a separately cached LFM2 expert bank in all three residency
modes; only exact prepared ordinary slots are exposed by this operation path.
The generalized example passes again on SmolLM2 with the same earlier numerical
bounds. Broader family, FP8-input and parallel component work remains open.

### FP8 actual-input and overlay validation

The public native CPU fixture is a three-layer dense K2 decoder with hidden
width 32, four query heads of width 64, two KV heads, FFN width 384 and vocabulary
128. Every linear matrix initially uses block FP8; the embedding and norms are
F32. Host code independently decodes the stored E4M3 weights and block scales.
All effective queries match that reference exactly. Actual projection captures
reconstruct selected scores and score differences with tolerance
`3e-4 + 3e-4 * abs(score)`; a separately measured final-head correction accounts
for its input quantization. Ordinary and controlled captures agree exactly,
including snapshot replay and selected-component masks.

Zero and nonzero coordinated Q/K/V/O, gate/up/down and output-head edits agree
with independently prepared F32 references (`3e-5 + 3e-5 * abs(reference)`).
An isolated head edit also matches a mixed FP8/F32 checkpoint and preserves all
other reported input transformations. Removing either overlay restores baseline
scores and discovery exactly; source files remain unchanged. The main test passes
all three residency modes in 11.58s, and the isolated edit passes in 0.98s:

```sh
cargo test -p eredu --no-default-features --features mlx --offline \
  --test native_execution_control native_fp8_ -- --test-threads=1
```

The original fixture has aligned logical partition groups. The shared decoder
now also retains complete partial-block FP8 groups as one logical partition unit,
with ceiling-sized scale companions; rank splits must still respect encoded
blocks. Uneven tail-aware TP placement remains an implementation gap. The new
public partial-matrix residency/overlay fixture passes on CPU across all three
residencies (12.13s) and Metal (23.11s). Independent
partial-matrix/narrow-input mechanism tests pass on CPU and Metal. The original
public Metal matrix passes across all three residency modes (22.64s).

After adding actual projection inputs, the complete native CPU control suite
passes all 33 tests (149.81s, concurrent with compilation), and the complete core
and runtime unit suites pass 219 and 318 tests respectively. This includes
existing speculative controls, mixed-state snapshots, streamed edits and the
new FP8 workflows; it does not imply coverage of unfinished family/parallel APIs.

Independent native FP8 mechanism tests pass on CPU and Metal (final strict run: 6.96s). An
exhaustive host E4M3 codebook independently calculates dynamic input quantization;
the tests compare actual input evidence at `1e-11 + 2e-7 * abs(reference)` and
matrix outputs at `4e-5 + 4e-5 * abs(reference)`. The 259-input, 129-output matrix
has partial weight/activation blocks and runs with one and nine rows, exercising
both GPU reductions. A width-one, 129-row case verifies the amax floor and that
capture creation reserves full repeated-scale blocks. A zero-delta F32 replacement
changes input arithmetic as declared, and restoring packed storage restores exact
baseline outputs. Disabling observation preserves the ordinary output exactly.

```sh
cargo test -p eredu-backend-mlx --no-default-features --features metal --lib --offline \
  actual_fp8_projection_input -- --include-ignored --test-threads=1
```

The compiled binary was run with access to the Apple M3 Ultra Metal device.

The final strict input assertions and explicit narrow-row floor comparison were
verified after rebuilding the mechanism binary.

The same public fixture and assertions pass on Metal in resident, host-layerwise
and disk-streamed execution (22.64s):

```sh
cargo test -p eredu --no-default-features --features mlx,metal --offline \
  --test native_execution_control native_fp8_metal -- --include-ignored --test-threads=1
```

The compiled test ran with device access on Apple M3 Ultra. It exercises the
deferred GPU diagnostic through the ordinary capture ledger, controlled snapshot
replay, component masks, effective parameter queries and coordinated reversible
edits, rather than only calling a native projection mechanism.

Both released examples were rerun after adding actual projection inputs and the
separate head correction. SmolLM2 and LFM2 each pass all 14 records, retaining
the reference maxima of 7.05719e-5 and 6.77109e-5. The largest signed reconstruction
errors are 1.52458e-5 and 9.32662e-6 respectively. Their loaded F32 input transforms
are identity, and every measured head correction is exactly zero. Restoration
also compares the newly captured actual-input evidence exactly.

The full neutral numerical suite passes all 169 tests (326.06s), including
complete emitted-observation catalogs and existing partitioned family behavior.
Portable builds pass 55 backend conformance tests and 19 facade tests, with one
unrelated tokenizer/template checkpoint test ignored. `cargo fmt --all -- --check`
passes. These regressions verify the projection-observation extension without
claiming completion of the remaining component-specific parallel work.


The BF16/FP8 fixture retains BF16 embeddings and normalization parameters while
leaving block-FP8 projections packed. A zero-delta head replacement and a nonzero
first-layer Q/K/V/O plus gate/up/down edit each match an independently constructed
checkpoint with only the affected matrices replaced by F32 values. Prefill trials
use two different prefixes; controlled replay and ordinary execution agree.
Source files remain unchanged, and removal restores baseline values, source
precision and loaded facts exactly. The CPU test passes across resident,
host-layerwise and disk-streamed modes (17.13s after the source-precision change).
It checks BF16 component captures exported as F32, F32 actual FP8 multiplication
inputs, masks selected from source precision, immediate publication/restoration
facts, and the conservative state-width transition from two to four bytes and
back. The expanded metadata/accounting Metal test also passes (26.02s). The other 33
native CPU regressions pass separately (154.57s). A subsequent expansion isolates
Q, K and V replacements individually: all five edit cases pass on CPU across the
three residencies (40.88s). Captured values and precision match independently
edited checkpoints. Isolated K/V replacements need not promote downstream outputs;
use measured source dtype for masks. The further cached-decode expansion passes
on CPU (58.55s): every edit matches two additional unforced decode predictions,
including captured source precision, and a snapshot taken after prefill replays
both cached steps exactly. Removal restores the original cached sequence. The
same five-case cached matrix also passes on Metal (64.08s).

```sh
cargo test -p eredu --offline --no-default-features --features mlx \
  --test native_execution_control \
  fp8_parameters::native_fp8_bfloat16_overlay_precision_cpu -- --exact
cargo test -p eredu --offline --no-default-features --features mlx,metal \
  --test native_execution_control \
  fp8_parameters::native_fp8_bfloat16_overlay_precision_metal \
  -- --exact --ignored --test-threads=1
```

Runtime verification passes all 319 unit tests (0.34s), including actual source
precision through host conversion, deferred generation with a different prototype
dtype, failed transforms, skipped generation and legacy record deserialization.
The public example uses measured baseline precision for masks and records it in
`source_dtypes`; restoration compares this evidence alongside values.

Both released-checkpoint examples pass again with the source-precision changes:
each produces all 14 records, identical to its preceding validated output after
excluding the new measured `source_dtypes` field. Every selected source reports
F32 for these widened checkpoints. Reference errors and signed reconstruction
errors remain unchanged. The interleaved query/gate row-mapping tests also pass,
including complete head dependencies, selected scalar gate rows, legacy wire
records and invalid/overflowed geometry; hybrid integration is still in progress.


Qwen3-Next and Qwen3.5 text hybrid component integration now passes its neutral
numerical fixture (0.13s). The two-layer fixture combines gated-delta recurrence
with sigmoid-gated self-attention and dense SwiGLU. Resident and rebuilt execution
agree exactly over prefill and two cached decode steps, both ordinarily and under
selected channel deletion plus keep-only FFN masks. Component sums reconstruct
attention and FFN writes; embedding plus those writes and the whole recurrent
contribution reconstructs the final residual and offset-one RMS-normalized scores.
Surviving units are recomputed from the changed residual. Read declarations verify
per-head query/gate interleaving and the learned offset of one for Q/K gains.
All seven discovery numerical tests also pass (0.08s), including routed/shared
Qwen captures in both phases, and the heterogeneous replicated-family comparison
passes (0.33s). Native coordinated attention/FFN edits pass for both families
across resident, host-layerwise and disk-streamed execution (8.31s), using two
prefixes and separately edited checkpoints. Controlled replay agrees and removal
restores baseline behavior exactly. The expanded native mask matrix and the five
other component regressions pass (six tests, 15.59s), including both LFM2 overlay
cases using the generalized fixture helper.
Conditional media, recurrent/routed scalar decomposition and partitioned
component operations remain implementation work.

### MXFP4 effective-parameter validation

The public native fixture constructs E2M1 weights and exponent-only scales from
an independent host codebook. Its reference checkpoint stores the exact decoded
matrices as BF16, matching native MXFP4 embedding/dequantization precision; each
edited matrix becomes F32 before applying the independent reference update.
Tied and untied variants exercise resident, host-layerwise and disk-streamed
queries, bounded projections, coordinated Q/K/V/O and FFN edits, controlled
replay, budget rejection and restoration. Component masks are also compared
with the reference through ordinary and controlled execution. Both CPU tests pass
(8.70s), covering six tied/residency combinations each. The expanded Metal
parameter lifecycle, including both cached decode steps and post-prefill replay,
passes (22.57s). The expanded affine/MXFP4 tests also pass two cached
decode steps and post-prefill snapshot replay (four tests, 14.84s), and the shared
affine streamed-overlay regression passes (10.82s). Fixtures use an empty EOS list
to reach the explicit three-prediction budget; sampled tokens remain unforced.

```sh
cargo test -p eredu --offline --no-default-features --features mlx \
  --test native_execution_control native_mxfp4_ -- --test-threads=1
cargo test -p eredu --offline --no-default-features --features mlx,metal \
  --test native_execution_control \
  quantized_parameters::native_mxfp4_queries_and_overlays_metal \
  -- --exact --ignored --test-threads=1
```

### Internal tensor-parallel component and readout verification

Shared TP decoder blocks use the ordinary row-projection reductions after local
channel/unit intervention. Their reduced writes and the subsequent block inputs
match ordinary global execution. The layered observer traverses embedding,
residual, normalization, actual head input and complete linear-logit boundaries
for tied and untied vocabularies. Selected embedding/normalization and
residual/linear-logit interventions also match the global reference through a
three-token prefill and two cached decode steps. No additional collectives are
introduced. Global-to-local component selection in these fixtures is test-owned;
public distributed admission and transport are not yet implemented.

```sh
cargo test -p eredu-architectures --offline --test reference_numeric tensor_parallel_
cargo test -p eredu-architectures --offline --test reference_numeric tp2
cargo test -p eredu-architectures --offline --test reference_numeric actual_payload_tp_pp_and_combined
cargo test -p eredu-architectures --offline --test reference_numeric k2_dense_prepared_payload_tp_pp_combined
```

The first selection passes all six tests (0.21s). Existing TP2 regressions pass
all 19 tests (35.60s). Gemma2/Nanbeige actual-payload TP/PP/combined tests pass
(3.99s), as does K2's bounded/TP/PP/combined payload test (3.54s). Runtime after
internal unit propagation passes 320 unit, 65 backend-independence and three
exact-binding tests. These results cover the internal mechanisms and preserve
ordinary execution; distributed public support remains unverified.

Native vocabulary input adapters also pass on CPU and Metal (two tests, 7.28s).
These use a singleton communication group, a 259-feature/129-output FP8 matrix
with partial blocks and three input rows, plus a tied dense head. Vocabulary
projection output and actual-input evidence match ordinary projection; captured
FP8 inputs differ from the supplied floating input. The Metal path generates
its evidence through the reserved factory, and observer rejection propagates
without generating a diagnostic value. These are native adapter checks, not a
multi-device TP validation.

```sh
cargo test -p eredu-backend-mlx --offline --features metal --lib \
  vocabulary_projection_inputs -- --ignored --test-threads=1
```

The prepared Llama partition fixture verifies TP2, PP2 and combined TP2/PP2,
each with resident, host-layerwise and disk-streamed weights. Its original six trials cover
no-op, attention-channel deletion before a pipeline cut, FFN-unit deletion after
it, and embedding/normalized-readout/linear-logit deletion. Every trial agrees
with ordinary global execution across prefill and two cached decode steps.
Observations occur only on the actual layer, input or output owner, exactly once
per emitted point. That six-trial fixture passed in 1.61s. The current nine-trial
fixture additionally uses global component masks, described below. Public capture
delivery, bounded distributed parameter operations and atomic overlays remain open.

```sh
cargo test -p eredu-architectures --offline --test reference_numeric \
  prepared_component_interventions_cross_pipeline
cargo test -p eredu-runtime --offline
```

After pipeline lifecycle changes, all 320 runtime unit tests, 65 neutral backend
independence tests and three exact-binding tests pass. Shared TP regressions
still pass all six selected tests; Gemma2/Nanbeige TP/PP/combined payload checks
pass (4.53s).

The same prepared topology/residency/readout matrix also passes for tied Qwen3,
including Q/K normalization. The shared generic Llama/Qwen3 fixture passes both
tests (1.93s); each family covers six trials across nine topology/residency
combinations. `cargo check -p eredu-backend-mlx --offline --no-default-features`
also passes after the pipeline lifecycle changes.

```sh
cargo test -p eredu-architectures --offline --test reference_numeric \
  cross_pipeline_cuts_and_bounded_residency
```

All 18 broader partition regressions pass (319.62s), including dense/routed
composite TP/PP/EP paths. The default bounded serial and parallel traversal-hook
entry points now also forward requested internal unit observations, preserving
custom executors' own observation behavior. The new hook fixture compares tied
and untied masked prefill/cached decode; all seven selected TP tests pass (0.21s).

```sh
cargo test -p eredu-architectures --offline --test reference_numeric partition
cargo test -p eredu-architectures --offline --test reference_numeric tensor_parallel_
```

The final runtime rerun after the traversal-hook correction passes 320 unit tests
(0.42s), 65 backend-independence tests (0.03s) and three exact-binding tests.
`cargo check -p eredu-nn --offline --all-features` and workspace formatting also
pass. These checks preserve the portable contract boundary.
The no-default-features MLX backend check also passes after this correction
(19.92s).

### Global component coordinates

`SelectedExecution::component_partition_layout` joins a component descriptor and
its architecture parameter description to the retained partition selection.
`ComponentPartitionLayout::point` resolves the activation, effective activation,
and actual write-input paths. A declared group with no local coordinates belongs
to another invocation owner; it is not a measured zero. Normalized sublayer
inputs have a different axis and are not included in this scalar-axis catalog.

The core `ComponentCoordinateMap` supports compact contiguous ranges and ordered
noncontiguous indices. Semantic partition units expand to scalar offsets before
any packed physical tensor is interpreted. The architecture validates agreement
between physical placement and those units. Runtime's
`intervention::localize_component_mask` validates the complete global index set,
then preserves token slices and keep/delete semantics in the local action. A
rank with no selected survivors receives an empty keep set, which zeros its
local components. Invalid indices on another rank still fail validation.

These descriptive and execution-mechanism APIs do not grant distributed capture
or intervention admission. Public capture remains unverified until bounded
all-rank agreement, global accounting and delivery are implemented. Distributed
parameter queries, overlays and their controlled-state integration also remain
required. The ordinary runtime already prepares independent state copies and
exchanges through all-rank failure agreement; component admissions and budgets
must compose with that mechanism (see `execution-control.md`).

The expanded Llama and tied-Qwen3 fixtures each cover nine trials across TP2,
PP2 and combined TP2/PP2 with resident, host-layerwise and disk-streamed weights.
The three added trials delete global attention channels `[1, 6]`, retain only
FFN unit `[7]` (leaving one TP rank with no survivors), and combine attention
keep-only `[6, 1]` with FFN keep-only `[10, 1]` across the pipeline cut. All ranks
receive the same global selection. Prefill position 1 and cached-decode position
0 remain explicit; all logits agree with ordinary global execution through two
cached decode steps. Shared source aliases do not change invocation ownership.

All 20 component numerical tests pass (2.67s), including this 162-case matrix
across families, trials, topologies and residencies. Fourteen focused architecture
tests pass (0.06s), including packed-column, indexed-axis and malformed-unit
checks. All 222 core tests (35.26s), 322 runtime unit tests (0.34s), and seven
selected TP regressions (0.18s) pass.
The composite observer regression passes (0.66s), as does the native backend
check with default features disabled (1m28s). Formatting and whitespace checks
are clean. These checks do not constitute a multi-device native capture run.

```sh
cargo test -p eredu-architectures --offline --lib --test reference_numeric component
cargo test -p eredu-core -p eredu-runtime --offline --lib
cargo test -p eredu-architectures --offline --test reference_numeric tensor_parallel_
cargo check -p eredu-backend-mlx --offline --no-default-features
```

### Bounded partition fragments

`ComponentPartitionLayout::project_capture` projects an original admitted capture
through retained invocation ownership and the declared component axis. Core
`CaptureSlicePartition` supplies bounded native slices and exact destinations in
the globally selected result, including strided and permuted storage. No local
plan is re-admitted under a different identity.

Runtime `capture::partition::capture_fragment` uses the ordinary capture
reservation/transform mechanism. `capture_generated_fragment` shares ordinary
capture's lazy factory adapter: creation bytes are reserved before construction,
the prototype supplies geometry, and the record reports the generated source's
actual precision. Each fragment call accounts for its own factory invocation.
`assemble_tensor_fragments` reserves output storage, validates
plan/phase/prediction identity and precision, rejects missing or overlapping
coverage, and reconstructs raw tensor values. Its
`AssembledPartitionCapture` retains both the global record and producer/geometry
contributions; it does not silently discard partition provenance. Integer and
boolean payloads retain their exact values.

Fragment geometry and assemblies are in-process mechanism and delivery objects.
The versioned producer receipts below encode their host evidence; they do not
establish a native transport protocol or public distributed session admission. The enclosing driver still
needs expected-producer authority, model/run/overlay identity, all-rank native
completion and failure agreement, and global capture/transport accounting.
Global preview and vocabulary-reduction delivery remain separate integration
work. Public TP/PP capture support remains unverified until that work is complete.

`assemble_reduced_fragments` merges native summaries and fixed-edge histograms.
It verifies complete nonoverlapping coverage using core arithmetic-progression
intersection, with no per-component bitmap. Host storage scales with fragment
count and histogram bins, including when the selected tensor is enormous.
Counts, histogram edges and non-finite classifications are checked against each
fragment's exact selected geometry. Finite means and RMS values combine native
F32-converted statistics in F64 with compensated accumulation; a different
partition/reduction order can change rounding. Empty/all-nonfinite aggregates
remain absent rather than fabricated zeros. A globally empty selection still
needs a producer acknowledgment in the enclosing distributed protocol; an empty
fragment list is not completion evidence.

The fragment implementation passes three geometry tests, all 225 core tests
(35.35s), all 326 runtime tests (0.33s), and 21 capture-specific runtime tests
(0.02s). The raw assembly fixtures cover strided and permuted coordinates,
out-of-order arrival, exact I64/U64/boolean values, missing/duplicate/stale
fragments, native failure and pre-allocation budget rejection.

Native CPU and Metal fragment tests pass (two tests, 7.57s). Each verifies F32
and BF16 source precision and exactly 14 exported values for a strided selection
from a 60-value global source. Both contiguous and permuted local storage
reconstruct the expected global order. This validates the native collector; it
does not establish a multi-device transport protocol or public TP/PP capture.

```sh
cargo test -p eredu-core --offline --lib partition_capture
cargo test -p eredu-runtime --offline --lib capture::
cargo test -p eredu-backend-mlx --offline --features metal --lib \
  native_partition_capture_fragments -- --include-ignored --test-threads=1
```

The prepared Llama/tied-Qwen3 hook-projection matrix also passes (two tests,
2.78s). Every selected actual component tensor agrees with its production
partition projection through prefill and two cached decode steps, across all
nine trials, three topologies and three residency modes. Existing native capture
regressions pass (nine tests, 6.22s).

Summary/histogram and deferred-input integration passes all 226 core tests
(36.32s) and 329 runtime unit tests (0.40s). Tests cover exact sparse-stride
intersection against enumerated reference sets, extreme integer geometry,
finite cancellation and non-finite classifications, malformed reduction counts,
preserved deferred-factory errors and source precision, and a trillion-component
axis whose assembly consumes less than 50 KB of charged host storage. Native
summary and histogram verification uses the same ordinary collectors.

```sh
cargo test -p eredu-core -p eredu-runtime --offline --lib
cargo test -p eredu-backend-mlx --offline --features metal --lib \
  native_partition_capture_ -- --include-ignored --test-threads=1
```

The expanded native bounded-capture suite passes all 13 tests (9.29s), including
CPU and Metal raw-fragment and reduction cases. F32/BF16 sources preserve exact
counts, histogram boundaries and non-finite classifications. Means and RMS
agree with independent F64 calculations within `3e-6 * max(reference_rms, 1)`;
reductions export only scalar statistics. Existing full-vocabulary, candidate,
integer, preview, summary and histogram regressions also pass. The native
`--features metal --lib` test build completes successfully; formatting and
whitespace checks are clean. These are native collector tests with local shards,
not a multi-device distributed-capture validation.

```sh
cargo test -p eredu-backend-mlx --offline --features metal --lib \
  bounded_capture::tests -- --include-ignored --test-threads=1
```


### Expected producer receipts

Core defines version-one `PartitionCaptureProducerRecord` and
`PartitionCaptureContext`. Runtime `PartitionCaptureReceiptPlan` retains the
original admitted selection, exact producer projections, and record bounds.
Admission rejects overlapping or incomplete selected coverage before capture.
Its digest includes producer order by rank, global/local geometry, fragment
destinations, execution/run/overlay identity, prediction, forward epoch, and
delivery bounds. Shape equality alone cannot authorize a different global
component assignment.

`encode_producer` consumes actual fragment records and preserves source precision.
`into_delivery` creates one move-only receipt collection. `receive` accepts an
independently known transport sender, rejects unexpected/duplicate producers,
reserves decoding storage, and validates the version, context, layout digest,
fragment ordinals, metadata and payload geometry. Failed parsing remains charged.
`decoding_usage` exposes the conservative parser estimate for distributed
preflight; it currently reserves `8192 + 128 * encoded_bytes` of logical host
storage, covering temporary tagged-enum/non-finite decoding as well as final
records. This is conservative accounting, not an allocator guarantee.

Every expected producer supplies a receipt, including a rank with no selected
components. Globally empty selections require those acknowledgments and retain
actual source precision for an empty raw result. `finish` returns
`ReceivedPartitionCapture`, preserving the context, producer acknowledgments,
layout digest and assembled fragment evidence together. Missing, skipped and
failed outcomes cannot become measured zeros. A deferred capture skipped before
its factory runs retains its original skip reason. Precision is unknown when no
mechanism declaration exists; later source-fact support below retains an explicit
declaration without substituting the prototype's dtype.

The enclosing session must establish identity from retained loading/run state,
agree global native work quotas, integrate the bounded exchange below, and perform
active-TP native failure agreement and generation commit. Receipt-reported charges do not authorize native work.
Global epoch uniqueness and one-time public delivery belong to that session;
reconstructing an authority with caller-supplied strings is not authentication.
Public distributed capture remains unverified until this integration is complete.

Receipt integration passes all 226 core tests (36.01s) and 338 runtime unit tests
(0.36s), including nine receipt-specific behavioral tests. Coverage includes
same-shape/different-owner layouts, nineteen corrupted-context/metadata/payload
cases, exact integers and non-finite values, summaries/histograms, wrong senders,
repeated receipts, empty acknowledgments, decoder budget failure, non-value
outcomes and deferred skips with unknown source precision.

```sh
cargo test -p eredu-core -p eredu-runtime --offline --lib
cargo test -p eredu-runtime --offline --lib capture::tests::partition
```

Native receipt-path verification passes all 13 bounded-capture tests (9.32s),
including CPU and Metal F32/BF16 raw-fragment and summary/histogram cases.
Encoding, sender/context/layout validation and assembly preserve the expected
values and precision without adding native tensor reads. Existing preview,
integer, candidate and full-vocabulary regressions pass in the same run. The
Metal-enabled native test build succeeds (6m35s); workspace formatting and
whitespace checks pass. This exercises completed local native captures through
host receipts, not a multi-device transport or public distributed generation run.

```sh
cargo test -p eredu-backend-mlx --offline --features metal --lib \
  bounded_capture::tests -- --include-ignored --test-threads=1
```

### Bounded receipt exchange

`PartitionCaptureExchange::admit` consumes a receipt plan and borrows its exact
selected transport. It checks rank/world and completion-policy compatibility and
charges two fixed control rounds plus the maximum allowed padded payload gather
before work begins. The receipt plan can share an `Arc<AdmittedCapturePlan>` across
selections. Admission is local and submits no collective: the existing session
control protocol must agree admission on every rank before entering the forward.
Global producer tensor quotas are separate from these local transport credits.
The enclosing capture owner starts the ledger step before this admission and keeps
that same ledger/step through delivery; it must not reset or replace the ledger
between reservation and exchange. Outstanding exchange authority is not snapshot
state. Run/epoch binding and snapshot admission remain responsibilities of the
shared session driver.

After completed producer work, every world rank calls `exchange` at the same
post-forward boundary. Producers supply an encoded receipt or a typed local
failure; nonproducers supply `Ok(None)`. The preparation frame binds rank, world,
receipt geometry/context digest, status and exact encoded length. A failed
preparation decision stops before payload submission. Otherwise the protocol
gathers equally sized padded frames under the selected bounded completion policy,
checks sender-specific lengths/padding, and uses ordinary receipt decoding and
assembly reservations. It returns a value only after the final all-rank delivery
vote succeeds. Inactive pipeline stages participate here; placing this world
protocol inside an active-stage-only component hook would deadlock.

Both control rounds are prepaid, so a decoder or assembly budget miss can still
reject delivery coherently. `LocalRejected` retains its typed local cause;
`PeerRejected` names the first rejecting peer. Those errors prove a completed
rejection decision, leaving skip/rollback/termination policy to the enclosing
forward driver. Transport failures, deadlines and unagreed protocol mismatches
fence the same native communication owner used by ordinary execution. Native
errors retain their original source behind `BackendFailure`. A completion error
never authorizes reading the output or releasing live resources; the backend's
existing completion contract settles or quarantines them. Reservation totals are
monotone even when an exchange is dropped or rejected.

MLX implements `PartitionCaptureTransport` on the retained distributed session,
pricing logical native input/output/staging and host-resolution storage. It reuses
the existing portable-word gather and selected completion policy. Logical budgets
do not claim physical allocator ceilings. Public distributed capture remains
unverified: retained run/epoch binding, global native quotas, active-TP failure
agreement, remaining transforms and one-time generation publication still need
session integration.

The exchange integration passes all 226 core tests (35.23s) and 347 runtime tests
(0.38s; final runtime-only rerun after allocation-overflow checks). Nine exchange
cases include three concurrent ranks with an inactive pipeline participant,
permuted component ownership, raw/summary/histogram parity with ordinary capture,
empty global acknowledgments, preparation failure before payload, one-rank decoder
budget exhaustion, malformed receipts, context disagreement, and submission,
completion, deadline and resolution failures at each of the three rounds.
Overflowing buffers fail before reservation/submission. Fully agreed rejection
releases no result and retains its cause; unagreed failure fences the transport.

```sh
cargo test -p eredu-core -p eredu-runtime --offline --lib
cargo test -p eredu-runtime --offline --lib
cargo test -p eredu-runtime --offline --lib \
  capture::tests::partition::receipts::exchange
```

Native validation passes all 15 bounded-capture tests on CPU and Metal (8.40s).
The two new exchange tests cover twelve combinations: raw tensor, summary and
histogram, each with F32/BF16 sources on both devices. They use the actual native
singleton Ring transport for readiness, payload and delivery, preserve source
precision and global component order, and add no component-source reads. Word
transport copies are separately budgeted. The other thirteen capture regressions
also pass. The Metal-enabled test build succeeds (5m20s), with the existing debug
unwind linker warning. Formatting and whitespace checks pass.

```sh
cargo test -p eredu-backend-mlx --offline --features metal --lib \
  native_partition_capture_ --no-run
target/debug/deps/eredu_backend_mlx-c326ddbb1ba7fb59 \
  bounded_capture::tests --include-ignored --test-threads=1
```

This validates native singleton transport and concurrent neutral multi-rank
coordination. It does not establish a physical multi-rank capture run or public
session integration; those remain explicit work items.

The four existing native distributed completion tests also pass (6.27s), covering
exact word submission/resolution, retained buffers/groups/streams after session
drop, bounded deadlines, and the shared failure owner between public and partition
runtime views:

```sh
target/debug/deps/eredu_backend_mlx-c326ddbb1ba7fb59 \
  backend::distributed::tests --test-threads=1
```

### Global producer declarations from the retained selection

`SelectedExecution::component_partition_layout_for_rank` describes another rank
of the same retained partition selection. It uses the ownership compiler from
cold admission, including execution-group units, input/output roles and static
roles. The local-rank result uses retained local ownership directly. It neither
selects a different execution class nor allocates native state or communication.
`ComponentPartitionLayout::topology` identifies the described rank explicitly.
Callers supply the architecture descriptor and parameter description retained
with that selected model.
`component_partition_layouts` compiles a reusable `ComponentPartitionLayouts`
set after checking an explicit rank bound. Its `rank` lookup and `capture_producers`
reuse the retained maps across all selections and forwards, avoiding repeated
whole-model layout derivation for an all-layer experiment.

`ComponentPartitionLayouts::capture_producers` accepts a `ComponentCaptureProjectionRequest` with
the original global admission, selection, phase/prediction and producer/fragment
bounds. It returns the expected rank projections for runtime receipt admission.
For identical coordinate replicas of one logical invocation, the lowest world
rank supplies capture evidence; every consuming replica still needs interventions.
Distinct shards remain expected producers even when their selected overlap is
empty. Shared checkpoint aliases never transfer ownership between invocations.
Producer and fragment bounds reject excess geometry instead of returning a partial
list. Receipt admission remains responsible for complete/disjoint coverage, live
identity binding and native work quotas. These APIs currently describe the scalar
component activation/effective/write-input paths registered by component topology;
replicated normalized inputs and final readout require their corresponding
observation-placement integration.

The reusable layout/projection implementation passes 15 selected architecture
unit tests (0.07s) and all 20 numerical component tests (3.19s). The Llama and
Qwen prepared fixtures compare every remotely declared rank layout with that
rank's independent admission, then compare each expected producer projection with
actual component hooks through prefill and two cached decode steps. The matrix
retains nine causal trials across TP2, PP2 and combined TP2/PP2, each in resident,
host-layerwise and disk-streamed execution. This is neutral numerical execution,
not a physical multi-device capture run.

Coverage also checks rank/producer/fragment bounds, invalid prediction ordinals,
shared-source aliases, empty global selections, replica selection, and complete
runtime receipt admission. One shared rank-layout set is reused across selections
and forwards. Existing gated/non-gated, normalization, readout, mixed-state and
causal-mask component tests remain green.

```sh
cargo test -p eredu-architectures --offline --lib --test reference_numeric component
```

The native adapter also passes its feature-boundary check (1m37s); workspace
formatting and whitespace checks pass:

```sh
cargo check -p eredu-backend-mlx --offline --no-default-features
cargo fmt --all -- --check
```

Prepared discovery now retains the architecture descriptor, common execution
identity and partition selection. The shared runtime constructor carries its
validated global parameter description into the completed session, and native
model assembly binds that description to discovery. Calling
`PreparedModelDiscovery::component_partition_layouts(max_ranks)` uses only these
retained declarations. It rejects missing or mismatched partition metadata and
rank bounds before compiling layouts; it does not hash or reopen weights.
Descriptions are shared across the runtime and discovery with `Arc`.

Verification of this handoff passes all 65 runtime conformance tests (0.02s),
15 selected architecture unit tests (0.07s), 20 component numerical tests
(3.23s), and the ordinary discovery/validation-reuse test (0.11s). The final
two prepared numerical cases, including missing/mismatched metadata rejection,
pass in 3.72s. Native loader, capture, intervention and controlled-session
regressions pass all 14 inspection tests (41.55s). The native adapter also passes
`--no-default-features` checking (26.60s).

Existing public-loader Ring regressions verify that the retained declarations
survive real native partition construction and ordinary cached execution:

| Native CPU loopback case | Result |
| --- | --- |
| Qwen3 TP=2 resident reference | Pass, 17.97s |
| Qwen3 PP=2 resident reference | Pass, 18.44s |
| Qwen3 TP=2/PP=2 resident reference | Pass, 25.40s |
| Gemma4 PP=2 neutral composite session | Pass, 16.61s |

These are local multi-process native executions. Component capture in those
public distributed sessions is still unverified; its global layouts are checked
against actual hooks in the neutral numerical matrix above.

```sh
cargo test -p eredu-runtime --offline --test backend_independence
cargo test -p eredu-architectures --offline --test reference_numeric components::prepared
cargo test -p eredu-backend-mlx --offline --features metal --lib composition::mlx::inspection::tests:: -- --test-threads=1
for case in ring_two_process_qwen3_tensor_parallel_resident_reference \
  ring_two_process_qwen3_pipeline_resident_reference \
  ring_four_process_qwen3_tensor_pipeline_resident_reference \
  ring_two_process_gemma4_pipeline_neutral_composite_session
do
  cargo test -p eredu-backend-mlx --offline --features metal --lib \
    "tests::distributed_pipeline_ring::$case" -- --ignored --exact
done
```

Public distributed capture remains unverified. The shared capture-session driver
must consume these retained layouts with live identity, ledger, quota and
completion authority before that support changes.

### Observation delivery within the shared forward transaction

`ActivationObserver` supplies optional local preparation, common coordination,
post-completion delivery and infallible commit/discard callbacks. The ordinary
and controlled `CaptureObserver` now uses them through the same
`ReplicatedTextSession` driver. Distributed forwards first agree whether every
rank has a transactional observer, then agree local preparation before entering
any observer collective. Delivery runs only after exact output/state completion
and remains provisional until the existing final commit decision. Failure uses
the existing state rollback and native completion owners.

The selected realization retains exact-completion availability separately from
whether ordinary execution requested it. Active capture requires that capability
before work and forces completion; unobserved execution retains its ordinary
asynchronous behavior. Distributed unobserved forwards participate in one extra
status agreement to detect mixed observer configurations. They create no extra
component tensors.

`CaptureSession` retains the actual monotone forward epoch and prevents pending
records from being drained or checkpointed. Restoring a snapshot does not reset
epochs or cumulative usage. Existing local capture and intervention records keep
their boundary meanings; an applied intervention is not itself evidence that
generation committed. Aborted attempts cannot become resumable capture boundaries.

All 351 runtime unit tests (0.37s) and 68 backend-independence tests (0.03s) pass. Concurrent
rank tests reject mixed participation, local preparation, coordination, delivery
and native completion failures; final abort and indeterminate decisions discard
staged delivery. Completion capability tests cover resident and host-layerwise
execution, including the unobserved asynchronous control.

`CaptureObserver::for_step` now starts the envelope inside local preparation and
uses the actual forward phase. Rejected reservation consumes its epoch; pending
failure blocks restore even if it created no envelope. Runtime marks rejection
before mutable model-state access explicitly, letting native recovery distinguish
it from rollback without weakening completion ownership. Model-call adaptation
preserves the runtime error object and its available sources. The later
[observation error audit](#observation-error-causes) covers ordinary native
neural/observer adapters and public failure recovery.

Native verification after deferred admission passes all 21 recovery tests (9.21s)
and 14 inspection/capture/control tests (39.75s). The new CPU case rejects an
out-of-range capture prediction before model work, then reuses both the capture
owner and model session for prefill and decode after native settlement. Separate
failure injection proves that pre-mutation evidence cannot override late native
failure or unresolved resource ownership. A cause-chain test reaches the original
typed error through the public neutral `BackendFailure` wrapper. The backend
also passes `--no-default-features` checking (1m16s).

The participation phase was additionally verified with existing native CPU
loopback Ring references: Qwen3 TP=2 (18.07s), PP=2 (16.73s), TP=2/PP=2 (20.54s),
and Gemma4 PP=2 composite execution (16.63s). These cover ordinary distributed
execution; they do not verify public distributed component capture.

```sh
cargo test -p eredu-runtime --offline --lib --test backend_independence
```

```sh
cargo test -p eredu-backend-mlx --offline --features metal --lib composition::mlx::session::recovery_tests:: -- --test-threads=1
cargo test -p eredu-backend-mlx --offline --features metal --lib composition::mlx::inspection::tests:: -- --test-threads=1
cargo check -p eredu-backend-mlx --offline --no-default-features
cargo fmt --all -- --check
```

At this earlier stage public distributed component capture remained gated; the
loaded integration described below supersedes that restriction. The runtime composition
below now connects producer reservations, common receipt identity, active-group
votes and globally assembled records to the live ledger and shared callbacks.
Loaded native admission, coordinated run creation and the remaining parallel
intervention/query/overlay paths are still required.

## Live partition observer and accounting

Core `CaptureReservation` and move-only `CaptureQuota` let the runtime prepay a
global step and restrict every producer to a local allowance. Work handles cannot
be moved between capture owners or replayed after restore. `PartitionCaptureIdentity`
requires bounded retained artifact/execution/run/overlay labels; loaded composition
must establish their provenance and coordinated run identity.

`PartitionCaptureObserver::for_step` reserves scheduled producers and all ranks'
transport, decode, assembly and encoded evidence during local preparation. The
common coordination callback compares the complete selection/layout/cost digest
before any source factory. Architecture `ComponentPartitionLayouts` supplies
distinct producers and exact hook participants through `PartitionCaptureLayout`.
Separate prepaid status tokens let active tensor ranks agree source failure before
the next model collective, including nonproducing replicas. Inactive pipeline
ranks enter only common coordination and delivery. Global records and
`CapturedStep.partitions` evidence remain provisional until ordinary final commit.

The current neutral suite passes 363 runtime unit tests (0.35s), 68 conformance
tests (0.04s) and 226 core tests (35.38s). The 37 partition tests include raw,
summary and histogram agreement with ordinary capture; foreign/stale authority;
global versus local budget admission; nonrefundable nested quotas; generated-source
rejection before creation; source failures before the next model collective;
replica/inactive-rank participation; missing hooks; and exact completion failures
that never resolve or publish their output. Runtime capture errors preserve the
original native source through the observer boundary.

```sh
cargo test -p eredu-core
cargo test -p eredu-runtime --lib --test backend_independence
cargo test -p eredu-runtime --lib capture::tests::partition
```

Native CPU verification passes the new actual-driver comparison (19.06s):
partition-observer payloads and model logits exactly match ordinary component
capture for prefill and two cached decode steps on a nonzero Llama fixture.
This uses singleton receipt transport and verifies the shared model lifecycle.
The separate four-process Ring test passes (20.08s): only the active neighboring
pair votes, inactive pipeline ranks enter the common boundary, and true/false/true
decisions preserve communication reuse while switching stages. Inactive ranks
validate retained remote group facts without realizing those groups. These tests
do not establish public multi-process component generation support.

Regression verification also passes all 15 native bounded-capture cases: 12
ordinary cases (11.73s) and three explicitly selected Metal cases (7.07s), plus
21 native recovery cases (11.69s). The ordinary Qwen3 TP=2/PP=2 Ring reference
passes (26.17s) with the extended tensor-group requirement. The backend passes
`--no-default-features` checking (1m46s); formatting and whitespace checks pass.

```sh
cargo test -p eredu-architectures --lib component_partition
cargo test -p eredu-backend-mlx --offline --features metal --lib native_partition_observer_commits_real_component_prefill_and_cached_decode -- --test-threads=1
cargo test -p eredu-backend-mlx --offline --features metal --lib backend::distributed::tests::capture_hook::ring_capture_hooks_exclude_inactive_pipeline_ranks_and_allow_reuse -- --ignored --test-threads=1
cargo test -p eredu-backend-mlx --offline --features metal --lib composition::mlx::session::bounded_capture::tests:: -- --include-ignored --test-threads=1
cargo test -p eredu-backend-mlx --offline --features metal --lib composition::mlx::session::recovery_tests:: -- --test-threads=1
cargo test -p eredu-backend-mlx --offline --features metal --lib tests::distributed_pipeline_ring::ring_four_process_qwen3_tensor_pipeline_resident_reference -- --ignored --test-threads=1
cargo check -p eredu-backend-mlx --offline --no-default-features
```

At this earlier stage public distributed capture was gated. The loaded capture
integration below now supplies common run identity and component collection.
Initial prompt/run failure handling, partition interventions, normalized/readout/
other transform ownership, parallel parameter operations and overlays remain
implementation or audit work. The native hook adapter currently admits full-world groups, singleton
groups and independent neighbor pairs. Larger or routed subgroups that require a
world participation wave need another native mechanism before they can run from
a pipeline-local hook. This is an implementation gap, not a model limitation.

## Common distributed session identity

Runtime `establish_communication_session` exchanges versioned manifests and
fresh native instance nonces in the existing two setup gathers. Every rank
retains the same `CommunicationSessionIdentity`; identical manifests in a new
setup produce a distinct identity. MLX retains this value with its communicators
and exposes it through `MlxDistributedSession::session_identity`. Failed nonce
creation is exchanged before rejection. Serialized metadata is limited to
16 MiB per rank and 128 MiB for the padded gather before allocating padded gather
buffers or transferring payloads; local proposal serialization precedes the size
check. This is neither a physical allocation ceiling nor an authentication protocol.

`PartitionCaptureIdentity::for_session` combines that setup with the first
attempted shared forward epoch. It binds before fallible capture admission;
restore cannot refund the epoch or change the run's identity. Each later record
also retains its actual forward epoch. Production construction creates one setup
per model and retains clones with that model's shared runtime and native owner.
Composition must provide the matching artifact, execution and overlay facts.
These common descriptive labels do not replace live owner/epoch authority, and
they must not be inferred from independently created facade trace IDs.

The complete runtime suite passes 367 unit tests, 69 conformance tests and three
binding tests. Identity tests cover agreement across rank views, repeated setups,
missing/zero nonces, incompatible versions/manifests, oversized peer lengths,
failed first admission, stale-epoch rejection, restoration and later capture
owners. Loaded public distributed observer selection remains unfinished.

Native verification passes the three observer/input/admission cases (19.34s).
The actual component observer retains `capture:<setup>:1` through prefill and
two cached decode steps while matching ordinary captured values and logits.
The four-process Ring hook test passes (19.99s): every rank sees the same setup
identity, an identical repeated setup has a different identity, clones retain
their original identity, and the previous rejection/reuse checks still pass.
The public-loader Qwen3 TP=2/PP=2 reference also passes (22.14s), including cached
inference after the setup change. The backend no-default-features check passes.
Metal setup and receipt exchange pass (7.13s) for raw slices, summaries and
histograms over F32/BF16 inputs. Formatting and whitespace checks pass.

```sh
cargo test -p eredu-runtime
cargo test -p eredu-backend-mlx --features metal --lib model_session::partition_capture_tests -- --test-threads=1
cargo test -p eredu-backend-mlx --features metal --lib backend::distributed::tests::capture_hook::ring_capture_hooks_exclude_inactive_pipeline_ranks_and_allow_reuse -- --ignored --test-threads=1
cargo test -p eredu-backend-mlx --features metal --lib ring_four_process_qwen3_tensor_pipeline_resident_reference -- --ignored --test-threads=1
cargo test -p eredu-backend-mlx --features metal --lib native_partition_capture_exchange_metal -- --ignored --test-threads=1
cargo check -p eredu-backend-mlx --no-default-features
```

## Loaded distributed component capture

The ordinary MLX capture entrypoints now select the shared partition collector
when the loaded model retains a distributed setup. Discovery compiles the actual
architecture component layouts once, then combines each hook's members with the
selected native failure-agreement mechanism. Existing phase, selection and
instrumentation gates still apply. Unknown global observation ownership remains
unverified; component capture does not enable distributed masks or parameter edits.

Applications use the same public capture plan and generation API as on one rank.
Every rank prepares the same global plan and advances the same model step. The
loaded adapter supplies artifact/execution/setup/overlay facts; callers do not
construct native identities or reinterpret local checkpoint names. The public
`CapturedStep` contains assembled global records and `partitions` evidence.
Raw slices, full tensors, summaries and histograms use this route. Normalized
inputs, residual/readout ownership, other transforms and wider native hook groups
remain separate implementation work.

The loaded collector bounds layout compilation to 1024 ranks and a selection to
65,536 fragments. It derives a receipt allowance from actual fragments, declared
metadata and native encoded-size estimates, subject to the admitted byte ceiling.
Each rank prepays the global producer/transport/assembly cost before model work;
logical native credits include a conservative deferred-input factory allowance.
Private allocator workspace is excluded. Cold configuration is local; exact
all-rank costs are checked by the shared forward's preparation phase.

Controlled snapshots retain a setup-scoped child seed, with its host storage
included in the estimate. A re-admitted child gets a fresh owner and claims its
own first epoch; parent restore keeps the parent's run identity and all consumed
budgets. Each observed forward validates the seed against the actual loaded
setup and parameter version. Capture participation or composition failures reach
the shared preparation protocol before source work.

Runtime verification passes 368 unit tests, 69 conformance tests and three binding
tests. All three focused architecture placement tests pass. Native public Qwen3
verification passes TP2/PP2 (26.64s), resident TP2 (18.89s), disk-streamed PP2
(19.57s), and host-layerwise TP2 (19.43s). Each case compares every actual/effective
FFN-unit and attention-channel tensor with an ordinary resident session over
prefill and two cached decode steps, using nonzero F32 fixtures and tolerance
`2e-4 + 2e-4 * abs(reference)`. Sampled token IDs match exactly. Snapshot replay
and two sibling branches produce exactly equal payloads and tokens; siblings
have distinct run identities, parent restore preserves its identity, actual
forward epochs increase, and capture/copy budgets are not refunded.

The native partition-observer regression passes all three tests (10.58s), and
ordinary inspection/control passes all fourteen tests (42.53s).

The collector also implements coordinated skip-on-limit admission. It prepays
mandatory coordination, prepares selections in plan order, then compares the
successful claims, skipped indices/reasons and all consumed credits on every rank.
A limit skip performs no source or hook work, retains partial preparation charges,
and publishes its typed record only on final commit. Rank disagreement rejects
before model work. Record envelopes and coordination remain mandatory; geometry,
unsupported-mechanism and native execution errors are not successful captures.
Child re-admission accepts the same policy. Exceeding a prepaid child allowance
returns `PrepaidBound` with its typed cause instead of being treated as ordinary
budget exhaustion. Runtime verification now passes 372 unit tests, including
mixed/all-skipped selections, rank disagreement, aborted delivery, per-step and
cumulative exhaustion after producer reservation, mandatory-envelope costs and
changing native bounds. The 69 conformance and three binding tests also pass.
The expanded native matrix passes all four public Ring cases in 64.64s. Each
also runs a zero-capture-budget root over prefill and two cached decode steps,
then re-admits a skipped child from a captured snapshot. Skipped records carry
typed capture-count limits, no payloads and no partition evidence; token IDs
still match ordinary inference. The parent and ordinary siblings retain their
existing exact replay checks. `eredu-backend-mlx --no-default-features` passes
(24.04s), as do focused formatting and whitespace checks.

```sh
cargo test -p eredu-runtime
cargo test -p eredu-architectures --lib component_partition
cargo test -p eredu-backend-mlx --features metal --lib ring_public_component_capture_qwen3_ -- --ignored --test-threads=1
```

## Shared input preparation

The shared text session accepts fallible prepared inputs through
`prefill_input_result_with_observer` and `decode_input_result_with_observer`.
Every rank agrees preparation before capture admission, checkpointing or model
execution. MLX carries token extraction, composite-input admission, generation
parameter-version checks, capture prompt alignment and decode framing into this
phase. Native framing remains inside ordinary submission/recovery ownership.
A rejected input consumes the transaction epoch, preserves prior model state,
discards capture delivery and leaves the committed prompt identity unchanged.
Retry requires native settlement and uses a fresh epoch. The new fallible-input
entrypoints reject partitioned execution without bounded phase agreement, even
when the local input is valid; final-commit support alone is insufficient.

`eredu_nn::Error::backend_source` retains the original typed preparation cause.
This does not establish source preservation for every existing native error
conversion. Initial native prompt construction and run admission remain part of the
cold-failure audit. Common run identity and loaded capture integration are now
implemented as described above.

Verification passes 363 runtime unit tests, 69 conformance tests and 3 binding
tests. The concurrent conformance fixture covers one-rank prefill/decode failure,
mixed observer participation, unchanged nonempty prior state, original error
recovery, and a successful retry at the next epoch. Native CPU tests pass the
rejected-input/cached-decode comparison, the partition-observer comparison, and
exact core admission after a session capability change (3 cases, 11.01s).
Successful ordinary prefill retains the core runtime's admission check and
execution-plan target proof. Rejected prefill and decode steps produce
no observer admission; their successful successor matches the uninterrupted
cached logits exactly. Native recovery (21 cases, 14.43s), inspection/control
(14 cases, 43.89s), ordinary bounded capture (12 cases, 10.20s), and explicit
Metal capture (3 cases, 10.36s) also pass.

Ring verification passes Qwen3 TP=2 (28.76s), PP=2 (28.41s), combined TP=2/PP=2
(29.50s), and Gemma4 composite PP=2 (28.63s). Each fixture first rejects an empty
prompt on one rank while its peers supply valid input, synchronizes the native
session, and then exercises its existing ordinary prefill/decode, reference-logit
and prompt-cache checks. These are preparation/recovery checks through public
native sessions. The loaded distributed capture tests above subsequently verify
the public component collector itself.

`eredu-nn --all-features` and `eredu-backend-mlx --no-default-features` checks
pass, as do formatting and whitespace checks. The native test build retains the
existing linker warning about its large debug unwind section.

```sh
cargo test -p eredu-runtime
cargo test -p eredu-backend-mlx --features metal --lib model_session::partition_capture_tests -- --test-threads=1
cargo test -p eredu-backend-mlx --features metal --lib session::recovery_tests -- --test-threads=1
cargo test -p eredu-backend-mlx --features metal --lib mlx::inspection::tests -- --test-threads=1
cargo test -p eredu-backend-mlx --features metal --lib session::bounded_capture::tests -- --include-ignored --test-threads=1
cargo test -p eredu-backend-mlx --features metal --lib ring_two_process_qwen3_tensor_parallel_resident_reference -- --ignored --test-threads=1
cargo test -p eredu-backend-mlx --features metal --lib ring_two_process_qwen3_pipeline_resident_reference -- --ignored --test-threads=1
cargo test -p eredu-backend-mlx --features metal --lib ring_four_process_qwen3_tensor_pipeline_resident_reference -- --ignored --test-threads=1
cargo test -p eredu-backend-mlx --features metal --lib ring_two_process_gemma4_pipeline_neutral_composite_session -- --ignored --test-threads=1
cargo check -p eredu-nn --all-features
cargo check -p eredu-backend-mlx --no-default-features
```


## Deferred observation source precision

Neural mechanisms now provide `GeneratedTensorSource` facts containing creation
storage and optional actual `TensorElementType`. The architecture adapter uses
runtime's `generated_capture_source` conversion to supply core
`GeneratedCaptureSource`. This preserves the independent neural and core contract
crates. MLX FP8 declares F32 for its reconstructed multiplication input; the
prototype provides geometry only. Dtype-dependent capture estimators can use
`CaptureBackend::estimate_generated` before any deferred source work.

For a nonempty capture, runtime checks the generated element type against the
declaration before the native transform. Geometry or precision violations produce
an invalid failure record; unavailable empty raw precision produces an unsupported
record. Returned errors and record categories agree. Failure retains consumed
credits and actual dtype metadata, including a factory that returns wrong geometry. Ordinary empty raw, preview, summary and histogram
selections use a shared host constructor after reservation and never call the
source factory. Partition delivery shares it for raw, summary and histogram
transforms; this does not add distributed preview support.
Raw empty values require a known supported source precision. Partition receipts
carry each empty producer's declaration and reject disagreement before publishing
any result. Conservative creation credits reserved by ordinary capture are not
refunded merely because an empty selection does not need the factory.

Runtime verification passes 377 unit tests, 69 conformance tests and three binding
tests. Coverage includes nonzero-source dtype mismatch, empty F32/U64/bool payloads,
empty reductions, unknown-source rejection, no factory/native transform work for
empty selections, globally empty generated receipts, producer disagreement and
a mixed empty/nonempty producer selection that reproduces ordinary global values.
All seven public FP8 tests pass on CPU and Metal (189.16s), including empty
projection-input records in ordinary, controlled and snapshot-replay runs across
all three residencies and both complete and partial FP8 blocks. Logits remain
exactly equal with empty projection-input selections. Existing nonzero component
reconstruction, BF16 precision, independently edited references, overlay transitions
and restoration checks pass in the same suite. Native mechanism declaration
checks also pass on CPU and Metal: two FP8 input tests (10.79s) and two vocabulary
input tests (7.21s). The neural all-features check and the backend check without
default features pass. Portable verification passes 55 backend conformance tests
and 19 facade tests; one existing tokenizer fixture requires an external checkpoint
and remains ignored. All 19 architecture component tests pass, including tensor
parallel masks, traversal hooks, pipeline cuts and bounded residency.

```sh
cargo test -p eredu-runtime
cargo test -p eredu --no-default-features --features mlx,metal --test native_execution_control fp8_parameters:: -- --include-ignored --test-threads=1
cargo test -p eredu-backend-mlx --features metal --lib actual_fp8_projection_input -- --include-ignored --test-threads=1
cargo test -p eredu-backend-mlx --features metal --lib vocabulary_projection_inputs -- --include-ignored --test-threads=1
cargo check -p eredu-nn --all-features
cargo check -p eredu-backend-mlx --no-default-features
cargo test -p eredu --no-default-features --test portable_facade --test backend_conformance
cargo test -p eredu-architectures --test reference_numeric components::
```


## Observation error causes

Ordinary MLX capture and intervention hooks carry structured backend errors through
native array/tensor adapters and retain them with the neural error's source. Capture
admission errors remain available as typed `CaptureError` causes. Native exceptions
preserve their original message and creation location through the facade's
`BackendFailure`; no native error type is added to application operation signatures.
Processor observation failures retain their native causes too. Embedded and external
speculative observers retain their separate native mechanism contracts.

A failure's source chain does not establish completion or prove state restoration.
The existing shared transaction records successful rollback or preflight rejection;
native recovery independently governs safe reuse. Failed capture reservations
remain consumed when the capture checkpoint is restored. Initial prompt construction
and cold distributed run admission remain part of the broader failure audit.

The current native verification passes:

- Two CPU/Metal failure-and-retry tests (15.23s) cover all three residencies,
  prefill and cached decode, typed cold rejection and an original native exception
  injected after actual component evaluation. They verify state preservation,
  exact native settlement, failed capture records, nonrefunded capture charges,
  exact retry payloads and the following cached decode against ordinary execution.
- Two public facade tests (11.61s) cover those residencies on CPU/Metal. A controlled
  snapshot replay exhausts its previously consumed allowance with a typed cumulative
  `CaptureError::Limit` and emits no token. A scratch checkpoint containing one NaN
  readout coefficient exercises the full-vocabulary score transform's finite-logit
  rejection; ordinary and controlled runs retain the same original native message
  and creation location, emit no token and permit reset after settlement.
- Two native neural source tests (13.21s) preserve an actual invalid-reshape
  exception through deferred generation and ordinary compute adapters. The native
  duplicate-observation test (13.27s) retains `ObservationError::DuplicatePath`.
- All 21 session recovery tests (8.34s), 15 native capture-transform tests (12.67s)
  and 14 inspection/control tests (54.97s) pass. The four native Ring component
  capture configurations, including controlled branches and coordinated skipping,
  pass after the error-boundary change (70.81s).
- All seven public FP8 regression tests pass on CPU/Metal (177.74s), including
  component reconstruction, effective projection inputs, empty captures, atomic
  overlays and controlled snapshot replay across all three residencies.
- `eredu-backend-mlx` checks without default features and with optional `image,audio`
  enabled (1m24s), including the processor observation adapter.

The source chain is separate from bounded diagnostic record text: consumers can
downcast typed portable errors while record sizes and failure accounting remain
unchanged. These checks do not claim that every unrelated native error conversion
has been audited.

```sh
cargo test -p eredu-backend-mlx --features metal --lib native_capture_failure_causes_and_retry -- --include-ignored --test-threads=1
cargo test -p eredu-backend-mlx --features metal --lib observation_errors:: -- --test-threads=1
cargo test -p eredu-backend-mlx --features metal --lib native_duplicate_observation_retains_portable_error -- --test-threads=1
cargo test -p eredu-backend-mlx --features metal --lib session::recovery_tests -- --test-threads=1
cargo test -p eredu-backend-mlx --features metal --lib session::bounded_capture::tests -- --include-ignored --test-threads=1
cargo test -p eredu-backend-mlx --features metal --lib mlx::inspection::tests -- --test-threads=1
cargo test -p eredu-backend-mlx --features metal --lib ring_public_component_capture_qwen3_ -- --ignored --test-threads=1
cargo test -p eredu --no-default-features --features mlx,metal --test native_execution_control capture_failures:: -- --include-ignored --test-threads=1
cargo test -p eredu --no-default-features --features mlx,metal --test native_execution_control fp8_parameters:: -- --include-ignored --test-threads=1
cargo check -p eredu-backend-mlx --no-default-features --features image,audio
```

## Global normalized inputs and affine readout

Global capture placement now includes each declared component's normalized input
and its effective companion, plus `readout.embedding`, `readout.residual`,
`readout.normalized` and `readout.linear` with their effective companions. The
normalized and residual values use the complete hidden axis; affine readout uses
the vocabulary axis after the selected vocabulary projection gathers its output.
These points use the same bounded collector, producer receipts and shared
completion/delivery policy as scalar components.

`ComponentPartitionLayout::observation` exposes the retained axis and local
coordinates. Its existing `group` and `point` methods retain scalar component
coordinates for mask lowering. Input embeddings follow the actual input
invocation, even when tied weights are also stored on an output stage. Final
normalization and affine scores follow output invocations. Every executing
replica participates in hook failure agreement, while one producer exports each
identical coordinate map.

The final `model.logits` seam now retains separate authoritative publication
ownership from architecture admission. Plan construction and capture placement
consume that same owner. Other output ranks offer ordinary dependencies through
a replica callback whose default is inert; ordinary observation and intervention
remain owner-only. Final-logits integration passes all four native Ring
configurations (77.43s), including a logits-only plan across prefill and two
cached decode steps. That plan checks final scores against ordinary execution,
one authoritative producer and positive collector timing without depending on
upstream readout captures to evaluate the lazy source.

The expanded Llama and tied-Qwen3 numerical cases pass (two tests, 4.53s).
A Gemma2 case also passes (3.97s) across TP2, PP2 and combined TP2/PP2 and all
three residencies. It uses attention softcap 0.3 and final softcap 0.7, verifies
that affine and final scores differ, and checks the declared final transform
`0.7 * tanh(affine / 0.7)` within `2e-6`. Partition captures match the ordinary
final values through cached decode and existing component interventions.
All 18 focused architecture ownership/execution tests pass, including a later
rank as the explicit publisher. All 386 runtime tests pass (0.39s), and the
backend checks without default features (33.79s).

Transformed `readout.projection_input` values still need evidence about the
actual multiplication transforms before they can be treated as identical
replicas. Global support for transformed inputs and whole residual writes on
remaining specialized drivers remains open. Block residual normalization placement now follows the declared
logical gain/bias parameter's invocation owner and passes native verification.
Its neutral Nanbeige test passes (6.79s): two physical blocks are invoked twice,
with a normalization at the loop boundary, across TP2, PP2 and combined TP2/PP2
and resident, host-layerwise and disk-streamed execution. Captured original and
effective values match the ordinary reference through prefill, cached decode and
existing component intervention trials. Shared physical sources do not move the
normalization to the final-output stage. Expanded Llama, Qwen and Gemma regression
cases also pass (three tests, 11.91s).

Global support also requires hook facts from the actual architecture and selected
unit executor. Input, unit, readout and final-publication sites are distinct.
The current propagation work corrects reports that previously inferred internal
hook availability from a session-wide inspection flag and valid coordinates.
Missing specialized hooks report `Unverified` before capture admission; final
logits retain the common publication seam. The eight Qwen/Nanbeige capture cases
pass with these facts. Two further native Ring tests pass (9.29s) for dense and
routed LFM2: internal plans reject during admission while final-logits capture
matches ordinary execution through prefill and two decode steps. All 396 runtime
tests (0.29s), 18 architecture tests (0.02s), no-default backend compilation
(2.97s) and formatting checks pass. Specialized family/provider implementation
remains open.

Final-logits placement is verified independently of component/readout
declarations. Its ownership comes from the selected output and publication
roles, and its vocabulary extent comes from the ordinary observation catalog.
This enables final-score capture while an undeclared component topology remains
an explicit integration gap; it does not imply that component attribution is
available for that family. The neutral Nanbeige placement test removes
component/readout metadata and verifies this independent ownership. The original
native publication-only checks for Qwen MoE have been expanded into full positive
component capture checks now that its shared decoder declares attention topology;
current evidence is recorded below.

The native Nanbeige fixture invokes one physical block twice and shares the
inter-loop gain with final normalization. All four Ring configurations pass:
resident TP2, host-layerwise TP2, disk-streamed PP2 and combined TP2/PP2. Each
prediction checks captured loop input/output against the independent RMS equation
and known nonuniform gains within `2e-4`, as well as ordinary execution. The
fixture retains full capture, previews, vocabulary reductions, controlled branches,
skip and failure-recovery checks. Architecture ownership tests pass (18 tests,
0.04s), and the backend checks without default features (16.20s). The combined
Nanbeige/Qwen regression run passes all eight native cases (154.82s).

```sh
cargo test -p eredu-architectures --test reference_numeric prepared_nanbeige_components_and_loop_normalization_cross_pipeline_cuts -- --test-threads=1
cargo test -p eredu-backend-mlx --features metal --lib ring_public_component_capture_nanbeige_ -- --ignored --test-threads=1
```

The expanded Llama and tied-Qwen3 numerical fixtures pass (two tests, 7.71s).
They exercise TP2, PP2 and combined TP2/PP2 across resident, host-layerwise and
disk-streamed execution, with prefill and cached decode. Every selected local
tensor is compared with its corresponding ordinary tensor or axis slice;
inactive invocations must be absent, and replicas must match the declared
producer projection. Existing deletion and keep-only trials remain included.
All three coordinate/producer unit tests also pass (0.03s). The expanded native
Ring matrix passes all four configurations (76.77s): resident TP2, host-layerwise
TP2, disk-streamed PP2 and combined TP2/PP2. Normalized inputs and affine-readout
values, including effective companions, match ordinary execution across prefill
and two cached decode steps within the existing `2e-4 + 2e-4 * abs(reference)`
tolerance. Controlled replay, sibling isolation, skipping, cold failure/retry and
terminal communication fencing remain part of those same tests.

Replicated capture needs dependency execution even on nonexporting ranks: an
ordinary source can contain lazy tensor-parallel collectives. The shared observer
therefore admits exact source geometry and completion resources for every active
rank, agrees geometry before submission, and prepares each ordinary source with
the selected bounded completion policy. Only admitted producers transform and
export values. Deferred captures prepare the ordinary prototype; their factories
remain uncalled on replicas without fragments. Globally empty or skipped captures
perform no additional source execution. Preflight and final success agreement use
separate credits and authority, and failed work never refunds reservations.

All 386 runtime unit tests pass (0.37s), including source-dependent replicas,
empty local shards, globally empty/skipped captures, preflight geometry rejection
and nonrefunded native-error/deadline reservations. Two focused native tests pass
(10.29s): bounded source completion without host export, including deadline fencing,
and real component prefill/cached decode through the singleton partition observer.
The backend also checks without default features (40.48s).

```sh
cargo test -p eredu-runtime --lib
cargo test -p eredu-architectures --lib component_partition
cargo test -p eredu-backend-mlx --features metal --lib partition_capture_source_has_bounded_completion_without_host_export -- --test-threads=1
cargo test -p eredu-backend-mlx --features metal --lib native_partition_observer_commits_real_component_prefill_and_cached_decode -- --test-threads=1
cargo test -p eredu-backend-mlx --features metal --lib ring_public_component_capture_qwen3_ -- --ignored --test-threads=1
```

The expanded native fixture allows 1 GiB of logical host reservation per step.
This includes conservative receipt decoding on every world rank; the small
tensor payloads alone do not describe that cost. The existing 128-fold encoded
receipt decoding allowance remains unchanged. This fixture allowance is not a
physical memory guarantee or completion of the realistic all-layer budget audit.

Partition `capture_seconds` now includes source preparation, transformation,
coordination and receipt delivery on the local rank, including failed attempts.
It measures synchronous collector wall time, so it can overlap ordinary source
evaluation; it is not an exclusive device-kernel profiler or a sum across ranks.

Distributed `TokenScores` and `TopCandidates` now have a complete-source receipt
path verified on all four native Ring configurations (77.82s). Admission requires one producer with the
complete final logits in original vocabulary order. Multiple shards, permuted
coordinates and duplicate exporting replicas reject before capture or factory
work. Output replicas retain their normal source participation. The bounded
reduction payload moves into the global record with producer/context evidence;
delivery does not export or assemble the full vocabulary tensor. Token-score
normalization still includes the entire final model distribution. Each native
test selects only one reduction, then compares three predictions with an independent
host reduction over previously captured complete final logits. It checks selected
scores, full-vocabulary log partition and log probabilities, ranks, strongest
alternatives and top candidates within `2e-5`. Sampled tokens agree with ordinary
execution. The backend also checks without default features (1m13s).

Preview prefix assembly passes all four expanded native Ring configurations
(79.31s), covering resident TP2, host-layerwise TP2, disk-streamed PP2 and combined
TP2/PP2. It validates
complete source coverage, exact local truncation and payload geometry, then
maps bounded fragment prefixes into the first requested global elements. Result
payloads remain flat prefixes; selected shapes and truncation counts describe the
original global selection. Each fragment contributes at most the requested prefix
length, and only that global prefix is allocated for assembly.

The native fixture selects previews of seven elements across all supported
component/readout points, checking original global selected shapes, explicit
truncation and ordinary-reference prefix values through prefill and two cached
decode steps. It retains reduction-only, snapshot/sibling, skip, retry and terminal
fencing coverage. All 394 runtime unit tests pass (0.41s), including strided,
permuted, multi-row and nonfinite/integer/boolean previews, malformed truncation,
empty receipts and failed assembly budgets. The backend checks without default
features (24.23s); repository formatting and whitespace checks pass.

All 391 runtime unit tests pass (0.40s). Vocabulary receipt tests cover exact
requested ordering, tied ranks, competitor identities, 24 malformed payload
variants, rejection before deferred source creation, exhausted delivery quotas,
and stable log probabilities for equal scores at positive and negative
`f32::MAX`. Host delivery bounds are unchanged between vocabularies of four and
one million IDs with the same requested reduction size; native computation and
retention remain separately charged by the collector's shape estimate.

Stable probability receipt checks now bound the shifted denominator using the
target, strongest alternative and vocabulary size, independently of rounded
`log_partition`. The runtime suite passes all 396 tests (0.41s), including 16
impossible-probability variants at extreme offsets and valid ordinary, near-equal,
large-gap and tied-score distributions. Native verification of the tighter
receipt check passes all eight Nanbeige/Qwen Ring cases (109.11s). The backend
checks without default features (23.75s), and workspace formatting and whitespace
checks pass.

## Cold text-run preparation

Ordinary, observed and controlled text entry points now agree local request,
prompt, sampling, instrumentation and initial-delivery readiness before the first
forward. The shared core machine handles prompt/sampler failures; the facade
supplies tokenizer, semantic-policy and callback/cancellation decisions. A local
failure retains its native source or explicit portable policy variant. Successful
peers receive a typed `TextPreparationRejected` containing the stage and rejecting
rank. Initial cancellation propagates across participants without publishing a
token. Cancellation and failure remain different outcomes.

`PreparedChatInput::TokenIds` carries exact prefixes into this preparation path.
An opaque `PreparedBackendInput` has already been constructed by its caller;
errors before that call cannot be recovered retrospectively. Low-level consumers
can use `ModelRuntime::finish_text_preparation` and its cancellable counterpart
to carry local results through the same agreement. Each participant must enter
the same stages in the same order. Local artifact inspection and immutable plan
preparation remain local operations, without hidden native collectives.

`TextPreparationCoordinator` binds two fixed-size metadata exchanges to the actual
communication setup, a monotone attempt and the stage. The second exchange
confirms that ranks validated the same first-round disposition. Protocol and
transport failures fence the retained communication authority, and the existing
forward input agreement remains in place before state mutation. All exchanges
use the selected bounded completion policy; an error alone never releases native
resources or establishes completion.

A local error while resolving the final confirmation can occur after a peer has
received its own confirmation. That peer must still enter the next shared
preparation or forward-input boundary; a ready result does not prove every peer
has observed completion. The failed rank fences its owner, and the next boundary
must reject or reach its bounded safe disposition before model-state work.

`LoadedModel::text_preparation_usage` reports cumulative attempts, logical native/
framing retention reservations and host-copy reservations. Both exchanges are
reserved before submission. Rejection and cancellation consume their charges;
model reset, snapshot restore and branch switching do not rewind this session
owner. These reservations are separate from capture usage and do not describe
physical allocator peaks. Raw core `TextGeneration` now returns `BackendFailure`.
Capture installation returns `TextCaptureSetupError`, separating a typed local
`CaptureError` from preparation failure; facade errors retain their existing
portable variants.

Current portable verification passes 226 core tests (35.33s), 383 runtime tests
(0.37s), 59 backend conformance tests (6.54s), and 19 portable facade tests
(47.36s; one existing external-checkpoint fixture ignored). Six concurrent runtime
tests cover all stages, corrected retry, cancellation precedence, setup/stage/
attempt mismatch, decoding disagreement, transport submission/completion/resolution
errors, deadline disposition, exhausted accounting and unavailable completion.
Four facade tests verify local/peer failures before forward, original local error
handling, exact corrected retry and initial cancellation through ordinary,
controlled and asynchronous token generation.

Native Ring verification passes all four configurations in 71.63s: resident TP2,
host-layerwise TP2, disk-streamed PP2 and combined TP2/PP2. Each injects a rank-zero
empty prompt, a native sampler exception and insufficient capture retention both
before and after a successful captured run. Every rank rejects before forward;
the failing rank retains its exact cause, peers retain the rejecting stage/rank,
and cached state remains exactly unchanged. Corrected runs retain the existing
component/reference tolerance, exact tokens and snapshot/sibling checks. The
session ledger charges six attempts per failure sequence. A final deliberate
stage mismatch fences agreement, synchronization and reset without refunding usage.
This completed-gather mismatch does not simulate an orphaned native queue.

The current public CPU/Metal regression run passes all nine capture-failure and
FP8 tests in 183.75s, covering resident, host-layerwise and disk-streamed execution,
ordinary/controlled captures, complete and partial quantization blocks, BF16
precision, overlays, error causes and replay accounting. Optional image/audio
compilation without default backend features passes in 1m46s. Focused formatting
and whitespace checks pass.

Detached native prompt and sampler preparation now preserves the original
operation error even when its submission scope also reports failure or blocked
completion. The scope is sealed and progressed before returning; unresolved
owners remain retained until terminal completion. All six native recovery tests
pass, including failure/blocked probes with exact original causes and no early
resource destruction (`/tmp/eredu-startup-error-preservation-recovery1.log`,
Metal build 103, 0.00 s). Two focused native unit-submission tests also pass,
covering delayed retirement and nonblocking polling/drop under a foreign runtime
(`/tmp/eredu-prediction-paging-submission-audit1.log`, CPU build 102, 0.00 s).

Speculative facade setup now gathers every lane's host preparation before one
request agreement, then all native prompts before one prompt agreement. A later
lane's local failure therefore cannot bypass the batch boundary. Controlled
speculation agrees its initial capture/activation validation too. The public
`LoadedModel::finish_text_preparation` and cancellable counterpart let callers
coordinate their own media or opaque-prompt results without backend access.
All 66 portable backend conformance tests pass in 5.47 s
(`/tmp/eredu-speculative-startup-facade-full1.log`). Five new tests cover original
prompt errors and corrected retry in ordinary/controlled runs, peer rejection,
later-lane host failure before any native prompt, invalid controlled capture and
caller-owned local/peer/cancellation outcomes; all nine preparation tests pass in
0.92 s (`/tmp/eredu-speculative-startup-facade-tests3.log`). These results precede
the subsequent shared-core extraction and native sampler/cache agreement checks.

After extracting the common result policy into core, all 252 core tests pass
(6.34 s), all 511 runtime tests pass (0.29 s), and all 66 portable backend
conformance tests pass again (0.76 s). Native session recovery passes all 23 tests
(0.57 s), including original resource/session failures with pending authority.
The new speculative Ring rejection check passes all four placements in 3.13 s:
resident TP2, host-layerwise TP2, disk-streamed PP2 and resident TP2/PP2
(`/tmp/eredu-speculative-startup-ring-failures2.log`, Metal build 105). Rank zero
requests a proposal width exceeding the selected realization; every peer rejects
at sampling preparation, emits no events and retains cumulative charges. The
original local policy and peer stage/rank remain available as typed causes.

This test found that cache construction itself enters architecture preparation
collectives. Sampler/lane readiness is now agreed before any cache construction;
cache results receive a separate agreement. A combined vote after both phases
would strand peers when the first phase failed locally. A failed broad native
operation remains fenced without a whole-run restoration witness. The successful
Qwen Next component workflow also passes all nine placements in 62.04 s with this
ordering, actual prediction paging, idle-owner checks, effective queries,
coordinated overlays, restoration and replay
(`/tmp/eredu-qwen-prediction-paging-matrix3.log`, Metal build 105).

Media preparation now holds detached native recovery across processor conversion
and observer calls while preserving caller-owned tokenizer errors. The image/audio
feature check passes in 21.19 s. The expanded native retained-Qwen-processor test
passes in 0.42 s: its sidecar is removed, nonzero image preparation succeeds, a
native observer error retains its exact cause, and clean retry reproduces the
baseline values. All seven detached-recovery tests pass (0.00 s), including typed
caller errors (`/tmp/eredu-media-startup-processor-test1.log` and
`/tmp/eredu-media-startup-recovery-tests2.log`, Metal/image/audio build 107).
Speculative scheduler invariants are also checked in whole-batch host readiness;
invalid capacities, lookahead and timeout fail before native prompt work. All
67 portable backend conformance tests pass in 0.79 s
(`/tmp/eredu-speculative-startup-facade-full3.log`).

The ordinary and controlled facade now share token-step readiness around the
committed cursor. The source carries typed host failures/cancellation through the
existing core agreement; successful peers stop before another prediction. Record
budget failures are observed through a read-only callback probe, and the original
local capture error is preserved. Native completion is explicitly drained even
when no capture plan is enabled. New neutral facade cases verify peer cancellation
after exactly one committed token, peer delivery rejection without a second
forward, and local encoded-record exhaustion after computation. All 13 preparation/
delivery tests pass in 1.43 s (`/tmp/eredu-step-delivery-conformance2.log`), followed
by 182 facade library tests (two existing ignores, 9.94 s) and all 70 backend
conformance tests (0.83 s, `/tmp/eredu-step-delivery-portable-full1.log`). Native
public verification is recorded below. Subsequent lifecycle/terminal record
checks extend these guarantees; the complete delivery-failure matrix also includes
specialized speculative paths.

The broader failure audit still includes specialized speculative/media startup,
caller-owned preparation outside run entry points, and late transport/record
failure across the complete execution matrix.

```sh
cargo test -p eredu-core --lib
cargo test -p eredu-runtime --lib
cargo test -p eredu --no-default-features --test backend_conformance --test portable_facade
cargo test -p eredu-backend-mlx --features metal --lib ring_public_component_capture_qwen3_ -- --ignored --test-threads=1
cargo test -p eredu --no-default-features --features mlx,metal --test native_execution_control -- capture_failures:: fp8_parameters:: --include-ignored --test-threads=1
cargo check -p eredu-backend-mlx --no-default-features --features image,audio
```

## LFM2 partition observation propagation

LFM2 now routes its ordinary and tensor-parallel blocks through shared component
instrumentation. Partition input/readout hooks use the same normalization and
projection mechanisms as other decoders. Dense FFN captures precede the actual
row-parallel projection, attention captures precede output projection, and whole
convolution writes follow the completed mixer output. Disabled instrumentation
keeps the ordinary operator path without observation tensor creation.

Whole residual-write placement follows the owning node's canonical parameter
namespaces and retained logical unit ownership. It rejects ambiguous or static
owners and does not infer invocation ownership from physical aliases or layer
ordinals. The new neutral test passes (0.94s) across TP2, PP2 and combined
TP2/PP2 in resident, host-layerwise and disk-streamed execution. Four mixed
attention/convolution blocks use distinguishable, nonzero parameters; the
standalone convolution kernels are explicitly loaded in the independent ordinary
reference. Nine trials cover baseline, whole-write and component deletion,
readout/embedding interventions and component keep-only masks over prefill and
two cached decode steps. All intervention trials produce a score change greater
than `1e-5`. Missing, static and conflicting whole-write owners reject before
execution. All 22 numerical component tests pass (3.80s), and six additional
LFM2 mixed-state, ordinary parallel and routed-provider tests pass (0.02s).
All four native LFM2 configurations pass: resident TP2, host-layerwise TP2,
disk-streamed PP2 and combined TP2/PP2. Native captures agree with ordinary
execution within `2e-4 + 2e-4 * abs(reference)` across three predictions; preview,
full-vocabulary reductions, controlled branches, skipping and cold failure/retry
remain included. The initial 17-case run passed 16 cases (53.50s); the host LFM2
case stopped in fixture setup because the helper lacked that family. After adding
the fixture entry, that case passed (6.65s), completing the same-production
matrix. The native build passed (33.97s; fixture rebuild 6.97s).

The full architecture library suite passes 537 tests with one existing ignored
test (0.36s). Its first run exposed an older assertion tied to communication
operation order; the test now verifies the four required operations independently
of their storage order. Compilation without default features passes (4.21s),
as do formatting and whitespace checks. The following section records the
subsequent routed LFM2 propagation.

```sh
cargo test -p eredu-architectures --test reference_numeric prepared_lfm2_components_and_convolution_cross_partitions_and_residency -- --test-threads=1
cargo test -p eredu-architectures --test reference_numeric components:: -- --test-threads=1
cargo test -p eredu-architectures --test reference_numeric lfm2 -- --skip components:: --test-threads=1
cargo test -p eredu-backend-mlx --features metal --lib ring_public_component_capture_lfm2_ -- --ignored --test-threads=1
```

## Routed provider observation propagation

Provider-backed ordinary and parallel units now have distinct internal-hook
facts. The shared traversal passes capture/intervention callbacks into the
selected architecture call while retaining its ordinary group, residency and
forward-resource lifecycle. Pipeline provider selection is shared between
observed and ordinary calls. LFM2-MoE uses the same dense/attention component
boundaries and emits its whole routed contribution after the existing expert
reduction. Other families retain conservative defaults until their actual
provider calls implement internal hooks. The neutral routed LFM2 case passes
(1.08s) across TP2, PP2 and combined TP2/PP2 in all three residencies. It compares
ten baseline/intervention trials, including complete expert-write deletion,
through prefill and two cached decode steps. All 23 component regressions pass
(4.61s), six further LFM2 tests pass (0.02s), and the 396 runtime tests pass
(0.30s). All four native routed LFM2-MoE configurations pass: resident TP2,
host-layerwise TP2, disk-streamed PP2 and combined TP2/PP2. They capture actual
and effective dense units, attention channels and complete convolution/expert
writes through prefill and two cached decode steps, comparing ordinary native
execution with the established `2e-4 + 2e-4 * abs(reference)` tolerance. Controlled
branches, previews, vocabulary reductions, skipping and cold failure/retry remain
included. The full native Ring matrix passes all 20 cases (65.57s).

All 17 broader routed numerical regressions pass (63.44s), including shared and
composite expert execution, TP/PP/EP schedules and provider observations. The
full architecture library suite passes 537 tests with one existing ignored test
(0.36s). Native compilation passes (33.83s), compilation without default features
passes (5.36s), and formatting/whitespace checks pass. Internal hooks on other
specialized family/provider paths still require propagation.

```sh
cargo test -p eredu-architectures --test reference_numeric prepared_lfm2_moe_components_and_complete_expert_writes_cross_partitions_and_residency -- --test-threads=1
cargo test -p eredu-architectures --test reference_numeric routed -- --skip components:: --test-threads=1
cargo test -p eredu-runtime --lib
cargo test -p eredu-architectures --lib
cargo test -p eredu-backend-mlx --features metal --lib ring_public_component_capture_ -- --ignored --test-threads=1
cargo check -p eredu-backend-mlx --no-default-features
```

## Shared routed decoder attention and residual terms

Qwen MoE and GPT-OSS now declare attention channels independently of their routed
FFNs. Each unit exposes its normalized FFN input and completed expert contribution
as `feed_forward.contribution` plus the effective companion. The whole-write
record's optional `input` names the normalized input; it is also populated for
existing convolution, recurrent and routed whole writes. Expert contributions
include routing, expert biases and the completed parallel reduction. They remain
whole terms: individual expert-neuron attribution is an implementation gap.

The shared provider path uses the ordinary decoder's actual attention, cache,
normalization and residual driver. Attention masks change values consumed by the
output projection; the whole expert hook follows its reduction. Provider-owned
routing observations keep their existing timing. Generic partitioned input and
readout callbacks also use the actual computation, with separate loaded hook facts.

GPT-OSS attention metadata names its learned normalization sink logits in
`AttentionAttributes::sink_logits`, alongside the existing GQA mappings and Q/K/V/O
biases. Sink logits contribute to the softmax denominator with zero value vectors.
They do not add separate residual vectors. The numerical test confirms their
actual effect, reconstructs attention writes including output bias once, adds
whole expert and embedding terms, applies the declared final RMS equation and
reconstructs full scores and selected score differences through prefill and two
cached decode steps. It uses expanded scalar expert fixtures; native coverage
separately loads the published MXFP4 representation.

The source-backed Qwen MoE numerical test passes all nine combinations of TP2,
PP2 and TP2/PP2 with resident, host-layerwise and disk-streamed loading (0.76s).
Nine baseline/intervention trials cover attention deletion/keep-only selection,
normalized expert input, whole expert contribution and readout boundaries. Every
intervention has a score effect above `1e-5`, and partition captures match ordinary
values through prefill and two decode steps. The new GPT-OSS reconstruction test
passes (0.06s).

The native Ring capture matrix passes all 24 cases (18.10s, default-feature CPU
build), including four each for Qwen MoE and GPT-OSS: resident TP2, host-layerwise
TP2, disk-streamed PP2 and resident TP2/PP2. Captures match ordinary native execution
within `2e-4 + 2e-4 * abs(reference)`; tokens match exactly. The same tests retain
controlled branch, replay, previews, full-vocabulary scores, budget/skip and cold
failure/retry checks. The neutral stripped-descriptor test continues to verify
final publication independently of component declarations.

The Metal-enabled build passes the same 24 CPU Ring fixtures (86.41s); this is
feature-build coverage, not a distributed GPU result. The current regression run
also passes all 25 component numerical tests (6.05s), 17 broader routed tests
(63.79s), 537 architecture tests (one existing ignored), 226 core tests and 396
runtime tests. Portable facade/conformance tests pass without backend features
(19 plus 59, one existing ignored); backend compilation without default features
passes (9.48s). Formatting and whitespace checks pass.

```sh
cargo test -p eredu-architectures --test reference_numeric prepared_qwen_moe_attention_and_complete_expert_writes_cross_partitions_and_residency -- --test-threads=1
cargo test -p eredu-architectures --test reference_numeric gpt_oss_sink_attention_and_biased_expert_terms_reconstruct_scores -- --test-threads=1
cargo test -p eredu-backend-mlx --lib ring_public_component_capture -- --ignored --test-threads=1
```

## Partition intervention preparation

The local execution mechanism preserves the original admitted operation and its
prediction schedule while projecting selected positions through retained
architecture coordinates. `PartitionActivationLayout` enumerates every actual
invocation member, including replicas that do not export captures.
`PartitionActivationProjection` computes bounded native regions and keeps compact
component masks as one local operation even when storage is permuted. Whole-value
Mask/Replace/Add payloads preserve their global selected-result order and exact
F16/BF16 bits. A valid global logit mask may exclude a complete local vocabulary
shard; an empty local exclusion list submits no update.

`reserve` charges projected host storage before copying payloads, then reserves
all native updates using the backend estimator. Its move-only
`ReservedPartitionActivation` retains the original plan identity, intent identity,
operation, phase, prediction and exact local dtype/shape. Applying or dropping it
does not refund credits. Native updates reuse the ordinary activation mechanism;
they neither mutate the input nor establish completion or global publication.

Core admission now exposes `intent_identity()` for exact source/semantic/request/
operation agreement across distinct local sessions. The ordinary `identity()`
continues to bind the loaded session and run. Both digests are streamed from
canonical serialization without a second complete JSON payload buffer, and both
strings count toward snapshot and child-preparation storage.

The public distributed path now uses the shared capture owner for invocation/epoch
authority, pre-forward agreement, coordinated failure handling and global outcome/
evidence publication. The native integration validation below is separate from
the earlier local-mechanism validation.

The current local-mechanism validation passes six native tests on MLX CPU/Metal
(3.57s). New tests cover permuted F32/F16/BF16 replacement payloads, input
preservation, unchanged prepaid charges at execution, keep-only sets with no
local survivor, and complete local-vocabulary exclusions. Existing activation
conformance, patch/scale/mask/bias, exact half-payload and estimator tests also pass.
Neutral tests compare all eight action variants against ordinary values across
contiguous, permuted, replicated, empty and outer-axis partitions. They verify
fragment limits, compact mask size, host/native budget rejection, failed-work
charges and source shape/dtype rejection before native application. Architecture
coverage includes all replica members, inactive invocation owners, prefill/decode
shapes and member/region bounds. Current library suites pass 537 architecture
(one existing ignored), 227 core and 400 runtime tests.
Portable conformance/facade suites pass without backend features (59 and 19,
one existing ignored); the backend check without default features passes (11.02s).
Formatting and whitespace checks pass.

```sh
cargo test -p eredu-core --lib distributed_intent_preserves_semantics_without_sharing_session_authority
cargo test -p eredu-runtime --lib intervention::
cargo test -p eredu-architectures --lib component_partition::
cargo test -p eredu-backend-mlx --features metal --lib composition::mlx::session::intervention::tests -- --test-threads=1
```


## Public partition interventions and controlled branches

`TextGenerationBackend::intervention_discovery` now combines retained partition
ownership and invocation-group support with the ordinary native activation facts.
Admit the same host `InterventionPlan` independently against each rank's loaded
session, then use `ControlledTextGeneration::enable_interventions` or the ordinary
`TextGenerationDriver` setup. Local admission identities remain distinct; common
preparation compares global intent, exact ordered coordinates, schedules, geometry,
membership and prepaid costs. The model's existing execution driver applies each
local operation to the actual downstream input.

The live capture owner owns operation authority and global publication. Replicas
apply the operation even when they export no capture; empty local selections
acknowledge without manufactured tensors. Inactive pipeline ranks stay outside
invocation collectives. Input geometry agreement precedes dependency execution;
native edits and per-operation before/after evidence complete before the final
invocation vote. After shared forward completion, bounded world receipts reject
missing members. Global `Applied` and evidence records publish only on final commit.
Ordinary capture retains its pre-intervention meaning. Evidence uses separate
operation/side keys and surrounds every ordered operation, including two operations
on the same target. Summary reduction can differ by floating reduction order.

Use the existing `TextContinuationSnapshot`/`TextBranchRequest` flow to re-admit
future global masks from identical native context. The child receives fresh owner
and run identity; parent and sibling plans and state remain isolated. Restore
reproduces values and evidence without refunding projection, capture, transport or
copy budgets. Direct ordinary editing on a partition-bound owner is rejected.
An intervention limit is a failed attempt, including with capture skip policy;
ordinary optional capture skipping retains its existing coordinated behavior.

Validation covers all 24 CPU Ring configurations: Qwen3, Qwen MoE, GPT-OSS,
Nanbeige, LFM2 and LFM2 MoE, each with resident TP2, resident TP2/PP2,
host-layerwise TP2 and disk-streamed PP2. Each compares public deletion and
keep-only trials plus ordered scaling against ordinary inference for prefill and
two cached decode steps, using the last prompt row and first decode row explicitly.
Sampled token IDs agree; logits and tensor evidence use `3e-4 + 3e-4*abs(reference)`.
Controlled children verify distinct mask effects, exact same-child replay,
before/after evidence replay, unchanged parent behavior, fresh sibling provenance,
and cumulative charges after restore. The five unchanged family fixtures passed
20 cases; the four GPT-OSS cases passed in 16.99s with patterned signed read/write
weights and the published MXFP4 expert representation. Patterned weights avoid a
uniform-readout fixture that cannot distinguish causal trials.

The neutral five-rank fixture includes a permuted shard, a replica, an empty shard
and an inactive pipeline rank. It checks ordered effects against explicit scalar
values and ordinary execution, Preview/Summary evidence, commit/abort publication,
intent and coordinate disagreement, missing invocations, member geometry/native
failures, foreign owners, direct-authority bypass and stale work after restore.
The current runtime suite passes 403 tests (0.28s); architecture passes 537 with
one existing ignored test. Portable conformance/facade pass 59 and 19 tests
(one existing ignored), and the backend check without default features passes
(11.70s). Local CPU/Metal activation tests cover exact F16/BF16 payloads separately;
the multi-rank results here use CPU execution and F32 activations. They do not
establish distributed GPU validation or parallel parameter queries/overlays.

```sh
cargo test -p eredu-runtime --lib
cargo test -p eredu-architectures --lib
cargo test -p eredu-backend-mlx --features metal --lib ring_public_component_capture_ -- --ignored --test-threads=1
cargo test -p eredu-backend-mlx --features metal --lib composition::mlx::session::intervention::tests -- --test-threads=1
cargo test -p eredu --no-default-features --test portable_facade --test backend_conformance
cargo check -p eredu-backend-mlx --no-default-features
```

Global accounting includes replicated host projection/admission storage, each
actual member's native edits and dependency completion, invocation votes, world
outcomes, evidence transport, decoding and assembly. Receipt decoder allowances
remain conservative and can dominate small tensors. Tight retained-layout region
bounds keep compact masks from reserving the maximum fragment count. These are
logical storage bounds; native private allocator workspace is not a physical cap.
Parallel parameter operations, remaining family hooks and the complete accounting/
released-checkpoint audit remain on the implementation checklist.

## Effective parameter coordinates for parallel execution

`eredu_core::parameters::ParameterCoordinateMap` describes every logical axis of
one effective local parameter. `project_region` returns bounded contiguous local
rectangles and their destinations in the original selected global rectangle.
`PartitionParameterRegion::project_update` selects the corresponding Add/Replace
payload. `project_contraction` selects only the required direction coefficients
and retains both the output destination and contributing source rectangle.
Contractions along a sharded axis require summing partial results; other axes
produce separate output regions. Complete source coverage must be established
before either result is presented as a global measurement.

`eredu_architectures::prepared_sources::PreparedModelDiscovery::parameter_partition_layout_for_rank`
obtains these coordinates from the retained execution, parameter declaration and
selected task aliases. It shares the loading planner's TP/EP and pipeline ownership
logic. Only the requested atomic parameter group is lowered. Packed affine,
MXFP4 and GGML layouts map complete encoding blocks to logical scalars; invalid
partial blocks or reordered bytes are rejected. Uncompressed FP8 axes retain their
scalar coordinates, including uneven ranges. A stage with no parameter returns
no local coordinates; a local empty shard retains an explicit empty map. Tied
static consumers may own copies on more than one stage.

These contracts do not grant loaded-session or communication authority. They
perform no native work or source-content resolution. Metadata and selected
payload copies consume caller-supplied reservations before allocation, and
failure or dropping a result does not refund them. Range identities use compact
canonical runs, so even very large vocabulary dimensions need no full-axis index
buffer or iteration. The bounds describe logical storage, not allocator capacity.

Neutral tests compare nonzero signed queries, Add/Replace edits and contractions
over all axes against independent full-tensor calculations, with simultaneous
three-axis shards and permutations. Architecture tests cover packed expert axes,
fused rows, complete-block permutations, malformed block boundaries, uneven FP8
ranges and empty owners. Prepared TP2/PP2 discovery covers every selected dense
parameter with tied and untied embeddings and verifies that content identity stays
unresolved. The current suites pass 230 core tests and 541 architecture tests
(one existing ignored architecture test).

```sh
cargo test -p eredu-core --lib parameters::partition
cargo test -p eredu-architectures --lib parameter_partition
cargo test -p eredu-core -p eredu-architectures --lib
cargo check -p eredu-backend-mlx --no-default-features
```

Prepared parameter access now borrows each executor's actual architecture and
residency policy through `eredu_runtime::parameter_operations::LayeredParameterOwner`.
Ordinary, direct/routed partition, pipeline and composite execution share static
traversal, completion-safe unit loans and replacement publication. The selected
policy validates the exact local ordinal and semantic unit address, restores
resident loans after callback rejection, and applies retained replacements when
bounded units reload. These local mechanisms grant no global transaction authority.

The scalar prepared-construction test covers 18 configurations: tied and untied
embeddings, three residencies, and TP2, PP2 and TP2/PP2. Actual local parameter
values and projected edits agree with independently localized global values.
Coordinated attention, FFN, normalization and embedding replacements agree with
an independently edited ordinary model over prefill and two cached decode steps
within `2e-4` absolute error; restoration returns to the ordinary baseline.
Injected parameter callback failures leave the local owners usable. All 69 runtime
backend-independence tests also pass. These checks exercise real prepared
partition executors, but do not implement or validate atomic peer publication.

```sh
cargo test -p eredu-runtime --test backend_independence
cargo test -p eredu-architectures --test reference_numeric parameter_access:: --no-default-features
```

Four native Qwen Ring configurations also verify every prepared local dense slot:
resident TP2 and TP2/PP2, host-layerwise TP2, and disk-streamed PP2. They reject
foreign unit addresses and invalid ordinals before traversal, recover after a
completed read's callback rejection, read the published replacement values after
bounded reload, and restore the original values exactly. Later ordinary-reference
capture, intervention and controlled-branch checks still pass. The native adapter
retains both the architecture unit address and local residency address; this
distinction fixes access on later pipeline stages without changing unit geometry.
The complete existing CPU Ring component matrix passes all 24 tests in 106.57s
after this change. This validates local parameter mechanisms and existing global
capture/intervention behavior, not atomic distributed parameter transactions.
The public native parameter regression suite also passes all 19 CPU/Metal tests
in 37.63s, including ordinary and shared overlays, streamed failure atomicity,
prepared BF16 metadata, FP8 inputs and partial blocks, and affine/MXFP4 queries,
edits and restoration. Runtime and architecture unit suites pass 403 and 541 tests
respectively; portable backend conformance and facade suites pass 59 and 19 tests.

```sh
cargo test -p eredu-backend-mlx --features metal --lib ring_public_component_capture_ -- --ignored --test-threads=1
cargo test -p eredu --no-default-features --features mlx,metal --test native_execution_control parameters:: -- --include-ignored --test-threads=1
```

The runtime now provides a complete-source read plan and a live operation driver.
`PartitionParameterReadPlan` selects one producer per source cell, including partial
replica overlaps and differently fragmented permutations. It rejects missing
coverage and incorrect or non-finite payloads instead of supplying zeros. Native
partial contractions are combined in deterministic rank/tile order using F64 sums
and one F32 conversion. All geometry, selected host copies and assembly buffers
require reservations; compact huge axes need no full-weight index buffer.

`ParameterOperationCoordinator` retains a setup-bound monotone attempt sequence.
Its separately admitted control allowance permits shared rejection even when a
rank has no parameter-work budget. `ParameterReadPreparation` supplies layout and
payload facts only after local preparation succeeds. Ready peers agree those
facts, and every phase confirms the same completed decision before advancing.
Reads expose results only after all ranks complete assembly. Transactions prepare
all ranks before publication and restore all ranks after a completed publication
rejection; failed restoration or ambiguous communication fences execution.
Backend callbacks retain responsibility for native completion and compatible
parameter/cache/snapshot state. Control usage is reported separately in
`ParameterDiscovery::coordination_usage` and is never refunded by restoration.
Coordinator counters remain inspectable after an unwind; a poisoned owner rejects
new operations as fenced rather than appearing temporarily busy.

`AdmittedParameterOverlay::intent_identity` compares exact global edits, sharing,
source and prior overlay across independent local admissions. Its ordinary
`identity` remains bound to the local loaded execution. Neither digest alone grants
communication, branch, version or native publication authority. Both digests use
bounded streaming serialization; the original local identity format is preserved.

The runtime also issues one `ParameterModelIdentity` for each loaded model in a
communication setup. MLX retains it at model construction. Corresponding ranks
pair constructions in the same order, then bind operations to that particular
model, artifact, execution and parameter version. Two models with identical
weights cannot exchange operations merely because they share a communicator.
This host registration performs no native work and is not rewound by reset or
snapshot restoration.

Native CPU Ring validation passes all four dense Qwen configurations (resident
TP2 and TP2/PP2, host-layerwise TP2, and disk-streamed PP2). It reads selected global
O-projection entries and contracts both axes against ordinary public-API results,
with signed coefficients and absolute tolerance `2e-5`. A failure injected after
one rank publishes replacements restores every rank's originals; global reads
confirm restoration, successful retry, activation and removal. Budget rejection
and local preparation failure both complete without source reads. These fixtures
exercise the real retained parameter owners and transport; public global
cache/version publication remains a separate integration requirement.
The complete six-family CPU Ring regression also passes all 24 configurations
after model identity registration was added. The current library checks pass 231
core, 413 runtime and 541 architecture tests, with one existing architecture test
ignored; the public native parameter regression passes 19 tests.

```sh
cargo test -p eredu-backend-mlx --features metal --lib ring_public_component_capture_qwen3_ -- --skip qwen3_moe --ignored --test-threads=1
cargo test -p eredu-core -p eredu-runtime -p eredu-architectures --lib
```

Public distributed discovery, queries and projections now use the retained
model/version binding. All 24 CPU Ring configurations across Qwen3, Qwen MoE,
GPT-OSS, Nanbeige and dense/routed LFM2 passed public discovery, selected queries,
signed contractions, asymmetric budget rejection and stale-identity retry. The
first broad run exposed missing emitted quantization companions and an incorrect
assumption that sharing must stay within one task; the affected eight GPT-OSS and
Nanbeige cases pass after those fixes (38.14s). The architecture library passes
542 tests, including scalar companion geometry through expert/fused permutations.
The portable facade/conformance regression passes 19/59 tests, with one existing
external tokenizer fixture ignored.

Discovery is collective metadata work and does not evaluate weights. Read
`LoadedParameter::access()` for exact query/projection/replacement support. Every
query includes bounded metadata exchange and cumulative charges; a failed attempt
does not refund them. GPT-OSS's dense slots and grouped MXFP4 expert banks are covered.
Grouped affine/MXFP4 access validates exact packed geometry and preserves per-expert edits. Distributed install/remove
passes the same 24 configurations with shared provenance and reversible cache
exchange: four Qwen cases in 23.13s and the remaining 20 in 95.76s. These exercise
public edits against an independently edited ordinary model, budget rejection,
post-publication failure and retry, prefill plus cached decode, stale removal
authority, snapshot invalidation and exact restoration. The expanded active-overlay controlled capture and sibling replay matrix passes
all 24 cases in 117.18s, including shared provenance, exact replay, skipped-capture
siblings and nonrefunded copy/observation budgets. The ordinary public parameter
regression passes 21 CPU/Metal cases in 40.40s. Current portable checks pass
232 core, 419 runtime, 70 backend-independence, 59 facade-conformance and 19
facade tests (one existing external tokenizer fixture ignored). The backend
without default features and formatting/diff checks also pass.

For a distributed Inspector worker, every rank participates in parameter discovery,
queries, activation and removal in matching order. Use each rank's returned local
`identity` as its plan's `base_identity`; keep the global edits and provenance
identical. Local `AdmittedParameterOverlay` values bind that local authority, while
`intent_identity()` describes the common edit. Successful distributed activation
returns that common digest in `overlay_identity`; capture receipts retain it through
controlled replay and sibling branches. Ordinary single-model activation retains
its local admitted identity. Neither digest alone grants authority on another model.

Activation reserves projected payloads, affected native parameters, conversion and
update temporaries, metadata and fresh-state initialization before publication.
Every actual alias and replica receives the edit. An affected packed parameter is
promoted on all nonempty owners, including shards outside the selected rectangle;
this keeps projection input arithmetic consistent. Only affected parameters are
promoted, and original packed storage is retained for exact restoration.

A completed peer rejection restores original weights, cached state and prompt
identity and preserves existing snapshot validity. Confirmed success clears cached
state and invalidates old prepared requests and snapshots on every rank. Prepare a
fresh exact-ID prefix after activation or removal. Removal retains active handles
until its own publication is confirmed, uses the already-known fresh-state bound,
and charges state/metadata work without reloading weights. Parameter and control
counters never rewind. An unresolved transport or failed rollback fences the owner;
it is not reported as a usable partially edited model.

Reproduce the distributed public read/edit/control matrix and ordinary regression:

```sh
cargo test -p eredu-backend-mlx --features metal --lib ring_public_component_capture_ -- --ignored --test-threads=1
cargo test -p eredu --features metal --test native_execution_control parameter -- --ignored --test-threads=1
cargo test -p eredu-core -p eredu-runtime --lib
cargo test -p eredu-runtime --test backend_independence
cargo test -p eredu --no-default-features --test portable_facade --test backend_conformance
cargo check -p eredu-backend-mlx --no-default-features
```

These distributed tests use native MLX CPU Ring processes. They do not establish
distributed Metal/CUDA validation or individual expert-neuron topology/capture. Existing pinned
released-checkpoint results above remain separate evidence; no new released model
was substituted for the numerical partition fixtures in this matrix.

The grouped-parameter extension passes two independent native CPU/Metal mechanism
tests (3.97s). Hand-packed affine and MXFP4 banks with shape `[3, 4, 64]`, varied
scales, signed values and known decoded entries verify a three-axis rectangle,
signed contraction along every axis, grouped multiplication, a single-expert row
edit and exact restoration. Public discovery, queries, projections and coordinated
edits additionally pass all 12 GPT-OSS/Qwen MoE/LFM2 MoE Ring configurations
(59.56s). Each fixture explicitly requires both grouped read and write parameters
to participate. The tests retain the active-overlay controlled capture, shared
provenance, snapshot/sibling replay, peer rollback and restoration checks. GPT-OSS
uses its published MXFP4 layout; the other routed fixtures use dense banks. The
ordinary public parameter regression passes all 21 CPU/Metal tests again (42.74s).

```sh
cargo test -p eredu-backend-mlx --features metal --lib grouped_affine_mxfp4_parameters_project_edit_and_restore_ -- --ignored --test-threads=1
cargo test -p eredu-backend-mlx --features metal --lib ring_public_component_capture_ -- --skip nanbeige --skip ring_public_component_capture_lfm2_pipeline_streamed --skip ring_public_component_capture_lfm2_tensor --skip ring_public_component_capture_qwen3_pipeline_streamed --skip ring_public_component_capture_qwen3_tensor --ignored --test-threads=1
```

Grouped FP8/GGUF now have native effective-parameter mechanisms. Public grouped
GGUF parallel acceptance and ordinary FP8 lifecycle acceptance are recorded below.
Grouped FP8 parallel acceptance is recorded in the later
[parallel lifecycle section](#grouped-fp8-parallel-parameter-lifecycle), including its
remaining encoding and sharding gaps. Subsequent sections record the
individual expert-neuron integration and its remaining family coverage.
The supported grouped parameter operations do not imply that neuron-level routed
attribution is complete.

### Grouped FP8/GGUF and public GGUF parameter validation

Native grouped effective access now includes E4M3 with floating or E8M0 scales,
and GGUF matrix banks. Admission checks logical geometry and physical byte extent;
decoding preserves singleton group axes. FP8 reservations include every group and
both padded 128-value block axes. The retained format and companions survive
publication, while actual floating storage selects ordinary grouped multiplication.
For FP8, this also selects dense input arithmetic for the entire affected bank.
Quantized routing projections similarly accept floating parameter replacements
while retaining their declared routing precision and input transformations.

The numerical mechanism tests use independent hand-packed `[1|3, 130, 129]`
FP8 and `[1|3, 4, 64]` Q8_0/IQ4_NL banks, including both GGUF endiannesses. They
verify effective entries, rectangular reads, signed contractions on every axis,
single-expert edits, grouped multiplication and exact restoration. Additional
inexact FP8 inputs distinguish dense overlay arithmetic from activation
quantization. Native results agree with F64 contractions within
`1e-4 + 1e-6 * abs(reference)`; power-of-two fixture results are exact.
The final CPU/Metal pair passes in 2.74s. Routing projection edits and restoration
pass on CPU/Metal for affine and Q8_0 storage. The combined encoding, routing and
existing FP8 regression passes all 17 tests (3.07s).

The public GGUF fixture writes Q8_0 or IQ4_NL Qwen2 checkpoints directly and
independently constructs F32 SafeTensors reference weights from the same scalar
equations. No library decoder creates the reference. Both CPU/Metal tests pass
(9.55s), covering 12 format/device/residency combinations. Every effective
parameter matches the reference exactly. Native projections, prefill and cached
decode, coordinated Q/K/V/O plus FFN and embedding edits, changed prefixes,
controlled snapshot replay, exhausted-budget rejection and exact restoration are
verified through public APIs. Edited inference agrees with independently edited
reference checkpoints within `3e-5 + 3e-5 * abs(reference)`; original source bytes
remain unchanged. The full public parameter regression passes 23 tests (43.15s).
The backend builds without default features; formatting and diff checks pass.

```sh
cargo test -p eredu-backend-mlx --features metal --lib grouped_fp8_gguf_parameters -- --ignored --test-threads=1
cargo test -p eredu-backend-mlx --features metal --lib packed_selector_parameter -- --ignored --test-threads=1
cargo test -p eredu-backend-mlx --features metal --lib parameters::encoding::tests -- --include-ignored --test-threads=1
cargo test -p eredu --features metal --test native_execution_control native_gguf_parameter_lifecycle -- --ignored --test-threads=1
cargo test -p eredu --features metal --test native_execution_control parameter -- --ignored --test-threads=1
cargo check -p eredu-backend-mlx --no-default-features
```

These are deterministic synthetic fixtures with explicit arithmetic and no RNG.
They supplement the pinned released-checkpoint evidence above. Ordinary grouped
FP8 acceptance follows; the later [parallel lifecycle section](#grouped-fp8-parallel-parameter-lifecycle)
records parallel coverage and the remaining partial-tail and family gaps.

### Public grouped FP8 parameter lifecycle

The public K2 Horizon fixture has three layers: one dense layer and two routed
SwiGLU layers with three experts each. All three experts participate, with distinct
nonzero encoded weights and block scales. Only the routed banks use E4M3/F32
128-by-128 blocks. Hidden widths 128 and 130 cover complete blocks and partial
column/output-row tails; each expert has 128 units so fused gate/up scale rows
remain aligned. The reference expands bytes with independent scalar E4M3 and
block-scale arithmetic, then assembles each expert's gate/value rows in order.

All four loaded grouped read/write parameters match that reference exactly.
Selected expert rows and bounded signed contractions use `query_parameter` and
`project_parameter`; contraction errors stay within
`2e-6 + 2e-6 * abs(reference)` against F64 accumulation. Exhausted query and overlay
budgets reject without consuming work. A serialized multi-parameter plan edits one
expert's gate row and write column together with attention Q/K/V/O parameters.

The independent edited checkpoint promotes both affected fused banks to F32,
including unedited experts in those banks. That is the advertised execution
semantics: promotion removes dynamic FP8 input rounding for the entire affected
matrix. The second routed layer retains its encoded weights and FP8 input
transform. New prefixes and prefill plus two cached decode steps agree with the
independently edited checkpoint within `3e-5 + 3e-5 * abs(reference)`; sampled
tokens match exactly. Ordinary and controlled records agree, controlled snapshots
replay exactly, edit identities propagate, and removal restores the original
predictions and discovery facts exactly. Source checkpoint bytes remain unchanged.

Both CPU/Metal tests pass (10.47s), covering twelve device/geometry/residency
configurations across resident, host-layerwise and disk-streamed execution. These
tests use the public facade and existing mechanisms; this acceptance increment
adds no production behavior. Fixture construction is deterministic; generation
uses seed 17 and greedy sampling. The later
[parallel lifecycle section](#grouped-fp8-parallel-parameter-lifecycle) records
parallel validation. Arbitrary partial blocks on a partitioned FP8 axis and
released sparse-checkpoint validation remain separate gaps.

```sh
cargo test -j 1 -p eredu --no-default-features --features mlx,metal --test native_execution_control native_grouped_fp8_parameter_lifecycle -- --ignored --test-threads=1
```

### Packed GGUF MoE parallel parameter lifecycle

The Qwen3-MoE fixture now uses actual Q8_0 expert banks in its first layer and
IQ4_NL banks in its second layer, with four experts, intermediate width 64 and
hidden width 32. Both split gate/up inputs are packed independently. An independent
SafeTensors model is written from the scalar equations before loading either
execution. All four grouped read/write parameters must participate in the public
queries and coordinated edit, so neither encoding can be skipped by the test.

This exposed and fixed a cold admission bug: the metadata-only Qwen configuration
had left derived packed banks marked dense. Architecture admission now retains
the physical encodings before recipe selection, normalizes gate/up and down
identities using the declared parameter root, and rejects incompatible split
gate/up formats. Exact runtime task/descriptor checks remain enforced. Neutral
tests cover real GGUF catalog admission, mixed layers, canonical identities and
rejection; the architecture suite passes 544 tests with one existing ignored test.

The native CPU Ring matrix passes resident TP2, TP2/PP2, host-layerwise TP2 and
disk-streamed PP2 (four tests, 24.21s). Effective queries match the independent
reference exactly; signed projections agree within `2e-5` and cached predictions
within `1e-4`. The existing public suite also verifies multi-parameter activation,
asymmetric budget rejection, completed peer-publication rollback of weights and
caches, stale removal, active-overlay capture provenance, exact controlled
snapshot/sibling replay and restoration. Ordinary public GGUF CPU/Metal lifecycle
regression passes across all three residencies (two tests, 6.93s). EP2 and
combined TP2/EP2 additionally pass the same lifecycle with experts owned by
different ranks (two tests, 16.03s), giving six packed GGUF configurations in total.
The backend build without default features and formatting/diff checks pass.

```sh
cargo test -p eredu-backend-mlx --features metal --lib ring_public_component_capture_qwen_moe_packed_gguf -- --ignored --test-threads=1
cargo test -p eredu-architectures --lib
cargo test -p eredu --features metal --test native_execution_control native_gguf_parameter_lifecycle -- --ignored --test-threads=1
```

This matrix uses deterministic synthetic weights and native CPU communication.
It does not establish distributed GPU validation or expert-neuron attribution.

### Selected expert-unit mechanism boundary

`eredu_nn::GroupedUnitObserver` now exposes actual post-activation/gating values
inside MLX gated-product and ReLU-squared banks. This is a low-level neutral
mechanism, not yet a loaded public expert-neuron capture capability. Public
architecture topology, runtime admission and controlled/distributed delivery
remain required integration work. Provider coordinate mapping is described below.

A `GroupedUnitBatch` borrows values in sorted route order, local bank group IDs,
original flattened selection indices, chunk-relative token indices and the
original coefficient matrix. Its token offset preserves coordinates across
operator chunks. The observer sees only selected routes; an inactive expert has
no fabricated zero-valued row. Duplicate selections remain distinct through
selection indices. Local bank IDs and provider token rows are not global expert
or sequence identities; compact-bank and expert-exchange mappings must be applied
by their owning providers.

Callbacks run before the down projection: original observation, optional
replacement, and effective observation. The replacement must preserve exact
shape and dtype. A callback failure stops the projection and retains its original
error source. The effective hook reports values before any projection input
quantization. Quantized multiplication-input evidence is a separate requirement.
An absent observer uses the ordinary computation without extra observation
tensors, copies or evaluation. Borrowing native values does not authorize
retention, host copies or transport; bounded public composition must reserve
those resources first.

The deterministic numerical fixture uses four distinct expert banks, two selected
routes per token, signed weights, duplicate selections, an entirely inactive
expert, gated read/output biases, and five-token and 67-token requests. The longer
gated request crosses three native chunks. Independent scalar equations verify
original and effective units plus the final route-weighted write under no-op,
Zero, Scale, Add, Replace and keep-only modifications. Two unit partitions are
summed and the gated output bias is added once. Failure cases put an invalid
matrix downstream and verify that observer errors and invalid replacements stop
before that matrix can execute.

Native CPU/Metal verification passes all three selected-unit tests (3.35s).
Original/effective unit values agree with independent F64 scalar equations within
`2e-6`; complete and partitioned writes agree within `2e-5`. No-op observation and
the absent-observer path reproduce ordinary output exactly in this fixture.
The existing grouped-operator tests pass 9 cases (2.33s); effective-parameter
encoding regressions pass 6 cases (1.76s), including affine, MXFP4, FP8 and GGUF.
The portable neural suite passes 37 tests, the backend checks without default
features, and formatting/diff checks pass. These are synthetic mechanism results,
not a new released-checkpoint validation run.

```sh
cargo test -p eredu-backend-mlx --features metal --lib grouped_unit -- --ignored --test-threads=1
cargo check -p eredu-backend-mlx --no-default-features
cargo test -p eredu-nn --all-features
```

This fixture validates the low-level selected-unit mechanism. It does not claim
facade capture admission, routed residency/EP delivery, global expert topology or
released-checkpoint expert-neuron attribution.

### Routed-unit provider integration

`ActivationObserver::routed_unit_observer` now supplies a borrowed
`RoutedUnitObserver` for one architecture invocation. The canonical resident,
planned resident, boxed/bank-dispatched and addressable gated/ReLU-squared
providers forward it, including tensor-parallel operator dispatch. Shared
decoder adapters and the custom Qwen hybrid, Kimi Linear, Nemotron-H and DeepSeek
routed/shared methods attach it before the actual expert computation. The
combined shared/routed output observation retains its existing timing.

`RoutedUnitBatch` adds the original pre-compaction route tensor, retained global
expert map and provider chunk offset to the native batch. Resolve the original
source route before mapping the expert ID; a compact native bank index is not a
checkpoint-global expert. `route_origin` composes provider and native chunks.
For expert exchange, it also resolves existing receive counts and original route
tags into sending-peer ordinal, token row and top-k slot. Empty peers and duplicate
expert selections retain their identities. Token rows remain separate from
generation prediction indices. This mapping borrows existing metadata and adds
no exchange, expanded activation tensor or host route vector.

An original, intervention or effective callback failure retains its neural error
source through provider adaptation. A neural architecture wrapper uses
`ObservedExpertProvider::execute_neural` to retain that cause even when its generic
provider errors are otherwise formatted as text. Competing observer attachments
fail before execution. These mechanisms do not provide capture authority or
reserve host/native retention; those obligations belong to the eventual admitted
public sparse collector.

Neutral numerical checks compare actual addressable compact banks with resident
gated and ReLU-squared banks through prefill/decode, ordinary and tensor-parallel
dispatch, unchanged observers and selected-unit edits. Separate exchange fixtures
exercise idle peers, provider chunks, sorted rows and mutation of the consumed
values. A complete Qwen3-Next hybrid fixture checks ordinary/observed equality over
prefill and two cached decode steps, a nonzero expert-unit ablation, fresh-trial
restoration and preservation of a callback failure through the model. Native
CPU/Metal provider tests use the existing independent scalar fixture with 67
tokens, routed wrappers, gated/ReLU-squared banks and ordinary/TP helper dispatch.
The provider TP fixture has one partition; the native operator fixture separately
checks two actual unit partitions and one-time output bias.

The full neutral numerical suite passes all 187 tests (131.24s), including the
provider fixtures and the existing family/residency/parallel regressions. That
run exposed duplicate DeepSeek V3 layer-output observation in pure EP. Its model
entry points now call internal block hooks, leaving unit input/output observation
to the traversal; standalone blocks still emit their own boundaries. The
Cartesian regression verifies a single replacement and rollback after observer
failure. The native two-process DeepSeek V3 EP session also passes (8.26s).

The five native unit/provider CPU/Metal tests pass (3.23s), the runtime suite
passes 419 tests (0.28s), and portable facade/conformance pass 19/59 tests with one
existing external-fixture test ignored. Backend compilation without default
features and the final Metal-enabled native test build pass. These results use
synthetic fixtures; no new released-checkpoint claim is made here.

```sh
cargo test -p eredu-architectures --test reference_numeric -- --test-threads=1
cargo test -p eredu-runtime --lib
cargo test -p eredu --no-default-features --test portable_facade --test backend_conformance
cargo test -p eredu-backend-mlx --features metal --lib grouped_unit -- --ignored --test-threads=1
cargo test -p eredu-backend-mlx --features metal --lib ring_two_process_deepseek_v3_expert_parallel_opaque_session -- --ignored --test-threads=1
cargo check -p eredu-backend-mlx --no-default-features
```

This integration still does not advertise public expert-neuron capture or
intervention. The following topology extension supplies expert identities and
parameter joins; bounded sparse records and receipts, controlled capture budgets,
actual EP delivery tests and remaining custom execution paths remain open. The exchange-coordinate fixture
does not establish a complete distributed capture lifecycle or native EP evidence.

### Routed expert-unit topology and effective parameter joins

`ArchitectureDescriptor::routed_components` describes sparse expert banks
separately from dense `components`. `RoutedComponentId { group, expert, index }`
identifies one logical invocation, checkpoint-global expert and unit within that
expert. Token rows and duplicate route slots are participation coordinates, not
additional component identities. A compact provider bank index must first be
resolved through the retained global expert map.

The architecture projects these declarations from the same canonical bank
construction specifications used by execution. Qwen MoE, GPT-OSS, Qwen hybrid,
LFM2 MoE, Kimi Linear, Nemotron-H sparse layers and DeepSeek V3/V4 target banks
declare exact packed or independent read/write slots. Gated declarations include
both read branches, fused row offsets, optional affine biases, activation,
clamps and value offset; Nemotron-H declares ReLU-squared units. Quantization
companions are not declared as affine biases. A missing residual scalar means
that transformation is not described as scalar; in particular V4's hyper-residual
mapping must not be treated as ordinary addition. This extension does not declare
shared experts, MTP invocations or complete specialized-family equations.

Join a group to `LoadedModel::parameter_discovery()` before accessing weights.
`read_weight`, `read_bias`, `write_column` and `write_bias` validate identities,
row mapping, expert count and the actual loaded rank/shape. They return a borrowed
`LoadedParameter` plus an exact `ParameterRegion`; packed regions retain the
leading expert dimension of extent one. An absent bias returns `None`; a declared
slot missing from the loaded model returns `RoutedComponentError::MissingParameter`.
Invalid geometry and duplicate slots are errors. The helpers do no native work,
grant no authority and do not upgrade the loaded slot's `access()` capability.

```rust,ignore
use eredu_core::component::{ComponentReadRole, RoutedComponentId};

let graph = eredu::api::inspect_architecture(&checkpoint)?;
let group = &graph.routed_components[group_index];
let unit = RoutedComponentId {
    group: group.id.clone(),
    expert: global_expert_index,
    index: unit_index,
};
let facts = model.parameter_discovery()?;
let read = group.read_weight(&unit, ComponentReadRole::Gate, &facts)?;
let write = group.write_column(&unit, &facts)?;
let row = model.query_parameter(
    &facts.identity, &read.parameter.id, read.region, cumulative_limits,
)?;
let column = model.query_parameter(
    &facts.identity, &write.parameter.id, write.region, cumulative_limits,
)?;
```

Use those same regions and loaded shape/dtype facts in ordinary `ParameterEdit`
plans. Gate and value rows can share one fused slot; disjoint edits remain subject
to the existing conflict and alias checks. Re-read loaded discovery after an
overlay changes the parameter version. Queries, projections, activation and
removal retain their existing cumulative limits, state invalidation and source
preservation semantics.

The group's unit activation paths now declare sparse observation points using
`CaptureTransform::RoutedUnits`. Consult loaded support before planning captures;
topology alone still does not establish collector or distributed support.
Sparse intervention plans now use global expert/unit identities as described below.
Inactive experts have no measured activation row. Exact
per-unit reconstruction must include each selected route's coefficient and its
expert output bias once, plus shared-expert and other residual terms. Existing
`readout.other_writes` still contains the complete routed contribution and must
not be counted again alongside its unit decomposition. `write_input: None` does
not establish that a quantized projection has identity input arithmetic.

The topology extension passes 545 architecture tests (one existing ignored test)
and 235 core tests. The core fixtures distinguish packed rank-three matrices from
independently named experts, cover absent and missing biases, overflow, malformed
row maps, duplicate loaded slots and unavailable loaded capabilities. Architecture
fixtures check fused gate/value offsets, GPT-OSS's biased/clamped equation, hybrid
normalization and ReLU-squared banks.

Two public native tests pass on CPU and Metal (5.47s), each covering resident,
host-layerwise and disk-streamed Qwen MoE execution with four distinct experts,
six units per expert and nonzero parameters. They independently index full queried
matrices to verify discovered expert/unit selections, reject insufficient budgets
without consuming them, activate coordinated gate-row/write-column edits and
verify every edited and untouched matrix entry. The edits change two different
prefixes, ordinary and controlled results agree exactly, removal restores both
baselines exactly, and checkpoint bytes stay unchanged. This is public parameter
validation, not released-checkpoint sparse attribution or complete family coverage.
Portable facade/conformance tests also pass without backend features (19/59;
one existing tokenizer fixture is ignored). Formatting and diff checks pass.

```sh
cargo test -p eredu-core -p eredu-architectures --lib
cargo test -p eredu --no-default-features --features mlx,metal --test native_execution_control native_routed_component_parameters -- --ignored --test-threads=1
cargo test -p eredu --no-default-features --test portable_facade --test backend_conformance
```

### Public sparse expert-unit capture

Use the loaded capture report to admit the group's original or effective unit
path with `CaptureTransform::RoutedUnits`. The axes are original flattened `token`
rows, original top-k `route` slots and per-expert `component` indices. For example,
this selection captures three alternating units for the last prefill token:

```rust,ignore
CaptureSelection {
    id: "expert-units".into(),
    path: group.activation.clone(),
    schedule: CaptureSchedule { decode: false, ..Default::default() },
    slices: vec![
        CaptureSlice { axis: "token".into(), start: prompt_len - 1, end: prompt_len, stride: 1 },
        CaptureSlice { axis: "component".into(), start: 1, end: 6, stride: 2 },
    ],
    transform: CaptureTransform::RoutedUnits,
}
```

The payload is `CapturePayload::RoutedUnits`. For each row, combine its `expert`
with `unit_start + offset * unit_stride` and the group's ID to obtain a
`RoutedComponentId`. Retain `token`, `slot` and `coefficient` separately as
participation evidence. The record's enclosing `prediction_index` is not a token
row. A decode selection normally addresses token row zero; give it a separate
schedule when the prefill selection addresses a later prompt row. Rows are ordered
by original token/slot after complete receipt validation, independent of native
sorting. Duplicate expert selections retain their separate route coefficients.

Capture plans, ledgers, stepping, replay and forks use the existing public APIs.
Large sparse records need adequate branch `TraceLimits`; snapshot retention also
reserves the branch's worst-case trace delivery storage. These are logical
reservations, separate from an allocator ceiling. Empty token or route selections
produce an empty sparse payload without native retention or route-value copies.

The neutral suite covers once-per-invocation reservation, selected nonzero values,
commit-time delivery, skipped work, incomplete/duplicate chunk receipts, malformed
global coordinates and shapes, non-finite value encoding, native failures and
nonrefunded allowances. Draining an incomplete low-level step marks every pending
sparse record failed and drops its unverified payload. Legacy tensor collectors
report sparse points unsupported through the separate collector fact rather than
silently advertising values they do not emit.

Verification passes 545 architecture, 235 core and 424 runtime tests, plus seven
numerical discovery, three routed-unit and two addressable-provider tests. The
final public CPU/Metal routed suite passes all four capture/parameter tests in
6.35s; portable facade/conformance passes 19/59 tests without backend features.
One existing architecture fixture and one external tokenizer fixture remain ignored.
The backend check without default features passes (19.09s), as do formatting and
diff checks.

The public native fixture uses Qwen MoE with four experts, six units each and a
67-token prefix, followed by two cached decode steps. Captured units are checked
against scalar SiLU gate/value equations using public loaded parameter queries;
coefficient-weighted write columns reconstruct the actual routed output within
`2e-5 + 2e-5 * abs(reference)`. Ordinary, controlled replay and snapshot-fork payloads
match exactly. Token/route/unit slices match independently filtered full captures,
empty route selections retain zero native capture bytes, and scheduled decode skips
remain explicit. This validates the F32 fixture; it does not establish released
checkpoint sparse attribution, complete family acceptance or quantized additive
reconstruction. Subsequent distributed delivery and editing validation appears
below.

```sh
cargo test -p eredu-core -p eredu-runtime -p eredu-architectures --lib
cargo test -p eredu-architectures --test reference_numeric discovery::
cargo test -p eredu-architectures --test reference_numeric routed_unit
cargo test -p eredu-architectures --test reference_numeric architecture_driver_executes_
cargo test -p eredu --no-default-features --features mlx,metal --test native_execution_control routed_components:: -- --ignored --test-threads=1
```

### Public sparse expert-unit interventions

`LoadedModel::intervention_discovery()` exposes sparse scalar targets with
`InterventionPoint.routed_units`. Each target retains its architecture-owned
provider path and global bank geometry. Its edit axes are `[token, component]`,
where `component = expert * units_per_expert + unit`. Use the checked
`RoutedComponentGroup::component_index(&RoutedComponentId)` helper to construct
compact masks without interpreting checkpoint names or freezing baseline routes.

For example, build an ordinary `InterventionOperation` targeting the group's
`activation` path, with `MaskComponents { dtype, indices, keep_selected: true }`,
an explicit `token` slice and a prefill-only schedule. A separate decode schedule
can select decode row zero. Pass the plan and original/effective sparse capture
selections to `prepare_intervened_token_ids`. The prepared request runs through
either `generate_observed_text` or `start_controlled_text`. Take the experimental
snapshot before advancing the forward being tested. Future-plan re-admission does
not retroactively recompute cached context; replay exact token IDs when needed.

Zero, Scale, Mask, Replace and Add are also supported. Payloads address the exact
selected virtual token/component rectangle, retaining singleton dimensions and
exact F32/F16/BF16 storage. Runtime selects only entries whose experts participate
in the current forward. Native tensors contain participating route rows only;
no virtual expert-dense activation is allocated. Every duplicate expert slot is
edited separately. Down-projection consumes the modified units, while surviving
units and later-layer routing are recomputed from the current residual.

Original and effective `RoutedUnits` captures bracket all operations at a boundary.
Their capture axes remain `[token, route, per_expert_component]`; expert identity
is part of every row. Dense Preview/Summary operation evidence is rejected because
it omits sparse participation coordinates. Use `InterventionEvidence::None` with
these independent captures. Per-operation sparse evidence remains unavailable.

One full-invocation reservation precedes route metadata copies and native edits.
It includes actual participating source/gather/update storage, index and payload
temporaries, host coordinate validation and compact mask storage. Each operation
retains a fixed-size receipt with expected/completed token counts and addressed
values. Only contiguous coverage can complete it. `Unmatched` reports no addressed
participating values, including all-keep masks; it never reports an absent expert
as a measured zero. Failures retain existing charges and invalidate completion.
The ledger is shared with captures, snapshots and isolated controlled branches.

The public ordinary implementation is separate from unfinished distributed sparse
delivery and editing. TP/EP/PP execution needs committed receipts and global unit
placement; local native hooks alone do not establish that support. Broader family,
quantized, callback-failure and released-checkpoint acceptance remain part of the
open full-goal checklist.

Native public validation uses a nonzero Qwen MoE fixture with two layers, four
experts, six units per expert and top-k two. A 67-token prefix crosses native
chunk boundaries and is followed by two cached decode steps. CPU and Metal each
exercise resident, host-layerwise and disk-streamed execution. Every participating
scalar is checked for exact token/global-expert/unit selection under deletion,
keep-only, all-keep, Scale, explicit Mask, Replace, Add and ordered Scale→Add.
All-keep matches baseline payloads exactly; later-layer original units change
under upstream masks, demonstrating recomputation. Controlled execution, initial
snapshot replay and an isolated fork match ordinary keep-only payloads exactly,
with increasing cumulative usage. Both new native tests plus the four existing
sparse capture/parameter regression tests pass (six tests, 13.49s).

The F32 native fixture complements neutral tests for duplicate expert slots,
changed route order, strided selections, exact F16/BF16 payload bits, absent experts,
incomplete and duplicate chunks, stale geometry, dtype mismatch, unknown estimates,
failure ordering and budgets before host copies. This is fixture acceptance, not
a claim of released-checkpoint sparse attribution or native distributed support.

Regression results for this change: core 235 pass; runtime 431 pass; architectures
545 pass with one existing ignored fixture; backend conformance 59 pass; portable
facade 19 pass with one existing ignored fixture. The MLX backend checks with no
default features (11.46s); formatting and diff checks pass. A previous sparse
capture serialization test now compares wall-clock duration within floating-point
rounding tolerance while retaining exact payload, receipt and budget comparisons.

### Sparse partition coordinates and local edit recipes

The neutral `RoutedComponentCoordinateMap` keeps selected expert ownership and
within-expert scalar coordinates separate. Both axes support checked ranges and
permutations without expanding an expert-by-unit matrix. Architecture-owned
`derive_routed_component_coordinates` uses the selected `RoutedGroupedPlan` for
expert ownership and retained write layouts for unit storage. This distinction
matters for independently stored banks: a stored remote expert is not an executing
local expert. Packed expert selections and TP columns are validated independently;
packed scalar offsets reuse the existing semantic-unit proof. A bank's independent
matrices must agree on the unit map.

Prepared routed bank construction also joins the original grouped specification
to its localized specification and exact write layouts, retaining each invocation's
expert/unit map. Resident and addressable partition provider factories forward
the retained scalar map through ordinary, compact, TP and expert-exchanged calls.
The localized specification's actual intermediate width must match the derived
unit map before the provider is constructed.

`RoutedUnitBatch::unit_coordinates` and `with_partition_unit_observer` carry that
borrowed map through provider chunks and expert exchange. They preserve source
peer/token/slot identities, original values, replacement values and effective
values. `lower_partition_routed_intervention` lowers each current participating
row using its global expert identity and local unit map; exact payload coordinates
still refer to the original token/global-component rectangle. Source peers remain
distinct even when their token/slot coordinates coincide.

These are coordinate and execution primitives, not distributed session admission.
The recipe does not prove global route coverage or grant quotas, peer authority,
transport or commit rights. Ordinary owners reject partition-tagged batches.
The loaded capture integration described below adds live producer admission,
native coordinate collection and committed receipt delivery for supported native
TP/EP/PP groups. The subsequent sparse operation integration supplies edit authority
and committed outcomes; coordinate lowering alone does not establish either
capability. Remaining native groups still need integration.

Neutral validation covers unequal expert ownership, physical expert permutations,
independent remote weights, packed semantic columns, permuted local units, empty
local selections and invalid ownership/geometry. Every activation action is
compared against the corresponding global edit, including peers sharing a token
and slot. A numerical provider-chain test carries coordinates through chunking
and exchange, modifies the actual consumed values, checks effective callbacks,
rejects mismatched local widths and preserves the disabled-observer path. The
current suites pass: core 235, runtime 432, architectures 548 (one existing
ignored fixture), and reference numerics 188 (47.92s).
The rebuilt native public suite also passes all six routed capture, intervention
and parameter lifecycle tests on CPU/Metal across resident, host-layerwise and
disk-streamed execution (11.53s). This validates ordinary-path compatibility of
the coordinate changes; it does not validate distributed sparse delivery.
Portable facade regression tests pass: backend conformance 59 (0.72s) and portable
facade 19 (6.22s, one existing ignored fixture).
The MLX backend also checks with no default features (18.59s); formatting and
whitespace checks pass.

The subsequent prepared-provider integration has an end-to-end neutral test with
five experts (unequal EP ownership), six units, two layers, top-k two, three prompt
tokens and two cached decode steps. It compares actual original/effective unit
values and independently applied keep-only edits with ordinary routed execution.
Every TP2, PP2, TP2/PP2, EP2, TP2/EP2, PP2/EP2 and TP2/PP2/EP2 combination passes
under resident, host-layerwise and disk-streamed execution (21 combinations).
Unit values agree within `5e-6`; final logits agree within `2e-5`. The fixture checks
every EP source peer and assembles one complete logical capture from the replicated
input, rejecting duplicate or missing global units. That test assembly is separate
from the receipt protocol below; public committed delivery remains unfinished.
The full architecture suite passes 548 tests (one existing ignored fixture), and
the reference numerical suite passes 189 tests (48.02s).
All six existing native packed-GGUF Ring capture/overlay regressions also pass
(29.94s): TP2, TP2/PP2, host-layerwise TP2, disk-streamed PP2, EP2 and TP2/EP2.
These verify native preparation and existing public component/parameter behavior
with the retained maps; they do not enable public sparse distributed capture.
The backend checks with no default features (11.46s), and formatting checks pass.

The runtime receipt protocol now accepts `PartitionRoutedCaptureProducer` through
`PartitionCaptureReceiptPlan::new_routed`. It retains factorized expert/unit maps
and one authoritative EP source peer. Static admission checks complete expert/unit
ownership without expanding an expert-dense activation. Dynamic delivery checks
original token/slot coordinates, expert ownership, selected global units and native
chunk bounds. Assembly merges TP columns only when expert IDs and coefficient
bits agree; missing routes or columns fail instead of becoming measured zeros.
Empty producers still acknowledge explicitly. An empty scalar selection produces
no route rows and still requires every producer's acknowledgment.

Each partition contribution retains `RoutedUnitCaptureProvenance`, including its
native receive-order chunks. The global sparse payload leaves
`source_token_ranges` empty because exchanged receive order is not a global token
range. Host and encoded delivery bounds cover both route assembly and provenance
copies. The ordinary bounded exchange establishes the same native completion and
all-rank delivery verdict as dense captures; receipt arrival alone never commits
an inference step. Loaded sparse capture additionally uses the provider hook facts
and native group checks described in the integration results below.

Neutral receipt fixtures cover five experts with unequal ownership, permuted TP
columns, strided selections, duplicate expert choices in separate slots, empty EP
owners and an inactive pipeline rank. An eight-rank mock exchange checks exact
assembled values and agreed rejection of producer failure and incomplete routes.
These supplement the prepared-provider numerical tests; they do not replace the
remaining native collector, session and controlled-public integration tests.
The current receipt change passes core 235 tests (5.66s), runtime 439 tests
(0.32s), backend conformance 59 tests (0.71s) and portable facade 19 tests
(6.29s, one existing ignored fixture). The MLX backend checks without default
features (16.27s). Native execution was not rerun for this receipt increment.

The following native collector integration adds borrowed
`PartitionRoutedUnitCaptureSource` and `PartitionRoutedUnitCaptureRequest`
mechanism contracts. Runtime providers lend their original peer/token/slot tags
and actual local unit map; MLX checks these against expected ownership before
copying. Selected global columns are gathered in requested order, including
permuted local storage. Only the admitted publication peer is exported. The
estimator prices incoming routes from every source peer and the complete native
chunk history, rather than only the selected logical tokens. Contiguous cold unit
ranges are validated without expanding them.

Native fixtures connect actual gated and ReLU² providers, exchanged origin tags,
TP unit slices, original/effective callbacks and receipt assembly. They use
independent scalar activation calculations and check that downstream values
consume the effective units. These are mechanism tests with local native
execution. The later loaded Ring results below separately verify distributed
admission and controlled delivery. Inspector uses loaded facade capabilities;
these backend mechanisms are not an application capture API.

Sparse fragments now connect to the existing live `CaptureSession` owner. `prepare_partition_routed_capture` shares dense
preparation, identity binding, coordination, global/local quota and receipt
delivery. It charges native collection and retained fragment storage before
allocating accumulators. `begin_partition_routed_capture` pins actual native
rows, dtype, unit order and origin topology. Each chunk must advance exactly
that receive extent; `finish_partition_routed_capture` requires provider success
and complete chunk evidence. An idle EP producer explicitly begins zero rows and
finishes, while a missing invocation fails. Completed records remain provisional
until the ordinary transaction commits; abort and failures preserve spent budgets.

Three new neutral tests cover eight-rank commit/abort with unequal/permuted
ownership, idle owners, an inactive rank, omitted callbacks/chunks, malformed
chunks, provider failure and repeated completion; invalid owner, coordination,
dtype/maps/origins; and budget exhaustion before collection. The full runtime
suite passes 442 tests (0.35s). Native gated/ReLU² fixtures send real provider
chunks through these live session operations and singleton Ring receipt delivery
on CPU and Metal. They check four selected routes against independent scalar
activations at `2e-6` absolute tolerance, exact coefficients, native chunk
provenance, prepaid budgets and commit/abort publication. All 23 native
bounded-capture tests pass (4.38s).

The existing native partition observer's prefill/cached-decode regression also
passes (1.30s). Portable backend conformance passes 59 tests (0.73s), and the
portable facade passes 19 (6.25s, one existing ignored fixture). Formatting and
diff checks pass.

The local provider integration now uses a shared runtime invocation scope.
Prepared resident/host/disk banks enter it around all chunks; EP adapters enter it
before the idle-owner return and finish before reverse exchange. Nested bank
adapters suppress duplicate begin/end callbacks. The MLX adapter lends the same
arrays, unit maps and source origins. Nemotron-H's partition adapter now forwards
its routed ReLU² unit observer through the ordinary and TP paths. Input, ordinary
unit and readout hooks are also implemented across partitioned execution; see
[Nemotron-H partition component execution](#nemotron-h-partition-component-execution)
for native component coverage and remaining shared-expert/MTP acceptance work.

The prepared numerical tests exercise 72 configurations: Qwen gated and Nemotron-H
ReLU², EP2 with TP1/TP2, resident/host/disk, and successful execution or a failure
at active/idle start, active unit observation, or active/idle finish. Equal router
rows force a real idle owner. Success agrees with ordinary inference; failures
preserve the original injecting error source, restore state, prevent commit and
perform no reverse route exchange after the local failure vote. The vote in this
fixture is a bounded neutral test transport, not loaded capture admission. The
existing 21-mode Qwen matrix still verifies original/effective units, interventions,
prefill and two cached decode steps. All six routed numerical tests pass (0.61s),
and the full runtime suite passes 445 tests (0.30s).

The native live-session fixture now drives begin/end through this shared scope.
It supplies exchange tags and singleton Ring delivery. All 23 native
bounded-capture tests pass (4.42s). Three adapter tests verify the original typed
failure, inherited scopes and transparent borrows through both loaded-session
adapters; the composed case passes in 3.30s. Native dense partition prefill/cached
decode also passes (1.46s). The six public routed capture, intervention and
parameter-edit regressions pass across CPU/Metal and all three residencies
(12.27s). These tests retain their existing numerical tolerances.
Portable backend conformance passes 59 tests (0.73s), and the portable facade
passes 19 (6.26s, one existing ignored fixture). The backend checks without default
features (10.85s); formatting and whitespace checks pass.

The shared partition observer now consumes sparse placement through
`PartitionCaptureLayout::routed_capture_placement`. It checks the admitted
invocation and timing, reserves worst-case received input rows for all members,
validates actual rows/width/dtype/maps/origins before source work, and invokes the
existing bounded native source-completion method. Original and effective records
consume separate prepaid source and final votes in plan order. An earlier failure
prevents later source submission. Native source failure fences transport and
requires its existing settlement/teardown policy; otherwise prepaid group votes
settle local rejection. Native chunks
flow into ordinary sparse receipt delivery, and the common final transaction
controls publication.

Three new runtime tests cover nine world ranks, eight source members and seven
producers with unequal expert ownership, permuted unit columns, two idle owners,
a nonexporting replica and an inactive pipeline rank. They check independent
original/effective scalar values, empty selection, commit/abort, source geometry
and provider/chunk failures, source budget exhaustion and changed invocation or
timing. Both source and both final votes complete before provider return. Credits
are identical across ranks and are not refunded on failure or abort. The full
runtime suite passes 448 tests (0.29s).

The CPU/Metal live provider fixture now uses `PartitionCaptureObserver`, actual
native setup identity, prepaid source completion and automatically sized receipt
bounds. Real gated/ReLU² chunks still match independent scalar activations at
`2e-6` absolute tolerance, exact source/route coefficients and full native chunk
coverage. Commit/abort and both equations retain identical prepaid usage. All 23
native bounded-capture tests pass (4.21s), and the native dense partition observer
prefill/cached-decode regression passes (1.44s). Portable backend conformance
passes 59 tests (0.71s), and the portable facade passes 19 (6.23s, one existing
ignored fixture). The backend checks without default features (10.49s), and
formatting and whitespace checks pass. This fixture supplies sparse placement
and exchanged origin tags; it does not establish loaded multiprocess EP coverage.

Retained architecture layouts now supply exact sparse bank placement from the
same expert ownership policy used by construction, the declared logical invocation
and the local physical write-column layout. No artifact is reopened and no native
resource is created by this projection. One retained rank's projection agrees with
every independently prepared rank, including uneven expert partitions and PP cuts.
Malformed input widths, expert counts, route cardinality and invocation ownership
reject during cold placement.

The actual prepared Qwen matrix now drives the shared `PartitionCaptureObserver`
through prefill and two cached decodes in all 21 TP/EP/PP/residency combinations.
Delivered original/effective records match ordinary scalar values within `5e-6`
absolute tolerance; logits retain the existing `2e-5` tolerance. Gated and ReLU²
EP2/TP1–2 fixtures also drive the shared observer through 48 configurations across
all three residencies: success, active-source failure, idle-source failure and
collector failure. Aborted runs preserve model positions and publish no payload
or partition evidence; failure timing does not refund prepaid credits. Injected
local causes survive the error chain. The eager test transport wakes pending
waiters on teardown, so source failures cannot pass merely by timing out.
The existing 72 provider-boundary failure configurations remain covered. All six
routed numerical tests pass in 1.08s, including the stricter `2e-5` shared-observer
logit comparison. The broader run passes all 548 architecture
unit tests (one existing ignored test) and all 190 numerical tests in 47.78s.
The rebuilt native binary passes all 23 CPU/Metal bounded-capture tests (4.21s)
and the dense partition observer (1.83s). Portable backend conformance passes
59 tests (0.72s), and the portable facade passes 19 (6.24s, one existing ignored
fixture). Native results are regressions of the existing collectors and do not
establish newly loaded distributed sparse support. The backend also checks
without default features (9.55s); formatting and whitespace checks pass.

These are neutral prepared-executor tests. Current balanced construction rejects
EP topologies with more ranks than experts; idle received owners in this matrix
own experts but receive no routes. Remaining native TP/EP/PP invocation groups and
public controlled capture/intervention coverage remain unfinished. Those neutral results alone did not broaden public distributed sparse capability
facts; the subsequent loaded integration below establishes the selected native paths.

Loaded MLX now uses the retained placement in public discovery and capture.
The runtime distinguishes `ObservationHookSite::RoutedUnits` from complete internal
layer hooks. Provider-aware ordinary/TP calls and pipeline strategies declare
sparse coverage independently, so a sparse-capable hybrid path does not accidentally
advertise all internal components. Discovery additionally requires native routed
collection and exact selected invocation-group status transport. A group needing
world participation waves remains explicitly unverified.

Use the same `CaptureTransform::RoutedUnits` selections with the loaded discovery.
Every rank admits the same global plan and runs the ordinary or controlled text
driver. Committed rows contain the global expert, token and route coordinates,
coefficient, and selected unit values. EP replicas publish the retained source peer
zero; producer evidence preserves each rank's actual receive chunks. The result is
not an expert-dense tensor. Original/effective timing remains unchanged, and source,
transport and receipt credits remain nonrefundable through replay and branches.

The loaded packed-Qwen GGUF Ring tests pass all six configurations: TP2, EP2,
TP2/EP2, TP2/PP2, host TP2 and disk PP2. They compare full and odd-stride original
and effective sparse values and logits with a separately loaded ordinary native
session through prefill and two cached decodes. Scalar/logit tolerance is
`2e-4 + 2e-4 * abs(reference)`; route coefficients use `2e-5` absolute tolerance,
while route identities and selected geometry must match exactly. The tests also
exercise snapshots, replay, sibling isolation and skipped child captures through
the public controlled backend contract. These are synthetic native fixtures, not
released-checkpoint attribution evidence. Sparse edit support additionally requires
the native coordinate/gather/scatter mechanisms and shared operation authority
described below.

The broader native run passes 30 CPU Ring tests in the Metal-enabled build
(143.88s): 18 sparse configurations across packed Qwen MoE GGUF, Qwen MoE
SafeTensors, GPT-OSS and LFM2 MoE, plus 12 dense regressions. Qwen SafeTensors,
GPT-OSS and LFM2 MoE each cover resident TP2, TP2/PP2, host TP2 and disk PP2.
All 449 runtime tests pass (0.24s), along with 548 architecture unit tests
(0.37s, one existing ignored fixture) and 190 numerical tests (50.70s).
The portable facade passes 19 tests (6.23s, one existing ignored fixture), backend
conformance passes 59 (0.71s), and the MLX backend checks without default features
(4.75s). Formatting and whitespace checks pass.

Partitioned sparse edits now use the existing admitted operation owner, common
pre-forward agreement and committed outcome frame. Retained architecture placement
supplies exact expert/unit ownership; runtime validates complete coverage and
reserves all members' work, coordinate copies, completion, two invocation votes
and outcome transport. The native adapter maps received rows back to their original
source peer, token and route slot. Every source peer and replica receives edits;
only retained publication ownership contributes to logical affected counts.
Contiguous native chunk coverage and all members' final agreement are required
before reverse exchange. Outcomes become `Applied` or `Unmatched` only after the
ordinary distributed forward commits. Separate original/effective sparse captures
retain their established timing, and sparse operations also work without them.

The edit matrix passes all 18 CPU Ring configurations in the Metal-enabled build:
six packed-Qwen GGUF cases (38.60s) and twelve Qwen SafeTensors, GPT-OSS and LFM2
MoE cases (66.24s). Each case exercises all-keep, component deletion, keep-only,
ordered Scale/Add, Replace, Zero and boolean masks at prefill token row 1 and
decode row 0 through two cached decodes. Global values, logits and affected counts
match a separately loaded ordinary native session using the capture tolerances
above. All-keep matches baseline; every other edit changes the tested prefill
logits. Logits-only child capture proves operation-only sparse hooks work through
future re-admission, replay and sibling isolation without refunding credits.

Three neutral session tests cover nine ranks with permuted TP units, multiple EP
source peers, replicas, idle/empty owners and an inactive pipeline rank. They check
ordered edits and exact counts, failure/abort accounting, invalid or incomplete
ownership, missing/repeated chunks, native edit errors, exhausted budgets and
foreign/stale authority. Source/final votes settle before provider return, and
failed work never publishes a successful operation. All 237 core and 452 runtime
tests pass (5.68s and 0.29s). The twelve dense Ring regressions also pass (50.50s),
along with 59 portable backend conformance tests (0.71s) and 19 portable facade
tests (6.48s, one existing ignored fixture). The MLX backend checks without default
features (12.31s); formatting and whitespace checks pass.
These fixtures do not replace released-checkpoint
sparse validation. Subsequent stage-group transport validation follows below.

Prepared parallel execution now selects an exact failure-agreement group containing
every tensor/expert rank at each pipeline stage. `TopologyCommunicationPlan` projects
these groups after existing axis groups, preserving their opaque IDs. The loaded
observation layout supplies invocation membership, and MLX checks that the selected
group can complete status agreement independently. For the retained pipeline-major
rank order, stage members form a connected Ring segment. A native neighbor reduction
and broadcast carries status entirely within that segment; inactive stages continue
their ordinary execution protocol. Disconnected memberships still require a relay
mechanism and do not acquire support from this fact.

Both intermediate sends and final status use the existing bounded completion owner.
They retain native arrays, groups and streams, fence live work on timeout, and release
it only after established completion. This creates no component tensors when
instrumentation is disabled. Inspector uses the same capture and intervention plans
and the loaded discovery result; it does not configure native groups.

The member-status fixture verifies three-rank, wraparound, two-rank and singleton
groups in a four-rank native Ring. Nonmembers submit no native collectives; first-
and last-member failures reach every participant. A separate timeout fixture verifies
retained work prevents group reuse until completion. Neutral membership tests compare
every nonempty subset of worlds of size 1–10 with independent graph connectivity,
and stage-projection tests verify tensor/expert membership and preserved axis IDs.
The loaded packed-Qwen resident EP2/PP2 and TP2/EP2/PP2 workflows pass, including
component capture, sparse edits and controlled replay. The latter plus the timeout
fixture pass in 21.92s. The eight-rank component fixture reserves 4 GiB of host work per
step and 32 GiB cumulatively, covering conservative producer-by-receiver decoder
credits; these allowances do not allocate that amount of physical memory.
All 453 runtime and 548 architecture tests pass (0.27s and 0.37s, one existing
architecture test ignored).

The expanded CPU Ring matrix in the Metal-enabled build verifies 42 component
configurations. Twelve additions cover packed Qwen GGUF EP2/PP2 and TP2/EP2/PP2
in all three residencies, plus both resident combinations for Qwen SafeTensors,
GPT-OSS and LFM2 MoE. The initial run passed 41 cases in 300.48s; GPT-OSS's
eight-rank case rejected the earlier 1 GiB host-work allowance before execution.
After increasing the shared fixture allowance above, five focused cases pass in
70.06s: GPT-OSS TP2/EP2/PP2, packed Qwen TP2/EP2/PP2 in all three residencies,
and intermediate-timeout retention. Production resource charging is unchanged.
All thirty preceding configurations passed the expanded run.

Six routed numerical tests also pass (1.09s), including all 21 prepared
topology/residency combinations and gated/ReLU-squared provider failure checks.
Portable backend conformance and facade tests pass 59/19 cases (0.77s/7.27s,
one existing facade fixture ignored). The backend checks without default features
(5.92s); formatting and whitespace checks pass. These are synthetic CPU Ring
results; distributed GPU and released-checkpoint sparse validation remain open.

Reproduce the invocation integration with:

```sh
cargo test -p eredu-runtime --lib
cargo test -p eredu-backend-mlx --lib --features metal group::status:: -- --include-ignored --test-threads=1
cargo test -p eredu-backend-mlx --lib --features metal ring_member_status -- --include-ignored --test-threads=1
cargo test -p eredu-backend-mlx --lib --features metal ring_public_component_capture_ -- --include-ignored --test-threads=1
cargo test -p eredu-architectures --test reference_numeric routed_units::
cargo test -p eredu-backend-mlx --lib --features metal composition::mlx::session::bounded_capture:: -- --include-ignored --test-threads=1
cargo test -p eredu-backend-mlx --lib --features metal composition::neutral_observer_tests:: -- --include-ignored --test-threads=1
cargo test -p eredu-backend-mlx --lib --features metal composed_array_and_neutral_adapters_preserve_routed_invocation_scope -- --include-ignored --test-threads=1
cargo test -p eredu-backend-mlx --lib --features metal native_partition_observer_commits_real_component_prefill_and_cached_decode -- --include-ignored --test-threads=1
cargo test -p eredu --test native_execution_control --features mlx,metal routed_components:: -- --include-ignored --test-threads=1
```

The preceding collector increment passed 21 native bounded-capture tests, including its five new CPU,
Metal and estimator tests (3.68s). Six ordinary public routed capture,
intervention and parameter regressions pass on CPU/Metal across resident,
host-layerwise and disk-streamed execution (11.62s). Core passes 237 tests
(5.63s), runtime passes 439 (0.32s), and the five routed numerical tests pass
(0.59s), including all 21 prepared TP/EP/PP residency combinations. The backend
checks without default features (16.35s). The native provider fixtures use `2e-6`
absolute tolerance for scalar activations and doubled downstream outputs; the
permuted-copy fixture checks exact representable F32/F16/BF16 values and exact
original route coefficients.
Portable facade regression checks also pass: 59 backend conformance tests
(0.73s) and 19 portable facade tests (6.24s, one existing ignored fixture).
Formatting and whitespace checks pass.

```sh
cargo test -j 1 -p eredu-core -p eredu-architectures --lib
cargo test -j 1 -p eredu-runtime --lib
cargo test -p eredu-runtime --lib capture::tests::partition::receipts
cargo test -p eredu-backend-mlx --features metal --lib bounded_capture:: -- --include-ignored --test-threads=1
cargo test -j 1 -p eredu-architectures --test reference_numeric
cargo test -j 1 -p eredu-architectures --test reference_numeric routed_units::partition::
cargo test -j 1 -p eredu-backend-mlx --features metal --lib ring_public_component_capture_qwen_moe_packed_gguf_ -- --ignored --test-threads=1
cargo test -p eredu-runtime --lib intervention::tests::sparse
cargo test -p eredu --no-default-features --features mlx,metal --test native_execution_control routed_components:: -- --ignored --test-threads=1
```

## K2 Horizon component propagation

K2 dense prefixes, routed SwiGLU units, shared SwiGLU units and MoVA attention
channels now execute through the ordinary decoder component driver. Direct and
provider observation entry points share the same value-routing and FFN code.
Routing outputs retain their declared batch/sequence/hidden geometry; expert IDs,
scores and coefficients retain token-row/selected-expert geometry. Shared masks
modify the actual values consumed by the down projection and subsequent sum.

The neutral discovery test joins every ordinary read and routed value row to
actual loaded parameters, checks serialized descriptors and rejects foreign or
out-of-range component/expert identities and malformed loaded geometry. It
captures every declared path, reconstructs component writes, and changes shared
units and attention channels across prefill and two cached decode steps. The
prepared-provider test covers both K2 MoE and MoVA across all seven TP/EP/PP
combinations and resident, host-layerwise and disk-streamed execution: 42
configurations with original/effective sparse units, global masks, logits,
retained ownership, idle invocations and completed delivery.

The focused K2 suite passes 13 tests (5.24s), including existing independent
oracle, cached execution, routing, speculation and residency/parallel regressions.
After adding complete-write placement for all declared scalar groups, the full
neutral numerical suite passes 192 tests (47.96s). The shared partition capture
fixture now verifies original/effective affine and normalized writes alongside
unit/channel values.

```sh
cargo test -j 1 -p eredu-architectures --test reference_numeric k2_
cargo test -j 1 -p eredu-architectures --test reference_numeric
cargo test -j 1 -p eredu-backend-mlx --features metal --lib ring_public_component_capture_k2_ -- --ignored --test-threads=1
```

All twelve native K2 CPU Ring cases pass in the Metal-enabled build (105.97s):
four dense cases cover TP2, TP2/PP2, host-layerwise TP2 and disk-streamed PP2;
eight MoVA cases add EP2, TP2/EP2, EP2/PP2 and eight-rank TP2/EP2/PP2. Loaded
capability checks require shared input/unit/write/output and complete routed
contribution support. Dense and shared scalar masks include selected prefill and
decode positions, deletion, keep-only, all-keep and ordered scaling. Sparse unit
checks exercise original/effective full and strided payloads and interventions.
Captures use the public neutral API and match an ordinary native reference within
`2e-4 + 2e-4 * abs(reference)`; sampled tokens, restored payloads and sibling
replays match exactly. Existing typed skipping and nonrefunded budget checks are
retained.

Public parameter tests query the actual expert banks and apply a coordinated
edit to dense Q/K/V/O and gate/up/down parameters, MoVA Q/K/value-bank/O
parameters, shared gate/up/down projections and routed expert banks. The complete
edit agrees with the same edit in a separately loaded ordinary execution through
cached predictions. Peer failure rolls back weights and caches; activation/removal,
controlled capture, snapshot/sibling isolation and exact baseline restoration
pass. These synthetic CPU Ring cases establish the API path; they do not replace
released sparse-checkpoint or distributed GPU validation.

Current core/runtime/architecture library checks pass 237/453/548 tests (one
existing architecture test remains ignored).

Portable facade/conformance tests pass 19/59 cases (7.24s/0.75s, one existing
facade fixture ignored). The final focused K2 numerical run, using declared write
paths directly, passes all 13 tests (5.91s).

The extended K2 discovery test also reconstructs the full residual from captured
embedding, attention and feed-forward contributions, including the shared term
only through the complete sparse contribution. It projects those signed terms
through independently computed, per-trial grouped RMSNorm directions for two
selected tokens. Scores and their difference agree within `2e-5` across dense,
MoE and MoVA, direct/provider entry points, prefill, cached decode and shared-unit/
attention-channel masks (1 test, 0.09s). Backend compilation without default
features also passes (18.90s).

The existing 42 native CPU Ring component configurations also pass on the same
production source (318.08s), after expanding capture to complete affine and
normalized writes. Together with the twelve K2 cases, all 54 configurations are
verified. This retains Qwen dense/routed, packed Q8_0/IQ4_NL, GPT-OSS, Nanbeige and
dense/routed LFM2 coverage, including controlled capture/interventions and public
parameter lifecycle checks. Formatting and whitespace checks pass. Reproduce the
complete native regression with:

```sh
cargo test -j 1 -p eredu-backend-mlx --features metal --lib ring_public_component_capture_ -- --ignored --test-threads=1
```

## Routed provider failures and forward outcomes

Routed provider errors retain their original owned cause through resident,
addressable-bank, grouped-operation, partition-provider and family adapters.
Applications receive neutral `BackendFailure` errors. Source inspection can recover
original native diagnostics without exposing native types in application operation
signatures. A diagnostic alone never proves completion, rollback or peer termination.

Provider failure agreement is independent of component observation:

- TP provider calls agree before the next model collective.
- Expert exchange agrees after received-route validation, local bank execution and
  output validation, before reverse exchange. Idle owners participate too. TP votes
  precede EP votes, and a TP rejection still reaches the EP vote.
- Shared pipeline waves additionally agree on the session group retained from the
  prepared execution plan. Inactive stages participate before advancing to another
  collective. Stage-local votes alone cannot certify a shared wave.
- Pipeline execution agrees before boundary preparation. A completed negative
  execution vote records that all participants reached that boundary, allowing
  rollback without inventing a failed boundary transfer. Real submission,
  completion and transport failures retain terminal fencing and native ownership.

The origin keeps its first local cause even if agreement also fails. Healthy peers
receive typed provider or distributed-phase rejection, including through
`ReplicatedTextSessionError::Partition`. No later reverse exchange or model
collective is submitted after an agreed provider rejection.

Check `CapturedStep.outcome` before using a batch as evidence for a committed
prediction. `Committed` describes the model forward transaction; sampling can
still fail afterward. `Aborted` preserves any actual observations that completed
earlier in the unsuccessful forward. `Untracked` means a low-level caller or older
serialized record provided no transaction evidence. Individual observations retain
their original meaning. Aborted work remains charged, draining it does not permit
a new snapshot, and restoring an earlier snapshot refunds no work. Partition
provenance is published only after commit.

The native fixture injects a real MLX invalid-reshape exception at one grouped
mechanism entry. It checks the exact original message, creation file and line.
Qwen MoE exercises gated banks, Nemotron-H exercises ReLU² banks, and K2 MoVA
exercises linear banks. Ordinary execution has no observer; controlled execution
captures embeddings without installing a provider-unit observer. Each case fails
in prefill and after one successful prediction, then synchronizes, resets and
replays three tokens against its healthy baseline.

All 63 unique CPU Ring configurations pass: three bank families, three ordinary
weight residencies (resident, host-layerwise and disk-streamed), and seven
combinations (TP, EP, PP, TP/EP, TP/PP, EP/PP and TP/EP/PP). These cover 252
asymmetric failure scenarios, including inactive pipeline stages and eight-rank
execution. Twelve test entries, including three focused regressions, pass in
320.89s. The separate local CPU/Metal matrix passes all 96 scenarios in 5.02s,
including resident banks and independent-bank residency variants. Controlled cases
verify aborted outcomes; the local matrix additionally checks charged capture usage.

All 54 existing CPU Ring component configurations also pass on the same production
source (406.36s). They retain ordinary/controlled capture, masks, parameter queries,
coordinated overlays, cached replay and restoration coverage, including eight-rank
packed-GGUF execution.

The distributed fixture permits 100 captures, 512 MiB retained work, 4 GiB host
work and 512 MiB encoded work per step, with four times those cumulative credits.
These allowances include conservative producer/receiver transport envelopes; they
are work budgets, not physical-memory allocations or allocator guarantees.

```sh
cargo test -j 1 -p eredu-backend-mlx --features metal --lib \
  ring_provider_failure_ -- --ignored --test-threads=1
cargo test -j 1 -p eredu-backend-mlx --features metal --lib \
  provider_failures_retain_native_causes -- --ignored --test-threads=1
cargo test -j 1 -p eredu-backend-mlx --features metal --lib \
  ring_public_component_capture_ -- --ignored --test-threads=1
cargo test -j 1 -p eredu-runtime -p eredu-architectures --lib
cargo test -j 1 -p eredu-architectures --test reference_numeric
cargo test -j 1 -p eredu-runtime --test backend_independence
cargo test -j 1 -p eredu --no-default-features \
  --test backend_conformance --test portable_facade
cargo check -j 1 -p eredu-backend-mlx --no-default-features
```

Provider-agreement verification passed 455 runtime, 549 architecture, 193 numerical
and 70 backend-independence tests. One existing architecture test remains ignored.
Numerical tests finish in 47.62s. Tests verify idle participation, received-metadata
rejection, stopped reverse exchange, cause precedence, original transport fencing,
state rollback and exact replica deduplication. Portable backend-conformance and
facade suites pass 59 and 19 tests respectively, with one existing facade fixture
ignored. Backend compilation without default features, formatting and whitespace
checks pass.

Nemotron-H partition execution now reuses its ordinary block and readout hooks
across local, TP and pipeline paths. Effective embeddings remain in the typed
boundary payload. Discovery joins Mamba parameters using their canonical namespace
while preserving logical mixer identities. The fault matrix verifies native
embedding capture. The subsequent component matrix below verifies actual unit,
channel and whole-write values, interventions and parameter lifecycle behavior.

These injections are synchronous operator-entry failures. Earlier packing or
selection failures, asynchronous native completion failures, and cold preparation
or submission-policy source adaptation remain part of the wider failure audit.
Synthetic fault fixtures do not replace pinned released-checkpoint validation.

## Nemotron-H partition component execution

Nemotron-H's loaded target execution declares and emits ordinary unit hooks as well
as routed-provider hooks. The TP block driver now applies normalized-input,
attention-channel, dense ReLU²-unit, affine-write and residual interventions around
the same native operators used by ordinary execution. Readout hooks retain the
actual embedding, final residual, normalization and vocabulary projection boundaries.
Whole Mamba and routed-plus-shared writes retain their own original/effective timing.

A prepared numerical comparison exposed an existing Mamba TP placement error:
convolution channels combine value, input-state and output-state segments, so each
rank must select its group interval inside all three segments. The neutral parameter
plan now declares that placement for weights and bias. Materialization, effective
queries and overlays consume the same mapping. The scalar reference binder also now
implements the published `ln(-x)` transition recipe with negative-source validation.

The new neutral test covers eleven unmodified/deletion/keep-only/readout trials
through prefill and two cached decodes for all seven TP/EP/PP combinations and three
residencies. It compares actual component values, invocation ownership and downstream
predictions against ordinary execution with independently adapted source parameters.
The complete numerical suite passes 194 tests (47.90s); the architecture library
passes 549 tests (0.37s), with one existing ignored test. Backend compilation without
default features passes (3.53s).

Native fixtures use the published SafeTensors and GGUF schemas: `M-E*` includes a
dense ReLU² block, and `MEE*` includes two routed ReLU² blocks. Both include Mamba
and attention. Their parameters use deterministic name/coordinate mixing (FNV-1a
name seed 2166136261 followed by integer avalanche mixing). Embeddings and up reads
keep the tested ReLU² units active; distinct vocabulary rows avoid artificial exact
greedy ties. The earlier constant fixtures remain available for their original tests.

All 42 native CPU Ring configurations pass (six matrix entries, 346.87s): both
source formats, three residencies and all seven TP/EP/PP combinations, including
eight-rank execution. The public contract checks full and strided original/effective values, normalized
inputs, complete residual writes, dense-unit/channel and sparse-unit edits at
selected prefill/decode positions, controlled replay, sibling isolation, skipped
capture, parameter queries and signed contractions. Coordinated parameter edits
include Mamba input/convolution/output, dense or shared ReLU² projections, both
routed expert matrices, attention Q/K/V/O, normalization and readout parameters.
Budget/stale-identity rejection and a peer publication failure preserve rollback;
activation, removal and restoration are compared with a separately loaded ordinary
native session.

The ReLU² provider-failure regression also passes all 84 asymmetric scenarios
(three test entries, 96.10s): three residencies, seven TP/EP/PP combinations, and
ordinary/controlled prefill/decode failures. It preserves the original native cause,
requires typed peer rejection, verifies aborted captures, and synchronizes, resets
and replays against the healthy baseline after each fault.

Tensor captures use `2e-4 + 2e-4 * abs(reference)`; activation intervention evidence
uses `3e-4 + 3e-4 * abs(reference)`. Parameter queries compare exactly, and edited
prediction arrays use the family's existing `1e-3` absolute tolerance. Sampled
tokens, restored captures and sibling replay compare exactly. The shared fixture's
capture work allowance is 512 MiB retained, 4 GiB host and 512 MiB encoded per step,
with eight times those cumulative credits. These are conservative logical work
credits, not allocator guarantees.

```sh
cargo test -j 1 -p eredu-architectures --test reference_numeric
cargo test -j 1 -p eredu-architectures --lib
cargo test -j 1 -p eredu-backend-mlx --features metal --lib \
  ring_public_component_capture_nemotron_ -- --ignored --test-threads=1
cargo test -j 1 -p eredu-backend-mlx --features metal --lib \
  ring_provider_failure_relu2_ -- --ignored --test-threads=1
cargo check -j 1 -p eredu-backend-mlx --no-default-features
```

This initial coverage retained shared experts inside the complete sparse contribution;
the scalar extension below adds their individual units. MTP observation propagation,
released sparse-checkpoint validation and distributed GPU runs remain separate work.
The synthetic fixtures establish execution and API behavior, not editing efficacy.

### Shared-expert scalar extension

Nemotron-H target discovery now exposes `decoder.layers.N.operator.shared.units`.
The group joins the sparse operator's shared child and canonical parameter group,
with direct up-projection rows, down-projection columns, optional read/write biases,
ReLU² activation and the enclosing block's RMSNorm gain and epsilon. Consumers
should follow these descriptor fields to the public parameter-query and overlay
APIs rather than infer checkpoint names. The scalar width is the declared shared
intermediate width; tensor partition coordinates come from the actual down weight.

Its observation scope is `model.layers.N.shared.feed_forward`: `input`, `units`,
`write` and `output` each retain original and effective values. `write_input` uses
the existing effective projection-input contract. A shared-input edit changes only
the shared branch after routing input preparation. Unit edits modify the actual
ReLU² values consumed by the down projection. Write and output edits occur after
any TP reduction and before addition to the routed contribution. Ordinary,
controlled and partitioned execution reuse these same hooks. Disabled observation
does not construct these scopes or clone extra component tensors.

The shared scalar contributions reconstruct the shared affine write, including its
declared bias. That write is already inside the complete sparse term in
`ComponentReadout::other_writes`; adding both to the residual would count it twice.
Use the scalar decomposition to replace the shared subterm inside that sparse sum.
This follows the same constituent-write convention as K2's shared SwiGLU groups.

The neutral reconstruction fixture uses distinguishable positive up rows, signed
down columns and nonzero biases. It checks the discovered parameter identities,
F64 read/ReLU² and write-sum reconstruction (`2e-6` read and `2e-4` write tolerances), exact no-op
execution, position-local deletion, and keep-only survivors recomputed after a
shared-input intervention. The prepared mixed-model test now runs fifteen trials
across all 21 topology/residency configurations, including shared-input/output
edits, shared-unit deletion/keep-only and a combined upstream-write/attention/
shared/dense keep-only trial through prefill and two cached decodes.

All 42 native CPU Ring configurations pass again with these required shared hooks
(six matrix entries, 359.47s). SafeTensors requires the sparse layer's shared input,
units, write and output; GGUF requires both sparse layers. Every original/effective
point must have loaded support, and an explicit test assertion requires every
requested intervenable scalar point to participate in selected-position
deletion/keep-only plus scaling trials. This includes seven-wide shared FFNs split
across two TP ranks. The existing native workflow also checks full captures, bounded
previews and score reductions, controlled snapshots and sibling replay, parameter
queries/contractions, coordinated overlays, publication rollback and restoration.
The same tolerances, deterministic fixtures and reproduction commands above apply.

The complete numerical suite passes 195 tests (47.61s), and the architecture library
passes 549 tests (0.37s), with one existing ignored test. Backend compilation without
default features passes (2.77s); formatting and whitespace checks pass. The preceding
84 provider failure scenarios remain separate evidence; they were not rerun for this scalar
extension. No additional native family dispatch or parameter mechanism was needed.
MTP and broader compatible-family propagation, remaining failure/accounting audits
and released sparse-checkpoint validation remain on the full implementation
checklist. The following section records grouped FP8 parallel acceptance and its
remaining limits.

### Grouped FP8 parallel parameter lifecycle

K2's routed and shared FFNs can use different weight encodings and therefore have
independent tensor-parallel boundaries. The new native fixture exposed a construction
error for 384 routed FP8 units alongside 384 dense shared units: retained placement
assigned 256/128 routed units and 192/192 shared units, but construction reused the
routed width for the shared branch. Binding then rejected the shared down weight.
The architecture now retains the shared width from its own local down-projection
layout. No backend family branch or alternate execution driver was added.

The neutral regression constructs both mixed-encoding combinations, with one and
two shared experts, from real retained TP layouts. It checks local shared geometry
and reconstructs nonzero shared writes against ordinary execution. Native fixtures
use three K2 layers, including dense, shared SwiGLU, routed SwiGLU and MoVA values.
Three feed-forward experts participate in every route. Only the routed FFN banks
use E4M3 with F32 128-by-128 scale blocks. The two geometries are hidden width 128
with 256 expert units, and hidden width 130 with 384 expert units. The latter covers
partial hidden-axis blocks and unequal complete-block TP allocations.

Public queries must exactly match independent scalar E4M3/scale equations at both
ends of the expert axis, gate/value row boundaries and final output rows. Bounded
signed projections use F64 references and `2e-6 + 2e-6 * abs(reference)` tolerance.
Coordinated edits include attention, dense/shared projections, MoVA values and both
matrices in the first routed layer. A separately edited SafeTensors checkpoint promotes
all experts of the affected matrices to F32, applies logical edits directly to
individual source rows/columns, and leaves the second routed layer encoded. Its
prefill and two cached decodes must match active partition execution within `1e-4`.
Discovery must report dense input arithmetic for promoted matrices and FP8 input
arithmetic for the untouched layer. The original checkpoint bytes must remain
unchanged. Existing controlled capture, masks, replay, sibling isolation, budget
rejection, peer publication rollback and exact restoration checks also run.

Fixture bytes use a wrapping base-31 hash of the source parameter name, distinct
per-coordinate E4M3 codes and block multipliers `(1 + (phase + 3 * block) % 5) / 128`.
There is no fixture RNG; generation uses greedy sampling (`temperature = 0`).
Native tensor and intervention comparisons retain the common Ring tolerances
`2e-4 + 2e-4 * abs(reference)` and `3e-4 + 3e-4 * abs(reference)` respectively.
The tests use the existing conservative logical capture and parameter work limits;
they do not establish physical allocator bounds.

The matrix crosses both geometries with resident, host-layerwise and disk-streamed
ordinary weight policies and seven topologies: PP2, TP2, EP2, TP2/EP2, TP2/PP2,
PP2/EP2 and TP2/PP2/EP2. These are native CPU Ring processes in a Metal-enabled
build; they do not establish distributed GPU results.

All 42 configurations pass (three matrix entries, 606.83s). The complete numerical
suite passes 196 tests (48.52s), including the four mixed-encoding/shared-count
regression cases. The architecture library passes 549 tests (0.37s), with one
existing ignored test. Backend compilation without default features passes
(3.63s); formatting and whitespace checks pass.

```sh
cargo test -j 1 -p eredu-architectures --test reference_numeric
cargo test -j 1 -p eredu-architectures --lib
cargo test -j 1 -p eredu-backend-mlx --features metal --lib \
  ring_public_component_capture_k2_fp8_ -- --ignored --test-threads=1
cargo check -j 1 -p eredu-backend-mlx --no-default-features
```

These fixtures do not resolve arbitrary partial blocks on a partitioned FP8 axis.
The 130-wide hidden axis is replicated within each expert; the 384-wide sharded
axis contains three complete blocks. These earlier tests use ordinary bank
residency; the explicit expert-cache matrix is recorded below. General partial-tail
sharding, FP8 scale encodings beyond the fixture's F32 scales, wider-family
propagation and released sparse-checkpoint validation remain separately tracked work.

### Independent-bank parameter ownership and editing

Independent expert-cache selection previously omitted bank weights and companions
from loaded parameter discovery. The K2 Ring reproduction returned 39 parameters
where ordinary execution exposed 49. Ordinary module edits still worked, but this
was an implementation gap for bank reads and edits.

Runtime now joins architecture parameter declarations to exact lowered member
bindings. Bank-owned slots retain their global member IDs, rank-local geometry,
sharing metadata and companion relationships. A selected query copies members in
order through the existing cache manager and completion-safe transfer loans, then
assembles only its requested parameter and companions. It does not reopen the
checkpoint or require every member to remain resident together. Additional native
copy and host metadata costs are reserved before this work. The existing effective
parameter decoding and signed projection mechanisms consume these values.

Publication validates every involved bank pool before changing ordinary handles
or bank overrides. Providers substitute the immutable overrides when constructing
compact groups. Original disk/cache entries remain unchanged, and removal restores
them. Promoting a packed matrix updates member byte costs seen by the neutral chunk
planner and the native compact-scratch check. Retained overlay storage belongs to
the parameter budget; cache and compact-scratch limits continue to apply during
inference. An insufficient compact-scratch allowance produces a typed execution
failure before compact-bank construction.

Compact binding validates floating replacements against the operator's declared
logical dimensions. Packed source weights retain their encoded dimensions and
dtype; companion and ordinary bias shapes remain exact. This matters for MXFP4,
affine and GGUF weights, whose packed column counts differ from floating weights.
All bindings validate before any compact parameter slot changes. The same rule is
used by gated, ReLU² and selected-linear mechanisms.

The cache fixtures reuse the independent decoders and independently edited
checkpoints described above. Ring tests use two MiB of compact scratch, no host
cache, and a device cache sized to one rank-local routed FFN bank. Competing MoVA
banks force eviction even when a pipeline rank owns only one routed layer. The
public facade fixture uses one MiB of compact scratch and routes through either
one or three experts, with room for exactly that many encoded experts in the
device cache. Its one-expert variant queries all three members, exercising
sequential loans beyond the cache's capacity. Its two routed layers also compete
for the cache. Both retain effective queries, independent edited references, budgets,
provenance, controlled replay, cached predictions and restoration. The Ring workflow additionally checks
component captures, masks and peer rollback. A repeated query/forward check loads
every MoVA owner, then requires new FFN cache misses and evictions while comparing
prefill and two cached decodes with ordinary execution. An edited layer must also
report the larger FP32 compact-bank size.

Reproduction commands:

```sh
cargo test -j 1 -p eredu-runtime --lib parameter_operations::banks
cargo test -j 1 -p eredu-runtime --lib
cargo test -j 1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  ring_public_component_capture_k2_fp8_cached_ -- --ignored --test-threads=1
cargo test -j 1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  ring_public_component_capture_cached_banks_ -- --ignored --test-threads=1
cargo test -j 1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  compact_packed_linear_accepts_logical_float_weights_and_rejects_partial_updates \
  -- --ignored --test-threads=1
cargo test -j 1 -p eredu --no-default-features --features mlx,metal \
  --test native_execution_control native_grouped_fp8_cached_parameter_lifecycle \
  -- --ignored --test-threads=1
```

The four neutral ownership/accounting tests and all 459 runtime library tests pass
(0.30s for the full suite), including repeated shared declarations without duplicate
storage. All 24 facade configurations pass on CPU and Metal (25.46s after the
packed-binding fix): two hidden geometries, one/three selected experts and all
three ordinary residencies on each device. All 42 explicitly evicting K2 Ring configurations pass (737.64s),
including both geometries, all seven TP/EP/PP combinations and all three ordinary
residencies. These runs include the additional copy reservations, FP32 compact
accounting, independently edited checkpoints and repeated query/reload checks.

The additional 84 cached-bank configurations pass (745.65s): MXFP4 GPT-OSS,
grouped GGUF Qwen, and SafeTensors/GGUF Nemotron, each across the same seven
TP/EP/PP combinations and three ordinary residencies. They use default independent
bank budgets and compare global effective reads, signed projections, coordinated
edits, cached predictions, peer rollback, component capture/masks and controlled
replay with ordinary native owners. This covers packed column counts, ordinary
projection biases and ReLU² banks. The 42 FP8 configurations above separately
enforce cache eviction and compare with independently edited checkpoints.

The promoted selected-linear regression passes for F32, F16 and BF16 (3.28s),
including invalid floating geometry and a valid changed weight paired with an
invalid companion. Both rejections preserve the previous scalar-reference output.

The 17 native bank regressions pass (2.54s), including preflight rejection,
companion geometry, rank-local lowering and eviction. Runtime backend-independence
passes 70 tests, portable backend conformance passes 59, and the portable facade
passes 19 with one existing ignored test. The backend check without default
features passes (0.89s after the final typed-error preservation check). These results do not complete the broader family, encoding,
released sparse-checkpoint or final failure/accounting audits in the checklist.

### Partial FP8 blocks on the dense FFN partition axis

Dense decoder placement now retains complete FP8 blocks and a short final block
as shared logical chunks. The 259-unit fixture assigns units `0..256` and
`256..259` to two tensor ranks. Gate/up rows, down-projection columns and their
scale companions follow the same partition. Component capture and intervention
coordinates retain the exact scalar range, including the last three units.

The K2 fixture uses independently generated E4M3 bytes and F32 block scales,
hidden width 130, dense-prefix width 259 and routed width 384. Its scalar oracle
checks dense read rows at block boundaries and the final row, and checks full
write rows crossing the TP cut. Effective queries agree exactly; signed native
projections agree with F64 accumulation within `2e-6 + 2e-6 * abs(reference)`.
Coordinated edits modify gate/up rows `257..259` alongside a write edit crossing
the cut. The independent reference promotes and edits source matrices directly,
then removes their scale companions and quantization selection. The workflow also
checks restoration, peer rollback, cached predictions, component masks and
controlled replay. The two-rank native reproduction passes (12.87s).

```sh
cargo test -j 1 -p eredu-runtime --lib parallel::tests
cargo test -j 1 -p eredu-architectures --lib chunk
cargo test -j 1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  ring_public_component_capture_fp8_partition_tail_tensor \
  -- --ignored --test-threads=1
cargo test -j 1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  ring_public_component_capture_fp8_partition_tail_matrix_ \
  -- --ignored --test-threads=1
```

The neutral tests cover overflow, malformed chunks, mismatched companions,
per-segment tail placement and exact component indices. Companion declarations
cover F32 and UE8M0 scales; the native fixture uses F32 scales. All 42 CPU Ring
configurations pass (763.96s): seven TP/EP/PP combinations, three ordinary
residencies, and ordinary or independently cached bank ownership. These include
eight-rank execution and the explicit cache-eviction checks described above.
Independently blocked fused gate/up tails are covered in the next section.
Attention head tails are described in the dedicated section below.

The updated architecture suite passes 552 tests with one existing ignored test;
runtime passes 463, backend independence 70, portable backend conformance 59,
and the portable facade 19 with one existing ignored test. Formatting and
whitespace checks pass. Ring prediction comparisons use the fixture's `1e-4`
tolerance; the independent effective-query and signed-projection checks use the
stricter criteria above.

The generic chunk-format expander also retains GGUF byte coordinates after
converting scalar chunks to encoded blocks. A neutral regression covers
contiguous and independently segmented input axes, including a final chunk
containing one complete Q8_0 block. This is a storage-placement contract test;
it does not advertise a new native GGUF component execution path.

### Independently blocked fused FP8 gate/up tails

Fused gate/up projections now retain separate scale-block origins for each read
branch. An expert bank with shape `[3, 518, 130]` represents two 259-row branches
and uses scale shape `[3, 6, 2]`. A tensor rank owning the final three units retains
six weight rows and two scale rows, preserving one partial block in each branch.
`eredu-nn::LinearRowLayout` carries this physical declaration through architecture
topology, parallel placement, native projection and effective-parameter decoding.
The native kernels share the usual dynamically quantized activation input; the
row-layout declaration changes weight-scale coordinates only.

The primitive CPU and Metal tests use two groups, two independent row partitions,
widths 3 and 259, and hidden width 130. They cover F32 and UE8M0 scales and route
counts 1 and 9, exercising both grouped Metal dispatches. Independent scalar E4M3
decoding exactly matches two- and three-dimensional native dequantization. Native
projection results agree with F64 accumulation within
`2e-6 * max(sum(abs(terms)), 1)`, a magnitude bound that remains meaningful under
cancellation. Inputs include exactly representable FP8 values and a magnitude-448
value in each activation block so the independent activation scale is exactly one.
Both tests pass (4.39s); CUDA kernel changes have no native hardware validation.

The public two-rank CPU Ring workflow passes (12.28s). It uses the same 130-wide
hidden axis and 259-unit dense prefix, routed experts and shared FFN. Dense-prefix
and routed reads are FP8; the shared FFN remains F32. Queries cross block, branch
and ownership boundaries and agree exactly with independently generated source
values. Coordinated edits modify gate rows `257..259` and up rows `516..518` of
the same fused primary parameter, together with the other attention and FFN edits.
An independently rewritten checkpoint promotes each affected source once and
applies both regions. Edited prefill and two cached decodes, removal, peer rollback,
component masks and controlled replay retain the ordinary lifecycle checks.

The eviction fixture budgets for the larger of one local FFN bank and the local
value-member batch required by one token. TP determines each member's output rows;
EP limits how many routed members that rank can need together. For the short tail,
one encoded FFN member occupies 1,194 bytes and one local value member 2,080 bytes.
Two local value routes require 4,160 bytes; an EP owner with one value expert needs
2,080 bytes. The ledger's reservation error reports the batch total. The replay
check requires fresh FFN-bank misses after competing value queries, plus cache
evictions. The promotion check likewise uses the FFN bank's own compact peak and
encoded-member size, preserving its strict greater-than-three-members bound.

```sh
cargo test -j 1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  independent_row_tails -- --ignored --test-threads=1
cargo test -j 1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  ring_public_component_capture_fp8_fused_tail_tensor \
  -- --ignored --test-threads=1
```

All 42 CPU Ring configurations now pass: ordinary and independently cached banks,
each across resident, host-layerwise and disk-streamed execution and all seven
TP/EP/PP combinations. The corrected cached run passes all 21 configurations in
391.18s, including eight-rank eviction and FFN-specific reload checks. The ordinary
21 configurations passed in the preceding matrix run. Prediction comparisons use
`1e-4`; effective queries are exact and signed projections use
`2e-6 + 2e-6 * abs(reference)`. These synthetic tests establish parameter and
execution behavior; they do not replace released sparse-checkpoint validation.

```sh
cargo test -j 1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  ring_public_component_capture_fp8_fused_tail_matrix_ \
  -- --ignored --test-threads=1
# Reproduce the independently cached subset:
cargo test -j 1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  ring_public_component_capture_fp8_fused_tail_matrix_cached_ \
  -- --ignored --test-threads=1
```

The neutral NN, runtime and architecture library suites pass 38, 464 and 553 tests;
the architecture suite retains one existing ignored test. Backend independence,
portable backend conformance and portable facade pass 70, 59 and 19 tests, with
one existing facade ignore. These cover retained row metadata, global and local
companion geometry, checked block boundaries and independently segmented scale
placement. The broader architecture fixtures pass 196 numerical tests (47.51s),
9 construction tests (0.12s) and 46 structural tests (0.10s). Their independent local
layout helpers retain contiguous and segmented chunks and exact chunk metadata.

```sh
cargo test -j 1 -p eredu-architectures --test reference_numeric \
  --test reference_structural --test reference_realtime_construction
```

Public parallel UE8M0 acceptance remains open. Attention head tails are covered
in the following section. Formatting and whitespace checks pass.

### Partial FP8 attention heads and mixed per-layer state widths

Attention placement now chooses one joint partition of complete grouped-query
heads. Query, key, value, optional output-gate and output-projection encodings
constrain that partition; separately supplied value banks declare their physical
encoding through the architecture contract. Per-head gains and sinks follow the
same ownership. FP8 scale chunks derive from the primary matrix and its block
geometry, including a final partial block. Cache construction preserves each
layer's actual local KV-head count across pipeline boundaries.

The K2 fixture has hidden width 130, 66 query heads, 33 KV heads and head width 4.
Its first two attention layers use E4M3 weights with F32 block scales; the final
attention layer remains dense. On two tensor ranks the respective local KV-head
counts are `[32, 32, 17]` and `[1, 1, 16]`. Query/key normalization is enabled.
The dense-prefix and fused routed FFNs also retain their 259-unit tails, while
the shared FFN remains F32. Admission uses generated SafeTensors source bytes
and ordinary quantization metadata.

Independent scalar decoding checks effective Q/K/V/O rows, routed value-bank
rows, and both FFN read branches, including block and ownership boundaries.
Queries are exact; signed projections use F64 reference accumulation with
`2e-6 + 2e-6 * abs(reference)` tolerance. Coordinated edits affect the final two
Q/K/V rows, value-bank rows `130..132`, and output columns `255..264`, alongside
FFN edits. Separately rewritten source tensors provide the edited reference
through prefill and two cached decodes. Global captures and selected-position
masks include final attention channel 263; controlled replay, rollback and
restoration use the same retained ownership. Prediction tolerance is `1e-4`.
The native two-rank CPU Ring test passes (14.70s).

Independently cached runs derive a budget from the larger of one local FFN bank
and the largest one-token value-member batch among the stage's actual layers.
The latter uses each bank's retained TP/EP shape and encoding; a dense final
value bank can cost more than an earlier FP8 bank. Replay requires fresh FFN
misses after competing value queries, cache evictions, and the original strict
FFN promotion bound. Parameter-query allowances include already consumed work
and scale with participant metadata delivery. Resets and evictions must preserve
all cumulative parameter counters. These allowances describe bounded work;
they are not simultaneous physical-memory allocations.

```sh
cargo test -j 1 -p eredu-architectures --lib attention_head_tails
cargo test -j 1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  ring_public_component_capture_fp8_attention_tail_tensor \
  -- --ignored --test-threads=1
cargo test -j 1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  ring_public_component_capture_fp8_attention_tail_matrix_ \
  -- --ignored --test-threads=1
```

All 42 CPU Ring configurations pass (840.52s across six harness tests): ordinary
and independently cached banks, resident/host-layerwise/disk-streamed weights,
and PP, TP, EP, TP/EP, TP/PP, PP/EP and TP/PP/EP. This includes the eight-rank
cases with different per-layer cache widths. Effective queries, signed projections,
coordinated edits, independent edited-checkpoint comparisons, captures, masks,
rollback, restoration and controlled replay pass throughout. Every cached case
also verifies fresh FFN misses, evictions, promotion accounting and monotone
parameter-query charges. Query and edit phases preserve earlier cumulative work
when admitting their additional finite allowances.

Neutral geometry covers both F32 and UE8M0 scales across all eight TP/PP/EP
ranks, with mixed per-layer head counts, exact primary/companion shapes and
query/key gains. The architecture suite passes 554 library tests with one
existing ignore, 196 numerical tests, 9 construction tests and 46 structural
tests. Backend independence passes 70; portable conformance and facade pass
59 and 19, with one existing facade ignore. Backend compilation without default
features also passes.

This acceptance concerns split attention projections and separate routed value
banks. The generic helper rejects unequal fused-QKV segments with partial FP8
blocks: that extension needs distinct segment chunk widths and scale origins.
The current shared fused-attention family, Moshi, exposes dense/packed formats,
and the shared cold decoder requires split attention; this result makes no new
fused-FP8 source-format claim. The public native matrix uses F32 scale companions;
it does not establish public parallel UE8M0, distributed GPU or released
sparse-checkpoint acceptance.

### Released sparse checkpoint validation

The official [LiquidAI/LFM2-8B-A1B checkpoint](https://huggingface.co/LiquidAI/LFM2-8B-A1B/tree/c1c44ff9fc00db3ebf4516970563f5f383d23670)
is pinned at `c1c44ff9fc00db3ebf4516970563f5f383d23670`. Its four SafeTensors
shards total 16,680,151,504 bytes. Download and provenance verification pass:
every local SHA-256 matches both the pinned fixture manifest and remote LFS
metadata, and the checkpoint index names exactly those four shards.

```sh
/tmp/eredu-qwen-reference-env/bin/python \
  eredu-evaluation/scripts/download_component_fixture.py \
  /tmp/eredu-lfm2-sparse-validation --model lfm2-moe
```

The downloader records the verified snapshot path and all four digests in
`provenance.json` outside the repository. Its original single-file SmolLM2
workflow also passes after the sharded-download extension. The pinned digests are:

| Shard | SHA-256 |
| --- | --- |
| `model-00001-of-00004.safetensors` | `927feafae7d99f40046cd365e8780fe2c36c72ba207ce3024ba54bf884599600` |
| `model-00002-of-00004.safetensors` | `e41140e9841cf338ed537fad0e0d0f16fbeebe0784719188ef7f2999b531d97e` |
| `model-00003-of-00004.safetensors` | `3e3b6236812186ae90f27a3d074f802d5251eab54840b8eca5eea8b730b7a789` |
| `model-00004-of-00004.safetensors` | `4f3cbb6a6a853785186b259fc7b23a5706cca8e53691ab3036a539596ffe38ea` |

`component_sparse_reference.py` uses Transformers 5.16.1, Torch 2.14.0 and
SafeTensors 0.8.0 with BF16 weights, F32 expert correction biases, eager attention
and eager experts. It captures the actual dense/attention down-projection inputs
and routed expert down-projection inputs. Its instrumented baseline exactly
matches four uninstrumented predictions. Source weights remain BF16; no full-model
F32 derivative is created.

The public `component_sparse_probe` discovers component and effective-parameter
coordinates without checkpoint-name parsing. Ten exact parameter queries and
signed projections cover dense gate/value/write, attention Q/K/V/O, and an active
expert's gate/value/write. Five trials exercise baseline, deletion, keep-only,
coordinated BF16-representable replacements and exact restoration. Both masks
select units 1 and 3 only at the last prefill token; routed masks address those
units across all 32 experts. Surviving values and later layers recompute normally.

The optional `controlled` argument additionally compares ordinary and controlled
predictions, restores a post-prefill snapshot, runs a sibling from the initial
boundary, and checks the parent again after exchanging branches. Each trial
checks fourteen controlled predictions against four ordinary predictions. Capture
work permits sixteen steps; restoring snapshots does not refund it. Fork retention
includes the facade's worst-case semantic-event storage derived from the child
trace allowance, plus bounded cache and capture-plan storage. These are logical
reservations, not physical allocator limits.

The initial released comparison exposed arithmetic mismatches: LFM2 was using
FP32 routing coefficients, fused rotary/attention rounding, and the native BF16
sigmoid's intermediate rounding. LFM2 now selects the existing neutral input-dtype
routing, rotary-product, attention-score and sequential expert-reduction policies.
The shared sigmoid operator computes its intermediates in FP32 and rounds once.
Routing normalization likewise accumulates selected scores in FP32 before casting
the completed sum back to the declared score dtype; forced and ordinary routes
share the same normalization driver.
Independent BF16 regression values pass on CPU and Metal; all fourteen grouped
operator tests and eleven neutral LFM2 execution tests pass. Ten native CPU Ring
component tests also pass across TP, PP, EP, combined TP/PP/EP, host-layerwise and
disk-streamed execution (71.07 seconds).

The native Metal public probe and strict independent comparator pass all five
trials: baseline, deletion, keep-only, overlay and restoration. The comparison
checks 18,971,520 values across all thirty component groups, every prefill position
and three cached decode steps. All 17,660,800 non-logit values match exactly,
including original/effective units, routing coefficients, captured write vectors,
residual boundaries and final-head inputs. All seventy controlled predictions
match their twenty ordinary counterparts, including cached replay, parent/sibling
isolation and exact restoration. The all-position run reserves 4,898,947,072 bytes
for snapshots and retained events, with a separate 4 GiB cumulative copy limit.

The 1,310,720 logits pass `atol=rtol=0.001` with the explicit
`--bf16-logit-ulps 1` allowance. Only 49 values need that allowance; their measured
head inputs are identical and the additional difference is at most one BF16 step.
Maximum logit absolute error is 0.03125 and maximum per-prediction RMS error is
0.000185681. Near-zero values may span more BF16 steps while remaining within the
absolute tolerance. This rule never relaxes component or route comparisons.
Baseline generates `[4481, 856, 4481, 856]`; keep-only recomputes survivors and
changes the sequence to `[4481, 803, 4481, 803]` without forcing tokens. These
illustrative interventions and edits demonstrate API behavior, not research efficacy.

The last mismatch was layer 3's feed-forward RMS normalization on the first
cached deletion decode. Exact residual input produced nineteen differing Metal
coordinates near BF16 rounding midpoints. The shared learned RMS mechanism now
uses the existing float32 row reduction and rounds the normalized value before
the gain. Its compact independent fixture fails before the fix and passes exactly
on CPU and Metal afterward, through both parameterized and direct weighted
operations and for an independently calculated F16 variant. The existing mixed-
precision gain/output-dtype regression and backend build without default features
also pass. Fixture provenance and regeneration are recorded in
`eredu-backend-mlx/validation/bf16_rms_input_rounding.*`; aggregate released evidence
and artifact hashes are in `eredu-backend-mlx/validation/lfm2_sparse_components.json`.
Selected-token and token-difference score reconstruction is accepted below.
These released results do not close the full component-analysis goal.

The optional `writes` consumer mode reconstructs every scalar and routed write at
the final prefill position using public bounded parameter projections. It compares
unit-axis write projections with output-axis signed column projections, accounts
for the released equation's BF16 expert/weight/reduction rounding, and retains
the mathematical signed sum separately. An independent float64 reference checks
all five trials: 921,600 component terms and 150 writes. Of these writes, 136 are
exact; the largest physical-write error is 0.00048828125. The two native projection
orders differ by at most 2.30284e-7; independent signed sums differ by at most
1.34125e-7 and pass absolute/relative tolerances of 1e-6. Physical writes retain
the existing absolute/relative tolerance of 0.001. These directions are periodic
signed fixtures, not selected vocabulary rows.

The all-layer calculation sizes a finite cumulative work allowance from loaded
source/selection geometry, the planned routes and two projections per route.
Packed-source retention is charged again on each projection; it is not refunded
between trials. The combined reconstruction/control run passes all seventy
controlled predictions and consumes 931,657,150,624 logical
retention-work bytes, 46,952,096 host bytes and 64,920,824 encoded bytes across 991
parameter operations. These counters do not describe simultaneous allocation.
Provenance, tolerances, artifact hashes and results are recorded in
`eredu-backend-mlx/validation/lfm2_sparse_write_reconstruction.json`.

```sh
/tmp/eredu-qwen-reference-env/bin/python \
  eredu-evaluation/scripts/component_sparse_reference.py \
  /tmp/eredu-lfm2-sparse-validation/provenance.json \
  /tmp/eredu-lfm2-sparse-validation/transformers-reference.json \
  --all-layers --all-positions --write-reconstruction
cargo run -j 1 -p eredu --no-default-features --features mlx,metal \
  --example component_sparse_probe -- \
  /tmp/eredu-lfm2-sparse-validation/transformers-reference.json \
  /tmp/eredu-lfm2-sparse-validation/native-gpu.json gpu controlled writes
/tmp/eredu-qwen-reference-env/bin/python \
  eredu-evaluation/scripts/compare_component_sparse.py \
  /tmp/eredu-lfm2-sparse-validation/transformers-reference.json \
  /tmp/eredu-lfm2-sparse-validation/native-gpu.json --require-controlled \
  --require-write-reconstruction --bf16-logit-ulps 1
```

The final command passes with zero failures. The rounding allowance requires exact
measured head inputs and BF16-representable output values; it is not a blanket
relative-tolerance increase.
`--report-only` emits diagnostic metrics without declaring acceptance. The reference
supports `--trials baseline --all-layers --all-positions` to locate an earlier
prefill divergence, including convolution inputs influenced by prior positions.

### Released sparse selected-token and token-difference reconstruction

The `scores` consumer mode uses public effective-parameter queries for two head
rows and the normalization gain. At each prediction, the baseline winner and its
strongest competitor define fixed target/alternative IDs for every trial. Each
trial then follows its own unforced trajectory: cached prefixes can differ between
baseline and keep-only. The independent reference and native run compare matching
contexts within each trial. The selected position is the final prefill token,
followed by three cached predictions; the preceding all-position capture validation
remains separate evidence.

Measured residuals determine target and difference covectors through the declared
RMS equation. Public output-axis contractions return signed per-unit projections
for all thirty component groups. The consumer also captures the effective embedding,
all eighteen convolution writes, final residual and actual head input. It replays
the forty-eight ordered BF16 residual additions exactly at all twenty predictions.
Convolution writes remain explicit whole terms rather than neuron attributions.

The reported accounting separates component terms, operator rounding, embedding,
other writes, residual-addition rounding, direction conversion and normalization
rounding. These measured numerical corrections are engineering extensions; they
are not claims about the paper's sparsity or editing efficacy. No correction is
derived from the final score error. The sum is checked against an independent
float64 dot product of the actual head row and captured head input. This fixture
requires additive residuals, identity output transformation, BF16 head rows and
outputs, and its declared bias-free RMS equation; incompatible equations receive
an explicit consumer error.

The complete Metal run passes baseline, deletion, keep-only, coordinated overlay
and restoration, with all seventy controlled replay/isolation checks. The strict
independent comparator checks 10,944,290 values with zero failures. All 9,629,120
captured non-logit values match exactly. Reconstruction covers 600 component groups,
3,686,400 component terms and forty selected/difference scores. Maximum error
between the explicit accounting sum and direct float64 head dot is
`3.552713678800501e-15`. Individual independent signed component projections use
`atol=rtol=1e-6`; aggregate sums propagate those per-group bounds. The small
operator-rounding remainder propagates the uncertainty of both large quantities
being subtracted, rather than scaling tolerance only by the remainder.

Actual BF16 target scores are allowed one representable output step; differences
use the sum of target and alternative step bounds. This is an absolute bound even
for nearly cancelling scores. The maximum measured score error is
`0.051227287272922695`, using at most `0.36512885708361864` of the applicable bound.
This output-rounding comparison is separate from the much tighter affine accounting
check. Final explicit head/dtype guards also pass a fresh baseline run and strict
comparison through all four predictions and fourteen controlled checks.

The combined run consumes 3,931 parameter operations, 3,649,564,913,824 cumulative
logical retention-work bytes, 367,684,256 host bytes and 495,514,544 encoded bytes.
Finite allowances derive from loaded geometry and all planned predictions; restore
does not refund prior queries. These totals do not describe simultaneous allocation.
Provenance, fixed token IDs, tolerances, artifact hashes, counters and results are
recorded in `eredu-backend-mlx/validation/lfm2_sparse_scores.json`.

```sh
/tmp/eredu-qwen-reference-env/bin/python \
  eredu-evaluation/scripts/component_sparse_reference.py \
  /tmp/eredu-lfm2-sparse-validation/provenance.json \
  /tmp/eredu-lfm2-sparse-validation/transformers-score-reconstruction.json \
  --score-reconstruction
cargo run -j 1 -p eredu --no-default-features --features mlx,metal \
  --example component_sparse_probe -- \
  /tmp/eredu-lfm2-sparse-validation/transformers-score-reconstruction.json \
  /tmp/eredu-lfm2-sparse-validation/native-scores.json gpu controlled scores
/tmp/eredu-qwen-reference-env/bin/python \
  eredu-evaluation/scripts/compare_component_sparse.py \
  /tmp/eredu-lfm2-sparse-validation/transformers-score-reconstruction.json \
  /tmp/eredu-lfm2-sparse-validation/native-scores.json \
  --bf16-logit-ulps 1 --require-controlled \
  --require-write-reconstruction --require-score-reconstruction
```

### DeepSeek V3 compressed component hooks

Resident and paged latent-attention execution now expose the actual aggregated
channels and output-projection multiplication input through the shared component
instrumentation. The dense-only and routed-capable V3 blocks share normalized
attention/FFN inputs, dense SwiGLU units, write vectors and residual boundaries.
The existing complete `compressed_attention.output` intervention is preserved.
Compressed replicated traversal forwards its admitted observer into this same
family code; disabled instrumentation creates no additional component tensors.

The nonzero neutral fixture checks direct and normalized low-rank queries, unequal
query/value head widths, both cache modes, prefill and two cached decodes. Forty-eight
candidate predictions exercise no-op, channel deletion, channel keep-only and dense
unit deletion. All ninety-six attention/FFN writes reconstruct from their actual
inputs and independent float64 sums. Masks affect exactly the requested row and
channel; downstream gate/value reads recompute from the changed residual. Both
V3 block adapters agree exactly, and resident/paged results agree within the
existing numerical fixture tolerance.

Descriptor version 3 declares direct and low-rank query reads, normalized KV-latent
reads and the separate shared rotary-key rows. Component read chains preserve
their ordered projections, normalization gains and current-position observations;
they are not collapsed across a nonlinear normalization. Additional neutral tests
reconstruct both latent stages from their declared parameters and prove that
effective KV latents enter the cache. Whole-model observed execution agrees with
ordinary execution through prefill and two cached decodes. Declared embedding,
attention, dense and complete sparse contributions reconstruct the residual,
selected-token score and token-score difference with an absolute `2e-4` tolerance.
All three component tests and the focused V3 regressions pass, including TP2,
target execution and V3/V4 prediction reuse; all ten discovery tests also pass.

```sh
cargo test -j 1 -p eredu-architectures --test reference_numeric v3_
```

Public native acceptance covers twelve configurations: direct or low-rank queries,
dense-only or dense/sparse blocks, and resident, host-layerwise or disk-streamed
weights. CPU capture and component-mask trials agree across residency modes and
with controlled snapshot/fork execution. CPU and Metal parameter tests use the
descriptor to select complete query-head rows, strided key/value dependencies,
latent first-stage rows, normalization gains and attention/FFN write columns.
Signed native projections agree with independent float64 contractions (`2e-6`
absolute tolerance). Coordinated additive edits agree with a separately edited
F32 checkpoint for two fixed prefixes through prefill and two cached decodes
(`3e-5 + 3e-5 * abs(reference)` tolerance). Snapshot replay is exact, removing the
overlay restores the baseline, cumulative parameter usage is retained, and source
checkpoint bytes are unchanged. The three public tests pass (CPU capture 5.82s,
CPU edits 8.20s, Metal edits 8.52s).

```sh
cargo test -j 1 -p eredu --no-default-features --features mlx,metal \
  --test native_execution_control v3_components:: -- --ignored --test-threads=1
```

The tests exposed and closed a native compressed-cache snapshot gap. Resident
snapshots now compact logical latent/rotary arrays into independent storage;
paged copies additionally isolate the catalog and tails. Partition hooks and
MTP propagation still remain required for the family's component integration.
These local results do not establish those capabilities.

All five native cache-isolation regressions also pass (6.79s). The compressed
fixture uses strided latent/rotary input, three interleaved 260-token histories,
a saved prefix and a copy of a mature catalog. It crosses resident allocation
chunks, forces paged blocks to host, verifies independent buffers/catalogs and
checks that clearing parents leaves copied values unchanged. Its finite pool
separates a 256-byte per-cache device limit from aggregate room for five catalogs
and copy work; host capacity includes page-aligned transfer buffers.

```sh
cargo test -j 1 -p eredu-backend-mlx --no-default-features --features metal \
  --lib isolated_snapshot_tests:: -- --test-threads=1
```

Shared-expert scalar integration uses the existing `RoutedPlusShared` equation.
The `model.layers.N.feed_forward.shared` scope exposes normalized input, gated
units, actual down-projection input, write and output, including distinct effective
values. The shared child retains its canonical gate/value rows and down columns;
its write is a constituent of the complete sparse sum. Target-only V3 catalogs
describe these groups completely; catalogs with embedded prediction layers retain
an explicit partial result. Actual loaded support remains a separate report.

Sixty nonzero neutral predictions cover one and two shared experts, resident and
paged caches, prefill and two cached decodes, deletion, keep-only masks and changed
inputs. Shared reads recompute from current inputs, float64 sums reconstruct the
write, and the effective shared write enters the complete sparse contribution
exactly once. A separate bounded-SiLU fixture checks the preserved gate/value
limits and agreement with identity TP reduction. All five focused component tests,
eight V3 regressions and ten discovery tests pass.

The native capture matrix now has eighteen cases, adding six shared-unit mask
cases to the original twelve. All twelve CPU/Metal parameter configurations edit
both layers, including shared gate/value rows and write columns. The same public
reference consumer reconstructs a selected score and score difference on CPU and
Metal across all three residency modes, under ordinary and controlled baseline
and masked execution. It preserves nested signed contributions separately; adding
them to their complete sparse parent exceeds the reconstruction tolerance in every
tested case. This verifies the consumer's accounting rather than merely declaring
the nested topology.


V3 pipeline component propagation now uses the same target input/readout driver.
Only the input stage observes embedding lookup; its effective value travels in
the typed target boundary. Only the output stage observes final residual,
normalization, actual vocabulary-projection input and affine scores. Latent Q/KV
observations have retained partition coordinates derived from declared projection
rows; pre-intervention and effective seams share invocation ownership, independently
of checkpoint aliases. These coordinates preserve row permutations and short FP8
row chunks and reject incomplete column-sharded products.

The checkpoint-backed neutral test
`v3_pipeline_components_match_whole_execution_across_residencies_and_cached_steps`
compares direct/low-rank query variants over two pipeline stages, cutting between
the dense and sparse layers. Nine direct-query and ten low-rank trials cover
readout/input interventions, attention/dense/shared units, Q/KV latents and a
combined keep-only plan, with prefill plus two cached decode steps. All three
weight residencies agree with a whole-model reference bound to the same checkpoint
bytes: 171 pipeline predictions, each with original/effective capture geometry and
producer receipt checks. Unselected static placeholders in the scalar adapter are
poisoned so accidental execution cannot pass. All 34 component conformance tests
and all nine partition-coordinate tests pass.

```sh
cargo test -p eredu-architectures --test reference_numeric components::
cargo test -p eredu-architectures --lib component_partition::
```

Remaining V3 work includes embedded prediction internals and remaining
format/route combinations of distributed
query/edit acceptance. All-dense target partitions now use direct construction and the dense
TP hooks described below. The sparse TP path still combines routed and shared
partials in one ordinary reduction. The neutral additive receipt and intervention
mechanisms below preserve that behavior. Mixed V3 TP execution integration passes
neutral conformance and the native F32/affine matrix detailed below.


The native two-process Ring test
`ring_two_process_deepseek_v3_pipeline_components` passes resident, disk-streamed
and host-layerwise execution (21.28 seconds for all three cases). Its nonzero F32
SafeTensors fixture uses a low-rank query, one dense layer and one routed/shared
layer. Public loaded discovery must advertise both Q/KV latent stages, channel
and dense/shared unit seams, complete sparse writes and readout. Global captures
match an ordinary local reference within `2e-4 + 2e-4 * abs(reference)` through
prefill and two cached decode steps. The shared harness also checks bounded
transforms, skipped capture, final-output-only capture, cold preparation, controlled
snapshots, parent/sibling replay and position-specific deletion/keep-only masks.
The same harness also verifies routed expert-unit captures and sparse interventions
against the ordinary native reference through cached decode.

```sh
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal \
  --lib ring_two_process_deepseek_v3_pipeline_components \
  -- --ignored --nocapture --test-threads=1
```

Cold checks additionally prove Q/KV latent replication and one publisher under
TP2 and TP2/PP2, unchanged ownership under shared-storage aliases, and rejection
of mismatched stage extents. These layout checks do not establish TP block-hook
or TP native execution acceptance. No new TP support claim follows from the PP
result.


### All-dense V3 partition construction and rotary correction

All-dense prediction-free V3 now consumes the direct retained selection for TP,
PP and combined TP/PP. No expert bank or grouped operation is requested. The
architecture owns source/target transform layouts, parameter companions, static
ownership and local compressed-cache ordinals. Selected embedding/head encodings
are honored, including load-time affine transforms. Sparse schedules retain their
existing routed construction.

The dense block observes complete attention and FFN writes after the same two
reductions used by ordinary execution. Exact no-op tests check that observation
adds no reduction and preserves cached values. Direct and low-rank query variants
pass 513 neutral partition predictions (nine topology/residency combinations,
nine/ten trials, prefill and two cached steps), including original/effective
captures, latent coordinates, input/readout edits and combined keep-only masks.
All 38 component conformance tests pass (4.69 seconds). Expanded scalar affine
execution additionally agrees across all nine topology/residency combinations
(27 cached predictions, 0.32 seconds); this is numerical and construction evidence,
not native packed-storage evidence.

The native `ring_deepseek_v3_dense_partition_components` test passes all nine
SafeTensors combinations (52.12 seconds). It queries and projects every weight,
atomically edits embedding, final readout, normalization, both MLA bottlenecks,
head projections and both layers' gated FFN reads/writes, and compares the resulting
predictions with an ordinary native reference. It checks rejected publication
rollback, stale state/version rejection, active-overlay capture provenance,
restoration, snapshot replay and siblings. Captures and position-specific
zero/keep-only trials use the same public controlled driver. Global captures use
`2e-4 + 2e-4 * abs(reference)`; edited logits use absolute `1e-4` for this F32 fixture.

The corresponding native GGUF test also passes all nine combinations (49.82
seconds), preserving the published split KV-B key/value representation through
the direct constructor. The post-correction public CPU/Metal regression passes
all four V3 tests (12.57 seconds), and mixed dense/routed V3 PP capture, edits and
replay pass all three residencies (15.83 seconds).

Native load-time affine quantization also passes all nine combinations (64.59
seconds): 4 bits, group size 32, a seeded F32 source with hidden width 64, FFN width
128, four heads, query/KV latent ranks 32 and vocabulary 32. The fixture uses
parameter-name FNV seeds starting at `2166136261` and index step `0x9e3779b9`, with
matrix amplitudes up to `0.2` and normalization gains in `[0.9, 1.0]`. Its ordinary
reference retains the same quantization request. The test requires query,
projection and replacement support for every matrix and normalization weight, then checks their
effective values and all coordinated edits. It exercises the actual transformed
packed tensors and companions, including embedding and readout, through cached
execution and restoration. Edits publish F32 copies of the affected parameters'
effective values; removal restores their original packed storage. Capture and
edited-logit tolerances remain the F32 comparison tolerances above. Together these
are 27 native topology/residency/format
cases; they do not establish mixed sparse or FP8/UE8M0 V3 acceptance.

This validation exposed a pre-existing query rotary-axis error: V3 rotated
`[batch, sequence, heads, features]` as if the head axis were sequence. Queries now
rotate in `[batch, heads, sequence, features]`. The correction applies to all V3
execution paths. A stronger neutral fixture uses identical heads and sixfold
matrix amplitudes; direct/low-rank queries and resident/paged caches agree between
five-token prefill and sequential cached execution within `2e-6`, with head equality
within `1e-6`. MLA equation revision 2 in the cache fingerprint rejects older V3
states; recreate those caches from exact token IDs.

```sh
cargo test -p eredu-architectures --test reference_numeric components::
cargo test -p eredu-architectures --test reference_numeric v3_dense_partition_transforms
cargo test -p eredu-architectures --lib deepseek::
cargo test -p eredu-architectures --lib v3_rejects_cache_identity
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal \
  --lib ring_deepseek_v3_dense_partition_components \
  -- --ignored --nocapture --test-threads=1
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal \
  --lib ring_deepseek_v3_dense_gguf_partition_components \
  -- --ignored --nocapture --test-threads=1
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal \
  --lib ring_deepseek_v3_dense_affine_partition_components \
  -- --ignored --nocapture --test-threads=1
cargo test -j1 -p eredu --no-default-features --features mlx,metal \
  --test native_execution_control v3_components:: \
  -- --ignored --nocapture --test-threads=1
```

The full Inspector integration goal remains open. These results establish the
listed dense V3 paths; they do not establish sparse TP hooks, prediction-internal
components or untested native encoded-format combinations.


### Additive tensor-parallel write mechanism

Component topology schema 3 adds `ComponentGroup::write_partition`. `Complete`
retains existing reduced-write semantics; `TensorParallelSum` describes actual
full-hidden-width terms before an ordinary TP reduction, including reductions
fused with another branch. V3 shared experts declare the latter. This declaration
does not by itself enable execution hooks. Both target-only mixed V3 TP traversals
now emit those hooks. Neutral and native acceptance is detailed below.

Partition receipt/evidence schema 3 records `Disjoint` or `SumF64ToF32` explicitly.
Summed receipts require every authoritative TP term, including empty acknowledgments;
identical hidden coordinates do not make different TP terms replicas. Raw floating
terms use the ordinary F32 collector, compensated F64 host summation in world-rank
order, and a final F32 conversion. Summary and histogram follow that assembly;
preview exports only its bounded prefix. Source precision describes the terms.
The result is a measured host sum, not evidence of a separate complete native tensor
or bitwise equality with the fused native reduction. Nonfinite terms remain explicit.
Routed-unit and vocabulary transforms are outside this activation-sum contract.

`PartitionActivationProjection::as_sum_term` requires complete ordered coordinates.
It applies Zero, Scale and masks to every term; Add only to the designated offset
owner; Replace to that owner and Zero to the other terms. Each actual replica sum
has one owner. Geometry agreement binds this choice. Native per-term rounding can
differ from editing an already reduced tensor. Unchanged Add peers still validate
their source and participate in completion, without copying payloads or submitting
an update. Vocabulary sentinel masking receives a typed rejection.

The runtime tests cover cancellation-heavy sums, raw/preview/summary/histogram,
nonfinite values, strided positions, forged equations, missing and duplicate peers,
empty selections, reservation rejection before export, generated sources, common
completion, abort/failure publication and restore without budget refunds. The six
partition intervention tests also compare complete edits with summed edited terms,
including sequential overlapping operations and one offset owner per operation.
All 471 runtime library tests pass (0.34 seconds), including rejection of schedule-
skipped metadata reservations before allocation or export.

```sh
cargo test -j1 -p eredu-runtime --lib capture::tests::partition::
cargo test -j1 -p eredu-runtime --lib intervention::tests::partition::
cargo test -j1 -p eredu-architectures --lib component_partition::
```


The V3 provider-aware TP traversal now invokes the additive shared-write hooks and
preserves its ordinary fused reduction through a shared helper. A nonzero neutral
test covers complete/partial provider results and literal post-bias across five
shared-branch experiments (15 cases), with exactly one reduction. Nine
partition-coordinate tests pass, including TP2/EP2 term ownership and one offset
owner per replica sum. The current complete-suite counts appear below.


The backend's portable-feature build also passes:

```sh
cargo check -j1 -p eredu-backend-mlx --no-default-features
cargo test -j1 -p eredu-architectures --test reference_numeric components::compressed_v3::
```

These commands verify compilation and the neutral fixtures. Native commands and
results are recorded separately below.


The target-only mixed V3 TP integration now uses the common block and observed
routed/shared driver for both ordinary resident banks and runtime providers. Shared
write/output terms enter the existing fused reduction; provider post-bias remains
a once-only addition. No extra diagnostic reduction is introduced. The ordinary
resident path retains its general grouped-backend requirement.

The prepared neutral matrix covers TP2 and TP2/PP2 across resident, host-layerwise
and disk-streamed execution, direct and low-rank queries, twelve/thirteen intervention
trials, prefill and two cached decodes: 450 logical predictions. Shared input,
write and output edits join channel/unit/latent experiments. Actual TP terms go
through receipt encoding, reversed arrival and runtime sum assembly before being
compared with the whole-model reference. The expanded mixed PP matrix has 225
predictions. These comparisons exercise the actual runtime receipt assembly,
including the partitioned execution paths used by native sessions.


Mixed V3 quantized partition integration uncovered portable construction gaps:
partition units aligned to encoding blocks were mistaken for scalar expert units,
and the routed handoff omitted the source architecture needed for load-time
transforms. The range conversion now expands validated uniform chunks into scalar
coordinates. The handoff retains and revalidates the typed source layout, expert
ownership, execution addresses and compressed-state geometry before materialization.
Both native local and pipeline adapters consume that retained authority.

The prepared scalar affine matrix covers mixed dense/routed TP2, PP2 and TP2/PP2
across all three residencies, with prefill and two cached decode steps. It validates
real selected packed geometry and companions while retaining expanded scalar
weights for numerical execution. All thirteen V3 tests pass (2.51 seconds), the
full numerical fixture suite passes 209 tests (99.57 seconds), and the architecture
library passes 560 tests (0.37 seconds; one manual FP8 scaling measurement ignored).
These scalar results validate construction and equations; native quantization
acceptance is recorded below. The shared router change also passes seven V4
numerical tests.


The expanded native F32 mixed-V3 matrix passes 42 cases: SafeTensors and GGUF,
TP2/PP2/EP2 and every combination, each across resident, host-layerwise and
disk-streamed execution. Both packed expert read halves and the expert write
banks participate in atomic edits, including separate EP owners. Shared additive
writes exercise Zero, Scale, Add, Replace and strided masks at prefill position 1
and decode position 0, with ordinary-reference logits and before/after evidence
compared at `4e-4 + 4e-4 * abs(reference)`. Inactive operations must report schedule
skips without payloads; every active operation must apply and provide measurements.
Summary and histogram trials capture only the additive sources, so native source
completion cannot depend on exporting logits. The summaries follow complete
compensated sum assembly; fixed-edge histogram counts agree exactly.

The GGUF component fixture scales non-normalization weights by eight relative to
the existing patterned fixture, preserving its published split KV-B representation.
This makes component writes distinguishable at the test tolerance. These are
synthetic F32 native fixtures.

The affine native rerun also exposed a router constructor that forced dense
storage after selection requested packed weights. Shared V3/V4 router policy now
consumes the explicit retained format, with matching parallel companion geometry;
the default without a per-parameter override remains dense.

The mixed affine native matrix passes all 21 cases: TP2/PP2/TP2-PP2 and
EP2/TP2-EP2/PP2-EP2/TP2-PP2-EP2 across the three residencies. A fresh run after
the source-precision correction below passes these 21 cases plus all nine dense
V3 affine cases (30 cases in 317.73 seconds). The seeded fixture has hidden width 64, dense FFN width 128,
expert width 64, four heads, Q/KV latent ranks 32 and vocabulary 32. The first
layer is dense; the second has four routed experts and one shared expert. It uses
the same parameter-name seeds and amplitudes as the all-dense affine fixture.
Each TP shard retains a complete 32-element quantization group. Both ordinary and
distributed references use the same load-time 4-bit/group-32 request. Effective
queries, both expert read halves, write-bank edits, capture transforms and all five
additive actions pass without relaxing their existing tolerances. Packed edits
promote affected matrices to F32; removal restores original packed sources.

Exact materialization checks the admitted global derived recipe before applying
rank placement, then checks local source geometry and dtype. This fixes the
previous comparison of global metadata with an already sharded recipe. The final
portable backend check passes (6.79 seconds). The combined ordinary-residency mixed
matrix has 63 passing cases. Qwen3's three native exact-transform TP/PP/TP+PP
regression tests also pass with source-preserving precision (19.72 seconds).
Independent-bank results are listed below; public MTP
component integration and other V3 encodings remain separate gaps.

Independent-bank validation subsequently identified an ordinary affine transform
that used an unloaded F16 placeholder as a cast request. Exact replicated-text
materialization now preserves the floating source dtype, matching bank transforms
and selected-byte accounting. Both the ordinary affine matrix above and the
cached affine matrix below now pass with this correction. Reusable prompt-cache
schema 9 rejects older persisted prefixes; replay the exact token IDs to rebuild
them. The schema boundary also invalidates older unquantized caches. The neutral
cached-affine matrix passes nine TP/PP/residency combinations through prefill and
two decodes. At this change, all 210 numerical tests, 560 architecture tests (one
existing manual test ignored), 238 core tests, 471 runtime tests, 59 backend
conformance tests and 19 portable facade tests pass. The native cache suite passes
all 48 cases, including its seven explicitly selected CPU/Metal tests. These are
cache compatibility and lifecycle checks, not native distributed GPU validation.

```sh
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal \
  --lib ring_deepseek_v3_mixed_affine_partition_components \
  -- --ignored --test-threads=1 --nocapture
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal \
  --lib ring_deepseek_v3_mixed_expert_affine_partition_components \
  -- --ignored --test-threads=1 --nocapture
```

### V3 independently cached component parameters

The affine cached matrix passes all 21 native CPU Ring cases (270.99 seconds):
TP2, PP2, TP2-PP2, EP2, TP2-EP2, PP2-EP2 and TP2-PP2-EP2, each with resident,
host-layerwise and disk-streamed ordinary parameters. It uses the same nonzero
mixed V3 fixture and 4-bit/group-32 request described above. Every cached member
retains the selected generated scale/bias destinations; ordinary binding rejects
unclaimed destinations. Companion precision agrees with ordinary materialization.

The matrix checks effective queries and projections, atomic edits to both packed
expert read halves and write banks, all five additive actions, raw/preview/summary/
histogram capture, controlled snapshots, sibling isolation and restoration. It
checks exact PP/EP expert ownership, positive device occupancy and actual compact
bank construction. Queries and edits can populate a cache before inference, so a
miss is not required. These cases use default bank budgets; they do not establish
forced-eviction acceptance.

The cached F32 GGUF CPU Ring matrix passes all 21 cases (251.90 seconds). Some EP
owners receive no routes despite owning queried/edited parameters: those owners must
report zero selections and construct no compact banks. Every routed PP stage must
still have an active cache owner, with positive occupancy and compact-bank work;
dense stages own no banks. The final F32 SafeTensors and affine rerun passes all
42 cases with the same checks (525.52 seconds). Together, the three source variants
pass all 63 independently cached CPU Ring cases. These results include exact
source-preserving companion precision and generated companion ownership; they do
not establish additional encodings, forced-eviction component trials or distributed
GPU acceptance.

```sh
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal \
  --lib ring_deepseek_v3_mixed_cached_ -- --ignored --test-threads=1 --nocapture
```

### V3 prediction component integration in progress

`V3PredictionLayer` now uses one fusion/head driver for ordinary and observed
execution. Provider and TP variants reuse the existing block hooks. Under the
canonical prediction unit path, `prediction.embedding.normalized` and
`prediction.hidden.normalized` are mutable effective inputs to the fusion;
`prediction.fusion.output` feeds the decoder. Attention and shared/routed FFN
hooks retain their ordinary suffixes. `prediction.readout.residual` precedes head
normalization, `prediction.readout.normalized` feeds the actual head, and
`prediction.readout.linear` is the resulting draft score. Each mutable seam has a
separate `.effective` observation. Fusion and head projection inputs are read-only
evidence. The older outer `mtp.{depth}.output` remains after head computation.

A nonzero neutral test passes 216 forwards: direct/low-rank query, resident/paged
compressed cache, ordinary-provider/resident-TP/provider-TP dispatch, six trials,
and prefill plus two decodes. It checks no-op parity, attention deletion and
keep-only, shared-unit deletion, normalized-hidden and pre-head residual edits,
write/head reconstruction, exact selected positions and head renormalization.
The TP variants in this test use one rank. A second test passes 48 actual
layerwise observed forwards across four ordinary/provider/TP dispatch paths and
two prediction depths, plus sixteen fresh sibling replays. It verifies only the
selected prediction cache advances and target/other-depth state remains unchanged.
Both prediction tests pass (0.17 seconds); the full 211-test numerical suite before
the second test was added also passes (97.75 seconds). All 560 architecture tests
and the portable backend check pass with the production hooks.

This is primitive and neutral-traversal acceptance only: phase-aware speculative
admission and delivery, public queries/overlays, distributed construction and
native acceptance remain required.
Loaded prediction component support remains unadvertised.

The existing public V3 embedded scheduler passes on Metal after the shared-driver
change (7.59 seconds). This validates ordinary embedded drafting, not native
prediction component capture or public speculative component admission. The native
TP2 ordinary-target/typed-extension regression also passes (8.12 seconds).

```sh
cargo test -j1 -p eredu-architectures --test reference_numeric v3_prediction_ \
  -- --test-threads=1
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal \
  --lib public_deepseek_v3_embedded_scheduler_executes_on_metal \
  -- --ignored --test-threads=1 --nocapture
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal \
  --lib qwen3_transformed_ -- --ignored --test-threads=1 --nocapture
```

Prediction topology is now available in `ArchitectureDescriptor::component_scopes`
(descriptor schema 6, component schema 5). Select one scope before computing draft
score attribution. Its `LinearFusion` residual base names normalized sources, their
concatenation columns and the effective fusion matrix. Add that whole fused base
to the scope's decoder writes, counting the complete sparse write once, and use
that scope's normalization and head. Do not mix these groups into primary target
attribution or distribute input-dependent RMS normalization over earlier hidden
components. Use the observed invocation input: extension prefill/replay and
sequential proposals can supply different hidden sources.

Scope declarations include original/effective fusion and head seams, actual
read-only projection inputs and the separate after-head legacy hidden output.
The ordinary selected-call-path support gate rejects all prediction requirements;
public phase-aware capture must be wired before that gate can be enabled.

The scope-driven numerical check passes all 216 primitive forwards, reconstructing
the fusion from declared input columns and its effective weight, then the
pre-head residual and selected-token scores/differences from the scope's terms.
Direct residual edits are separate terms. The scalar F32 reference uses F64
accumulation and a score tolerance of `2e-4 * (1 + abs(expected))`; tensor
reconstruction uses the suite's existing absolute `2e-4` tolerance. Ten discovery
tests pass, including scope/group/parameter joins, original/effective timing,
legacy JSON and rejection by target-only capture support. Core's 238 and runtime's
472 tests pass with the new schema and support gate.

Typed V3 extension operations now forward optional internal observers during
prefill, proposals and retained-input replay. The neutral typed test passes for
both query forms and serial/TP dispatch (TP uses one rank): eight observed prefill and eight
observed replay unit forwards, each paired with ordinary execution, then 24 proposal trials each comparing ordinary and observed
execution, a channel edit, an injected observer failure after state mutation and
restored retry. The separately owned prediction lane is restored by its caller;
the target cache stays unchanged. Invalid depth fails before invoking the target.
Serial mode checks every declared ordinary tensor point. TP checks component and
outer invocation hooks independently of routing-output records, whose separate
provider collector is absent from this fixture. This is not native distributed
or public speculative collector acceptance.

```sh
cargo test -j1 -p eredu-core -p eredu-runtime --lib
cargo test -j1 -p eredu-architectures --lib discovery::tests
cargo test -j1 -p eredu-architectures --test reference_numeric v3_typed_extension_hooks
```

After the scope and typed-extension changes, the full architecture suite passes:
560 unit tests (one existing manual test ignored) and 213 numerical tests, including
all component and prediction regressions. The numerical run took 52.76 seconds.

```sh
cargo test -j1 -p eredu-architectures --lib --test reference_numeric
```

Portable integration validation also passes: 59 backend conformance tests,
19 facade tests (one existing optional fixture ignored), the backend check with
no default features, and all portable facade examples.

```sh
cargo test -j1 -p eredu --no-default-features --test portable_facade --test backend_conformance
cargo check -j1 -p eredu-backend-mlx --no-default-features
cargo check -j1 -p eredu --no-default-features --examples
```

A fresh native MLX build after these changes passes the existing public V3 embedded
scheduler on Metal (5.22 seconds) and the two-process CPU Ring target retaining
its typed prediction extension (5.92 seconds). These confirm ordinary speculative
regression behavior; they do not establish public prediction component capture.
The build emitted only the existing linker warning about the large unwind table.

```sh
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  public_deepseek_v3_embedded_scheduler_executes_on_metal -- --ignored --test-threads=1 --nocapture
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  ring_two_process_deepseek_v3_mtp_target_tensor_parallel_opaque_session -- --ignored --test-threads=1 --nocapture
```

### Shared speculative phase observer bridge

The embedded executor can now receive a separate internal observer through
`EmbeddedPredictionObservers::with_internal`. The architecture-owned strategy
forwards it to actual target prefill, prediction seed, sequential proposals,
verification and both replay paths. `SpeculativeActivationPhase` identifies the
operation and proposal depth; each invocation supplies its physical sequence
width. These are not committed prediction ordinals. The MLX session binds the
complete observer set without reconstructing family or scheduling policy.

Runtime's lossless observer bridge retains the original error and all reservation,
routing and transaction callbacks. Target operations reuse the existing observed
replicated transaction, including exact hidden capture, publication, completion
and rollback. Failed invocation admission and unwinding close the observer without
refunding consumed allowance. A successful callback stages tentative evidence;
commitment remains the shared speculative driver's responsibility.

The focused neutral checks pass: twelve inspection tests, ten shared embedded-driver
tests and two target-capture tests. They cover phase/depth/width ordering, causal
logit edits, original error identity, generated-source admission before creation,
prefill/replay delivery failures, and target prefill/decode rollback and retry.

```sh
cargo test -j1 -p eredu-runtime --lib inspection::
cargo test -j1 -p eredu-architectures --lib speculative_execution::
cargo test -j1 -p eredu-runtime --test backend_independence prediction_target
```

The hook bridge alone does not establish bounded public capture. The separate
invocation admission, scheduler provenance, cumulative collector and public
installation described below complete that path for V3 sequential execution.
Public single-rank and distributed F32 prediction parameter/overlay acceptance is recorded below.
Ordinary generation retains its prediction-inspection gate; other speculative
strategies remain implementation gaps.

The new native observer-binding test passes on CPU (5.49 seconds) and Metal
(2.32 seconds). Each device runs five deterministic nonzero V3 cases with two
target and two prediction layers: disabled instrumentation, passthrough observation,
a channel mask at the last physical position, target-prefill failure and proposal
failure. Passthrough retains identical tokens, acceptance lengths and draft rows.
The mask changes the actual draft head and preserves all unselected channel values;
original and effective captures identify the exact edited position. Failures retain
the injected cause, and target-prefill failure publishes no token. These are native
hook/binding results, not public bounded-capture admission or distributed component
acceptance.

```sh
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  native_v3_internal_phases_reach_components_on_cpu -- --test-threads=1 --nocapture
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  native_v3_internal_phases_reach_components_on_metal -- --ignored --test-threads=1 --nocapture
```

The same build passes the ordinary V3 Metal scheduler regression (2.64 seconds)
and two-process CPU Ring prediction-target regression (7.19 seconds). The complete
neutral core/runtime check passes 238 core tests, 476 runtime tests and 71 backend
independence tests, including the new target transaction checks.


After the phase bridge, the full architecture run passes 564 unit tests (one
existing manual test ignored) and all 213 numerical tests in 48.35 seconds. The
existing outer-observer native regression also passes (2.05 seconds), preserving
its causal callbacks and failure behavior.


Portable validation after the bridge also passes all 59 backend conformance and
19 facade tests (one existing optional fixture ignored), the backend check without
default features, and every portable facade example. The support gate remains
disabled for public prediction-internal capture throughout these checks.

### Independent invocation geometry and speculative collection

`CapturePlan::admit_invocations` and `InterventionPlan::admit_invocations` retain
explicit `CaptureInvocationBounds`. At execution, `CaptureInvocationShape` supplies
physical batch/sequence/context extents independently from the prediction used by
capture schedules. Thus cached verification can have several physical rows at one
logical prediction coordinate. Existing ordinary admission retains its original
prefill/decode geometry. Payloads remain exact; masks and replacements do not
acquire broadcasting semantics. Re-admission, checkpoint restore and forks retain
the admission mode, and repeated invocations consume the existing cumulative
capture/intervention allowances.

The scheduler supplies `SpeculativeActivationOrigin` before each actual executor
operation. It includes request identity, canonical committed-token count, schedule
coordinate, a fixed-size digest of the exact assumed generated prefix and an
optimistic-work flag. The digest uses SHA-256 over the documented domain and
little-endian u32 IDs. It does not identify prompt contents independently of the
owning run. Canonical proposals are tentative even when `optimistic` is false.
The scope clears on failure and unwinding. No prefix hashing is added when the
executor has no internal observer.

Runtime `SpeculativeCaptureObserver` composes the existing `CaptureSession` and
`CaptureObserver` with borrowed native primitives. Architecture node ancestry
selects target versus prediction-depth applicability. Uninvoked selections become
explicit skips; uninvoked edits stay inactive. One collector owns one scheduler
request, and every queued invocation reserves its provenance/delivery envelope
before native work. `SpeculativeActivationCapture` distinguishes completed forward
work from accepted output tokens, retains exact physical shape, and exposes original
and effective evidence. Restore neither refunds accounting nor rewinds invocation
IDs. Native transform failures retain their original sources through the portable
`SpeculativeControlError`; admission failures remain typed `CaptureError` values.

`ControlledSpeculativeStep.activations` is the internal-evidence delivery field;
existing `captures` remains the one-row sampler interface. Internal records inherit
the controlled step's run/restore identity and trace transport limits. Public admission and installation now use a separate loaded speculative report.
Distributed producer agreement and context-axis supply remain integration work.
Explicit native collector binding is verified below; ordinary generation retains
its original prediction-inspection gate. Current neutral
checks pass 242 core and 487 runtime tests, including interleaved requests,
optimistic acceptance/rejection, actual masks/Scale evidence, sparse error sources,
and restore without refunds. Architecture checks pass 565 tests with one existing
manual test ignored; portable facade checks pass 59 conformance and 19 facade tests
with one existing optional test ignored. The runtime independence suite also passes
71 tests.

The bounded native phase fixture passes on MLX CPU (4.17 s) and Metal (2.27 s).
It uses deterministic nonzero weights with two target and two prediction layers,
existing `NativeCapture` transformations and the shared runtime collector. It
covers target prefill, prediction seeding, proposals and verification; checks
physical sequence shapes and scheduler origins; verifies encoded record bounds;
and confirms capture-only proposal logits and output tokens match ordinary
execution. A compact attention-channel mask zeros exactly the selected channel
at all physical positions, preserves every unselected value exactly and changes
the actual draft-head logits. Original/effective evidence comes from the shared
intervention path. A cumulative one-capture budget fails with typed capture
exhaustion during seed collection before any generated token is published.

```sh
cargo test -p eredu-backend-mlx --no-default-features --features metal --lib \
  native_v3_bounded_speculative_collector_cpu -- --test-threads=1 --nocapture
cargo test -p eredu-backend-mlx --no-default-features --features metal --lib \
  native_v3_bounded_speculative_collector_metal -- --ignored --test-threads=1 --nocapture
cargo test -p eredu-backend-mlx --no-default-features --features metal --lib \
  public_deepseek_v3_embedded_scheduler_executes_on_metal -- --ignored --test-threads=1 --nocapture
cargo test -p eredu-backend-mlx --no-default-features --features metal --lib \
  ring_two_process_deepseek_v3_mtp_target_tensor_parallel_opaque_session -- --ignored --test-threads=1 --nocapture
```

The ordinary Metal scheduler regression passes in 1.89 s; the two-process Ring TP
regression passes in 6.21 s. The bounded fixture explicitly supplies prediction
invocation applicability and the loaded session's intervention identity. It does
not turn conditional ordinary discovery into public speculative admission. The public installer and shared continuous/controlled entry points are now wired.
Their native acceptance is recorded separately from this explicit binding fixture;
partitioned invocation collection remains required integration work.


Public speculative activation plans use a separate loaded contract:

1. Obtain `LoadedModel::speculative_activation_discovery()`. Its bindings join
   each selectable node to the actual target or prediction depth; ordinary capture
   discovery still does not authorize prediction execution.
2. Build `SpeculativeActivationPlan` with captures, optional interventions and
   physical `CaptureInvocationBounds`, then call `prepare_speculative_activations`.
   Admission binds the exact source, selected execution, active overlay and session.
3. Supply the result in `ControlledSpeculativeOptions.activations`. The existing
   `capture` option still selects one-row sampler observations.
4. Use `with_controlled_chat_speculative` / `with_controlled_text_speculative`, or
   continuously deliver steps with `generate_observed_chat_speculative` /
   `generate_observed_text_speculative`. Both use the same advancement driver.

A scoped controller may drain `take_activation_evidence()` after cancellation or
a failed action. Those host envelopes retain run/epoch and invocation identity,
spend the ordinary trace budget and never make a failed session resumable. Public
installation is removed when the borrowed executor returns. The internal plan
keeps capture selections and allowances fixed for that scope. Sequential V3 now
supports complete native snapshots, prospective internal edits and isolated serial
branches across all three target weight residencies. Public prediction parameter
operations are verified below; other incomplete state profiles and distributed
invocation admission remain tasks.


The scoped public installer passes native backend acceptance on CPU (5.60 s) and
Metal (3.30 s). These tests use loaded discovery and immutable admission, exercise
preflight and cumulative failures, drain aborted evidence through trace accounting,
and verify that the next ordinary execution inherits no observer or edits.

Direct `LoadedModel` acceptance also passes CPU (6.79 s) and Metal (4.14 s) across
resident, host-layerwise and disk-streamed target execution. Its deterministic
nonzero SafeTensors fixture has MLA low-rank queries, routed/shared FFNs and two
prediction depths. It uses an exact three-token prefix, compares continuous and
controlled capture/intervention records, verifies channel masking and actual draft
logit changes, and restores ordinary baseline behavior on the next call.

```sh
cargo test -p eredu --no-default-features --features mlx,metal \
  --test native_execution_control \
  public_v3_internal_activations_match_continuous_and_controlled_cpu \
  -- --ignored --test-threads=1 --nocapture
cargo test -p eredu --no-default-features --features mlx,metal \
  --test native_execution_control \
  public_v3_internal_activations_match_continuous_and_controlled_metal \
  -- --ignored --test-threads=1 --nocapture
```

Current focused verification passes 244 core, 488 runtime and 566 architecture
tests (one existing architecture ignore), plus 61 backend-conformance and 19
portable-facade tests (one existing optional ignore). Facade conformance includes
continuous/controlled equality, stale-session rejection, bounded original failure
evidence and increasing capture usage through snapshot replay with a neutral
backend. Subsequent native sequential V3 snapshot acceptance is recorded below. The ordinary
Metal and two-process Ring TP regressions pass in 2.68 s and 7.20 s; these do not
establish distributed internal activation collection.


The same public CPU and Metal acceptance commands now also exercise complete
embedded sequential V3 snapshots (7.20 s CPU, 6.66 s Metal). After prefill, each
case saves a checkpoint, restores it twice and compares exact activation values,
intervention evidence and generated suffixes. Two serial siblings share the saved
immutable checkpoint; forcing a different next token in one leaves the parent and
other sibling unchanged. Invocation IDs, capture consumption and copy accounting
continue increasing. Both mask and capture-only cases cover resident, host-layerwise
and disk-streamed weights. The saved state includes both prediction replicas and
the capture tensor; transactional shallow checkpoints are not used as durable copies.
Two neutral architecture tests additionally prove that an unknown storage bound
prevents copying and that a late copy failure preserves live state and its original
error source. This first snapshot acceptance used fixed internal plans; the
additional prospective-edit acceptance below extends it to saved authority.


`ControlledSpeculativeSession::readmit_activation_interventions` now replaces future
internal edits at a drained canonical boundary, including the initial boundary
before prefill. Prepare alternate `SpeculativeActivationPlan` values against the
same loaded model before entering its borrowed controller. Preserve the original
capture selections, invocation bounds and limits; vary the intervention plan,
including an empty operation list to remove edits. Save a snapshot before the
forward being tested, install the alternate admitted plan, and advance normally.
Restore reinstates both model state and the saved internal plan. For first-prefill
trials, create a fresh run with the same exact token-ID prefix; speculative durable
snapshots currently require completed prefill and no pending proposals.

Re-admission rechecks the actual loaded discovery, preflights native costs against
spent capture usage, and charges retained host authority. It neither recomputes
earlier KV/prediction state nor refunds capture, transport or copy budgets. Capture
authority must be installed from run creation. Snapshot authority is scoped to its
collector and cannot be restored into another run. A prepared authority restore is
committed only after all native copies succeed.

The public CPU and Metal commands above now pass in 8.28 s and 6.62 s with
prospective-edit acceptance included. Across resident/host/disk execution, a child
adds or removes a channel mask, verifies effective values, saves and replays its
edited state, and returns to an unchanged parent and sibling. A stale admitted
execution is rejected before prefill. Neutral verification passes 244 core, 488
runtime and 568 architecture tests (one existing architecture ignore), plus 61
backend-conformance and 19 portable-facade tests (one existing optional ignore).
Runtime tests cover dropping prepared restoration, foreign collector rejection,
fixed capture allowances, edit removal/restoration and cumulative accounting.


### Effective prediction parameters and coordinated overlays

A loaded embedded execution now includes its actual prediction modules in
`LoadedModel::parameter_discovery()`. Use the same `query_parameter`,
`project_parameter`, `admit_parameter_overlay`, `activate_parameter_overlay` and
`remove_parameter_overlay` APIs as for target parameters. No separate prediction
parameter API or checkpoint-name parser is needed. Join IDs from
`ArchitectureDescriptor::component_scopes`: fusion weights and normalization gains
come from the residual base, decoder reads/writes from its component groups, and
prediction head weights/gains from its readout. Check each loaded parameter's
support, selected encoding and effective input transform before interpreting it.

The architecture enumerates physical module owners independently of proposal
depth. MLX retains exact prepared binding metadata and selected materialization
tasks beside each actual module. Queries borrow those slots under the ordinary
bounded parameter mechanism. A shared target embedding remains one target owner;
prediction fusion consumes the edited target embedding. V3's separately published
prediction heads remain distinct owners. The existing multi-parameter transaction
prepares all replacements before publishing target, bank and prediction handles.
Removal uses the same publication path and restores original handles. Source
artifacts remain unchanged.

For Inspector, place target and prediction edits in one admitted plan when an
experiment requires both. Activate it before creating the speculative controller;
then prepare activation admission against the resulting loaded identity. Use an
exact token-ID prefix for each new trial. Within that controller, ordinary durable
snapshots preserve the active overlay identity and complete prediction state.
An overlay transition changes the execution identity and invalidates incompatible
state/admissions; reusing an old KV cache does not become valid merely because the
edited parameter belongs to a prediction module. Parameter-query, copy, retention
and transaction limits are the existing cumulative limits, including failed
preflight and restoration rules described above.

Public F32 acceptance passes on CPU (7.31 s) and Metal (8.74 s). Each test covers
resident, host-layerwise and disk-streamed target execution with two target layers,
low-rank MLA, routed/shared gated FFNs, and two prediction depths. It queries every
effective parameter, normalization vectors and bounded matrix rows/columns, and
checks two-direction native projections against a host f64 contraction with
absolute tolerance `2e-6`. A coordinated edit touches target embeddings, attention,
FFNs, packed experts, prediction fusion and both prediction heads. An already-spent
budget rejects activation without changing usage.

The oracle edits a separate fixture's F32 source bytes using its published stacked
or split expert layout, without invoking production binding recipes. On prefixes
`[1, 2, 5]` and `[5, 2, 1]`, five-token greedy speculative runs compare exact tokens,
phase-tagged captures of target/prediction logits, normalized fusion inputs, fusion
outputs and normalized heads. Controlled snapshot replay matches the continuous
oracle exactly. Removal restores baseline captures/tokens, and original checkpoint
bytes remain unchanged. These fixtures verify numerical plumbing, not association
editing efficacy or released-checkpoint quality.

```sh
cargo test -j1 -p eredu --no-default-features --features mlx,metal \
  --test native_execution_control public_v3_prediction_parameters_and_overlays_cpu \
  -- --ignored --test-threads=1 --nocapture
cargo test -j1 -p eredu --no-default-features --features mlx,metal \
  --test native_execution_control public_v3_prediction_parameters_and_overlays_metal \
  -- --ignored --test-threads=1 --nocapture
cargo test -j1 -p eredu-architectures --test reference_numeric prediction_adapter \
  -- --test-threads=1
```

The two neutral prediction adapter tests pass. The V3 case edits an actual head
through the new module visitor, verifies exactly doubled logits and unchanged
hidden output, restores the head, then continues its existing causal-mask and retry
checks across query forms, serial/TP1 execution, both depths and three steps.
The native construction regression passes V3, Inkling, Qwen hybrid dense/MoE and
Nemotron (5.81 s). The existing Qwen hybrid dense/MoE/FP8 speculative scheduler
regression also passes (2.14 s), covering selected FP8 companion bindings. V4
DSpark and V3 two-process CPU Ring scheduler regressions pass
in 8.16 s and 5.07 s. Those construction/scheduler checks do not establish public
prediction parameter overlays or component invocation collection on distributed
executions. Other-family component hooks, additional selected encodings and those
distributed acceptance paths remain explicit implementation/validation work.

```sh
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  public_replicated_prediction_variants_install_only_the_neutral_extension \
  -- --test-threads=1 --nocapture
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  ring_two_process_deepseek_v4_dspark_tensor_parallel_neutral_scheduler \
  -- --ignored --test-threads=1 --nocapture
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  ring_two_process_deepseek_v3_mtp_target_tensor_parallel_opaque_session \
  -- --ignored --test-threads=1 --nocapture
```


After the parameter-owner change, the complete library checks pass 244 core, 488
runtime and 568 architecture tests (one existing architecture ignore). Portable
facade verification passes 61 backend-conformance and 19 portable-facade tests
(one existing optional ignore). Native builds emit the existing large unwind-table
linker warning; no new build warning was introduced.

### Complete pooling snapshots and long-context DSpark replay

The ordinary speculative controller now supports complete V4 sequential and DSpark
pooling-state snapshots through the same public APIs as V3. Supply bounded
`ControlledSpeculativeOptions::snapshots`, advance through prefill, and save only
when `can_snapshot()` reports a canonical boundary. A proposal-only step does not
necessarily establish that boundary. Restore preserves target state, both prediction
replicas, capture tensors and sampler state; it never refunds cumulative copy costs.

Native copies preserve logical elements of strided arrays, local keys, all pending
value/gate windows, pooled outputs and both overlap buffers. Paged copies retain
the bounded pool but receive independent catalogs, shared once within each complete
copied state. Estimates include the key-only persistence sentinel and catalog growth.
A focused cache fixture covers two layers and zero, one or two pooling streams with
both resident and paged local keys, interleaved siblings, independent recomputation,
transaction rollback and prediction-target forks. It passes in 0.28 s.

Public CPU and Metal tests each pass six cases: sequential V4 and DSpark across
resident, host-layerwise and disk-streamed weights. The final combined rerun passes
both tests in 20.38 s after all cache fixes. A deterministic
nonzero F32 fixture has two target layers with compression ratios 4 and 128 and two
local-only prediction layers. Its exact 255-token prefix cycles IDs 1 through 15;
seven greedy generated tokens cross both pool completion boundaries. Full-vocabulary
sampler records and tokens match exactly between continuous generation, controlled
execution and two reusable restores. A changed child saves and replays independently;
the parent and second child remain unchanged. A one-byte snapshot allowance produces
a typed limit error without changing usage, and generation still completes.

This acceptance exposed execution defects. Transaction rollback and explicit
pooling forks copied raw backing memory from strided windows; they now retain
immutable views or copy compact logical arrays on the owning stream. DSpark also
sized a bidirectional mask by clipping the frontier to the single-token local window,
although a multi-token append can return bounded history plus the entire block.
All six architecture proposal routes now expand the mask using the returned key
extent. A neutral numerical long-context regression checks finite logits and exact
transaction replay; all 568 architecture library tests pass (one existing ignore).

A longer two-rank DSpark fixture exposed another cache defect: paged sliding fetch
discarded keys outside the final window before returning the whole prefill span.
It now assembles bounded past plus all submitted keys before persistent eviction.
Blockwise attention retains visible blocks until the complete native attention call
finishes, then evicts history. Eighteen native ordinary/key-only cache cases cover
windows 1/3/8, block sizes 1/2/4 and successive 17/1/3/11/2-token submissions using
strided nonzero inputs and an independent exact-history oracle. Both direct paging
tests pass (0.08 s); append-failure rollback also passes (0.03 s). A scalar reference
verifies GQA, softcap, sink/no-sink and full/sliding attention on 9- and 7-token chunks
with finite device/host budgets (0.10 s).

The final two-process CPU Ring DSpark regression passes with a 17-token prefix,
seven generated tokens, fused proposals, fresh-lane replay and invalid-token fencing
(0.45 s). V4 pipeline persistence/MTP also passes (0.27 s). These tests exercise the
native scheduler and cache paths; they do not establish distributed component capture
or snapshot acceptance.

```sh
cargo test -j1 -p eredu --no-default-features --features mlx,metal \
  --test native_execution_control public_pooling_prediction_snapshots_cpu \
  -- --ignored --test-threads=1 --nocapture
cargo test -j1 -p eredu --no-default-features --features mlx,metal \
  --test native_execution_control public_pooling_prediction_snapshots_metal \
  -- --ignored --test-threads=1 --nocapture
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  isolated_pooling_snapshots_copy_all_streams_and_preserve_interleaved_siblings \
  -- --test-threads=1 --nocapture
cargo test -j1 -p eredu-architectures --test reference_numeric \
  deepseek_execution_graphs_run_target_mtp_and_dspark_transactions \
  -- --test-threads=1
cargo test -j1 -p eredu-architectures --lib
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  paged_sliding -- --ignored --test-threads=1 --nocapture
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  paged_score_softcap_matches_scalar_masked_gqa_through_chunked_prefill \
  -- --ignored --test-threads=1 --nocapture
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  ring_two_process_deepseek_v4_dspark_tensor_parallel_neutral_scheduler \
  -- --ignored --test-threads=1 --nocapture
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  ring_two_process_deepseek_v4_pipeline_persistence_and_mtp \
  -- --ignored --test-threads=1 --nocapture
```

These synthetic tests establish pooling state and sampler-capture behavior. They do
not establish released-checkpoint quality, internal V4 component hooks, V4 parameter
overlays or distributed component/snapshot acceptance. Those remain separate work
in the support matrix.


### Independently shaped partition invocations

The neutral partition protocol accepts the original `admit_invocations` authority
and exact `CaptureInvocationShape`. Producer projections, generated factories,
receipts, sparse placement and final assembly all validate that geometry. Schema-3
receipt/evidence contexts retain batch, physical sequence and optional context
separately from the scheduled prediction and forward epoch. All-rank coordination
rejects a context-only disagreement even if it leaves the selected shape unchanged.
No fallback converts an independently shaped plan into ordinary decode geometry.

The shared partition observer uses the invocation's capture and intervention scope
masks. Selected edits affect actual local tensors before downstream consumption;
replicas and empty shards participate in source and hook completion. Evidence and
captures remain unpublished until final commit. Invocation metadata is reserved
for every rank, and restoration preserves cumulative charges and monotone epochs.
Architecture-owned layouts implement both dense and routed geometry projection;
backends continue to supply native transformations and transport.

Three-rank conformance compares slice, summary and histogram assembly against
whole-tensor capture for physical widths 5, 2, 1 and 5 at prediction 4. It covers
aborted work, restoration, exact evidence serialization, idle producers and
geometry mismatch before native transformation. Independent summary comparison
uses the existing absolute tolerance `2e-15 * max(reference RMS, 1)` for the mean
and RMS, accounting for reduction ordering; same-layout
replay and cross-rank records are exact. A five-rank observer fixture applies
compact keep-only and Scale edits with changing applicability, checks actual local
values and before/after evidence, and exercises nonexporting replicas, an empty
shard, an inactive pipeline rank and replay without refunded allowances.

```sh
cargo test -j1 -p eredu-core -p eredu-architectures --lib
cargo test -j1 -p eredu-runtime --lib
cargo check -j1 -p eredu-backend-mlx --no-default-features
```

These checks pass 244 core, 568 architecture (one existing ignored), and 492 runtime
tests. After the borrowed-observer integration below, the runtime suite passes 493
tests. Backend compilation passes with both no default features and Metal. Nine
native partition regressions pass (0.54 s), including CPU/Metal fragment, exchange
and reduction checks. The new loaded native fixture runs prefill width 2 followed
by cached widths 3 and 1 at prediction 4, retaining exact context lengths 2, 5 and
6. Partition and whole-tensor collection produce exactly equal component payloads
and logits on both CPU and Metal. Its transport is singleton; it does not establish multiprocess
speculative support. Both sparse CPU/Metal live-session tests pass (0.17 s), and
the ordinary four-process Qwen3 TP/PP component-capture regression passes (2.42 s).

```sh
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  native_partition_ -- --include-ignored --test-threads=1 --nocapture
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  native_sparse_live_session_ -- --include-ignored --test-threads=1 --nocapture
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  ring_public_component_capture_qwen3_tensor_pipeline -- --ignored --test-threads=1 --nocapture
```

This neutral invocation support now feeds the selected V3 prediction producers
and native factory described under "Loaded partitioned prediction components"
below. Other prediction families still require their own complete hook acceptance.


The shared internal invocation wrapper also supports `with_activation_observer`,
which keeps one borrowed observer alive across the complete forward. Its default
uses the existing local collector without allocation. A five-rank fixture lends a
real partition observer with producer/source work, edits, evidence and completion;
separate admission, execution, completion and unwind failures verify that borrowed
work is dropped before outer finalization. Architecture operations receive only
the ordinary activation interface, while the enclosing owner retains admission
and speculative provenance. The actual speculative collector now has a partition provider and shared
auxiliary transactions. Selected V3 prediction layouts and public native binding
are verified under "Loaded partitioned prediction components" below.


Final validation of the shared borrowed-observer path passes 568 architecture
library tests (one existing ignored), 493 runtime tests, and 45 numerical component
tests (5.19 s). The explicit V3 bounded collector passes CPU and Metal tests (0.85 s
combined). Public V3 activation tests pass on both devices across resident,
host-layerwise and disk-streamed execution (7.03 s combined), including continuous/
controlled parity, snapshots, prospective intervention changes, saved-authority
replay and budget failures. Portable facade validation passes 61 conformance tests
and 19 facade tests (one existing optional test ignored).

```sh
cargo test -j1 -p eredu-architectures --test reference_numeric components::
cargo test -j1 -p eredu-runtime -p eredu-architectures --lib
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  native_v3_bounded_speculative_collector_ -- --include-ignored --test-threads=1 --nocapture
cargo test -j1 -p eredu --no-default-features --features mlx,metal \
  --test native_execution_control public_v3_internal_activations_ \
  -- --ignored --test-threads=1 --nocapture
cargo test -j1 -p eredu --no-default-features \
  --test portable_facade --test backend_conformance
```


### Auxiliary prediction observation transactions

Observed prediction prefill, proposal and replay now use the ordinary shared
session lifecycle. One transaction spans every depth inside a prediction-prefill
invocation. Observer preparation precedes mutation, each operation agrees its
checkpoint before execution, and completed records wait for native completion,
receipt agreement and the final commit. The architecture enumerates all retained
state in sequential, pooling and model-state profiles; MLX completes those actual
values together with deferred token checks and relevant proposal outputs.
Unobserved execution skips this observation transaction and enumeration.

The error bridge preserves native causes through `BackendFailure` and
`safemlx::Exception::from_source`. Two-rank neutral tests inject preparation,
coordination, initial and second-depth checkpoint, execution, completion and
delivery failures, plus mismatched participation. They check matching rank
protocols, no premature publication, full target rollback, original completion
causes and final-commit abort/indeterminate behavior. An uncertain decision keeps
the existing mutation fence. Prediction state remains owned by the existing
architecture/scheduler recovery path.

The full runtime suite passes 493 library tests, 73 backend-independence tests and
three binding-placement tests. Both numerical prediction-adapter tests pass;
the MLX adapter checks without default features. The architecture library passes
568 tests with one existing ignored test. Native CPU/Metal dependency-completion
and original-error tests pass (two tests, 0.20 s). The six V3 internal-hook,
bounded-collector and public-admission regressions pass (2.54 s). The two-process
V4 DSpark tensor-parallel scheduler passes (0.66 s), exercising the new operation
checkpoint agreement over real collectives. Public V3 continuous/controlled
activation parity passes on CPU and Metal (two tests, 6.88 s) across
resident/host/disk execution, including snapshots, prospective masks, saved-authority replay and budget failure.
Distributed internal producer/transport binding and V4 internal component hooks
remain open.

```sh
cargo test -j1 -p eredu-runtime
cargo test -j1 -p eredu-architectures --test reference_numeric prediction_adapter
cargo check -j1 -p eredu-backend-mlx --no-default-features
cargo test -j1 -p safemlx --no-default-features --lib \
  rust_exception_source_survives_nested_native_error_domains
```

```sh
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  prediction_completion_ -- --include-ignored --test-threads=1 --nocapture
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  native_v3_ -- --include-ignored --test-threads=1 --nocapture
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  ring_two_process_deepseek_v4_dspark_tensor_parallel_neutral_scheduler \
  -- --ignored --test-threads=1 --nocapture
cargo test -j1 -p eredu --no-default-features --features mlx,metal \
  --test native_execution_control public_v3_internal_activations_ \
  -- --ignored --test-threads=1 --nocapture
```

Portable facade checks pass 61 backend-conformance tests and 19 portable-facade
tests (one existing optional test ignored). Formatting and whitespace checks pass.

```sh
cargo test -j1 -p eredu --no-default-features \
  --test portable_facade --test backend_conformance
```

The real speculative collector now composes `PartitionCaptureBackendProvider` with
retained layout and transport owners. A five-rank fixture verifies scoped causal
edits, replicas, an empty shard, failed publication and nonrefundable replay. A
nine-rank sparse fixture verifies actual routed edits, idle owners, native failure
sources and missing-chunk rejection. Runtime library coverage passes 495 tests.
Native CPU/Metal partition coverage passes 11 tests (0.63 s), including the actual
provider capturing nonzero components across prefill and multirow cached decode.
The fixture transport is singleton. V3 internal regressions pass six tests (2.60 s);
public continuous/controlled regressions pass two tests (6.67 s) across all three
residency modes. These validate the reusable provider and preserve the existing
public local path. Selected V3 prediction layouts and the public distributed factory
are verified below. Reproduce the provider checks with the runtime,
`native_partition_`, `native_v3_` and `public_v3_internal_activations_` commands above.


### Loaded partitioned prediction components

The loaded V3 speculative activation report now uses the exact prediction layout
retained by successful module materialization. Its public admission and collector
factory combine target placement with the independent prediction scope, then use
the existing bounded partition observer and shared controlled inference driver.
Prediction scalar columns follow the same TP compiler as their weights. Prediction
banks and score heads execute on replicas over the other axes; publication removes
redundant replicas while interventions reach every actual consumer.

Prepared resident prediction banks carry their scalar coordinates into one provider
scope spanning all native chunks. Source preparation and failure votes finish before
the next tensor reduction. Serial capture uses its ordinary local metadata. Sparse
original/effective evidence retains the same meaning in both paths.

A predictor that borrows the target embedding needs that parameter on every pipeline
stage where it executes. Neutral auxiliary static-role declarations now retain those
storage dependencies. Physical binding, effective parameter coordinates and residency
estimates include the replicas; target input and readout observation ownership remain
separate. Disabled or external drafting removes auxiliary requirements before target
selection. The four drafting modes and storage-versus-invocation distinction have
neutral behavioral coverage.

The real CPU Ring fixture passes all **21** combinations of TP, PP, EP, TP×PP,
TP×EP, PP×EP and TP×PP×EP with resident, host-layerwise and disk-streamed targets
(42.88 s). It uses two, four or eight processes and nonzero F32 SafeTensors weights.
Every case compares four generated tokens, phase/origin/invocation evidence, target
logits, prediction normalization/head values, and dense/sparse original/effective
components with an ordinary local reference. Compact attention-channel deletion and
routed-unit deletion each change prediction scores and reproduce that reference.
Tensor/unit tolerance is `2e-5 * max(abs(reference), 1)`; route identities are exact.
Invalid target tokens preserve the input rank's cause, propagate failure to other
stages, and fence retries. These are synthetic integration fixtures; distributed GPU
execution and additional prediction encodings are not established by this run.

Architecture tests pass 569 with one existing ignored test; runtime passes 496.
The prepared V3 numerical scope test checks serial/TP1 callbacks, state replay and
failure cleanup. Six native CPU/Metal V3 tests pass (2.65 s), preserving local public
admission and continuous/controlled behavior. Portable conformance/facade pass 61
and 19 tests, with one existing ignored facade test.

```sh
cargo test -j1 -p eredu-runtime -p eredu-architectures --lib
cargo test -j1 -p eredu-architectures --test reference_numeric \
  v3_typed_extension_hooks_preserve_scope_state_and_retry -- --test-threads=1
cargo test -j1 -p eredu-backend-mlx --features metal --lib \
  ring_deepseek_v3_mtp_component_capture_residency_and_parallel_matrix \
  -- --ignored --test-threads=1 --nocapture
cargo test -j1 -p eredu-backend-mlx --features metal --lib \
  native_v3_ -- --include-ignored --test-threads=1 --nocapture
cargo test -j1 -p eredu --no-default-features \
  --test portable_facade --test backend_conformance
cargo check -j1 -p eredu-backend-mlx --no-default-features
```

V4 internal components, remaining embedded state profiles and additional encodings
remain separate implementation/validation work on the checklist.


### Distributed prediction parameter edits

The same 21-case F32 CPU Ring matrix now also passes loaded prediction parameter
queries, projections, coordinated overlays and controlled replay (148.02 s).
Discovery joins the retained ordinary and auxiliary materialization tasks. Each
parameter uses its actual architecture-owned placement: prediction units follow
their tensor shards and replicas over pipeline/expert axes; borrowed target
embeddings retain their separate storage ownership. Only the requested parameter
group is lowered for a query, after metadata reservation.

Every effective slot is queried through a bounded row, with matrix columns and
two signed native projections checked against a local runtime and host reduction.
One atomic edit spans target and prediction weights, norms, both gated read paths,
packed expert banks, fusion and separate prediction heads. A peer's insufficient
budget and a late publication rejection preserve the baseline. Active edits change
both target and prediction scores on nonzero inputs, match a separately loaded
local runtime on two exact prefixes, and retain their overlay identity in partition
evidence. Saved states restore and exchange isolated siblings under the active
version without refunding copy budgets. A failed removal preserves the active
version; successful removal restores original queries, captures and generated tokens.
The independent edited-checkpoint oracle remains the single-rank validation above;
this distributed fixture specifically verifies placement and replica publication.

The command is the existing
`ring_deepseek_v3_mtp_component_capture_residency_and_parallel_matrix` invocation
above. Native values use `2e-5 * max(abs(reference), 1)` tolerance, host projection
reductions use absolute `2e-6`, and restored F32 query values are exact. The neutral
`prepared_prediction_parameter_maps_preserve_tp_shards_and_pp_ep_replicas` test
compares eight-rank maps with the materialization compiler and checks cold missing
placement, absent targets, invalid ranks, zero budgets and unresolved source identity.
All 570 architecture tests pass (one existing ignore), as do 61 portable conformance
and 19 facade tests (one existing ignore), six CPU/Metal V3 regressions (2.45 s),
and the backend build without default features. Additional prediction encodings
and distributed GPU execution remain outside this recorded F32 validation.


### Affine prediction parameters

The same complete distributed component/parameter experiment passes all **21**
parallel/residency cases with SafeTensors weights converted to affine 4-bit,
group size 32 (231.54 s). The fixture has hidden width 64, four attention heads,
one dense and one routed target block, and one embedded prediction block. Weights
and prefixes are deterministic and nonzero. Each parameter uses its name hash
(FNV initial value `2166136261`, multiplier `16777619`) as the fixed seed;
`write_deepseek_config_fixture` defines the element mixing and scaling.

Preparation applies the selected auxiliary formats to the actual prediction
modules. It retains separate source and executable layouts when conversion is
needed. The shared native quantizer reads rank-local source tiles and the ordinary
binding builder consumes the resulting local packed weights and companions.
Conversion reports include this auxiliary work. Loading does not rewrite the
checkpoint.

The fixture checks effective weight rows and columns, signed projections, causal
channel/unit deletion, coordinated target/prediction overlays, peer-budget and
publication rejection, snapshot/sibling replay, failed removal and exact restoration.
It compares each distributed run with a separately loaded local runtime using the
same selected encoding. Standalone quantization companions report unsupported
queries/edits; the effective matrix exposes their decoded meaning through its
ordinary parameter operations. Editing promotes affected parameters to F32 using
the existing bounded overlay policy. It does not requantize the edit or change
untouched parameters. Original packed owners remain available for removal.

Native captures use `2e-5 * max(abs(reference), 1)` tolerance; signed host reductions
use absolute `2e-6`. Effective query values agree exactly with the local runtime,
updated regions equal the original decoded values plus the F32 edits, and removal
restores exact original query values. These are synthetic placement and lifecycle
checks, complementing the independent decoded/edit references above. They do not
establish other prediction encodings or distributed GPU execution.

```sh
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  ring_deepseek_v3_mtp_affine_components_and_parameters_matrix \
  -- --ignored --test-threads=1 --nocapture
cargo test -j1 -p eredu-architectures --test reference_numeric \
  v3_prepared_prediction_formats_preserve_sources_and_packed_layouts \
  -- --test-threads=1
```

The focused native conversion test compares the same artifact loaded with and
without embedded drafting. It verifies additional transformed weights and source/
output bytes for the predictor, a planned peak within the admitted conversion
allowance, and byte-for-byte unchanged SafeTensors storage. It also exercises the
retained source graph after disabled drafting removes auxiliary requirements.
All seven CPU/Metal V3 regressions pass (2.88 s), and the existing two-rank V4
sequential predictor regression passes (0.81 s) with the shared binding path.
The four-mode neutral handoff test reproduced the disabled-drafting mismatch before
the fix and now compares complete selected/construction requirements. Architecture
coverage passes 570 tests with one existing ignore; portable coverage passes 61
conformance and 19 facade tests with one existing ignore. The backend check without
default features also passes.

```sh
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  native_v3_ -- --include-ignored --test-threads=1 --nocapture
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  ring_two_process_deepseek_v4_mtp_target_tensor_parallel_opaque_session \
  -- --ignored --test-threads=1 --nocapture
cargo test -j1 -p eredu-architectures --lib
cargo test -j1 -p eredu --no-default-features \
  --test portable_facade --test backend_conformance
cargo check -j1 -p eredu-backend-mlx --no-default-features
```

The complete F32 component/parameter matrix was rerun after the shared
materializer and drafting-handoff changes: all 21 CPU Ring cases pass (154.06 s),
including every combined axis and all three residency modes.

The prepared-format numerical fixture now covers F32, affine and MXFP4 modules,
including unchanged floating source geometry and the exact one- or two-companion
packed destination declarations. The scalar fixture retains decoded logical
matrices; the physical packing assertion uses the retained execution layout. Its
ordinary MXFP4 linear visitor now retains scale handles, matching its existing
grouped visitor. All 214 numerical reference tests pass (144.15 s).

The public V3 continuous/controlled internal-activation regression also passes on
CPU and Metal across resident, host-layerwise and disk-streamed targets (two tests,
7.96 s), preserving the facade integration after these materialization changes.

```sh
cargo test -j1 -p eredu --no-default-features --features mlx,metal \
  --test native_execution_control public_v3_internal_activations_ \
  -- --ignored --test-threads=1 --nocapture
```


### MXFP4 prediction parameters

The identical 64-wide nonzero fixture and complete component/parameter experiment
also pass all **21** CPU Ring parallel/residency cases with load-time MXFP4
conversion (214.22 s). No additional production path was needed: selected auxiliary
formats, source layouts, bounded transformation and exact local bindings use the
same mechanisms as affine conversion. MXFP4 supplies one scale companion per
matrix; it has no affine-bias companion.

The matrix uses the same exact effective queries, signed-reduction tolerance,
component capture/mask tolerance, two edited prefixes, peer failure injection,
active-overlay snapshots/siblings and removal checks described above. Updates act
on decoded parameter values and promote the affected parameters to F32 under the
ordinary overlay allowance. Removal restores their packed owners and exact queried
values. This establishes the MXFP4 CPU distributed path; distributed GPU validation
and other prediction encodings remain separate work.

```sh
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  ring_deepseek_v3_mtp_mxfp4_components_and_parameters_matrix \
  -- --ignored --test-threads=1 --nocapture
cargo test -j1 -p eredu-architectures --test reference_numeric -- --test-threads=1
```

### V4 grouped channel and hyper-connection work in progress

Descriptor schema 6 / component schema 5 distinguish actual V4 attention channels
from the narrower grouped projection output. For `C` channels, `G` output groups
and rank `R`, the first effective matrix has shape `[G*R, C/G]`; the final matrix
has shape `[hidden, G*R]`. `ComponentGroup::write_column` joins a scalar identity
to the selected first-factor column `[R,1]` and final-factor columns `[hidden,R]`.
Use ordinary admitted parameter queries or two bounded contractions with those
regions. Discovery does not allocate a dense product or grant query authority.

The channel hook follows inverse rotary transformation of the aggregated values.
A channel intervention changes the input consumed by the first factor. Its actual
multiplication input is `write_input_projection.input`, with axes
`[batch, group, sequence, channel]`. The grouped output and final multiplication
input have `[batch, sequence, projection]` geometry. These separate observations
expose selected input rounding at both factors; a fixed matrix product cannot
explain that rounding alone. The grouped input keeps its original shape so
capture admission can precede any generated diagnostic copy.

Target blocks now share one serial/parallel/provider driver. It observes normalized
attention/FFN inputs, true attention channels, shared SwiGLU units, and existing
hyper-connection `pre`, `post` and `combination` values. The combination matrix is
indexed by input stream then output stream. Its coefficients depend on the current
residual. A collapsed-input intervention does not retroactively recompute the
coefficients already produced at that boundary. Shared V4 experts use unclamped
SwiGLU; the configured clamp applies to routed experts. Block input/output tensors
retain the stream axis and have distinct effective-value observations.

The neutral reference tests below pass (2 tests, 0.09 s). They exercise local,
compressed/indexed and ratio-128 attention, prefill and two cached decode steps,
exact ordinary/observed parity, deletion/keep-only masks, independent grouped-write
reconstruction, both hyper residual cycles, and survivor recomputation after an
upstream channel deletion. The single-rank collective driver preserves its two
reductions and captured values. This is not multi-rank acceptance. The architecture
suite passes 573 tests with one existing ignored test.

```sh
cargo test -j1 -p eredu-architectures --test reference_numeric components::compressed_v4 -- --nocapture
cargo test -j1 -p eredu-architectures --lib
```

The shared native grouped-projection observer separately passes CPU and Metal
checks for dense and FP8 inputs, deferred rejection and width overflow; existing
vocabulary-input observer tests and the V4 target TP Ring regression also pass.
This initial target milestone was followed by the prediction scopes, public
overlays, stream propagation and final-collapse evidence described below. Full
V4 acceptance remains open for the outstanding distributed and encoding cases
in the current support matrix.

Public native V4 target validation now passes on CPU and Metal across resident,
host-layerwise and disk-streamed loading (six cases in two tests, 1.58 s). The
private F32 fixture has deterministic mixed-sign, distinct read/write parameters
and embeddings. Controlled generation captures prefill and two cached decode
steps for baseline, deletion and keep-only trials. Queries of the actual loaded
factors reconstruct the channel write with absolute tolerance `2e-5`, and the
regions returned by `ComponentGroup::write_column` match the corresponding full
small-factor queries. Masks affect only the requested prefill position. This
acceptance covers the target's grouped channels and coefficient delivery; it leaves the broader overlay and prediction-scope work open. The subsequent
readout validation below extends this target acceptance.

```sh
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib native_v4_grouped_component_capture_ -- --ignored --test-threads=1 --nocapture
cargo test -j1 -p eredu-architectures --test reference_numeric v4_ -- --nocapture
```

All nine V4 numerical regressions pass (0.09 s), including target TP2 and prefill
chunking. The existing native V4 target TP Ring regression passes (1.01 s).


The target readout now carries `ComponentReadoutEquation::stream_residual`.
`ComponentStreamResidual` declares the initial embedding broadcast, ordered
attention/FFN cycles, and final learned collapse. For each cycle, at the measured
coefficients of that trial:

```text
next[i,h] = post[i] * write[h] + sum_j combination[j,i] * incoming[j,h]
collapsed[h] = sum_j pre[j] * incoming[j,h]
readout.residual[h] = sum_j head_coefficients[j] * readout.streams.effective[j,h]
```

The sequence and batch axes are implicit here. `write` is the complete effective
sublayer output after parallel reduction. The grouped attention write captured
before that reduction retains its separate sum-combination policy. Partition
ownership for cycles follows the declared coefficient parameters; embedding and
head evidence follow input/output execution ownership. Matrix observations use
input-stream then output-stream axes, and all stream extents are checked.

Inspector can propagate separate signed embedding and sublayer-write terms through
these measured matrices, then apply the declared final collapse and ordinary
measured RMS readout. Remeasure the matrices after every intervention or overlay.
They describe the current forward computation; they are not causal derivatives.
At a mutable residual boundary, add the effective-minus-original value as a
separate correction term. At rounded arithmetic boundaries, retain the measured
reconstruction residual as a numerical correction rather than attributing it to
an arbitrary component. The actual final projection input remains separate evidence.

Embedding hooks run before the original stream broadcast. Resident, bounded and
partitioned target traversal use the same observed block and readout drivers.
The generic `TensorValueObserver` borrows the actual final-collapse coefficients;
its existing projection-observer alias preserves API compatibility. Disabled
instrumentation does not create extra coefficient tensors or diagnostic views.

The V4 numerical component suite now passes three tests (0.10 s), including
whole-model reconstruction of two selected scores and their difference through
prefill plus two cached decode steps. Separate signed base/write terms reconstruct
all six stream cycles within `2e-4`, and selected scores/differences within `3e-4`.
Observed and ordinary resident scores agree exactly. The full neutral suites pass
247 core tests and 575 architecture tests (one existing ignored test).

Observed unit-boundary ownership is explicit in the shared traversal contract.
V4 emits its input/output hooks internally, so output edits take effect before
retained target captures are formed. The outer traversal omits duplicate delivery.
The numerical V4 suite passes 11 tests (0.17 s), including TP2 complete-stream
agreement, exact local channel coordinates and summed partial writes, plus both
pipeline cuts with provider and tensor-parallel variants, embedding/shared-unit
interventions, prefill and two cached decode steps. The expanded public native
score-readout check passes after this ownership fix (two tests, six CPU/Metal
residency cases, 1.90 s). Its prior run rejected duplicate block-output evidence.


The native target test now queries the actual loaded RMS gain and head matrix and
captures every stream cycle plus the actual final collapse coefficients. It
reconstructs collapsed inputs within `2e-5` and stream updates, selected scores and
score differences within `3e-5` in every baseline/deletion/keep-only trial across
resident, host-layerwise and disk-streamed execution. The CPU/Metal coefficient
mechanism tests also pass (two tests, 0.53 s), including exact ordinary/observed
parity and rejection preserving the original typed error source.

Shared-path regression checks pass after the boundary-ownership change: all five
V4 two-process Ring scenarios (4.02 s), the V3 CPU/Metal component scenarios
(1.03 s), 496 runtime unit tests, 73 backend-independence tests, and portable facade
and backend conformance. The full numerical suite passed 218 cases; its sole new
custom-executor test had an incorrect expectation of effective companions from
ordinary outer hooks. Correcting that test expectation produced a passing targeted
rerun, covering all 219 cases without further production changes.

```sh
cargo test -j1 -p eredu-runtime --lib --test backend_independence
cargo test -j1 -p eredu --no-default-features --test portable_facade --test backend_conformance
cargo test -j1 -p eredu-architectures --test reference_numeric
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib ring_two_process_deepseek_v4_ -- --ignored --test-threads=1 --nocapture
```

Sequential V4 prediction scopes declare `ComponentResidualBase::ProjectedSum`.
Each input names its actual source observation, normalization, effective
normalized value, selected matrix and actual multiplication input. The embedding
projection has rank three and broadcasts along the declared stream axis after
its intervention; the hidden projection retains rank four. Add their effective
outputs after that broadcast. This is a sum of independent projections, not a
single concatenated matrix. The scope's stream equation then supplies the two
measured decoder cycles and the learned final collapse. Use its final RMS gain
and the shared target vocabulary head to project signed terms into the declared
score stage. These input-dependent coefficients must be captured for each trial.

The predictor's outer `mtp.{depth}.input` and `.output` have distinct effective
companions; inner decoder boundaries use `mtp.{depth}.decoder`. An edit to the
outer output changes the state returned for a later prediction depth, after the
current prediction head has run. Editing the readout's stream input affects the
head branch. Prepared prediction hooks use the existing prefill/proposal/replay
and controlled-session drivers. DSpark uses the separate fused/context equations
and acceptance tests described below.

The sequential implementation passes `cargo test -j1 -p eredu-architectures --lib
prediction -- --nocapture` (32 tests, 0.06 s), including source/broadcast timing
validation and all seven TP/PP/EP producer layouts. `cargo test -j1 -p
eredu-architectures --test reference_numeric v4_ -- --nocapture` passes 14 tests
(0.17 s). These cover four observed prediction entry points and their complete
model traversals with prefill plus two cached decodes, exact no-edit parity,
projection deletion, unique outer/inner boundary delivery, and descriptor-driven
score/difference reconstruction within 3e-4. The predictor's parallel traversal
fixture uses a singleton collective; this result is not a native multi-rank
prediction acceptance claim.

Public native sequential V4 checks pass on CPU and Metal in resident, host-layerwise
and disk-streamed execution. Activation tests cover channel deletion, continuous
versus controlled parity, repeatable snapshot replay, independent sibling branches,
stale admission rejection and monotone budgets (six device/residency cases, 14.21 s).
The F32 equation tests query the effective fusion matrices, final gain and shared
head through the facade, reconstruct selected scores and their difference within
3e-5, and compare coordinated fusion/attention/FFN parameter overlays with an
independently edited checkpoint for two token prefixes. Removing the overlay
restores baseline captures exactly; the source checkpoint bytes are unchanged
(six device/residency cases, 12.55 s). These fixture results verify the operations;
they do not establish useful semantic editing or native multi-rank support.

Reproduce with:

```sh
cargo test -j1 -p eredu --no-default-features --features mlx,metal \
  --test native_execution_control public_v4_internal_activations_ \
  -- --ignored --test-threads=1
cargo test -j1 -p eredu --no-default-features --features mlx,metal \
  --test native_execution_control public_v4_prediction_equations_ \
  -- --ignored --test-threads=1
```

The complete numerical suite also passes after these changes: 221 tests in
63.95 s using the test binary with `--test-threads=4`. Later DSpark integration,
prediction encodings and expanded component-mask results are recorded below.

Shared regression checks pass after sequential prediction integration: 247 core
unit tests, 578 architecture unit tests (one existing ignored test), 496 runtime
unit tests, 73 backend-independence tests, 61 portable backend-conformance tests
and 19 portable-facade tests (one existing ignored test). Public V3 CPU/Metal
prediction activation and parameter-overlay regressions pass in 13.07 s. Public
pooling snapshots for sequential V4 and DSpark pass on CPU/Metal across all three
residency modes in 22.08 s; that snapshot regression does not add DSpark internal
component support.

The existing five native V4 distributed regressions also pass after integration
(4.20 s), covering tensor-parallel sequential prediction, DSpark scheduling,
pipeline persistence and prepared speculative capabilities. Reproduce with
`cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib
ring_two_process_deepseek_v4_ -- --ignored --test-threads=1`. These ordinary
execution regressions are separate from the multi-rank component matrix below.

The new pipeline component test found and fixed a shared-resource omission:
sequential V4 prediction uses the target vocabulary head, but an earlier retained
plan did not load that head on non-output pipeline stages. Those prediction
replicas could produce zero draft scores. Selected V4 requirements now replicate
the target embedding and output-head roles across prediction replicas while
preserving tensor shards and ordinary accounting. The focused two-process
pipeline test passes its complete capture/query/deletion/overlay/rollback checks
in 8.44 s. Cold preparation covers V3, sequential V4 and DSpark in all four
drafting modes, including removal of these consumers when prediction is disabled.
All 21 F32 CPU Ring configurations pass (285.18 s): TP, PP, EP and their four
combinations, each with resident, host-layerwise and disk-streamed weights.
Every case compares public prediction captures and channel/routed-unit deletion
against an ordinary native reference, including cached proposal work. Effective
rows, columns and signed projections agree across partitions; coordinated
target/prediction overlays, asymmetric reservation failure, late publication
rejection, rollback, snapshot/sibling replay and exact removal also pass. Routing
tables remain exact I32 buffers, outside floating parameter edits. Component
comparison uses `2e-5 * max(1, abs(reference))`; projected queries use `2e-6`.
The matrix also asserts one realization of the retained neutral communication
manifest. An expanded rerun also passes all 21 F32 cases with selected-token
channel/routed/shared keep-only masks, shared-unit deletion and exact all-keep
equivalence. Original/effective values verify that only the requested token and
components change. Combined channel-plus-shared keep-only trials verify that
the surviving shared unit recomputes from the changed residual. A three-token
target prefix seeds two shifted prediction rows; token slices refer to those
physical rows. Affine and MXFP4 quantized coverage are recorded below.

```sh
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  ring_deepseek_v4_mtp_components_and_parameters_matrix \
  -- --ignored --test-threads=1 --nocapture
```

V4 quantized admission now uses architecture-declared projection matrices rather
than treating every rank-two checkpoint tensor as a linear weight. Integer routing
tables, additive compressor positions and hyper-connection coefficients retain
their required storage. Canonical grouped expert transforms also retain the
operator's exact scale/bias companion names. A neutral cold-selection regression
covers sequential and DSpark checkpoints locally and with pipeline partitioning,
checks affine/MXFP4 eligibility, and preserves I32 source metadata. All 579
architecture unit tests pass with one existing ignored test. The affine
4-bit/group-32 native prediction matrix passes all 21 CPU Ring configurations
in 499.35 s, including every expanded mask and parameter lifecycle check above.
The F32 prediction matrix also passes all 21 configurations again after the
source-format and deferred-gather changes.
The source fixture is F32 with hidden/expert width 64 and head width 32, so
tensor partitions retain complete quantization groups. Source and selected
target architectures are constructed from retained formats before ordinary
materialization applies the transforms. MXFP4 prediction also passes all 21
CPU Ring configurations using this larger source fixture and the same expanded
capture, mask, parameter and replay checks. Target-stack matrices are separate
acceptance work; their first run exposed a missing ordinary/parallel hook
declaration despite implemented target traversal.

```sh
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  ring_deepseek_v4_mtp_affine_components_and_parameters_matrix \
  -- --ignored --test-threads=1 --nocapture
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  ring_deepseek_v4_mtp_mxfp4_components_and_parameters_matrix \
  -- --ignored --test-threads=1 --nocapture
```

The bounded affine regression also exposed a deferred-validation failure: an
invalid token reached V4's routing-table gather before the embedding assertion
was checked. The generic MLX gather now records its own bounds assertion and uses
safe indices until completion rejects the submission. CPU/Metal tests cover
signed negative indices, out-of-range I32/U32/I64/U64 indices and a dependent
second gather (0.95 s). The focused two-rank affine host-layerwise case now passes
captures, edits, restoration, invalid-token rejection and fencing in 18.17 s:

```sh
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  gather_bounds_settle_before_rejection_and_preserve_valid_indices \
  -- --ignored --test-threads=1
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  ring_two_process_deepseek_v4_mtp_affine_components_host \
  -- --ignored --test-threads=1
```


### Fused prediction declarations and phase applicability

Descriptor schema 7 and component schema 6 separate a fused score scope from
context-cache preparation. `ComponentExecutionScope.execution_groups` lists the
ordered physical decoder groups; `static_parameter_roles` describes pinned or
borrowed parameter storage independently. A fused proposal's row capacity is not
its decoder-block count. `ComponentResidualBase::Source` describes an actual
input and optional axis expansion without inventing an identity projection.

`ComponentReadoutEquation.score_writes` declares dynamic vocabulary-space
additions after the primary head's effective affine scores and before the output
transform. Each term exposes its input source, effective projection parameters,
actual multiplication input, original/effective output, and explicit singleton
axes broadcast to the final score geometry. These terms are separate from both
static vocabulary bias and residual-space writes. A consumer reconstructing final
scores or score differences must include them.

DSpark discovery now describes one fused multiblock scope, direct proposal-token
embeddings, each block's stream cycles, final stream collapse and the anchor's
Markov embedding/projection repeated over proposal rows. Its separately declared
context invocation exposes cache-input seams: context preparation does not run
FFNs or consume attention output writes in a scored residual equation.

Speculative activation schema 2 distinguishes `PredictionContext` from
`FusedProposal`. Sequential `Prediction { depth }` no longer applies to a fused
proposal phase. Context applies only during prediction prefill/replay; fused
proposal applies only during the fused invocation. Explicit invocation roots in
`ArchitectureDescriptor.speculative_invocations` also cover work without a score
equation. Old activation plans require fresh admission under the current schema.
Six neutral runtime tests pass, including actual phase-selective edits/evidence,
replay and nonrefundable usage; the new graph test verifies a two-block proposal
with a three-row capacity and a distinct Markov width. The shared DSpark context and proposal drivers now implement these hooks.
Public CPU/Metal capture, equations, parameters and overlays pass as recorded
below; the complete distributed matrix remains open.


The nonzero DSpark numerical fixture uses two physical blocks, three proposal
rows and a Markov rank of three. Serial and single-rank TP execution each run
three context updates and six proposal variants: baseline, a context-input edit,
channel removal, shared-unit keep-only, a Markov-input edit, and combined channel
removal/shared keep-only. Descriptor-driven stream terms plus the dynamic Markov
term reconstruct selected scores and their difference within `3e-4`. Original
and effective values identify exact edited coordinates; proposal-cache clones
replay exactly and surviving shared units are recomputed. All eight V4 component
numerical tests pass in 0.17 seconds, alongside 582 architecture unit tests with
one existing ignored test.

```sh
cargo test -j1 -p eredu-architectures --lib
cargo test -j1 -p eredu-architectures --test reference_numeric components::compressed_v4
```

The public DSpark activation tests pass on CPU and Metal across resident,
host-layerwise and disk-streamed execution (two tests, 14.62 seconds). They compare
continuous and controlled runs, actual channel deletion, original/effective
records, snapshot replay, sibling isolation, stale-authority rejection and
intervention re-admission. Context records occur only in context prefill/replay,
and proposal components only in fused execution. Dynamic Markov records retain
their declared singleton sequence axis. The initial test exposed a physical-row
mismatch: DSpark consumes the complete accepted target prefix, whereas sequential
prediction consumes shifted hidden/next-token pairs. The extension now supplies
that row count to the shared phase wrapper.

```sh
cargo test -j1 -p eredu --no-default-features --features mlx,metal \
  --test native_execution_control public_dspark_internal_activations_ \
  -- --ignored --test-threads=1 --nocapture
```

The expanded public DSpark suite passes all four CPU/Metal tests in 35.30 seconds after the cache fix,
each covering all three residencies. The additional tests query actual readout
and Markov matrices, reconstruct every captured stream cycle and selected score
and score difference within `3e-5`, and compare ordinary/controlled records.
Coordinated attention/FFN/Markov edits match a separately rewritten F32 checkpoint
on two token-ID prefixes; removal restores the exact baseline and leaves source
bytes unchanged. A one-token prefix also passes ordinary/controlled replay.
These tests demonstrate plumbing and numerical semantics, not editing efficacy.

```sh
cargo test -j1 -p eredu --no-default-features --features mlx,metal \
  --test native_execution_control public_dspark_ \
  -- --ignored --test-threads=1 --nocapture
```

The expanded two-block, nonconstant Ring fixture exposed a paged sliding-cache
rollback defect during ordinary long-prefix speculation: verification could
discard a sealed block still needed by its saved frontier. Native transient
checkpoints now retain that catalog history without preventing tier movement or
bypassing cache budgets. Both CPU/Metal cache regressions pass (1.11 seconds):
local, compressed and sparse states, resident and paged storage, a 17-token
prefix, rejected five-token work, repeated two-token replay against a fresh
reference, and resumed discard after checkpoint release. The two-process CPU
Ring DSpark test now passes (14.94 seconds), including long-prefix ordinary
speculation followed by public component/parameter checks. The complete
parallel/residency/encoding matrix remains in progress.

```sh
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  pooling_sliding_prediction_rollback_retains_sealed_history_ \
  -- --include-ignored --test-threads=1 --nocapture
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  ring_two_process_deepseek_v4_dspark_tensor_parallel_neutral_scheduler \
  -- --ignored --test-threads=1 --nocapture
```

The expanded Ring fixture uses two distinct F32 DSpark blocks, capacity two and
Markov rank four. Its first focused case compares all component records to an
ordinary local execution, including channel/shared/routed deletion and keep-only
at physical row one of the first fused proposal. Context and Markov edits affect
actual proposal scores. Every floating effective slot is queried, projected and
included in a coordinated overlay; asymmetric reservation failure, late peer
publication failure, active-overlay snapshot/sibling replay, restoration,
invalid-token rejection and transport fencing also pass. These checks retain
all existing cumulative accounting.


Following the fallible pooling-checkpoint change, the complete portable regression
passes: 248 core, 38 neural-contract, 497 runtime and 582 architecture unit tests
(one existing architecture ignore), all 222 numerical architecture tests, 61
backend-conformance tests, 19 portable-facade tests (one existing ignore), and 73
runtime backend-independence tests. These checks include the current component
and speculative-activation schemas and the cache rollback contracts.

```sh
cargo test -j1 -p eredu-core -p eredu-nn -p eredu-runtime -p eredu-architectures --lib
cargo test -j1 -p eredu-architectures --test reference_numeric
cargo test -j1 -p eredu --no-default-features --test portable_facade --test backend_conformance
cargo test -j1 -p eredu-runtime --test backend_independence
```


Native V3 and DSpark phase regressions pass on CPU and Metal (four tests, 2.34 s).
The DSpark cases inject observer failures during target prefill, accepted-context
prefill and fused proposal execution. Each invocation receives exactly one finish
status, failed work leaves no active invocation, and prefill failures publish no
output. Ordinary and observed execution agree; a channel intervention changes
actual proposal scores and only the selected channel/row. Existing V4 target
readout regressions also pass on both devices (two tests, 1.69 s).

The expanded two-rank F32 DSpark check passes in 16.33 s. All eleven mask variants
now verify the exact embedding broadcast, pre-attention stream collapse, every
residual-stream cycle, final residual, effective Markov projection and selected
scores/differences within `3e-5`. This includes context and Markov interventions.
The full F32/affine/MXFP4 parallel and residency matrices are still running.

```sh
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  internal_phases_reach_components -- --include-ignored --test-threads=1 --nocapture
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  native_v4_grouped_component_capture -- --include-ignored --test-threads=1 --nocapture
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  ring_two_process_deepseek_v4_dspark_tensor_parallel_neutral_scheduler \
  -- --ignored --test-threads=1 --nocapture
```


### Qwen hybrid prediction integration in progress

Qwen prediction units share their fusion and final-normalization driver between
ordinary, serial observed and tensor-parallel observed execution. The effective
final normalized hidden state conditions the next depth and the readout head.
The neutral fixtures cover Next and 3.5 dense and MoE variants, six masks,
three-row prefill and two cached steps, exact no-op behavior, component/fusion
reconstruction and learned-offset RMS. Use `qwen3_5_moe_text` for the routed 3.5
fixture: the dense model type intentionally clears expert settings.

The hybrid text layered adapter now forwards causal component observations through
serial and parallel blocks, embedding entry, pipeline entry and final readout.
Two-rank numerical fixtures verify globally selected channel deletion, keep-only
FFN sets, unchanged collective order, readout interventions and the sum of gated
shared-expert partial writes over prefill and two cached steps. All eighteen Qwen
regressions pass (8.24 s), followed by all 225 architecture numerical tests (51.35 s).
The prepared target and conditional media matrices are recorded below.

Prepared Qwen prediction now has per-depth discovery scopes and internal hooks for
prefill, proposals and accepted-token replay. Both depths identify the same four
shared fusion/normalization parameters and reuse the target embedding and output
head. The discovery fixture covers dense and MoE descriptions with tied and untied
heads (14 discovery tests pass in 0.40 s). The prepared dense numeric fixture checks
seven causal masks, independent attention/FFN write and fusion reconstruction,
no-op equality, separate caches and retry after an observation failure. Native
public prediction capture, query/edit, residency and distributed acceptance remain
in progress; the completed target and conditional media matrices below do not
establish prediction-path acceptance.

```sh
cargo test -j1 -p eredu-architectures --test reference_numeric qwen_ -- --test-threads=2
cargo test -j1 -p eredu-architectures --test reference_numeric
```


The new scalar gate declaration is `ComponentGroup.output_gate` (component schema
7, descriptor schema 8). A group's pre-gate affine write is still the sum of unit
values times effective write columns. Apply any declared output normalization,
then multiply each contribution by the effective scalar gate for that row. Its
own affine read and activation are separate dependencies in backward analysis.
Do not treat it as a constant residual scale or move it across the projection in
native execution. The Qwen shared-unit and gate hooks now follow this order;
the numerical fixtures include shared-unit and gate interventions. Two focused
partition-contract tests also pass: producer ownership is exact and malformed
shape, timing, row mappings and partial gate-weight shards are rejected. The
architecture library passes all 584 active tests (one existing ignore). Full
prepared/native partition acceptance remains in progress.


The expanded F32 DSpark CPU Ring matrix passes all 21 combinations in 532.36 s:
TP2, PP2, EP2, all pairings and TP2/PP2/EP2, each in resident, host-layerwise and
disk-streamed execution. Every case includes the eleven component/context/Markov
mask variants with stream and score reconstruction, effective queries/projections,
coordinated parameter edits, peer rollback, snapshots/siblings and restoration.
Affine and MXFP4 acceptance remains in progress.

```sh
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  ring_deepseek_v4_dspark_components_and_parameters_matrix \
  -- --ignored --test-threads=1 --nocapture
```


The DSpark affine CPU Ring matrix also passes all 21 topology/residency cases
(943.19 s), including the expanded stream/readout reconstruction and parameter
lifecycle checks. The corresponding MXFP4 matrix is running.

Prepared-source validation exposed missing Qwen hybrid direct/routed partition
constructors. Those architecture-owned constructors and cold state selection are
now implemented, with exact prepared tasks, global parameter authority, local
expert banks and PP state offsets. Validation is still running; the earlier
block/layered results alone do not establish prepared or native acceptance.

The prepared Qwen hybrid component matrix now passes (8.21 s): Qwen3-Next and
Qwen3.5, each dense and MoE, across 60 topology/residency combinations. Dense
variants cover TP2, PP2 and combined TP2/PP2; routed variants add EP2 and every
TP/PP/EP pairing. Each runs four causal trials through three-row prefill and two
cached decodes, compares actual producer captures and shared gated-write sums with
the nonzero serial reference, and checks exact invocation ownership. A middle-row
final-layer prefill mask changes that row's scores while preserving the published
last-row prediction.

The released-layout source fixture exposed a recurrent TP bug: convolution rows
were sliced contiguously despite consuming concatenated Q, K and V. Convolution
weights now use the same segmented head ownership as the input projection. The
prepared constructors also retain global parameter authority separately from
local modules and bind the exact selected local expert banks. This is neutral
prepared-execution evidence; the new native hybrid matrices still await execution.

```sh
cargo test -j1 -p eredu-architectures --test reference_numeric \
  prepared_qwen_hybrid_components_and_scalar_gates_cross_tp_ep_pp_and_residency \
  -- --nocapture
```

The expanded DSpark MXFP4 CPU Ring matrix passes all 21 topology/residency cases
(705.96 s). Together with F32 (532.36 s) and affine (943.19 s), this completes the
three-encoding distributed DSpark matrix, including effective stream/Markov score
reconstruction and parameter-query/edit/rollback/replay checks.

After the Qwen prepared integration and recurrent placement correction, the full
architecture library passes 584 tests (one existing ignore, 0.39 s), and all 226
numerical tests pass (50.52 s), including the existing multimodal and distributed
regressions.

The backend-independent construction changes also pass
`cargo check -j1 -p eredu-backend-mlx --no-default-features` (20.39 s). This compile
check does not replace the pending native hybrid execution matrix.

The ordinary V4 target affine CPU Ring rerun passes all 21 topology/residency cases
(346.60 s), including the expanded readout reconstruction and effective parameter
lifecycle checks. The ordinary target MXFP4 rerun also passes all 21 cases (305.08 s). The F32 rerun passes all 21 cases as well (258.96 s).

Conditional Qwen3.5 target hooks pass an initial nonzero serial/TP2 fixture for
dense and MoE configurations (0.09 s): six intervention modes, media prefill and
two cached decodes. Assembled input, text-layer writes and actual DeepStack
contributions reconstruct the final residual; observed no-op execution is exact.
The prepared-source conditional matrix also passes all 30 dense/MoE
TP/EP/PP/residency combinations (2.62 s). It binds nonzero checkpoint payloads,
uses six intervention modes, and compares media prefill plus two cached decodes
against the ordinary scalar execution. The later vision DeepStack projector
crosses the vision/text boundary; assembled inputs, component writes, DeepStack
residuals and final scores agree. This exposed and corrected pinned vision-merger
ownership and construction of local conditional expert banks from the retained
expert plan. Native conditional results are recorded below.

```sh
cargo test -j1 -p eredu-architectures --test reference_numeric \
  prepared_conditional_qwen_component_ownership_and_media_writes_cross_partitions \
  -- --nocapture
```

The conditional ownership/construction changes pass all 500 runtime tests
(0.32 s), 584 architecture library tests (one existing ignore, 0.39 s), and
228 numerical tests (54.57 s). DeepStack declarations now restrict capture to
media prefill, matching the actual execution boundary; cached decoding does not
perform that addition. Native public text and media results are recorded below.

Conditional media boundaries keep their `Conditional` discovery status while
allowing capture/intervention plans to select them. The neutral runtime tests
exercise present and absent boundaries with nonzero intervention inputs, ensure
missing captures have no payload or transform, and require scheduled
interventions to complete before commitment. `cargo test -j1 -p eredu-runtime
--lib` passes all 502 tests (0.25 s) after this admission change. Native
conditional TP/PP acceptance is included in the matrix below.

Native Qwen3.5 and Qwen3-Next dense target matrices each pass all nine TP/PP and
resident/host/disk combinations. Qwen3.5 text MoE also passes all 21 TP/EP/PP and
residency combinations through the public capture, intervention, parameter,
overlay and replay harness. The conditional Qwen3.5 TP/PP focus passes in 6.53 s;
the broader conditional matrix exposed a missing successful-policy finish in
composite streaming execution, whose fix is covered by the completed matrix below.
These Ring runs use native MLX CPU execution. Reproduce the text MoE matrix with
`cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib
ring_public_component_capture_qwen_35_hybrid_moe_matrix -- --ignored --nocapture
--test-threads=1`.

The conditional native matrix now passes all nine dense and 21 MoE topology/
residency cases, plus the focused TP/PP, PP host, and eight-rank TP/PP/EP tests
(five tests, 273.83 s).
The eight-rank overlay discrepancy came from lazy world collectives evaluating
independent embedding/vision branches in different orders. Ordered completion
of the logical subgroup fallback corrects ordinary and controlled execution;
a focused eight-rank regression independently checks sums and gathers consumed
in opposite branch orders (2.71 s). The earlier PP host failure was caused by a received byte
payload starting at an unaligned offset after a variable-length role header.
Native unpacking now materializes unaligned bytes before creating the typed
view. CPU and Metal tests cover every header alignment for F32/F16/BF16 and
verify both host values and actual typed arithmetic (two tests, 1.65 s). The
focused PP host regression passes without diagnostic instrumentation (3.73 s).

```sh
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  role_exact_payloads_preserve_values_at_every_header_alignment \
  -- --include-ignored --nocapture --test-threads=1
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  ring_public_component_capture_qwen_35_conditional_ \
  -- --ignored --nocapture --test-threads=1
```

Qwen3-Next dense/MoE public native acceptance also passes all 30 cases plus the
focused MoE TP test (186.40 s). Together with Qwen3.5 this establishes the 60-case
hybrid text matrix. Public prepared-input counts, controlled capture, causal
masks, parameter queries, coordinated overlays, rollback and branch replay are
included.

The shared Qwen MTP parameter refactor passes architecture and backend type
checking and all 21 Qwen numerical regressions (8.30 s). The two-depth graph now
validates one canonical owner for each shared fusion/normalization parameter,
with both logical prediction depths declared as consumers. Ordinary and observed
cached fusion tests consume that shared owner. The prepared and native prediction
acceptance below extends this ownership proof to public execution.

Prepared replicated text MTP now passes a real-payload fixture for both Qwen3-Next
and Qwen3.5 (0.10 s). It verifies the target and extension's exact bound values,
independent scalar unpacking of Next's fused rows, shared-head scaling,
shared-fusion changes, lane rollback after mutation, and canonical target-state
isolation. All 26 Qwen numerical regressions pass (26.84 s), and the existing
Nemotron prediction regression passes (0.02 s) after sharing replicated readout
helpers. Subsequent prepared internal scopes and public capture/edit acceptance
are recorded below; remaining formats and conditional prediction coverage stay
explicit in the implementation checklist.

```sh
cargo test -j1 -p eredu-architectures --test reference_numeric \
  qwen_text_prepared_prediction_binds_shared_fusion_and_restores_lane_state \
  -- --nocapture
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  ring_logical_world_collectives_preserve_order_across_independent_branches \
  -- --ignored --nocapture --test-threads=1
```

The bounded-completion refinement also passes the forced pending-completion
retention and group-fence regression (1.33 s), the independent-branch eight-rank
sum/gather ordering regression (2.75 s), and the public conditional Qwen TP/PP/EP
capture/edit/replay regression (22.16 s). Pending logical world operations retain
their native resources and fence the group until completion after a deadline.

Prepared Qwen prediction public capture admission now passes native MLX CPU
validation for eight two-depth configurations: Next/3.5 dense and MoE, each with
tied and untied output heads (3.13 s). Metal passes Next and 3.5 dense two-depth
fixtures (3.36 s). These tests use public speculative discovery and controlled
capture admission, compare disabled and no-op instrumentation, apply causal
channel masks, and exercise cumulative-budget failure evidence and fresh replay.
The synthetic checkpoint writer retains the complete architecture static module,
including the four canonical shared MTP parameters. Commands:

```sh
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  native_qwen_public_speculative_activation_admission_cpu -- --test-threads=1 --nocapture
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  native_qwen_public_speculative_activation_admission_metal -- --ignored --test-threads=1 --nocapture
```

Prepared producer placement covers original/effective normalization and raw
embedding inputs on every prediction replica, and rejects pinned fusion parameters
owned by another consuming group or static role (five tests, 0.10 s). The prepared
scalar fixture covers Next/3.5 dense and MoE at one and two prediction depths,
including independently packed expert source values, causal routed-unit masks,
scope completion after failure, and exact lane replay (0.66 s).

Native CPU Ring prediction capture, masks, queries, coordinated overlays and replay
pass for all four text variants under TP (21.25 s) and PP (22.20 s). Prepared
resident banks finish their routed capture scope before tensor reduction; the
neutral load requirements replicate shared target embedding/head shards across
pipeline stages and account for those copies. The PP run also compares actual
fusion inputs, original/effective normalization and readout values against the
resident reference. The F32 residency/distribution matrix is completed below;
additional source encodings remain in progress.

```sh
cargo test -j1 -p eredu-architectures --test reference_numeric \
  qwen_text_prepared_prediction_binds_shared_fusion_and_restores_lane_state \
  -- --test-threads=1
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  ring_qwen_text_prediction_components_tensor_parallel -- --ignored --nocapture --test-threads=1
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  ring_qwen_text_prediction_components_pipeline_parallel -- --ignored --nocapture --test-threads=1
```

Native Next and Qwen3.5 MoE prediction acceptance passes all 42 topology/residency
cases: TP, PP, EP and all combinations, each resident, host-layerwise and
disk-streamed. The public checks include original/effective captures, nine mask
modes, effective parameter queries, coordinated overlays and branch replay.
The initial combined run took 461.46 s. After correcting the dense fixture
launcher, both Next and Qwen3.5 dense matrices pass all eighteen cases, completing
all sixty text prediction cases. The dense rerun was part of a 118.13 s run whose
conditional cases stopped at harness assertions that omitted vision units. After
correcting those counts, the complete conditional rerun passes all nine dense and
21 MoE cases in 422.01 s, including DeepStack and all three eight-rank TP+PP+EP
residency cases. Together these runs establish all 90 F32 text/conditional
prediction cases.

```sh
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  ring_qwen_prediction_components_next_moe_matrix -- --ignored --nocapture --test-threads=1
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  ring_qwen_prediction_components_35_moe_matrix -- --ignored --nocapture --test-threads=1
```

Prediction input retains architecture-owned media placeholder identity. Qwen
interleaved text/image/video/projected spans match ordinary target tokens exactly
for dense and MoE configurations; Inkling's projected image identity matches too.
After sharing ingress token construction, all 27 Qwen numerical regressions pass
(20.04 s), as do all five Inkling regressions (4.71 s). These neutral checks cover
the input contract. Native image-backed dense TP prediction also passes (9.61 s),
including all masks, effective queries, coordinated overlays and replay with
DeepStack enabled. All thirty conditional cases now pass as recorded above;
additional source encodings remain pending.

The complete conditional F32 prediction acceptance command is:

```sh
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  ring_qwen_prediction_components_35_conditional -- --ignored --nocapture --test-threads=1
```

Nemotron prediction fusion, physical attention/MoE units and final normalization
now share ordinary and observed execution. A nonzero two-depth scalar fixture
covers six selected-value interventions, fusion reconstruction, prefill and two
cached steps, serial/TP1 comparison and target-state isolation (0.03 s). Disabled
and no-op paths remain exact within the same execution strategy; biased serial/TP
comparisons allow the existing F32 arithmetic tolerance. All fourteen Nemotron
numerical regressions pass (5.09 s).

Discovery validates attention-only, `*E` and `*E*E` physical schedules over two
depths, tied/untied heads and biased projections (0.05 s). Prepared prediction
coverage uses those declarations for actual parameter and observation joins.
It verifies one/two depths, fusion/head payload changes, shared and routed ReLU²
writes, causal masks, provider failure cleanup, accepted-token replay and exact
lane isolation. All five prepared prediction tests pass (1.26 s), including routed
Nemotron targets and the existing Qwen and V3 cases. The architecture suite at
that checkpoint passes 587 tests with one existing ignore. The thirty-case native
Nemotron F32 prediction matrix now passes as recorded below; additional encodings
remain pending.

```sh
cargo test -j1 -p eredu-architectures --test reference_numeric \
  nemotron_prediction_fusion_and_components_share_serial_parallel_execution -- --nocapture
cargo test -j1 -p eredu-architectures --lib \
  nemotron_prediction_scopes_preserve_physical_schedules_and_parameter_roots -- --nocapture
cargo test -j1 -p eredu-architectures --test reference_numeric prediction_adapter \
  -- --nocapture --test-threads=1
```

Inkling's neutral component fixture uses distinct nonzero parameters, including
all four convolution kernels and the learned dense/router scales. It reconstructs
attention and dense writes and the coefficient-weighted routed/shared expert sum
from actual component values. A separate scalar causal-convolution calculation
reconstructs complete residual contributions across a three-position prefill and
two cached decode steps. The fixture checks exact no-op instrumentation and
selected position/unit/channel deletion in serial and TP1 execution:

```sh
cargo test -p eredu-architectures --test reference_numeric \
  inkling_component_masks_feed_projections_and_convolution_histories -- --nocapture
```

This passes in 0.08 seconds; the five pre-existing Inkling numerical regressions
also pass (4.86 seconds), including TP2 multimodal execution and MTP rollback.
The public discovery, prepared MTP and native matrix are still being integrated;
these primitive checks alone do not establish full public Inkling support.

The Inkling layered fixture additionally exercises the real serial runtime for
both dense and routed targets. It reconstructs the residual from assembled input
and complete convolutional writes, verifies the declared muP scaling at the
actual head multiplication input, reconstructs scores, and checks causal
embedding/channel/unit/readout masks through prefill and cached decode. Run:

```sh
cargo test -p eredu-architectures --test reference_numeric \
  inkling_layered_component_and_readout_hooks_preserve_causal_execution -- --nocapture
```

The layered check passes in 0.07 seconds. The target driver changes also preserve
all six current Inkling regressions (5.37 seconds). Public discovery and embedded
prediction support remain separate items in the implementation checklist.

Nemotron-H's prepared prediction component matrix passes all 30 F32 cases:
nine dense-target cases (TP, PP and TP+PP across resident, host-layerwise and
disk-streamed execution) and 21 routed-target cases (all seven nonempty TP/PP/EP
combinations across those three residency modes). The sparse prediction schedule
contains both attention and ReLU-squared expert units. Each case checks actual
component masks, effective global parameter rows/columns, coordinated target and
prediction overlays, failed publication rollback, restoration and controlled
replay. The repeated routed TP focus also passes; the combined run takes 303.31
seconds on local CPU Ring workers:

```sh
cargo test -p eredu-backend-mlx --no-default-features --features metal --lib \
  ring_nemotron_prediction_components_ -- --ignored --nocapture --test-threads=1
```

Prediction units use a tensor-rank layout and replicate across target pipeline
and expert coordinates. Retained static roles supply shared embedding/readout
modules on every participating pipeline rank. Target-only inspection excludes
prediction depth; the paired extension retains and validates that separate state.
Additional Nemotron source encodings remain to be validated for these component
operations.

Inkling prediction primitive coverage preserves the released double application
of its shared hidden normalization and the hidden-before-embedding fusion order.
Seven causal masks cover both normalization applications, embedding normalization,
fusion, attention channels, dense FFN units and returned continuation. Both depths
are exercised with and without chain normalization, across prefill and two cached
steps; inactive depth state remains untouched and no-op captures preserve exact
outputs and state (0.07 seconds). Its existing MTP rollback regression also passes.

```sh
cargo test -p eredu-architectures --test reference_numeric \
  inkling_prediction_fusion_components_preserve_double_norm_and_depth_state -- --nocapture
```

## Archived implementation checklist

This snapshot preserves the detailed implementation and test history before the
checklist was condensed. Pending items here can be superseded by the current
[implementation checklist](component-analysis-checklist.md).



- Public portable APIs provide component discovery, bounded original/effective
  captures, compact selected-position masks, effective-parameter queries and signed
  projections, atomic reversible overlays, exact token-ID scoring, replay and
  isolated controlled branches. Disabled instrumentation creates no extra component
  tensors. Full-vocabulary probabilities and ranks use the complete distribution.
- Shared dense execution covers gated and unary GELU/ReLU/ReLU² FFNs, attention
  channels, RMSNorm/LayerNorm, biases, tied/shared parameters and readout equations.
  Resident, host-layerwise and disk-streamed paths share the same drivers.
- Distributed component placement retains logical invocation owners independently
  of shared parameters. TP/EP replicas lower global masks onto actual local values;
  PP transports effective embeddings and residual state. Source preparation,
  invocation votes, exact producer receipts and final commit govern capture delivery.
  Empty/skipped selections avoid source work; failed work retains consumed credits.
- Canonical routed banks expose expert/unit topology and checked read/write joins.
  Provider scopes preserve original experts, source-token/route identities and
  chunks through resident, addressable and EP execution, including idle owners.
  Loaded sparse captures and edits share the ordinary public contract.
- Native CPU Ring coverage includes Qwen dense/MoE, packed Qwen GGUF, GPT-OSS,
  Nanbeige, dense/routed LFM2, K2 dense/MoVA and Nemotron-H target execution.
  The detailed family/residency/topology matrix is in the integration guide.
  K2 includes shared SwiGLU and activated routed value reads. Nemotron includes
  dense/shared/routed ReLU², attention and complete Mamba/routed-plus-shared writes.
  Broader compatible-family propagation remains open below.
- Effective parameter operations use actual prepared source ownership and loaded
  transforms. Multi-parameter edits span attention, FFNs and expert banks, agree
  before publication, preserve originals, roll back peer failures and invalidate
  incompatible cached state. Controlled captures retain common edit provenance.
- Dense, affine, MXFP4, FP8 and GGUF Q8_0/IQ4_NL operations have numerical coverage.
  Packed edits promote only affected matrices to F32 and restore exactly. Deferred
  FP8 inputs retain actual precision and reserve storage before creation. Grouped
  FP8 lifecycles pass ordinary CPU/Metal and all seven CPU Ring TP/EP/PP combinations
  across resident/host/disk execution, including unequal complete-block allocations.
  Independently evicting expert caches retain these query and overlay lifecycles.
  Dense FP8 intermediate-axis tails, independently blocked fused gate/up tails,
  and joint attention-head tails each pass all 42 public Ring configurations.
  Attention placement preserves per-head gains and each layer's local cache width.
  Packed grouped GGUF has native TP/EP/PP lifecycle coverage.
- Independently cached parameters join exact lowered members to architecture
  declarations. Sequential completion-safe loans support queries larger than the
  source cache; atomic publication updates ordinary handles and bank providers.
  Packed floating replacements use logical operator shapes and retain original
  cache sources for removal. Gated, ReLU² and selected-linear mechanisms share this
  behavior.
- Cold request preparation, nonrewindable capture epochs, setup identity, source
  completion, transaction outcomes and state reuse remain in shared runtime drivers.
  Public failures retain neutral types and original native sources. Provider success
  agreement runs without observation; inactive pipeline stages join the retained
  session wave. Logical execution rejection does not establish native completion or
  remove actual transport fences.

## Verification evidence

Commands, pinned revisions, hashes, tolerances, exact execution matrices and test
history are retained in [component-validation.md](component-validation.md). The entries
below summarize completed validation; they do not mark the full requirements done.

- The current numerical suite passes 196 tests, including fifteen Nemotron trials
  through prefill and two cached decodes across 21 topology/residency combinations.
  Shared-unit reconstruction, bias joins and survivor recomputation pass, alongside
  segmented Mamba placement and the negative-transition transform. K2 mixed-encoding
  construction and shared-write reconstruction now cover independent routed/shared
  TP widths in both encoding combinations.
  The architecture library passes 554 tests with one existing ignored test;
  backend compilation without default features, formatting and whitespace checks pass.
- All 42 K2 grouped FP8 CPU Ring configurations pass (606.83s). Effective queries
  exactly match independent E4M3 decoding; signed projections, coordinated edits,
  captures, masks, rollback and controlled replay pass. A separately edited
  SafeTensors checkpoint agrees through prefill and two cached decodes, and removal
  restores the baseline exactly. Both tested geometries retain complete blocks
  along the sharded axis; the 130-wide hidden-axis tail is replicated.
- All 42 explicitly evicting FP8 bank configurations pass (737.64s), and all 24
  public CPU/Metal configurations pass (25.46s), including whole-bank queries
  through a one-member cache. Repeated competing-bank queries force fresh FFN
  reloads while preserving prefill and cached predictions. Added reservations cover
  sequential copies, and compact accounting includes floating overlay promotion.
- Another 84 independently cached configurations pass (745.65s): MXFP4 GPT-OSS,
  grouped GGUF Qwen, and F32/GGUF Nemotron across the same seven topologies and
  three residencies. Public parameter operations, captures, masks, coordinated
  edits, rollback and controlled replay agree with ordinary native owners. These
  use default bank budgets; explicit eviction is enforced by the FP8 matrix.
  Current runtime library/backend-independence and portable backend/facade suites
  pass 459/70/59/19 tests, alongside 17 native bank regressions.
- Dense FP8 intermediate-axis tails pass 42 CPU Ring configurations (763.96s):
  seven TP/EP/PP combinations, three residencies, ordinary and independently
  cached banks. Exact queries and signed projections cross scale/ownership
  boundaries; coordinated edits affect the three-unit tail and agree with an
  independently rewritten checkpoint. Capture, masks, rollback, restoration and
  controlled replay pass. The updated architecture/runtime suites pass 552/463
  tests; architecture retains one existing ignored test. Backend independence,
  portable conformance and facade pass 70/59/19, with one existing facade ignore.
- Independently blocked fused FP8 gate/up tails pass 42 CPU Ring configurations:
  21 ordinary and 21 independently cached (391.18s for the corrected cached run).
  Both 259-row branches retain their own final block; the short TP shard has three
  units per branch. Queries, signed projections, coordinated edits to both branches,
  independent edited checkpoints, rollback, restoration, masks and controlled replay
  pass. Cache tests account for the actual TP/EP-local value batch and require fresh
  FFN-bank misses and evictions. CPU/Metal primitive tests pass for F32 and UE8M0
  scales, widths 3/259 and both grouped dispatches. The public matrix uses F32 scales.
  NN/runtime/architecture library suites pass 38/464/553 tests, with one existing
  architecture ignore; portable backend independence/conformance/facade pass 70/59/19.
  Broader architecture fixtures pass 196 numerical, 9 construction and 46 structural
  tests; their independent local-layout helpers retain the new chunk placements.
- Attention-head FP8 tails pass all 42 CPU Ring configurations (840.52s), including
  query/key normalization and mixed FP8/dense attention layers with different local
  cache widths. Exact Q/K/V/O and value-bank queries, signed projections, last-head
  edits, independent edited checkpoints, capture/masks and controlled replay pass.
  Cached cases preserve FFN misses, evictions, promotion bounds and cumulative query
  charges across resets. Architecture library/numerical/construction/structural
  suites pass 554/196/9/46, with one existing library ignore; backend independence,
  portable conformance and facade pass 70/59/19, with one existing facade ignore.
  Backend compilation without default features, formatting and whitespace pass.
- All 42 Nemotron CPU Ring configurations pass with shared scalar hooks (359.47s),
  covering SafeTensors and GGUF, three residencies and all seven TP/EP/PP combinations. Actual captures,
  shared/dense/routed masks, coordinated parameter edits, rollback and controlled
  replay match ordinary native execution, including eight-rank cases.
  The preceding Nemotron provider-failure regression passed all 84 asymmetric
  scenarios (96.10s), including typed peer rejection and synchronized reset/replay.
- The preceding provider-agreement matrix passed 252 asymmetric CPU Ring failure
  scenarios across gated/ReLU²/linear banks, three residencies and seven TP/EP/PP
  combinations. Local CPU/Metal coverage passed 96 scenarios. Exact original causes,
  typed peer rejection, ordinary/control prefill/decode, aborted capture outcomes,
  synchronization, reset and healthy replay were checked. Runtime, backend-independence
  and portable conformance/facade checks passed 455/70/59/19 tests respectively.
- The preceding 54 native family configurations passed captures, scalar/shared/
  sparse masks, global parameter queries, coordinated edits, peer rollback,
  cached predictions and controlled replay. K2 residual, signed-score and
  score-difference reconstruction passes at `2e-5`, including grouped RMSNorm.
- Pinned official SmolLM2 and LFM2 examples pass all fourteen records each and
  restore exactly. They independently compare 414,740 and 921,615 values, with
  maximum errors 7.05719e-5 and 6.77109e-5. Illustrative edits establish API behavior,
  not the paper's general efficacy. Released sparse parity also passes as below.
- Quantized acceptance includes independently decoded effective reads and edited
  references, partial FP8 blocks, both scale encodings, big-endian GGUF, singleton
  banks, cached ordinary/control parity and nonrefunded budgets. See the guide for
  exact CPU/Metal and Ring configurations rather than inferring untested coverage.
- Pinned official LFM2-8B-A1B passes all five trials, twenty ordinary predictions
  and seventy controlled replay/isolation checks. All thirty groups and all
  prefill/cached positions pass 18,971,520 comparisons, including 17,660,800 exact
  non-logit values. Independent component-write reconstruction passes 150 writes
  and 921,600 terms with maximum physical error 0.00048828125.
  The additional selected-token and token-difference run passes 10,944,290
  comparisons, including 9,629,120 exact non-logit values, 600 component groups,
  3,686,400 terms and forty scores. Explicit embedding, convolution and rounding
  terms reproduce the float64 head dot within 3.55272e-15; BF16 score error uses
  at most 0.365129 of the declared absolute output-rounding bound. All twenty
  residual replays and seventy controlled checks pass. Final head/dtype guards
  pass a fresh baseline comparison. These are single-device Metal results.

## Earlier requirement audit

This audit records the state at that implementation stage. Later acceptance is
recorded below; the [current checklist](component-analysis-checklist.md) and
[integration matrix](component-analysis.md#execution-coverage) track the remaining
work for the full goal.

1. Complete compatible-family propagation and acceptance: remaining specialized
   normalized inputs and residual writes, compressed/hybrid attention, conditional
   media, recurrent relationships, custom projections and MTP execution. Nemotron's
   target shared-expert ReLU² topology/hooks are implemented; the shared write remains
   a constituent of the complete sparse contribution. Distinguish implementation
   gaps from architectural inapplicability and hardware validation gaps.
   V3 resident/paged MLA channels and dense SwiGLU hooks now share ordinary
   equations and pass nonzero neutral masks, write reconstruction and downstream
   recomputation. Versioned low-rank read chains, latent-stage capture/intervention,
   KV-cache insertion and whole-model score reconstruction also pass neutral tests.
   Public native CPU capture/masks and CPU/Metal queries, coordinated edits,
   independently edited references, cached replay and restoration pass all twelve
   direct/low-rank × dense/sparse × residency configurations. Compressed-cache
   isolated copies and continuation accounting close the snapshot gap found by
   these tests. Shared-expert joins, sixty neutral shared-unit predictions and
   native shared masks/edits also pass; the public score consumer avoids counting
   nested shared writes twice. Mixed dense/routed V3 pipeline input/readout hooks
   and latent-stage ownership now pass checkpoint-backed neutral comparison across
   all three residencies and cached decode. Native two-process Ring capture,
   component masks, snapshots and branch replay pass resident, host-layerwise and
   disk-streamed modes. All-dense V3 now has direct construction and complete TP/PP
   hooks, with all nine native cases for each of F32 SafeTensors, F32 GGUF and
   SafeTensors load-time affine 4-bit/group-32 passing public queries, coordinated
   edits, masks and branch replay. The query rotary-axis fix
   also has strict identical-head/chunked-prefill coverage and a revised cache
   identity. Target-only sparse TP/PP/EP hooks now pass all 42 native F32
   SafeTensors/GGUF residency cases and all 21 load-time affine 4-bit/group-32
   cases. The source-precision correction passes a fresh 30-case dense/mixed affine
   run. Independent-bank F32 SafeTensors/GGUF and affine acceptance passes all
   63 cases, including idle EP owners and actual activity on every routed PP stage.
   Prediction primitives have shared fusion/head hooks and 216 nonzero neutral
   forwards. Loaded public prediction capture and causal deletion now pass the
   21-case native parallel/residency matrix recorded below; distributed prediction
   queries, edits, rollback and controlled replay also pass. Additional encodings
   remain acceptance work.
2. Extend parallel acceptance beyond F32 scale companions and independent-bank
   acceptance across remaining compatible families/formats.
   Keep exact transformed reads, signed projections, coordinated edits, state
   compatibility and restoration in each case.
3. Finish failure/accounting audits across specialized startup, external preparation,
   earlier route packing/selection, cold/submission source adaptation, asynchronous
   completion and late transport/record delivery. Source retention or logical
   agreement alone never proves safe native completion. Preserve cumulative capture,
   transport, copy, parameter-query and overlay work through cancellation and restore.
4. Native callback-group audit completed: selected model paths use connected stage
   or session groups with bounded member-only agreement. Arbitrary disconnected
   low-level Ring memberships retain their explicit independent-status rejection;
   they are not selected by these component workflows.
5. Run applicable distributed GPU checks where hardware permits;
   retain neutral conformance and explicitly record any hardware-only validation gap.
6. Run the final crate/feature/facade/native matrix and audit every original
   requirement against implementation, public integration, behavioral coverage and
   documentation. Complete the feature-by-execution support matrix and verify the
   public example's attribution, ablation, backward queries and both edit workflows.

Research ranking, calibration, semantic naming, graph search and efficacy remain
Inspector/evaluation work. Inspector plans captures before execution and retains
prepared-boundary snapshots; historical uncaptured predictions require exact
original token-ID replay. The tested output token is never forced.


- [x] Neutral additive TP write receipts and live observer lifecycle: explicit
  equation, raw-term accounting, compensated host assembly, complete membership,
  failure/rollback and restore without refunds.
- [x] Neutral additive activation projection: masks/scaling on each term, one Add
  offset/replacement owner, geometry identity and pre-work resource checks.
- [x] Mixed V3 TP resident/provider hooks preserve the fused reduction; 450 neutral
  TP/TP+PP predictions cover three residencies and both query forms. Actual partial
  writes are encoded and assembled before comparison with whole execution.
- [x] Native mixed V3 F32 SafeTensors/GGUF: 42 cases across TP, PP, EP and all
  combinations, three residencies, capture/transforms, five summed-write actions,
  atomic packed-expert gate/value/write edits and controlled branches.
- [x] Mixed V3 load-time affine native acceptance: 21 TP/PP/EP and residency
  cases, including packed expert edits and additive capture/intervention evidence.
  Source handoff, selected router format and global/local metadata checks pass.
- [x] Mixed V3 independently cached affine native acceptance: all 21 TP/PP/EP
  residency cases, exact generated companion ownership, matching source precision,
  queries/atomic edits, actual occupancy and compact-bank construction.
- [x] Mixed V3 independently cached F32 SafeTensors/GGUF acceptance: all 42
  TP/PP/EP residency cases. Idle EP owners do no compact-bank work; every routed
  PP stage has active caching. The final SafeTensors/affine rerun also passes.
- [ ] Additional compatible V3 encodings and forced-eviction component acceptance.
- [x] V3 prediction primitive fusion/head hooks reuse ordinary execution and pass
  216 nonzero neutral forwards, including exact masks and head renormalization;
  48 observed traversal forwards and sixteen sibling replays preserve depth state.
- [x] V3 prediction component scopes declare their fused residual bases, separate
  heads, canonical parameter joins and original/effective capture paths. Ordinary
  target support cannot admit prediction points merely from collector support.
- [x] Typed V3 extension prefill/proposal/replay hooks preserve absent-observer
  execution, actual component edits, target-state isolation and prediction-lane
  retry in neutral serial/TP1 fixtures. Public collector wiring is verified below.
- [x] Architecture-owned traversal and retained exact native prediction parameter
  owners; public F32 queries/projections and coordinated target/prediction overlays
  pass CPU/Metal across resident/host/disk, with independent source comparisons,
  two prefixes, controlled snapshot replay, budget rejection and exact restoration.
- [x] Complete pooling target/prediction snapshots, including strided pending,
  pooled and overlap arrays and independent paging catalogs. Public V4 sequential/
  DSpark CPU/Metal replay and sibling isolation pass all three weight residencies.
- [ ] Remaining embedded state-profile acceptance, other-family component hooks,
  additional prediction encodings and distributed component/parameter acceptance.
  Sequential V3 snapshots and prospective internal edits are verified below.

- [x] Shared embedded strategy forwards separate internal observers with actual
  phase/depth/sequence geometry; target passes reuse the authoritative observed
  transaction. Lossless error bridging retains reservations and routing callbacks.
  Neutral phase failures and replay preserve cache recovery and spent authority.
- [x] Public phase/frontier admission, cumulative capture delivery and continuous/
  controlled parity on the selected V3 path, separate from one-row sampler logits.

- [x] Native V3 internal observer binding: deterministic nonzero CPU and Metal
  runs cover passthrough, exact channel masks, actual draft head changes and
  target/proposal failures. Public bounded admission is verified below.

- [x] Exact independently admitted invocation geometry in the shared capture and
  intervention owner, including physical sparse receipt rows and nonrefundable
  restore/fork accounting.
- [x] Scheduler-owned request/frontier/prefix provenance across canonical and
  optimistic work, verification and replay; origin scopes clear on failure/unwind.
- [x] Neutral speculative collector reuses bounded transforms, generated-value
  reservation and routing callbacks; scope filtering, original/effective evidence,
  charged queued records, monotonic invocation IDs and typed/native error retention
  have behavioral coverage. Controlled step transport has a distinct activations
  field; public plan installation and end-to-end parity are verified below.
- [x] Native bounded speculative collector binding on CPU/Metal: real phase shapes,
  scheduler origins, capture-only parity, compact channel masks reaching draft
  logits, exact original/effective evidence and typed cumulative-budget failure
  before publication. Ordinary Metal and two-process TP regressions pass. This
  explicit binding fixture does not complete public installer/parity coverage.

- [x] Combined versioned internal activation admission bound to source, execution,
  active overlay, native session and architecture scopes; typed executor installer
  restores prior observers and rejects allowance resets after execution.
- [x] Public loaded internal activation acceptance and continuous/controlled
  parity on native CPU/Metal across resident/host/disk, with exact channel masks,
  actual draft-logit changes, admitted identity and subsequent-run cleanup.
- [x] Complete sequential V3 embedded snapshots and reusable restores with both
  prediction replicas, isolated siblings and monotone capture/copy budgets; native
  CPU/Metal acceptance covers resident/host/disk and masks.
- [x] Prospective internal intervention re-admission with actual loaded validation,
  fixed capture allowances, exact saved authority and atomic native/authority restore.
  CPU/Metal child mask install/removal/replay preserves parent and sibling plans.
- [x] V4 sequential/DSpark sampler capture parity, reusable snapshots, changed child
  replay, parent/sibling isolation and typed budget rejection on CPU/Metal across
  resident/host/disk, crossing ratio-4 and ratio-128 pool completion boundaries.
- [x] Paged sliding fetch and blockwise attention retain the complete submitted
  span before eviction; nonzero native cache/scalar-reference coverage and the
  long-prefix two-rank DSpark scheduler regression pass.
- [x] Neutral partition invocation admission: exact physical axes in schema-3
  receipts, cross-rank geometry agreement, deferred-factory rejection, architecture
  placement, active scope masks, real local edits, original/effective evidence,
  final-commit publication and nonrefundable replay budgets. Live three-/five-rank
  fixtures include replicas, idle ranks and varying widths at one prediction.
- [x] Borrow-scoped internal observer lifetime through the complete forward, with
  live five-rank partition work, edits/evidence, and failure/unwind drop-order tests.
  Native CPU/Metal multi-row capture and public V3 continuous/controlled replay
  across all three target residencies pass with this shared lifetime.
- [x] Auxiliary prediction transactions reuse ordinary preparation, checkpoint,
  completion and final-commit policy across whole prefill/proposal/replay phases.
  Two-rank fault coverage includes the second prediction checkpoint; native
  CPU/Metal completion preserves deferred validation and original error sources.
  Public V3 continuous/controlled parity passes across resident/host/disk.
- [x] Runtime speculative collector provider retains partition layout and transport
  across a whole phase; five-rank coverage verifies scoped edits, replicas, an empty
  shard, aborted publication, replay and nonrefundable accounting. Nine-rank sparse
  coverage preserves causal edits, aborted evidence and original native causes.
  Native singleton CPU/Metal capture matches ordinary component values and logits.
- [x] Selected V3 prediction producer layouts and public native collector factory:
  all 21 CPU Ring TP/PP/EP combinations × resident/host/disk pass capture and
  attention/routed-unit deletion against a nonzero local reference (42.88 s).
  Neutral shared-static replicas fix borrowed embeddings on downstream PP ranks;
  storage, parameter coordinates and residency remain separate from target hooks.
- [x] Distributed V3 F32 prediction queries, projections, coordinated edits,
  peer budget/publication rollback, active-overlay snapshot/sibling replay and
  removal across all 21 CPU Ring parallel/residency cases (148.02 s). Discovery
  consumes separate auxiliary tasks and exact tensor-only prediction placement.
- [x] V3 affine 4-bit/group-32 prediction construction, effective queries, causal
  masks, coordinated edits, peer rollback and snapshot/sibling replay across all
  21 CPU Ring parallel/residency cases (231.54 s). Rank-local source conversion
  and packed target layouts reuse the ordinary quantizer and binding builder.
- [x] V3 MXFP4 prediction components, effective queries/projections, coordinated
  overlays, peer rollback and controlled replay across all 21 CPU Ring cases
  (214.22 s), using the same production source/target conversion mechanism.
  Neutral prepared-format checks cover exact scale companions; all 214 numerical
  reference tests and public CPU/Metal continuous/controlled regressions pass.
- [ ] Other prediction encodings and distributed GPU validation beyond the
  verified F32, affine and MXFP4 CPU component/parameter matrices.
- [ ] Remaining embedded state-profile acceptance for Qwen and other compatible
  families. V4 internal component acceptance is complete in the matrices below.
- [x] Grouped component write-factor joins and actual-input evidence, including
  CPU/Metal dense/FP8 mechanism checks. V4 target attention and shared block hooks
  pass neutral local/compressed/indexed, cached decode, masks, hyper reconstruction
  and survivor-recomputation checks; architecture discovery retains real channel,
  latent and stream geometry (573 unit tests passed, one existing ignored).
- [x] Typed V4 target stream cycles and final-collapse evidence, shared observed
  resident/bounded/parallel traversal, and whole-score/difference reconstruction
  in the nonzero numerical reference. Neutral producer mapping checks stream
  geometry, effective timing and invocation ownership.
- [x] V4 sequential/DSpark component scopes and native public query/edit
  acceptance across resident/host/disk execution and all seven CPU Ring
  topologies, with F32, affine and MXFP4 coverage recorded below.
- [x] Public native V4 target grouped-channel capture, deletion/keep-only and
  effective factor queries on CPU/Metal in resident, host and disk modes
  (six cases, 1.58 s). Mixed-sign factors and bounded component-column regions
  reconstruct actual writes through prefill plus two cached decode steps.

- [x] Public native V4 target whole-stream/score/difference reconstruction on
  CPU/Metal in all three residency modes, with controlled baseline, deletion and
  keep-only trials through prefill and two cached decodes (six cases, 1.90 s).
  Native coefficient rejection retains the original error source.
- [x] Sequential V4 prediction declaration and projected-sum geometry admission:
  independent embedding/hidden matrices, explicit stream broadcast, actual inputs,
  effective projections, two learned stream cycles and shared vocabulary head.
  Thirty-two prediction contract tests pass, including seven TP/PP/EP placements.
- [x] Complete the native distributed matrix and component ablation/edit variants
  for sequential V4 prediction in F32, affine 4-bit/group-32 and MXFP4. Fourteen V4
  numerical tests pass, including all four observed prediction traversals through
  cached decode and descriptor-driven score reconstruction; the public native
  baseline, deletion, query and coordinated overlay proofs below also pass.
- [x] Public sequential V4 prediction channel deletion, continuous/controlled
  parity, snapshot replay, independent siblings and stale-admission rejection on
  CPU/Metal in resident, host and disk modes (six cases, 14.21 s).
- [x] Public sequential V4 F32 prediction projected-sum/stream/score/difference
  reconstruction, effective fusion/norm/head queries and coordinated fusion,
  attention-plus-FFN overlays against an independently edited checkpoint, with
  exact restoration on CPU/Metal in all three residency modes (12.55 s).
  Affine and MXFP4 prediction acceptance is recorded below.
- [x] Sequential V4 F32 CPU Ring prediction captures, channel/routed-unit deletion,
  effective rows/columns/projections, coordinated overlays, peer reservation and
  publication rollback, snapshot/sibling replay and exact restoration across all
  21 TP/PP/EP and residency combinations (285.18 s). The shared target embedding
  and vocabulary head are retained on each prediction replica with tensor shards
  and storage accounting preserved.
- [x] Expanded sequential V4 F32 Ring matrix: exact selected-token channel,
  shared and routed keep-only masks, shared deletion, all-keep equivalence and
  surviving-unit recomputation across the same 21 execution cases. Original and
  effective captures preserve shifted prediction-row coordinates.
- [x] Sequential V4 affine 4-bit/group-32 prediction component and parameter
  matrix across all 21 CPU Ring configurations (499.35 s). This includes the
  expanded masks, survivor recomputation, global queries/projections, coordinated
  edits, peer rollback, snapshot/sibling replay, restoration and invalid-token
  rejection. Generic gather bounds protection also passes CPU/Metal regression.
- [x] Sequential V4 MXFP4 prediction component and parameter matrix across all
  21 CPU Ring configurations, alongside a fresh passing 21-case F32 regression.
  Target-stack component acceptance is recorded separately below.

- [x] Fused/context invocation contracts and descriptor-driven DSpark graph:
  one multiblock score scope, direct embedding expansion, separate cache inputs,
  pinned parameter roles and dynamic anchor-dependent vocabulary addition.
  Core admission and six shared runtime phase/edit/replay tests pass. Native
  internal hooks, local acceptance and distributed acceptance are recorded below.
- [x] V4 F32 target CPU Ring component/parameter matrix across all 21 TP/PP/EP
  and residency combinations, including whole-score/difference reconstruction
  and coordinated edits against an independently rewritten F32 checkpoint.
- [x] Expanded V4 target encoding matrices pass all 21 CPU Ring topology/residency
  cases: F32 (258.96 s), affine (346.60 s), MXFP4 (305.08 s). Readout queries
  include bounded catalogue exchange, and the full capture/score/parameter/edit
  lifecycle succeeds at eight ranks.

- [x] Public DSpark phase captures and channel interventions pass CPU/Metal across
  all three residencies (14.62 s), including continuous/controlled parity,
  snapshots, siblings, stale admission and re-admitted interventions. Actual
  context-prefill row counts come from the architecture extension.
- [x] DSpark public CPU/Metal score/parameter/independent-edit acceptance across
  all three residencies: four tests pass in 35.30 s after the cache fix, including original/effective
  Markov terms, stream/score/margin reconstruction, two edited prefixes, exact
  restoration, controlled replay and a one-token-prefix geometry regression.
- [x] DSpark distributed encoding acceptance. The expanded nonconstant two-block
  fixture exposed a paged sliding rollback defect; both CPU/Metal retention
  regressions pass (1.11 s). All 21 topology/residency combinations pass for F32
  (532.36 s), affine (943.19 s) and MXFP4 (705.96 s), including stream/score
  reconstruction, queries, overlays, rollback and replay.

- [x] Qwen hybrid prediction primitive hooks share ordinary serial/parallel
  execution. Next/3.5 dense and actual MoE variants cover prefill and two cached
  steps under six masks, exact no-op behavior, component/fusion reconstruction,
  learned-offset RMS and effective returned hidden state.
- [x] Measured shared-FFN output-gate contract and serial/parallel hooks. Preserve
  gate application after the affine write; validate scalar geometry, row mapping,
  parameter ownership and replicated gate weights.
- [x] Hybrid layered target parallel and partition observation adapters. Two-rank
  neutral fixtures cover component masks, readout, shared partial-write sums and
  collective order. All 18 Qwen regressions and 225 numerical tests pass.
- [x] Neutral prepared-source TP/EP/PP and residency acceptance for hybrid target
  components: Next/3.5 dense/MoE, 60 topology/residency combinations, four masks,
  prefill and two cached steps pass (8.21 s), including shared gate/write evidence.
- [x] Conditional target prepared-source component acceptance: dense/MoE,
  30 TP/EP/PP/residency cases, six interventions, media prefill and two cached
  decodes (2.62 s). Canonical unit-consumer ownership places pinned vision
  mergers on their consuming stages; conditional expert banks retain local plans.
- [x] Native public Qwen3.5 and Qwen3-Next dense target matrices: all nine
  topology/residency cases each pass, including bounded capture, masks, parameter
  queries/overlays and replay. The conditional dense TP/PP focus also passes.
- [x] Native Qwen3.5 text MoE: all 21 TP/EP/PP and residency cases pass with
  public capture, masks, queries, overlays and replay.
- [x] Native Qwen3-Next dense/MoE: 30 topology/residency cases plus a focused
  MoE TP run pass in 186.40 s, including public input counts and controlled edits.
- [x] Native conditional media matrices: all nine dense and 21 MoE topology/
  residency cases pass, plus three focused repeats. Native byte-frame alignment
  and ordered logical world collectives correct the two exposed pipeline faults.
- [x] Prepared replicated text MTP pairing for Next and Qwen3.5: exact source
  binding, shared fusion/head changes, lane rollback and target-state isolation.
  All 26 Qwen numerical regressions and the Nemotron prediction regression pass.
- [x] Qwen prediction discovery scopes and prepared internal hooks: canonical
  shared fusion/normalization, target embedding/head, distinct decoder depths;
  dense prepared numerical capture, seven causal masks, write reconstruction,
  prefill cache parity, accepted-token replay and failure/retry.
- [x] Public prediction capture, channel intervention, no-op parity and bounded
  failure/replay on native CPU (eight dense/MoE/tied-head configurations, two
  depths) and Metal (Next/3.5 dense, two depths).
- [x] Prepared routed prediction scope and scalar acceptance: Next/3.5 dense/MoE,
  one/two depths, actual packed source values, unit masks and failure cleanup.
- [x] Native text prediction TP/PP capture, masks, effective parameter queries,
  coordinated overlays and replay for all four variants. Shared target embedding
  and head replicas are declared and counted by neutral load requirements.
- [x] Native Next/3.5 MoE prediction capture/query/edit/replay across all 42
  topology/residency cases, including TP+PP+EP and bounded host/disk execution.
- [x] Composite prediction semantic token contract: Qwen interleaved media and
  Inkling projected image identity match ordinary target construction; all 27
  Qwen and five Inkling numerical regressions pass.
- [x] Native Next/3.5 dense prediction matrices: all eighteen cases pass,
  completing sixty text prediction topology/residency combinations.
- [x] Image-backed conditional dense prediction TP with DeepStack: complete
  public masks, effective queries, coordinated overlays and replay pass (9.61 s).
- [x] Complete image-backed conditional Qwen prediction public capture/masks,
  query/edit/replay across nine dense and 21 MoE topology/residency cases,
  including DeepStack and TP+PP+EP. The F32 text/conditional matrix totals 90 cases.
- [ ] Additional Qwen prediction source encodings and quantized materialization.
- [x] Nemotron prediction primitive, declared multi-unit schedules and prepared
  component hooks. Neutral coverage includes attention-only, *E and *E*E
  schedules, one/two depths, shared/routed ReLU², fusion and write reconstruction,
  causal edits, scope failures, head/fusion source changes and exact lane replay.
- [x] Nemotron-H native prediction F32 matrix: dense targets 9/9 and routed
  targets 21/21, plus repeated routed TP focus, including eight-rank TP+PP+EP.
  Actual masks, parameter queries, coordinated overlays, failed publication,
  restoration and controlled replay pass (303.31 seconds combined).
- [ ] Additional Nemotron prediction source encodings and final capability audit.

- [x] Inkling's neutral decoder shares ordinary and instrumented attention,
  dense SwiGLU, routed/shared banks and causal convolution execution. Nonzero
  fixtures reconstruct affine writes, learned dense scaling and complete causal
  residual contributions across prefill and two cached steps; selected scalar
  masks and exact no-op serial/TP1 parity pass.
- [ ] Complete Inkling discovery, target/readout and MTP preparation integration,
  native component/parameter/overlay verification and applicable execution matrix.

- [x] Inkling neutral layered target/readout reconstruction and causal masks;
  neutral MTP double-normalization/fusion/unit/channel masks at both depths, with
  optional chain normalization and independent cached depth state.

## Prediction source ownership and encoded placement

Qwen conditional prediction FP8 coverage uses distinguishable E4M3 values and
128-by-128 scale blocks. The focused TP run passes in 16.58 seconds, including
independent source-row decoding for seven encoded matrices, selected-position
masks, effective queries/projections, coordinated edits, rejected-publication
rollback, restoration and controlled replay. Captured floating comparisons allow
`2e-4 * max(abs(reference), 1)` for this FP8 fixture; decoded parameter rows,
edited rows, restored captures, replay and output token IDs remain exact.

The TP/PP residency cases pass with a finite 1 MiB host paging allowance for the
larger fixture's independent snapshots. Expert parallelism exposed omitted
other-owner source declarations in composite construction: the fix derives exact
keys from admitted expert recipes and reuses text partition source ownership.
The full EP focus passes in 17.29 seconds. The complete 21-case matrix then
passes in 477.02 seconds: TP, PP, EP, TP+PP, TP+EP, PP+EP and eight-rank
TP+PP+EP, each with resident, host-layerwise and disk-streamed execution. Every
case runs the same capture/query/edit/rollback/restoration/replay assertions.

```sh
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  ring_qwen_prediction_fp8_components_tensor_parallel -- --ignored --nocapture --test-threads=1
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  ring_qwen_prediction_fp8_components_expert_parallel -- --ignored --nocapture --test-threads=1
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  ring_qwen_prediction_fp8_components_conditional_moe_matrix -- --ignored --nocapture --test-threads=1
```

Nemotron MXFP4 preparation now derives executable prediction modules from exact
selected formats and retains separate local source layouts for transforms.
Expert ownership expands uniform encoded partition units into scalar intervals;
the runtime rejects missing, zero, incompatible and nonuniform mappings. That
shared helper also serves DeepSeek and Kimi Linear. Its focused runtime check
passes, and the architecture suite passes 588 tests with one existing ignore.
Five prepared prediction regressions pass in 1.37 seconds after the routed-source
construction change. After the source and packed-embedding dtype fixes, the
focused native routed TP test passes in 7.62 seconds. The complete matrix passes
all three tests in 351.64 seconds: nine dense-target cases, 21 routed-target cases
and the repeated TP focus. TP, PP and TP+PP cover dense targets; routed targets
also cover EP, TP+EP, PP+EP and eight-rank TP+PP+EP, each in resident, host-layerwise
and disk-streamed execution. Every case includes capture/mask, parameter-query,
coordinated-overlay, peer-rollback, restoration and controlled-replay assertions.
The separate CPU/Metal embedding tests pass in 1.82 seconds, with independent
E2M1/E8M0 decoding, repeated selected rows, floating replacement and restoration.

## Inkling materialized prediction observation driver

The typed materialized prediction adapter passes a nonzero neutral regression
with two physical depths (full and sliding attention), both chain-normalization
settings, three-token prefill and two cached decode steps. Eight selected-position
masks cover both hidden normalization applications, embedding normalization,
fusion, attention channels, FFN units, continuation normalization and linear
scores. Captured multiplication inputs reconstruct the unpadded vocabulary scores
with the target muP multiplier of 1.7. These multiplication inputs are read-only;
interventions use the declared mutable boundaries.

All-keep scores and retained state agree exactly with ordinary execution. Depths
advance independently; a late observer failure preserves its actual advancement,
and replay from the unchanged parent restores exact scores after every mask and
failed sibling. The target state remains unchanged. Learned dense scales are 1.3;
causal kernels use distinct nonzero taps to make each tested unit visible in F32.
The focused test passes in 0.05 seconds. This establishes the materialized driver;
Inkling discovery and its full public/native component matrix remain open.

```sh
cargo test -j1 -p eredu-architectures --test reference_numeric \
  inkling_materialized_prediction_observations_preserve_scoring_and_replay -- --nocapture
```

The full Inkling numeric regression filter then passes all nine tests in 5.02
seconds, including the materialized adapter, ordinary multimodal/TP1/TP2 paths,
rollback, chunk invariance, component masks, and independent key/value causal
convolution reconstruction before head normalization. This evidence covers only
current input positions; it does not export historical KV tensors.

```sh
cargo test -j1 -p eredu-architectures --test reference_numeric inkling -- --nocapture
```

Inkling's decoder transform declarations pass descriptor-driven reconstruction
in the same nonzero scalar fixture (0.09 seconds): five transforms per dense
layer and four per routed layer, each across serial/TP1 execution, masks, prefill
and cached decode. The test obtains kernel/scalar identities and dimensions from
`component_transforms`, joins them to real module parameters and advertised
observation paths, and independently reconstructs every transform output. This
covers the declared equations while full scalar topology and prediction-scope
integration continue.

```sh
cargo test -j1 -p eredu-architectures --test reference_numeric \
  inkling_component_masks_feed_projections_and_convolution_histories -- --nocapture
```


Nemotron MXFP4 acceptance commands:

```sh
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  mlx_mxfp4_embedding_lookup_preserves_f32_contract -- --include-ignored --nocapture --test-threads=1
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib \
  ring_nemotron_prediction_mxfp4_components_ -- --ignored --nocapture --test-threads=1
```

The relative-profile extension passes all nine Inkling numeric regressions in
4.93 seconds. The descriptor-driven fixture now reconstructs six transforms per
dense layer and five per routed layer, including the learned relative table's
shared per-head multiplication. That multiplication is independently accumulated
in F64. Readout tests distinguish the exact muP-scaled tensor from the selected
projection's actual multiplication input; the latter remains read-only evidence.

Scalar discovery passes all nine Inkling numeric regressions in 4.73 seconds:
attention Q/K/V and relative-query reads resolve to the observed affine outputs,
and declared attention/dense write columns reconstruct the actual projections.
The following readout extension passes the same nine tests in 4.75 seconds,
including descriptor-driven F64 embedding/final normalization, the declared muP
scale, complete causal residual contributions and unpadded 13-token score output.
Unit/channel masks, cached continuation and exact no-op/replay assertions remain
in that fixture. Routed unit declarations, prediction scopes and full native
component integration remain separate outstanding work.

Routed/shared discovery passes all nine Inkling numeric regressions in 4.63
seconds. The fixture resolves each bank's actual down-projection parameter through
its declared group and reconstructs the jointly weighted sparse write. It verifies
separate selected and always-on route counts against both captured coefficients
and the public routed-unit geometry. The shared branch retains its actual
always-on semantics; no standalone router events are advertised.

Inkling per-depth prediction preparation and scope reconstruction pass all ten
numeric Inkling regressions (4.77 seconds after a 25.26-second build). Reproduce:

```sh
cargo test -j1 -p eredu-architectures --test reference_numeric inkling -- --nocapture
```

The prepared fixture covers two full/sliding depths, with and without the shared
chain norm, for F32 and 4-bit affine conversion with group size 16. It checks exact
disjoint task/companion ownership, visits the optional shared norm once, binds the
prepared modules, and compares ordinary and observed cached replay. Descriptor
reconstruction uses actual effective inputs for hidden-first fusion, the repeated
hidden norm, source and prediction embedding norms, whole causal residual writes,
and optional chain normalization. The existing eight causal masks and failed-lane
replay checks remain active. Evidence: `/tmp/eredu-inkling-prediction-scopes3.log`.
The relative-position table remains a floating right-multiplied tensor; it is not
converted as a backend linear operator. Native capture/query/edit coverage and
prediction paging are not established by this neutral fixture.

Inkling transform placement: the architecture library's 25 component-partition
checks passed, including affine row ownership, channel-preserving convolution,
shared relative projections, reversed transform dependency order, and static MTP
scope placement on all TP2/PP2/EP2 replicas. Command:
`cargo test -j1 -p eredu-architectures --lib component_partition -- --nocapture`.
Result: 25 passed in 0.08 s; build 17.78 s. Log:
`/tmp/eredu-inkling-transform-placement2.log`.
The subsequent MLX test type-check also passed in 11.95 s:
`cargo check -j1 -p eredu-backend-mlx --tests --no-default-features`.
Log: `/tmp/eredu-inkling-transform-placement-native-check49.log`.
These results precede the catalog-based shared-bank placement correction and do
not establish native Inkling capture or overlay acceptance.

Inkling catalog and provider placement: two neutral integration tests passed in
0.75 s (build 1m 22s). Command:
`cargo test -j1 -p eredu-architectures --test reference_numeric inkling_selected_ -- --nocapture`.
Log: `/tmp/eredu-inkling-shared-placement6.log`.
Cold checks cover all seven TP/PP/EP combinations with one or four shared experts,
including shared and routed banks with equal cardinality. Prepared execution uses
varying signed weight entries and nonuniform gains, three prompt tokens and two
cached decode steps. Across TP2, EP2 and TP2/PP2/EP2 it checks actual provider
coordinates against cold declarations, original/effective values, routed and
shared masks, unchanged capture-only scores, and cross-topology agreement within
2e-5. This exposed and corrected both a hardcoded shared cardinality and expert
exchange being applied to replicated shared invocations. Native acceptance remains
separate from this neutral evidence.

After adding the shared-provider failure and addressable-ID cases, the focused
Inkling selection suite passed all four tests in 3.94 s (build 5.75 s):
`cargo test -j1 -p eredu-architectures --test reference_numeric inkling_selected_ -- --nocapture`.
Log: `/tmp/eredu-inkling-shared-addressable9.log`.
The added cases reject all EP2 and TP2/PP2/EP2 peers after one shared-provider
observer fails, preserve pre-forward positions and prohibit commit. Resident and
addressable execution also agree with ordinary execution when four shared experts
outnumber three routed experts. The fixture's compact-bank allowance derives from
the declared maximum route count while its member cache remains bounded.

The preceding full architecture regression passed 593 library tests (one existing
ignore) and 239 numerical tests in 0.37 s and 55.50 s respectively, with a 1m 25s
build. Command:
`cargo test -j1 -p eredu-architectures --lib --test reference_numeric`.
Log: `/tmp/eredu-inkling-shared-full-regression1.log`.
The focused four-test run above covers the later shared-failure and resident/
addressable-ID corrections that followed that full run.

Inkling compact-mask admission: the shared target/prediction decoder declaration
now identifies channel and FFN-unit axes as `component`. The neutral admission
test covers delete and keep-only plans, cold rejection, loaded mock support,
read-only effective values and out-of-range indices across both prediction depths
and optional chain normalization. Command:
`cargo test -j1 -p eredu-architectures --lib inkling_prediction_scopes_preserve_static_ownership_and_repeated_normalization -- --nocapture`.
Result: 1 passed in 0.06 s; build 11.09 s. Log:
`/tmp/eredu-inkling-component-admission2.log`. This fixes the declaration error
exposed by the first native CPU public-admission run; native reruns are separate.

Inkling public prediction capture and interventions pass on native CPU and Metal.
The fixture uses varying signed weights and nonuniform gains, two target layers
(full and sliding attention), and two MTP depths. CPU covers dense/routed targets
with chain normalization both enabled and disabled; Metal covers both chain
settings with a routed target. Capture-only output agrees with ordinary output,
channel masks change prediction logits, evidence is attributed and bounded, and
early/mid-run budget exhaustion and fresh-lane isolation pass. Commands:

```sh
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib --no-run
target/debug/deps/eredu_backend_mlx-f5971c4e042587d0 native_inkling_public_speculative_activation_admission_cpu --nocapture --test-threads=1
target/debug/deps/eredu_backend_mlx-f5971c4e042587d0 native_inkling_public_speculative_activation_admission_metal --ignored --nocapture --test-threads=1
```

Build: 31.42 s. CPU: one test/four fixture cases passed in 3.75 s; Metal:
one test/two cases passed in 2.43 s. Logs:
`/tmp/eredu-inkling-prediction-native-build48.log`,
`/tmp/eredu-inkling-public-prediction-cpu2.log`, and
`/tmp/eredu-inkling-public-prediction-metal1.log`. Distributed queries and overlays
remain separate validation; prediction weights remain resident.

A nonzero Inkling Ring test exposed lost history during paged relative attention:
only newly appended keys reached the direct operator. The neutral cache-aware
relative-attention dispatch now preserves ownership of historical keys.
`cargo test -j1 -p eredu-architectures --test reference_numeric inkling_ -- --nocapture`
passes 15 tests in 4.88 s (build 32.21 s), including a mock block cache that returns
only the append and supplies full visible history through its relative operation.
Log: `/tmp/eredu-inkling-relative-cache-neutral2.log`.
MLX test type-check passed in 5.24 s:
`cargo check -j1 -p eredu-backend-mlx --tests --no-default-features --features metal`.
Log: `/tmp/eredu-inkling-relative-cache-check53.log`. Native numerical and Ring
reruns are still required before accepting the paged correction.

The paged relative-attention correction now passes independent native CPU and
Metal tests. Both contiguous and paged outputs match a float64 scalar reference
within 2e-6 across two batches, four query heads/two KV heads, distance extent
four, full attention with logarithmic scaling, a three-token sliding window,
block sizes one and three, and four submitted spans including cached decode.
Cache device residency remains within its admitted limit. Command:
`target/debug/deps/eredu_backend_mlx-f5971c4e042587d0 paged_relative_profiles_match_scalar_reference_ --ignored --nocapture --test-threads=1`.
Result: 2 passed in 2.28 s after a 1m 04s native build. Logs:
`/tmp/eredu-inkling-prediction-native-build50.log` and
`/tmp/eredu-inkling-relative-cache-native1.log`.
The shared scan regression also passes six blockwise-attention tests, paged
full-attention/sink/softcap tests, and the public Inkling CPU/Metal cases (11 tests
total). Each selected test used `--include-ignored --nocapture --test-threads=1`;
log: `/tmp/eredu-inkling-relative-cache-regression1.log`.

Inkling routed-target prediction integration passes all 21 native CPU Ring cases:
TP2, PP2, EP2, TP2/PP2, TP2/EP2, PP2/EP2 and TP2/PP2/EP2, each with resident,
host-layerwise and disk-streamed target weights. All cases use paged target state
and resident prediction weights. The nonzero fixture checks target/prediction
captures against unpartitioned execution, prediction channel/FFN deletion and
keep-only masks, exact effective rows/columns, signed projections, coordinated
target-plus-prediction overlays, peer budget rejection, late-publication rollback,
restoration, controlled snapshots and sibling replay. Numerical capture tolerance
is 2e-5; queried/restored parameters and within-run replay remain exact. Command:
`target/debug/deps/eredu_backend_mlx-f5971c4e042587d0 ring_inkling_prediction_components_routed_target_matrix --ignored --nocapture --test-threads=1`.
Result: all 21 cases passed in 549.73 s. Log:
`/tmp/eredu-inkling-prediction-ring-routed-matrix1.log`. The preceding TP-only
diagnostic passed in 16.42 s (`/tmp/eredu-inkling-prediction-ring-tp3.log`).
These masks target prediction components; neutral target-component masks have
separate coverage. Native target-specific mask and source-format propagation,
and prediction paging, remain open.

The complementary Inkling dense-target matrix passes all nine CPU Ring cases:
TP2, PP2 and TP2/PP2 with each target weight residency. It uses the same component,
parameter, overlay, rollback and replay assertions as the routed matrix. Command:
`target/debug/deps/eredu_backend_mlx-f5971c4e042587d0 ring_inkling_prediction_components_dense_target_matrix --ignored --nocapture --test-threads=1`.
Result: nine cases passed in 168.07 s. Log:
`/tmp/eredu-inkling-prediction-ring-dense-matrix1.log`. Together the dense and
routed matrices cover 30 F32 cases; prediction weights remain resident.

Shared-family regression after the cache dispatch change: Qwen Next, Next MoE,
3.5 and 3.5 MoE all pass the existing TP2 prediction component/overlay workflow.
Command: `target/debug/deps/eredu_backend_mlx-f5971c4e042587d0 ring_qwen_text_prediction_components_tensor_parallel --ignored --nocapture --test-threads=1`.
Result: four fixture cases passed in 22.99 s. Log:
`/tmp/eredu-inkling-relative-qwen-regression1.log`.

The broader regression after the relative-attention correction passes:

- `cargo test -j1 -p eredu-architectures --lib --test reference_numeric`: 593 library
  tests passed (one existing ignore), then 242 numerical tests passed; 0.38 s and
  55.33 s respectively, build 6.65 s. Log:
  `/tmp/eredu-inkling-relative-full-regression1.log`.
- `cargo test -j1 -p eredu --no-default-features --test portable_facade --test backend_conformance`:
  61 backend conformance tests and 19 portable-facade tests passed (one existing
  ignore), in 0.73 s and 6.31 s; build 43.80 s. Log:
  `/tmp/eredu-inkling-relative-portable-facade1.log`.
- `cargo check -j1 -p eredu-nn --all-features` passed. Log:
  `/tmp/eredu-inkling-relative-nn-all1.log`.

Inkling ordinary-generation target components pass the native TP2 public workflow
for both dense and routed fixtures. The routed case has four shared experts and
two routed experts, exercising their distinct cardinalities. The existing target
harness verifies channel/unit delete and keep-only masks at prefill/decode
positions, effective-value evidence, bounded capture outcomes, global parameter
queries/projections, reversible overlays and controlled replay. Command:
`target/debug/deps/eredu_backend_mlx-f5971c4e042587d0 ring_inkling_target_components_tensor_parallel --ignored --nocapture --test-threads=1`.
Result: both fixture cases passed in 9.66 s after native build 39.10 s. Logs:
`/tmp/eredu-inkling-target-native-build51.log` and
`/tmp/eredu-inkling-target-ring-tp1.log`. The target residency/parallel matrices
remain separate validation from this focused run.

The ordinary-generation Inkling dense-target matrix also passes all nine CPU
Ring cases (TP2, PP2 and TP2/PP2, each resident/host-layerwise/disk-streamed).
Command: `target/debug/deps/eredu_backend_mlx-f5971c4e042587d0 ring_inkling_target_components_dense_matrix --ignored --nocapture --test-threads=1`.
Result: nine cases passed in 31.67 s. Log:
`/tmp/eredu-inkling-target-ring-dense-matrix1.log`.

The ordinary routed target matrix passes all 21 CPU Ring cases: TP2, PP2, EP2,
TP2/PP2, TP2/EP2, PP2/EP2 and TP2/PP2/EP2, each with resident, host-layerwise
and disk-streamed target weights. It explicitly joins all four routed/shared
invocations to loaded original/effective unit capture geometry before running
the public workflow. The fixture has four shared experts and two routed experts.
Command: `target/debug/deps/eredu_backend_mlx-f5971c4e042587d0 ring_inkling_target_components_routed_matrix --ignored --nocapture --test-threads=1`.
Result: 21 cases passed in 168.24 s, following native build 5.86 s. Logs:
`/tmp/eredu-inkling-target-native-build53.log` and
`/tmp/eredu-inkling-target-ring-routed-matrix1.log`.
Together with the dense cases, this completes all 30 F32 ordinary-target cases,
separately from the 30 F32 prediction cases. Quantized loading, other source
formats and prediction-weight paging remain separate work.

Quantized Inkling integration exposed two load-contract defects before inference:
composite partitions discarded the original source description needed for
transforms, and requirements classified the joint router as a convertible linear
module despite its floating tensor contract. The corrected handoff retains exact
source formats, local layout and routed/shared ownership. Model-wide conversion
preserves raw router matrices, relative tables and convolution kernels.

A subsequent native routed load exposed a checkpoint selection rewrite cycle.
Independent member, row and column restrictions could be exchanged recursively
without progress. Pushdown now traverses the original input once, preserves
leading source restrictions and reapplies each independent selection once.
`cargo test -j1 -p eredu-checkpoint --lib` passes all 99 tests in 0.19 s, including
all six axis-selection orders and bounded matrix-row tiles with distinct values.
Log: `/tmp/eredu-inkling-recipe-selection-tests2.log`.

The initial neutral affine regression passes nine dense and fifteen routed
partition/residency cases, with prefill and two cached decode steps (3.46 s).
It checks retained source authority, actual transformation tasks, floating-router
policy, replicated intermediate activations and final scores. The expanded
30-case version is included in the full regression. A scalar-reference GQA
mapping error found by this test was corrected to use contiguous query-head
groups. Log: `/tmp/eredu-inkling-transform-source-neutral8.log`.
Native quantized acceptance remains pending until the corrected recipe path is
rebuilt and rerun; the earlier dense affine TP success alone does not establish
routed or complete format support.

The complete regression after these changes passes with default thread stacks:
`cargo test -j1 -p eredu-architectures --lib --test reference_numeric` reports
593 library passes (one existing ignore, 0.37 s) and 244 numerical passes
(55.19 s), after a 33.05 s build. This includes all 30 affine Inkling cases,
replicated activation/score agreement at absolute tolerance `2e-4`, the invalid
relative-attention geometry checks, and ordinary composite-family regressions.
The optional original description is heap-owned across handoffs and admission
completion uses a separate construction frame. Log:
`/tmp/eredu-inkling-transform-full-regression5.log`.

Native target quantization now passes the focused CPU Ring TP2 workflow for
dense and routed Inkling, each with affine 4-bit/group-32 and MXFP4 loading.
The quantized fixture uses width 64, four query heads and two key/value heads,
so local quantization groups remain complete after TP2 partitioning. Command:
`target/debug/deps/eredu_backend_mlx-f5971c4e042587d0 ring_inkling_target_quantized_components_tensor_parallel --ignored --nocapture --test-threads=1`.
All four cases passed in 21.48 s with default thread stacks. Log:
`/tmp/eredu-inkling-target-quantized-ring-tp5.log`. Full quantized residency and
placement matrices are separate acceptance from this focused run.

Portable-facade regression after the transform handoff and selection fix passes:
`cargo test -j1 -p eredu --no-default-features --test portable_facade --test backend_conformance`:
61 conformance tests (0.74 s), 19 facade tests and one existing ignore (6.25 s),
after a 32.02 s build. Log: `/tmp/eredu-inkling-transform-portable-facade1.log`.

Quantized prediction exercised a rejected-draft replay path that the F32 fixture
had not reached. Inkling now forwards its replay observer through the same
prefill driver used for ordinary prediction-state advancement. The neutral
`inkling_prepared_prediction_depths_bind_exact_selected_formats` test passes all
four F32/affine and optional-shared-normalization variants (0.22 s). It compares
ordinary and observed replay exactly, then verifies that an intervention before
the attention cache update changes later predictions through retained state.
Command: `cargo test -j1 -p eredu-architectures --test reference_numeric inkling_prepared_prediction_depths_bind_exact_selected_formats -- --nocapture`.
Log: `/tmp/eredu-inkling-prediction-advance-neutral2.log`.

The rebuilt native prediction workflow passes all four dense/routed and
affine/MXFP4 TP2 cases, including rejected-draft replay (92.34 s; native build
44.54 s). Command:
`target/debug/deps/eredu_backend_mlx-f5971c4e042587d0 ring_inkling_prediction_quantized_components_tensor_parallel --ignored --nocapture --test-threads=1`.
Logs: `/tmp/eredu-inkling-prediction-advance-native-build58.log` and
`/tmp/eredu-inkling-prediction-quantized-ring-tp2.log`. This covers actual quantized
prediction inputs, effective parameter queries, masks, overlays and controlled
replay; the full quantized placement matrices remain separate validation.

Partitioned addressable-bank admission now accepts a replicated shared bank with
four members alongside a two-member routed bank partitioned over EP2. The focused
neutral test checks exact per-invocation route cardinalities and rejects missing
shared members, foreign units or bank IDs, nonlocal routed members, byte mismatch,
wrong distribution and wrong ownership before bank acquisition. Command:
`cargo test -j1 -p eredu-architectures --lib routed_text::tests -- --nocapture`.
All six focused tests passed in 0.01 s; build 23.10 s. Log:
`/tmp/eredu-inkling-partition-addressable-admission1.log`.

The complete neutral regression after the shared-bank constructor change and
prediction replay fix passes: `cargo test -j1 -p eredu-architectures --lib --test reference_numeric`
reports 594 library passes (one existing ignore, 0.38 s) and 244 numerical passes
(57.19 s), after a 52.81 s build. Log:
`/tmp/eredu-inkling-addressable-full-regression1.log`.

The full quantized ordinary-target matrices pass all 30 cases each for affine
4-bit/group-32 and MXFP4. Each covers nine dense and 21 routed CPU Ring cases,
with resident, host-layerwise and disk-streamed weights across the applicable
TP/PP/EP combinations. These run the same component masks, effective parameter
queries, projections, reversible overlays and controlled replay assertions as
the F32 workflow. Commands:

```sh
target/debug/deps/eredu_backend_mlx-f5971c4e042587d0 ring_inkling_target_affine_components_matrix --ignored --nocapture --test-threads=1
target/debug/deps/eredu_backend_mlx-f5971c4e042587d0 ring_inkling_target_mxfp4_components_matrix --ignored --nocapture --test-threads=1
```

Results: affine 30/30 in 252.26 s; MXFP4 30/30 in 231.28 s, using the native
binary from the prediction-replay fix. Logs:
`/tmp/eredu-inkling-target-affine-ring-matrix1.log` and
`/tmp/eredu-inkling-target-mxfp4-ring-matrix1.log`.

Kimi Linear's neutral component tests use signed, distinguishable parameters,
three-token prefill and two cached decode steps. They cover KDA and resident/paged
MLA, direct/low-rank queries and fused/split KV-B, dense and shared SwiGLU units,
last-position deletion/keep-only masks, original/effective parity and unchanged
preceding recurrent/KV updates after downstream channel masks. A PP cut between
KDA and MLA matches complete observed execution exactly for both tied and untied
heads. Declaration-driven checks reconstruct effective write columns, normalized
latent chains, causal SiLU convolution with retained history, residual writes,
final affine scores and a selected score difference. Reconstruction uses the
existing scalar tensor tolerance; the score-difference absolute bound is `2e-5`.

Command: `cargo test -j1 -p eredu-architectures --test reference_numeric kimi -- --nocapture`.
Eight focused tests passed in 0.07 s after a 37.24 s build. Log:
`/tmp/eredu-kimi-component-discovery-neutral1.log`. This is neutral validation;
native component source-format and placement coverage is tracked separately.

The complete neutral regression with these production changes passes 594 library
tests (one existing ignore, 0.38 s) and 248 numerical tests (56.96 s), after a
34.05 s build. Command:
`cargo test -j1 -p eredu-architectures --lib --test reference_numeric`.
Log: `/tmp/eredu-kimi-component-full-regression1.log`.

The additional declaration/physical-layout test and observed TP2×PP2 fixture pass
with the same focused command plus `--lib`: 17 library tests in 0.05 s and eight
numerical tests in 0.10 s. The TP/PP fixture captures both KDA and MLA on all four
layers and compares its cached outputs against ordinary replicated execution.
Log: `/tmp/eredu-kimi-component-observed-tp-neutral2.log`.

The first full affine Inkling prediction run completed 21 cases before its
TP+EP resident case exhausted the fixture's fixed 16 GiB cumulative host-query
allowance during final restored-parameter queries. Every query charges the
complete distributed catalog exchange. Admission rejected the operation across
the ranks; this run does not establish a full matrix pass. Log:
`/tmp/eredu-inkling-prediction-affine-ring-matrix1.log` (633.52 s).
The fixture now measures one catalog exchange on the selected topology and
budgets the planned repeated exchanges plus parameter work. Deliberate exhausted
budget and late-publication rejection trials remain in the workflow.

With the calibrated allowance, the focused routed TP2×EP2 prediction workflow
passes for both affine 4-bit/group-32 and MXFP4 (83.94 s), including complete
queries, projections, edits, rejection/rollback, restoration and cumulative-usage
assertions. Command:
`target/debug/deps/eredu_backend_mlx-f5971c4e042587d0 ring_inkling_prediction_quantized_components_tensor_expert_parallel --ignored --nocapture --test-threads=1`.
Log: `/tmp/eredu-inkling-prediction-quantized-ring-tp-ep1.log`.

Portable facade regression after the component-schema and Kimi integration:
`cargo test -j1 -p eredu --no-default-features --test portable_facade --test backend_conformance`
passes 61 conformance tests (0.80 s) and 19 facade tests (one existing ignore,
6.97 s), after a 41.10 s build. Log:
`/tmp/eredu-kimi-components-portable-facade1.log`.

Kimi's first native component workflows pass both SafeTensors and GGUF TP2
fixtures, including nonzero KDA/MLA channels, dense/shared units, routed units,
queries, projections, coordinated overlays and controlled replay. Independently
cached expert banks pass the same two-format focus. Commands:

```sh
target/debug/deps/eredu_backend_mlx-f5971c4e042587d0 ring_kimi_target_components_tensor_parallel --ignored --nocapture --test-threads=1
target/debug/deps/eredu_backend_mlx-f5971c4e042587d0 ring_kimi_addressable_components_tensor_parallel --ignored --nocapture --test-threads=1
target/debug/deps/eredu_backend_mlx-f5971c4e042587d0 ring_inkling_gguf_target_components_tensor_parallel --ignored --nocapture --test-threads=1
```

Results: Kimi target two-format focus 7.20 s; Kimi addressable-bank focus 7.77 s;
Inkling GGUF focus 6.47 s. Logs: `/tmp/eredu-kimi-target-ring-tp1.log`,
`/tmp/eredu-kimi-addressable-ring-tp1.log`, and
`/tmp/eredu-inkling-gguf-target-ring-tp1.log`. These use native build 60.

Kimi's first full F32 matrix passed TP and PP in resident/host/disk modes, then
stopped at EP on a harness assumption that pure EP never realizes a partition
manifest. The harness now compares realization with the retained cold selection.
The initial run is not a full matrix pass (23.03 s,
`/tmp/eredu-kimi-target-ring-matrix1.log`).

Broader neutral admission regression exposed two old tests that expected
conditional hooks to be rejected. Conditional admission is the documented
contract; runtime tests require actual application before commitment and retain
typed absence. Updated tests preserve unsupported/unverified rejection and check
conditional geometry and speculative invocation binding. No admission policy was
changed. `cargo test -j1 -p eredu-core -p eredu-runtime --lib` passes 251 core tests
(5.65 s) and 503 runtime tests (0.33 s), after a 40.42 s build.
Log: `/tmp/eredu-components-core-runtime-regression2.log`.

The corrected Kimi F32 native matrix passes all 21 TP, PP, EP and combined
configurations across resident, host-layerwise and disk execution (122.38 s).
Command: `target/debug/deps/eredu_backend_mlx-f5971c4e042587d0 ring_kimi_target_components_matrix --ignored --nocapture --test-threads=1`.
Log: `/tmp/eredu-kimi-target-ring-matrix2.log`, native build 61.

Quantized Kimi focus initially rejected transform tasks without a source
architecture, before parameter binding (`/tmp/eredu-kimi-quantized-ring-tp1.log`,
3.23 s). Routed construction now supplies that source; dense construction also
uses its exact admitted source formats. The first GGUF matrix stopped after TP
resident because its test harness lacked host-layerwise fixture dispatch
(`/tmp/eredu-kimi-gguf-target-ring-matrix1.log`, 3.43 s). The missing Kimi and
Inkling GGUF host fixture branches are supplied by the existing fixture writers.
These failing runs are not acceptance results; retries are recorded separately.

Kimi's new neutral transform fixture found replicated attention scale companions
on TP-sharded projections. The sharding declaration now uses each companion's
primary parameter identity. The fixture passes all 30 dense/routed affine
4-bit/group-16 cases, with source/target layout checks, three-token prefill and
two cached decode steps across resident/host/disk and applicable TP/PP/EP
combinations (1.92 s, build 6.00 s). Command:
`cargo test -j1 -p eredu-architectures --test reference_numeric kimi_prepared_transforms -- --nocapture`.
Log: `/tmp/eredu-kimi-transform-source-neutral3.log`. The scalar backend validates
construction and execution geometry; native quantized numerical acceptance is
reported separately.

Qwen recurrent-channel scalar validation exercises both Next and 3.5, equal and
unequal Q/K versus value heads, different key/value widths, three-token prefill
and two cached advances. An independent f64 calculation reconstructs affine
reads, fused causal SiLU convolution, additive-epsilon L2 normalization,
headwise decay/update, recurrent state, gated normalized channels and their
signed write. Delete/keep-only masks preserve earlier positions and the
operator's own subsequent recurrent state. This fixture passed in the Qwen
regression (`/tmp/eredu-qwen-recurrent-components-neutral3.log`).

The same regression passes 87 library tests (one existing ignore, 0.06 s) and
27 numerical tests including the prepared Next/3.5 dense/routed placement matrix
with recurrent-channel masks; one separate TP fixture remained zero because
its raw normalization/convolution parameters were unloaded. After binding
nonzero raw parameters in that fixture, all seven serial/TP2 trials pass for
both families and dense/routed targets (0.12 s, build 26.43 s):
`cargo test -j1 -p eredu-architectures --test reference_numeric qwen_target_components_preserve -- --nocapture`.
Log: `/tmp/eredu-qwen-recurrent-components-tp5.log`.

An early-layer mask at the middle prefill position can change later attention's
last-row prediction. The prepared fixture now asserts unchanged preceding
predictions and this causal downstream change; its final-layer masks retain the
stronger unchanged-last-row assertion. Final publication without decomposition
metadata is tested with a coherent descriptor that omits component transforms
along with their component inputs.

With projection-input evidence routed through the shared read-only projection
observer, the complete neutral regression passes 595 library tests (one existing
ignore, 0.39 s) and 250 numerical tests (57.37 s), after a 9.74 s build:
`cargo test -j1 -p eredu-architectures --lib --test reference_numeric`.
Log: `/tmp/eredu-recurrent-components-full-neutral2.log`. The preceding run caught
an undeclared effective-input record; the shared observer now reports the actual
multiplication input, including any selected input transformation.

### Public recurrent-component CPU and Metal acceptance

The ignored public tests `native_recurrent_component_analysis_cpu` and
`native_recurrent_component_analysis_metal` pass together (2 tests, 8.64 s) with:

```sh
cargo test -j1 -p eredu --no-default-features --features mlx,metal --test native_execution_control --no-run
target/debug/deps/native_execution_control-873ca620b303baf1 native_recurrent_component_analysis_ --ignored --nocapture --test-threads=1
```

Both devices exercise Qwen Next/3.5 gated-delta channels and Kimi KDA/MLA
channels with dense/shared FFN units in resident, host-layerwise and disk-streamed
execution. The public workflows compare uninterrupted and controlled trials,
selected-position masks, original/effective captures, snapshot replay, coordinated
recurrent read/control/write and FFN overlays, independent edited-source models,
and exact baseline restoration. The source fixture editor independently maps
canonical Qwen rows into its published group-major QKVZ/BA source packing;
it does not call the loader's layout recipes. An earlier run reached this
reference-editor mapping and failed before an overlay comparison; that fixture
mapping is corrected in the passing run. Logs:
`/tmp/eredu-recurrent-public-native-build3.log` (3.15 s build) and
`/tmp/eredu-recurrent-components-public-native2.log`.

The evaluation layer's additive-epsilon L2 normalization coverage passes all
three component-attribution tests (32.67 s build, <0.01 s tests):
`cargo test -j1 -p eredu-evaluation --lib component_attribution -- --nocapture`.
Log: `/tmp/eredu-recurrent-component-attribution-evaluation1.log`.

### Inkling affine prediction full matrix retry

`ring_inkling_prediction_affine_components_matrix` passes all 30 dense/routed
resident, host-layerwise and disk-streamed CPU Ring cases across TP, PP, EP and
their applicable combinations, including eight-rank TP+PP+EP (1109.83 s).
Command:

```sh
target/debug/deps/eredu_backend_mlx-f5971c4e042587d0 ring_inkling_prediction_affine_components_matrix --ignored --nocapture --test-threads=1
```

Log: `/tmp/eredu-inkling-prediction-affine-ring-matrix2.log` (native build 61).
The harness budgets repeated exact parameter catalog exchanges using measured
catalog cost and the bounded number of requests. Deliberate insufficient-budget
rejection, cumulative usage bounds, late-publication rollback, independent source
edits, restoration and controlled replay remain asserted. The earlier fixed
16 GiB cumulative host allowance failed after 21 complete cases; this passing
retry changes the fixture allowance, not production accounting.

### LFM2 transformed-source ownership and encoded expert ranges

`lfm2_prepared_transforms_retain_source_layout_across_residency_and_parallelism`
passes all 30 dense/routed TP/PP/EP and resident/host/disk cases (1.67 s, 29.93 s
build). The nonzero fixture executes three-token prefill and two cached decode
steps using F32 checkpoint sources and selected affine 4-bit/group-16 targets.
Every prepared transform must retain its typed source architecture and layout;
observed outputs agree across placements and residencies.

```sh
cargo test -j1 -p eredu-architectures --test reference_numeric lfm2_prepared_transforms -- --nocapture
```

Log: `/tmp/eredu-lfm2-transform-source-neutral6.log`. The test exposed missing
attention/convolution companion placement and an expert-range calculation that
mistook encoded partition units for individual FFN units. Both now consume
neutral parameter metadata and checked logical-range expansion. Routed
construction also retains the exact source architecture required by selected
transforms. This fixture uses aligned dense and expert widths; it does not prove
unequal source/target TP cuts caused by differing quantization alignment.

### Regression and native Kimi quantized TP after recurrent integration

The complete architecture regression passes after the LFM2 fixes: 595 library
tests with one existing ignore (0.38 s), and 251 numerical tests (56.45 s).
Logs: `/tmp/eredu-lfm2-components-library1.log` and
`/tmp/eredu-lfm2-components-full-numeric1.log`.

Native build 62 succeeds in 2 min 06 s. Its
`ring_kimi_quantized_components_tensor_parallel` test passes all three TP2 cases:
affine, MXFP4, and independently cached affine experts (18.33 s). This validates
the retained source construction and companion sharding with actual native
materialization, captures, masks, effective queries, coordinated parameter edits
and controlled replay. Log: `/tmp/eredu-kimi-quantized-ring-tp2.log`.

```sh
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib --no-run
target/debug/deps/eredu_backend_mlx-f5971c4e042587d0 ring_kimi_quantized_components_tensor_parallel --ignored --nocapture --test-threads=1
```

The Qwen Next hybrid MoE TP2 native component case passes with the new required
recurrent projected/convolved/control/channel captures (5.84 s):
`target/debug/deps/eredu_backend_mlx-f5971c4e042587d0 ring_public_component_capture_qwen_next_hybrid_moe_tensor --ignored --nocapture --test-threads=1`.
Log: `/tmp/eredu-qwen-recurrent-components-ring-tp1.log`.

Portable facade regression also passes after component schema 10 / architecture
schema 11: 61 backend-conformance tests (0.75 s) and 19 facade tests with one
existing ignore (6.74 s), build 11.20 s. Command:
`cargo test -j1 -p eredu --no-default-features --test portable_facade --test backend_conformance`.
Log: `/tmp/eredu-recurrent-components-portable-facade1.log`.

LFM2 native quantized TP2 acceptance passes all five cases (15.73 s): dense and
routed affine/MXFP4, plus independently cached routed affine experts. Command:
`target/debug/deps/eredu_backend_mlx-f5971c4e042587d0 ring_lfm2_quantized_components_tensor_parallel --ignored --nocapture --test-threads=1`.
Log: `/tmp/eredu-lfm2-quantized-components-ring-tp1.log`.

The component schema changes pass the complete neutral core/runtime suites:
251 core tests (6.60 s) and 503 runtime tests (0.30 s), build 16.79 s.
Command: `cargo test -j1 -p eredu-core -p eredu-runtime --lib`.
Log: `/tmp/eredu-recurrent-components-core-runtime1.log`.

### Inkling MXFP4 prediction full matrix

`ring_inkling_prediction_mxfp4_components_matrix` passes all 30 dense/routed
resident, host-layerwise and disk-streamed CPU Ring cases across the applicable
TP, PP and EP combinations, including eight-rank TP+PP+EP (929.39 s).

```sh
target/debug/deps/eredu_backend_mlx-f5971c4e042587d0 ring_inkling_prediction_mxfp4_components_matrix --ignored --nocapture --test-threads=1
```

Log: `/tmp/eredu-inkling-prediction-mxfp4-ring-matrix1.log`, native build 62.
The assertions include component masks, effective queries, signed projections,
coordinated target/prediction overlays, independent edited references, rollback,
restoration and controlled replay. Target residency is varied; prediction
extension weights remain pinned. This binary predates the subsequent composite
independent-bank construction changes, which require separate validation.

### Composite partition banks and shared invocation ownership

The focused numerical run passes 11 tests (23.79 s, 54.90 s build):

```sh
cargo test -j1 -p eredu-architectures --test reference_numeric independent -- --nocapture
```

Log: `/tmp/eredu-composite-partition-banks-neutral7.log`. New coverage includes:

- Qwen conditional MoE: 21 independent-bank TP/PP/EP and residency cases,
  with image/DeepStack input, six component-mask modes, prefill and two cached
  decode steps. Resident-bank coverage separately passes all 30 dense/routed cases.
- Inkling: 84 cases comparing resident and independent banks for F32 and
  affine 4-bit/group-16 weights across all seven applicable TP/PP/EP combinations
  and three residencies. Four shared experts remain distinct from two routed
  experts. The F32 fixture uses width 32; the affine fixture uses width 64 so
  TP2 retains the packed format's required 32 input columns. The shared cache
  has a 49,152-byte device budget, and the test requires actual acquisitions,
  completions, shared member 3, bounded peak usage and eviction.
- Muse, Qwen3-VL and sparse Gemma 4: 126 resident/independent-bank cases with
  nonzero checkpoints, layer-output comparisons and two cached decode steps.
  Muse and Qwen3-VL include actual image prefill. These bank tests compare
  execution and unit-boundary values; they do not establish complete component
  attribution support for those families.

The tests exposed and now cover retained task exclusions, shared-member
projection, distinct bank-invocation and execution-owner indices, Gemma 4's
stage-local state access, and its EP-local expert module construction. Cache
byte evidence is the scalar mechanism's selected-storage accounting, not an
assertion about a native allocator. Native composite-bank acceptance and full
regression checks are tracked separately.

The post-change library regression passes 595 architecture tests with one
existing ignore (0.37 s), and all 503 runtime tests (0.27 s), including foreign
member-owner rejection. Build time: 23.88 s. Command:
`cargo test -j1 -p eredu-runtime -p eredu-architectures --lib`.
Log: `/tmp/eredu-composite-banks-library1.log`.

The complete numerical executable also passes all 254 tests (57.63 s):
`target/debug/deps/reference_numeric-1b6ae52321494d51`.
Log: `/tmp/eredu-composite-banks-full-numeric1.log`.

Portable facade regression passes 61 conformance tests (0.75 s) and 19 facade
tests with one existing ignore (6.44 s), build 25.16 s:
`cargo test -j1 -p eredu --no-default-features --test portable_facade --test backend_conformance`.
Log: `/tmp/eredu-composite-banks-portable-facade1.log`.

### Muse normalization replacement after warm execution

The four centered sublayer normalizations now derive their gain from the
current parameter slot. A cached gain previously survived replacement: the
reproducer obtained `0.3100326` where the independently edited fresh model
produced `0.14306504` for its first output element. The regression warms the
model, edits each norm, compares two inputs against a fresh edited model, then
restores the original weights and checks baseline agreement.

```sh
cargo test -j1 -p eredu-architectures --test reference_numeric muse -- --nocapture
```

All three selected tests pass (0.05 s, 39.59 s build), including existing
chunked decode and TP2 image/text parity. Log:
`/tmp/eredu-muse-normalization-replacement-neutral2.log`.
This is normalization replacement coverage; complete Muse component capture and
public overlay acceptance remain separate work.

After this normalization fix, the complete numerical executable passes all
255 tests (57.69 s). Log: `/tmp/eredu-composite-banks-full-numeric2.log`.

### Native composite independent-bank TP2 acceptance

Native build 64 passes all six cases in
`ring_composite_independent_bank_components_tensor_parallel` (35.93 s):
Inkling F32 SafeTensors, GGUF and load-time affine; Qwen Next MoE, Qwen 3.5 MoE,
and Qwen 3.5 conditional MoE with image input. They exercise actual component
capture and masks, effective parameter access, coordinated overlays, restoration
and controlled replay through independently cached providers.

```sh
target/debug/deps/eredu_backend_mlx-f5971c4e042587d0 ring_composite_independent_bank_components_tensor_parallel --ignored --nocapture --test-threads=1
```

Log: `/tmp/eredu-composite-independent-components-native-tp2.log`.
The first attempt stopped on a test-only missing media-mode flag; the corrected
retry includes the image input and its complete expected unit construction.
These native cases use the ordinary cache limits; forced eviction is established
by the separate numerical matrices, not inferred from this TP2 run.
`cargo check -j1 -p eredu-backend-mlx --no-default-features` also passes (5.00 s),
log `/tmp/eredu-composite-banks-backend-check2.log`.

### Native Inkling independent-bank full placement matrices

The F32 and published GGUF fixtures each pass all 21 CPU Ring placements:
TP, PP, EP, TP/PP, TP/EP, PP/EP, and TP/PP/EP, each with resident,
host-layerwise, and disk-streamed target parameters. These exercise component
capture, masks, effective-parameter queries, overlays, restoration and replay
with independently acquired expert banks. Native cache defaults apply; forced
bank eviction is established by the neutral bounded-cache tests above.

```sh
target/debug/deps/eredu_backend_mlx-f5971c4e042587d0 ring_inkling_independent_bank_components_matrix --ignored --nocapture --test-threads=1
target/debug/deps/eredu_backend_mlx-f5971c4e042587d0 ring_inkling_gguf_independent_bank_components_matrix --ignored --nocapture --test-threads=1
```

Both used frozen native build 64: F32 **177.09 s**, GGUF **208.78 s**.
Logs: `/tmp/eredu-inkling-independent-components-ring-matrix1.log` and
`/tmp/eredu-inkling-gguf-independent-components-ring-matrix1.log`.

### Muse component equations and discovery

Five focused numerical tests pass for Muse's shared block/model driver. They
cover both HF centered gains and GGUF full gains, nonzero component sums before
the separate postnorms, epsilon/gain reconstruction after the projection,
selected-token channel/unit masks, recomputed keep-only survivors, tied and
untied vocabulary heads, and the non-unit output multiplier applied after the
actual projection. Existing image TP2 and cached-decode tests also pass.

```sh
cargo test -j1 -p eredu-architectures --test reference_numeric muse_ -- --nocapture
cargo test -j1 -p eredu-core -p eredu-architectures --lib
```

Focused result: **5 passed, 0.02 s** after a 76 s build,
`/tmp/eredu-muse-component-hooks-neutral3.log`. The GGUF block fixture explicitly
stores nonzero effective normalization gains; zero placeholder gains do not
provide a meaningful component test.

Library result: **596 architecture tests passed, 1 ignored**, and **252 core
tests passed** (0.38 s and 6.56 s; build 41.15 s),
`/tmp/eredu-muse-component-library1.log`. This includes discovery for dense and
sparse Muse under both weight conventions and schema round trips with an omitted
legacy head-scale field. Native Muse component/overlay acceptance and the new
full placement matrix remain separate validation steps.

The affine-quantized Inkling independent-bank matrix also passes all 21 CPU Ring
placements (215.39 s, frozen native build 64):

```sh
target/debug/deps/eredu_backend_mlx-f5971c4e042587d0 ring_inkling_affine_independent_bank_components_matrix --ignored --nocapture --test-threads=1
```

Log: `/tmp/eredu-inkling-affine-independent-components-ring-matrix1.log`.
This has the same capture, mask, effective-query, overlay and replay checks as
the F32/GGUF independent-bank matrices; the native cache uses its ordinary limits.

The first full Muse component-placement pass covers **51 placements** against
an ordinary resident reference loaded from the same exact checkpoint recipes:
9 dense TP/PP placements and 42 sparse placements across both bank policies,
all applicable TP/PP/EP combinations, and all three target residency policies.
Each runs six causal trials with image prefill and two cached decode steps.
The independent bank is bounded to 1,152 bytes and demonstrates eviction.
The test also rejects a postnorm with a foreign pipeline gain or mismatched
input width. Result: **6 tests passed, 6.53 s**, build 30.66 s,
`/tmp/eredu-muse-component-placement-neutral3.log`.

The serial reference uses ordinary execution; a one-rank topology is not passed
to the explicit parallel constructor. Sparse postnorm capture required extending
the shared placement rule to infer its input from the established whole-write
output while still checking exact geometry and parameter ownership.

The strengthened Muse placement test also passes (**6 tests, 5.44 s**; build
38.37 s), `/tmp/eredu-muse-component-placement-neutral4.log`. Its independent
serial checks sum actual effective sparse units times expert write columns and
route coefficients in F64, then compare with the executed complete write before
postnorm. For all six trial modes it reconstructs selected affine scores and a
target-versus-alternative difference from the assembled embedding, postnorm
residual writes, and an explicit actual-head-input correction. This includes
readout masks without treating the mask correction as a component contribution.

After the Muse placement change, the full numerical executable passes **258
tests in 61.59 s**, `/tmp/eredu-muse-components-full-numeric1.log`. The architecture
library passes **596 tests with 1 ignored** (0.39 s),
`/tmp/eredu-muse-component-library2.log`, and backend compilation without default
features passes (13.33 s), `/tmp/eredu-muse-components-backend-check1.log`.
The portable facade and backend conformance suites also pass,
`/tmp/eredu-muse-components-portable-facade1.log`.

Muse also explicitly advertises the implemented hooks through ordinary, tensor-parallel,
and partition selectors. The readout regression checks all three selectors and
passes (1 test, 0.01 s; build 15.78 s),
`/tmp/eredu-muse-hook-capability-neutral1.log`.

The native Qwen independent-bank component matrix passes **63 cases in 616.27 s**:
Qwen3-Next-MoE, Qwen3.5-MoE, and conditional Qwen3.5-MoE each cover seven
TP/PP/EP combinations across resident, host-layerwise, and disk-streamed target
execution. The cases exercise capture, causal masks, effective queries, overlays,
restoration and replay. This run uses the ordinary native bank cache limits;
forced eviction remains established by the separate neutral bank tests.

```sh
target/debug/deps/eredu_backend_mlx-f5971c4e042587d0 ring_qwen_independent_bank_components_matrix --ignored --nocapture --test-threads=1
```

Log: `/tmp/eredu-qwen-independent-components-ring-matrix1.log`. The binary was
built before the new Muse component integration.

The numerical tensor's exact GELU oracle now evaluates the erf equation in F64;
it previously used a tanh approximation with an incorrect coefficient. A fixed
Python `math.erf` table checks the replacement, and the existing non-gated
component test now covers exact GELU with independent Simpson integration.
The full numerical suite passes **259 tests in 60.21 s**,
`/tmp/eredu-exact-gelu-full-numeric1.log`. This also reruns Muse's 51 placements
with the corrected vision activation and current hook capability declarations.

### Muse public CPU and Metal components

The public facade tests pass on **CPU and Metal (2 tests, 6.30 s)**,
`/tmp/eredu-muse-components-public-native1.log`. Both tied and untied output heads
cover resident, host-layerwise and disk-streamed execution. Nonzero selected-position
channel/unit masks agree between uninterrupted and controlled runs; snapshot replay
preserves the captured effective values. The fixture uses query scale 1.3, output
multiplier 1.9, and softcap 7; F64 evaluation of the declared complete-score transform
agrees within `2e-6 + 2e-6 * abs(expected)`.

Coordinated overlays update Q/K/V/O, the channel gate, gated FFN read/write parameters,
all four pre/post-normalization gains, and the token embedding after baseline execution
has warmed the model. Two exact token prefixes match a separately edited physical
SafeTensors reference, and removing the overlay restores baseline scores exactly.
The source file remains unchanged. This public fixture exercises text execution;
actual image assembly is covered separately by the neutral and native Ring matrices.

```sh
cargo test -j1 -p eredu --no-default-features --features mlx,metal --test native_execution_control --no-run
target/debug/deps/native_execution_control-873ca620b303baf1 native_muse_component_analysis_ --ignored --nocapture --test-threads=1
```

Build: 1m49s, `/tmp/eredu-muse-public-native-build1.log`.

Muse's dense native Ring matrix passes **all 9 cases in 33.67 s**: TP, PP and
combined TP/PP across resident, host-layerwise and disk-streamed execution. Each
case includes actual image assembly, component capture, masks, effective queries,
coordinated overlays, restoration and controlled replay. The focused TP run also
passes (4.68 s). Logs: `/tmp/eredu-muse-components-dense-ring-matrix1.log` and
`/tmp/eredu-muse-components-ring-tp2.log`. The initial attempt exposed a test-harness
assumption that all media families use DeepStack; the harness now tests those
boundaries only when the architecture declares them.

```sh
target/debug/deps/eredu_backend_mlx-f5971c4e042587d0 ring_muse_components_dense_matrix --ignored --nocapture --test-threads=1
```

The first sparse native attempt exposed a checkpoint admission defect: packed
and split gate/up layouts both used their shared down projection as a layout
discriminator. The architecture plan now selects the representation from gate/up
identities while still requiring the complete down projection. Header-only tests
accept packed, split and individual-expert storage with released names or canonical
aliases, and reject missing down projections and mixed layouts. All **7 Muse
checkpoint tests pass (0.02 s)**, `/tmp/eredu-muse-expert-layout-admission1.log`.
Sparse native acceptance follows in a rebuilt binary.

Muse sparse TP now passes with both resident and independent expert providers
(**2 cases, 11.13 s**), `/tmp/eredu-muse-components-routed-ring-tp2.log`, after
the packed/split admission fix. The first quantized media trial then found that
a global format was incorrectly applied to a 12-column patch projection.
Load-time selection now uses the existing checked media geometry and retains
dense matrices when the requested blocks do not fit. Focused affine/MXFP4 policy
tests pass (**8 checkpoint tests, 0.01 s**),
`/tmp/eredu-muse-media-quantization2.log`; the complete architecture library passes
**598 tests with 1 ignored in 0.38 s**, `/tmp/eredu-muse-checkpoint-library3.log`.
Native quantized acceptance remains pending the rebuilt binary.

The retry exposed a second admission layer that still requested packed execution
for every vision matrix. The architecture now retains its optional alignment
policy in neutral parameter requirements, including after canonical recipe
finalization. Cold-admission tests compare every media projection with the
actual load-time configuration for affine groups 32/64 and MXFP4. GGUF text
geometry has separate coverage without a vision configuration. The full suites
pass **600 architecture tests (1 ignored) and 504 runtime tests**, in 0.37 s
and 0.28 s, respectively; build 33.62 s. Log:
`/tmp/eredu-muse-admission-library6.log`.

The initial sparse Ring matrix completed 18 cases before the first eight-rank
TP/PP/EP media prefill timed out. Diagnosis found text lookup collectives inside
the preceding replicated vision group. Muse now defers those lookups to text
group entry and declares their individual text-segment shapes. The seven focused
neutral Muse tests pass in 4.53 s, including all 51 component placements:
`/tmp/eredu-muse-segmented-ingress-neutral2.log`. Native eight-rank, quantized
and GGUF acceptance still require the rebuilt binary; these failures are not
recorded as passed matrix cases.

The full numerical regression suite after the segmented/deferred-ingress change
passes **260 tests in 59.15 s**, including Inkling shared-ingress and composite
placement regressions: `/tmp/eredu-muse-ingress-full-numeric1.log`. This binary
predates the subsequent admission-contract change, which is covered by the
600/504 library suites above.

The rebuilt native regression now passes the previously failing eight-rank
TP/PP/EP Muse case in **18.10 s**, including actual image ingress, component
capture, masks, queries, overlays, restoration and controlled replay:
`/tmp/eredu-muse-components-triple-native2.log`. Build: 2m00s,
`/tmp/eredu-muse-components-native-build70.log`.

```sh
target/debug/deps/eredu_backend_mlx-f5971c4e042587d0 ring_muse_components_tensor_pipeline_expert --ignored --nocapture --test-threads=1
```

The shared media-harness regression passes conditional Qwen 3.5 TP/PP
(**1 four-rank case, 7.00 s**),
`/tmp/eredu-qwen-conditional-component-harness-regression1.log`.

Subsequent Muse native trials advanced past the patch quantization and GGUF
schema failures but found mismatched generated companion names and a vision
static role on a text-only GGUF. Canonical source/companion naming now uses one
architecture function, and cold/loaded vision transport shares optional-role
ownership. The **42 focused Muse tests pass in 0.03 s**, including released
source provenance, canonical companions and absent-projector ownership:
`/tmp/eredu-muse-canonical-admission-neutral1.log`. The failed native attempts
are `/tmp/eredu-muse-components-quantized-ring-tp3.log` and
`/tmp/eredu-muse-components-gguf-ring-tp2.log`; quantized/GGUF acceptance is
still pending the next rebuilt binary.

Portable facade/conformance validation after these admission fixes passes
**61 conformance tests and 19 facade tests (1 existing ignore)**, in 0.78 s
and 7.31 s; build 32.25 s:
`/tmp/eredu-muse-admission-portable-facade2.log`.

Muse's complete F32 sparse Ring matrix now passes **all 21 cases in 155.80 s**:
TP, PP, EP, TP/PP, TP/EP, PP/EP and TP/PP/EP, each with resident, host-layerwise
and disk-streamed weights. This includes all three previously unaccepted
eight-rank cases. Every case exercises actual image prefill, cached decode,
component masks/capture, effective queries, coordinated overlays, restoration
and controlled replay. Log: `/tmp/eredu-muse-components-routed-ring-matrix2.log`.

```sh
target/debug/deps/eredu_backend_mlx-f5971c4e042587d0 ring_muse_components_routed_matrix --ignored --nocapture --test-threads=1
```

The next quantized trial reached partition materialization and exposed a missing
typed source architecture. Muse now retains its original-format partition and
physical layout through the existing composite handoff. A new neutral matrix
passes **12 affine cases in 3.29 s**: dense/sparse, TP and combined TP/PP
(with EP for sparse), each across all three residencies, using image prefill
and two cached decode steps. The common visitor verifies source retention
before materialization, and outputs agree across placements. Log:
`/tmp/eredu-muse-selected-transforms-neutral3.log`. The scalar payload binder
models affine transforms; MXFP4 numerical validation remains a native task.

A GGUF attempt subsequently exposed missing learned Q/K normalization gains
in partition metadata. Those head-width vectors now have replicated topology
entries; the Hugging Face convention declares no such learned weights. Native
quantized and GGUF retries remain pending this rebuild.

Native Muse quantized TP verification now passes **all 5 cases in 25.92 s**:
dense and sparse affine 4-bit/group-32 and MXFP4, plus sparse affine with
independent expert banks. The cases cover image prefill, cached decode,
component capture/masks, effective queries, coordinated overlays, restoration
and controlled replay. Log: `/tmp/eredu-muse-components-quantized-ring-tp5.log`.
Build: 1m55s, `/tmp/eredu-muse-components-native-build73.log`, after removing
regenerable incremental cache to recover build space.

```sh
target/debug/deps/eredu_backend_mlx-f5971c4e042587d0 ring_muse_components_quantized_tensor_parallel --ignored --nocapture --test-threads=1
```

Muse's full affine 4-bit/group-32 Ring matrix passes **all 30 cases in 221.19 s**:
9 dense and 21 sparse placements across resident, host-layerwise and disk-streamed
execution, including all applicable TP/PP/EP combinations. Each case includes
actual media, capture/masks, effective parameter operations, coordinated edits,
restoration and controlled replay. Log:
`/tmp/eredu-muse-components-affine-ring-matrix1.log` (native build 73).

```sh
target/debug/deps/eredu_backend_mlx-f5971c4e042587d0 ring_muse_components_affine_matrix --ignored --nocapture --test-threads=1
```

The subsequent sparse GGUF admission failure came from raw-versus-prepared
expert source names. Expert recipes now accept the exact raw GGUF or canonical
prepared identity, retaining the selected source keys. The focused neutral test
passes, checking both namespaces, packed geometry and every missing required
projection: `/tmp/eredu-muse-gguf-expert-aliases-neutral1.log`. The native GGUF
retry remains pending; the failed attempt is not counted as matrix acceptance.

Qwen3-VL now connects the shared decoder component hooks and prepared DeepStack
additions to ordinary, routed-provider, TP and partition traversal. Its focused
library regression passes **87 tests with 1 existing ignore in 0.08 s**:
`/tmp/eredu-qwen-vl-observation-library2.log`. Component numerical and native
matrices are in progress and are not yet acceptance evidence.

Qwen3-VL's new nonzero component matrix passes **all 51 placements in 8.25 s**:
9 dense plus 21 sparse ordinary-bank and 21 sparse independent-bank placements.
Every trial uses image prefill and two cached decode steps, with baseline,
channel deletion, FFN/expert keep-only, combined masks, assembled-embedding,
readout-normalization and DeepStack interventions. Selected sparse units and
actual routing coefficients reconstruct expert writes; dense units/channels
reconstruct projection writes; residual terms reconstruct two affine scores and
their difference using the actual readout input. Capture-only execution agrees
exactly with ordinary serial inference. A 4,608-byte independent-bank allowance
covers two selected experts and verifies cache completion, bounded peak usage
and eviction. Logs: `/tmp/eredu-qwen-vl-components-neutral3.log`.

The architecture library passes **602 tests with 1 existing ignore** after the
Qwen3-VL hooks and Muse GGUF source-alias fix:
`/tmp/eredu-qwen-vl-full-library1.log`. Native Qwen3-VL acceptance remains pending.

```sh
cargo test -j1 -p eredu-architectures --test reference_numeric qwen_vl_components_follow_media_residency_and_all_parallel_bank_placements -- --nocapture
cargo test -j1 -p eredu-architectures --lib
```

Muse's native F32 independent-bank matrix passes **all 21 cases in 178.91 s**,
covering every TP/PP/EP combination across resident, host-layerwise and
disk-streamed execution. Actual image prefill and cached decode exercise sparse
captures, masks, effective queries, coordinated overlays, restoration and
controlled replay. Log:
`/tmp/eredu-muse-components-independent-ring-matrix1.log` (native build 73).

```sh
target/debug/deps/eredu_backend_mlx-f5971c4e042587d0 ring_muse_components_independent_bank_matrix --ignored --nocapture --test-threads=1
```

The first two full numerical regressions exposed a legacy bank-trace mismatch.
The provider deliberately suppresses its local `.mlp.output` during TP because
that value still needs architecture reduction. The trace now compares the
architecture-declared complete component writes, excludes only those local MoE
provider diagnostics, and retains the original traversal/media evidence. Both
ordinary and replica observation callbacks retain strict duplicate checks.
The focused composite-bank regression passes in **3.95 s**; full-suite results
are recorded below. Logs: `/tmp/eredu-qwen-vl-full-numeric1.log`,
`/tmp/eredu-qwen-vl-full-numeric2.log`, and
`/tmp/eredu-qwen-vl-bank-trace-neutral1.log`.

The repaired Muse GGUF path passes the three focused native TP cases in
**12.85 s** (dense, sparse and independently cached sparse), followed by **all
30 dense/sparse matrix cases in 177.10 s** across the three residencies and every
applicable TP/PP/EP combination. These are F32 GGUF text-base fixtures with
learned Q/K normalization gains; they do not establish optional GGUF projector
or quantized-GGUF coverage. Logs:
`/tmp/eredu-muse-components-gguf-ring-tp5.log` and
`/tmp/eredu-muse-components-gguf-ring-matrix1.log` (native build 74).

```sh
target/debug/deps/eredu_backend_mlx-f5971c4e042587d0 ring_muse_components_gguf_tensor_parallel --ignored --nocapture --test-threads=1
target/debug/deps/eredu_backend_mlx-f5971c4e042587d0 ring_muse_components_gguf_matrix --ignored --nocapture --test-threads=1
```

Qwen3-VL's initial native TP attempt exposed the shared test harness's default
text prefix; explicit dense/sparse Qwen3-VL paths now replace that assumption.
Its first public CPU/Metal attempt correctly rejected a text-only capture plan
requesting conditional media terms in decode. The helper now selects applicable
terms for its exact token-only input; image coverage remains in the native
matrix. These attempts are failures, not accepted native component results:
`/tmp/eredu-qwen-vl-components-ring-tp1.log` and
`/tmp/eredu-qwen-vl-components-public-native1.log`.

Muse's native MXFP4 matrix passes **all 30 cases in 214.20 s**, covering dense
and sparse targets across the three residencies and every applicable TP/PP/EP
combination, with image prefill and cached decoding. Log:
`/tmp/eredu-muse-components-mxfp4-ring-matrix1.log` (native build 74).

```sh
target/debug/deps/eredu_backend_mlx-f5971c4e042587d0 ring_muse_components_mxfp4_matrix --ignored --nocapture --test-threads=1
```

Qwen3-VL's updated format and admission library tests pass **88 tests with one
existing ignore in 0.07 s**. They include canonical/relative vision-format
agreement, conflicting-alias rejection and cold media packing agreement for
affine group-32, group-64 and MXFP4. The fixed dense vision position table is
excluded from transformable projection geometry. Log:
`/tmp/eredu-qwen-vl-quantization-library5.log`. Numerical selected-transform
and native retries remain pending.

Qwen3-VL selected-source transform coverage passes **12 affine cases in 4.12 s**:
dense/sparse, TP and combined pipeline cuts (including sparse EP), and all three
residencies. These use actual image prefill plus two cached decode steps and
check typed prepared sources and cross-placement numerical agreement. Log:
`/tmp/eredu-qwen-vl-selected-transforms-neutral1.log`.

Native Qwen3-VL TP reaches component capture, then exposed a test-only numeric
state reporter that assumed every fixed tensor was Float32. Media rotary
positions are Int32. The reporter now converts integer fixture values exactly
and rejects unrepresentable values instead of silently rounding them; production
state storage is unchanged. Native retry is pending (failed diagnostic log:
`/tmp/eredu-qwen-vl-components-ring-tp3.log`).

After correcting the legacy trace, the full neutral numerical suite passes
**263 tests in 59.84 s**. A preceding run passed 262 tests but could not create
one fixture because the build volume was full; it is not numerical failure
evidence. The successful rerun follows cleanup of inactive compiler
intermediates. Log: `/tmp/eredu-qwen-vl-full-numeric4.log`; command:

```sh
target/debug/deps/reference_numeric-1b6ae52321494d51
```

Qwen3-VL native focused TP coverage now passes **three F32 cases in 12.53 s**
(dense, sparse, independently cached sparse) and **five transformed cases in
30.90 s** (dense/sparse affine and MXFP4, plus independently cached affine
sparse). These use actual image/DeepStack prefill, cached decoding, component
masks and the expanded coordinated edit plan covering all replaceable fixture
component weights, including grouped expert reads/writes. Logs:
`/tmp/eredu-qwen-vl-components-ring-tp4.log` and
`/tmp/eredu-qwen-vl-components-quant-ring-tp1.log` (native build 76b).
The broader native matrices remain in progress; public CPU/Metal results follow.

The public Qwen3-VL CPU and Metal component tests pass **2 tests in 7.22 s**,
covering tied/untied heads and three residencies, text-prefix capture/masks,
controlled replay, and eleven coordinated weights/norms/embeddings against
physically edited references and restoration. Image/DeepStack coverage is in
the separate native Ring and neutral tests. Public build 2; log:
`/tmp/eredu-qwen-vl-components-public-native2.log`.

Before Gemma4 additions, the updated architecture library passes **603 tests,
one existing ignore, in 0.37 s**. Portable checks pass **61 backend-conformance
tests in 0.76 s** and **19 facade tests, one ignore, in 7.08 s**. Logs:
`/tmp/eredu-qwen-vl-full-library2.log` and
`/tmp/eredu-qwen-vl-portable-facade1.log`.

The first larger Qwen3-VL native matrix completed **27 placements** before
an output-publication deadline in sparse TP2/PP2/EP2 resident replay. The cause
was the test harness: it polled worker exit before draining diagnostic pipes.
Four workers filled their 64 KiB stderr pipes and blocked before submitting the
broadcast. Both pipes now drain concurrently throughout worker execution.
A subprocess regression writes 256 KiB to each pipe and passes in **0.02 s**;
the focused eight-rank component replay passes in **15.38 s**, with the same
collective tracing enabled. No production collective change was required.
Logs: `/tmp/eredu-ring-output-drain2.log` and
`/tmp/eredu-qwen-vl-components-cartesian-replay2.log` (native build 78).
The complete F32 matrix retry passes **all 30 placements in 183.80 s**, including all three eight-rank residency modes. Log: `/tmp/eredu-qwen-vl-components-ring-matrix2.log`. Independent-bank and quantized matrices remain in progress.

Muse's affine independent-bank matrix passes **all 21 placements in 176.92 s**
with the expanded coordinated edit plan, actual media input, captures, masks,
queries and controlled replay. Log:
`/tmp/eredu-muse-components-affine-independent-ring-matrix1.log` (native build 76b).

Muse's F32 GGUF independent-bank matrix passes **all 21 placements in 110.54 s**
with the expanded coordinated component edit plan. This completes the tested
F32, affine, and F32 GGUF independent-bank matrices; optional GGUF projectors
and packed GGUF encodings remain separate gaps. Log:
`/tmp/eredu-muse-components-gguf-independent-ring-matrix1.log` (native build 76b).

Gemma4's first topology regression passes four dense/sparse and projected/reused
value configurations, including an actual shared-KV publisher, distinct dense
exact-GELU and expert approximate-GELU, branch/common postnorms, per-layer writes,
and learned scaling of the whole residual. The full discovery test selection
passes **19 tests in 0.38 s**. Log: `/tmp/eredu-gemma-components-discovery3.log`.

The focused Gemma4 numerical regression passes **one test with four cases in
0.04 s**: dense/sparse branches with/without per-layer inputs, nonuniform norm
gains, nonunit residual scaling, signed component writes plus projection bias,
exact component-position masks, recomputed surviving units, and warmed scalar
replacement/restoration. Log: `/tmp/eredu-gemma-components-numeric1.log`.
Shared-KV placement, native and public component validation remain in progress.

The full neutral numerical suite with the initial Gemma4 component hooks passes
**264 tests in 234.24 s**, serialized with `--test-threads=1`. Log:
`/tmp/eredu-gemma-components-full-numeric1.log`. This predates the new shared-KV
component placement test and its subsequent checkpoint/placement corrections.

The new Gemma4 fixture exposed missing attention-bias admission. SafeTensors
and GGUF now declare biases for exactly the projections the configured layer
executes: Q/O everywhere, K on publishers, and V when values are independently
projected. Sliding attention retains its independent V even when full attention
reuses K. Learned whole-residual transforms also require explicit invocation
ownership in component partition layouts; their inputs are now registered from
the declared block, while the generic transform checker verifies parameter and
axis placement. Numerical placement acceptance is still in progress.

Qwen3-VL's native F32 independent-bank matrix passes **all 21 placements in
156.28 s** (`/tmp/eredu-qwen-vl-components-independent-ring-matrix1.log`), and its
affine dense/sparse matrix passes **all 30 placements in 259.02 s**
(`/tmp/eredu-qwen-vl-components-affine-ring-matrix1.log`). These use the corrected
worker pipe drainage and the complete component-parameter edit plan.

Gemma's expanded neutral placement fixture exposed missing media continuation
geometry, installation of projected media at the decoder boundary, and transport
of shared KV across a publisher/consumer cut. The first two corrections now reach
the shared-state equation. Current-position transport and explicitly accounted
receiver caches are under validation; the full Gemma placement matrix is not yet
an accepted result.

Qwen3-VL's MXFP4 dense/sparse matrix passes **all 30 placements in 233.55 s**
(`/tmp/eredu-qwen-vl-components-mxfp4-ring-matrix1.log`), and affine independent
banks pass **all 21 placements in 224.30 s**
(`/tmp/eredu-qwen-vl-components-affine-independent-ring-matrix1.log`). Together
with F32 and affine matrices, this closes the planned native SafeTensors
residency/parallel component matrix for dense and sparse Qwen3-VL. GGUF fixture
coverage remains separate and open.

Inkling's F32 GGUF target component matrix passes **all 21 placements in 166.50 s**
(`/tmp/eredu-inkling-gguf-target-components-matrix1.log`), covering its seven
TP/PP/EP combinations across resident, host-layerwise, and disk execution.

Kimi Linear's F32 GGUF target component matrix passes **all 21 placements in
104.58 s** (`/tmp/eredu-kimi-gguf-target-components-matrix1.log`). Its affine
SafeTensors matrix passes **all 21 placements in 146.27 s**
(`/tmp/eredu-kimi-affine-components-matrix1.log`). Both use the expanded
coordinated component-parameter edit plan with masks, capture, effective queries,
projection, controlled replay and restoration (native build 78).

Kimi Linear's MXFP4 target component matrix passes **all 21 placements in
137.02 s** (`/tmp/eredu-kimi-mxfp4-components-matrix1.log`, native build 78).

Gemma's neutral component placement regression passes **all 51 standard
placements and all 30 early/shared-KV four-stage placements**, plus the exact
SafeTensors/GGUF attention-bias schema test. These cover nonzero image prefill,
two cached decode steps, dense/routed masks, independently cached expert banks,
whole-residual scaling and signed reconstruction. Log:
`/tmp/eredu-gemma-components-placement14.log` (3 passed, 1 failed, 28.97 s).
The separate padded image/audio batch case correctly exceeded its selected
8-position wire limit; its fixture admission is being expanded to the actual
16 padded encoder positions. This is not yet a complete Gemma acceptance result.

Kimi Linear's F32 independent-bank matrix passes **all 21 placements in
106.89 s** (`/tmp/eredu-kimi-independent-components-matrix1.log`, native build
78), exercising bounded expert acquisition with the complete component edit,
query, capture and replay plan.

Gemma's padded image/audio batch regression now passes **all 9 placements in
6.52 s** (`/tmp/eredu-gemma-components-padded16.log`): two distinct images and
two distinct audio inputs, unequal valid/padded extents, dense/sparse and
independent-bank execution, combined TP/PP/EP, and all three residencies.
The fixture admits its 16-position padded encoder transport. This also found
and fixed a portable delivery bug: a pending result from one independent media
root must retain its declared destination while another root executes. Together
with the earlier 51 standard and 30 four-stage cases, Gemma has 90 neutral
component placements passing; full regression and native/public validation follow.

Kimi Linear's affine independent-bank matrix passes **all 21 placements in
158.11 s** (`/tmp/eredu-kimi-affine-independent-components-matrix1.log`, native
build 78), including the complete coordinated component edit plan.

The shared Gemma pipeline changes pass **605 architecture tests** (one existing
ignore) and **252 core tests** in `/tmp/eredu-gemma-pipeline-libraries1.log`.
After aligning the new runtime fixture's exact state facts with its requested
paged cache, **all 505 runtime tests pass in 0.27 s**
(`/tmp/eredu-gemma-pipeline-runtime3.log`). The new test also verifies that stale
one-head capability facts cannot authorize a two-head replacement layout.

Kimi Linear's F32 GGUF independent-bank matrix passes **all 21 placements in
111.97 s** (`/tmp/eredu-kimi-gguf-independent-components-matrix1.log`, native
build 78). The planned target matrices now pass for F32, affine, MXFP4 and F32
GGUF, with independent-bank matrices passing for F32, affine and F32 GGUF.

The complete neutral numerical suite passes **all 268 tests in 308.97 s** after
the Gemma shared-state/media fixes and generic destination-preserving boundary
delivery (`/tmp/eredu-gemma-pipeline-full-numeric2.log`, serialized with
`--test-threads=1`). This includes all 90 Gemma placements, Qwen3-VL's shared
routed-observer regression and the remaining composite, prediction and prepared
materialization conformance suites.

Reproduce the focused Gemma neutral matrix with:

```sh
cargo test -j1 -p eredu-architectures --test reference_numeric gemma4_placement -- --nocapture --test-threads=1
cargo test -j1 -p eredu-core -p eredu-runtime -p eredu-architectures --lib
cargo test -j1 -p eredu-architectures --test reference_numeric -- --test-threads=1
```

LFM2's affine target component matrix passes **all 30 dense/sparse placements in
110.31 s** (`/tmp/eredu-lfm2-affine-components-matrix1.log`, native build 78).
This complements the earlier quantized TP checks; the MXFP4 full matrix and
unequal source/target partition-cut probe remain separate validation work.

Gemma's public component workflow passes **both CPU and Metal tests in 7.81 s**
(`/tmp/eredu-gemma-components-public-native1.log`). Each covers tied and untied
heads, attention-channel and dense-unit masks across resident/host/disk,
controlled replay, and a coordinated 20-parameter edit against a physically
edited source followed by restoration. The edit spans Q/K/V/O, projection biases,
FFN gate/up/down, norms, per-layer writes, embeddings and learned residual scales.

```sh
cargo test -j1 -p eredu --no-default-features --features mlx,metal --test native_execution_control native_gemma4_component_analysis -- --ignored --nocapture --test-threads=1
```

Portable facade regression passes **61 backend-conformance tests** and
**19 facade tests** (one existing ignore), with concrete backends disabled.
Log: `/tmp/eredu-gemma-pipeline-portable1.log`.

Gemma cached-history regression: the four focused placement tests pass again
(**90 placements, 39.70 s**) with a stricter neutral cache contract. Partitioned
updates now expose only the current K/V submission; attention reads older values
from the cache itself. The serial oracle retains full-history tensors, and an
additional ordinary cache-owned run agrees with it. This covers dense, sparse,
independent banks, all three weight residencies, TP/PP/EP, early/shared pipeline
cuts, and padded image/audio batches. The fixture crosses the sliding-window
frontier during decode. It caught both a paged-history bypass and shared
consumers bypassing sliding-window enforcement; both now use the architecture's
selected local publisher or pipeline receiver cache. Native revalidation is
still pending for this correction.

```sh
cargo test -j1 -p eredu-architectures --test reference_numeric gemma4_placement -- --nocapture --test-threads=1
```

LFM2's native MXFP4 component matrix passes **all 30 placements in 105.92 s**
(dense 9 and sparse 21), including cached parity, component capture/masks,
coordinated overlays, publication failure, restoration, and controlled replay.
The fixture explicitly resets both executions after its independent cached
parity control, before the edit trials begin. Run with:

```sh
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib ring_lfm2_mxfp4_components_matrix -- --ignored --nocapture --test-threads=1
```

The stricter Gemma history contract also passes the complete neutral numerical
suite: **268 tests, 283.40 s**, zero failures or ignored tests. The focused
pipeline-owner unit passes for every cut, including owners with additional
fixed state. Commands:

```sh
cargo test -j1 -p eredu-architectures --test reference_numeric -- --test-threads=1
cargo test -j1 -p eredu-architectures --lib gemma4::pipeline -- --nocapture
```

Gemma's corrected native shared-cache execution passes the tensor-parallel
component checks: **3 placements in 59.34 s**, dense, sparse, and independent
expert-bank TP2. Checks include prefill and ordinary cached decode before edits,
coordinated edits, rejected publication and restoration, effective parameter
queries, active-edit capture provenance, controlled replay, and sibling isolation.
The restoration query pass adds its allowance to actual cumulative usage; it
does not reset prior charges. Quantized and broader placement validation remain
in progress.

```sh
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib ring_gemma4_components_tensor_parallel -- --ignored --nocapture --test-threads=1
```

Qwen3-Next's native hybrid component matrix passes **all 9 dense placements in
22.31 s**: TP, PP, combined TP/PP, each with resident, host-layerwise, and
disk-streamed weights. This uses the public component capture, global masks,
parameter queries, coordinated overlays, and controlled replay harness.

```sh
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib ring_public_component_capture_qwen_next_hybrid_matrix -- --ignored --nocapture --test-threads=1
```

The Gemma cold-format regression now checks actual transform admission against
its family media policy, including a 12-column image projection and a 48-column
audio-to-text projection that remain dense. Eligible media weights select
affine or MXFP4 as requested. The focused unit passes in **0.02 s**; native
quantized revalidation is pending.

The full neutral numerical suite passes **269 tests in 265.95 s** after the
Gemma media-format and logical-shape corrections. A further focused Gemma
transform-source regression passes **12 placements in 7.22 s**: dense and sparse,
TP2 and combined TP2/PP2 (with EP2 for sparse), across all three residencies.
It retains the typed floating source and selected affine target, then compares
image prefill and two cached decode steps with nonzero parameters.

```sh
cargo test -j1 -p eredu-architectures --test reference_numeric gemma4_selected_transforms_retain_source_geometry_and_cached_execution -- --nocapture --test-threads=1
```

Qwen3.5's dense native hybrid component matrix also passes **all 9 placements
in 21.88 s**, with the same component, parameter-overlay and controlled replay
checks across TP, PP, combined TP/PP and all three residencies.

```sh
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib ring_public_component_capture_qwen_35_hybrid_matrix -- --ignored --nocapture --test-threads=1
```

LFM2's unequal quantized-partition regression passes **12 native placements in
35.41 s**: dense/sparse, TP2/TP2+PP2, and resident/host/disk weights. Floating
FFN width 96 would ordinarily split 48/48; affine group-32 target ownership
splits 64/32. Prepared transform sources now retain the latter exact cuts.
The common component harness verifies cached predictions, selected parameters,
causal masks, coordinated edits, publication rollback, restoration, and
controlled replay. The neutral fixture also passes **60 placements in 9.66 s**,
covering widths 96/128 and the full applicable TP/PP/EP combinations.

```sh
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib ring_lfm2_affine_components_unequal_source_target_cuts -- --ignored --nocapture --test-threads=1
cargo test -j1 -p eredu-architectures --test reference_numeric lfm2_prepared_transforms_retain_source_layout_across_residency_and_parallelism -- --nocapture --test-threads=1
```

After target-aligned source ownership and the corresponding composite scalar
adapter updates, the complete neutral numerical suite passes **270 tests in
286.90 s**, with no failures or ignored tests. This includes the expanded
60-placement LFM2 unequal-cut fixture and all selected composite transform
source checks.

Qwen3.5's dense conditional native component matrix passes **all 9 placements
in 41.45 s**, including image/DeepStack input, the three residencies, TP/PP and
combined TP/PP, effective parameter operations, coordinated edits and replay.

```sh
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib ring_public_component_capture_qwen_35_conditional_matrix -- --ignored --nocapture --test-threads=1
```

Qwen3-VL and Muse now each cover **102 neutral component placements** with tied
and untied readout, instead of 51 untied placements alone. Both expanded tests
pass together in **25.64 s**. Coverage includes nonzero signed reconstruction,
seven capture/mask trials with prefill and cached decode, all applicable TP/PP/EP
combinations, resident/host/disk weights, and independent routed banks. These
checks verify that shared embedding storage still serves the declared output
consumer on the final pipeline stage.

```sh
cargo test -j1 -p eredu-architectures --test reference_numeric components_follow_media_residency_and_all_parallel_bank_placements -- --nocapture --test-threads=1
```

Conditional Qwen hybrid passes its expanded tied/untied prepared-component
matrix: **102 placements across two tests in 10.54 s**, covering ordinary and
independent expert banks, image/DeepStack writes, selected masks, cached decode,
and all applicable residency/TP/PP/EP combinations. The final-stage consumer
role remains `output`; tied parameter ownership supplies the embedding replica.

```sh
cargo test -j1 -p eredu-architectures --test reference_numeric prepared_conditional_qwen_ -- --nocapture --test-threads=1
```

Qwen3-Next MoE passes **all 21 native component placements in 164.37 s**,
covering TP, PP, EP and all combined axes with resident/host/disk weights.
The common harness checks recurrent channels, component masks, effective
queries/projections, coordinated overlays, rollback, snapshots and replay.

```sh
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib ring_public_component_capture_qwen_next_hybrid_moe_matrix -- --ignored --nocapture --test-threads=1
```

Architecture and runtime libraries also pass after the source-layout and tied
consumer-role corrections: **606 architecture tests (one existing ignore)**
and **508 runtime tests**, with no failures.


Gemma 4's quantized native tensor-parallel component trial passes **all five
cases in 241.76 s**: dense and sparse affine/MXFP4 plus sparse affine with an
independent expert bank. It exercises effective parameter queries/projections,
component masks, coordinated edits, rollback and snapshot replay. Component
cases have a larger whole-case watchdog; each collective retains its existing
30-second completion deadline.

```sh
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib ring_gemma4_components_quantized_tensor_parallel -- --ignored --nocapture --test-threads=1
```

The Qwen3-VL projector mapping regression passes for token-pending and bound
plans. It retains both temporal GGUF slices and infers the combined five-axis
patch tensor. Native GGUF execution validation is still in progress.


The new prediction-module completion contract passes three focused neutral
checks: state changes followed by equation failure, simultaneous shared owners,
and native-completion failure preventing success publication. The current
architecture library passes **607 tests with one existing ignore**; runtime
passes **509 tests**, including exact per-unit catalogs under one shared budget.
Backend library and test code type-check with Metal enabled. Native primitive
acceptance and the first integrated prediction results are recorded below.


The complete neutral numerical suite passes **273 tests in 319.24 s** after
per-module prediction invocation was wired through V3, V4, DSpark, Inkling,
Qwen hybrid and Nemotron-H. This includes the three new completion tests and
the existing nonzero component, transform, placement, cached execution and
prediction conformance fixtures.

```sh
cargo test -j1 -p eredu-architectures --test reference_numeric -- --test-threads=1
```


Native shared-source residency and module completion primitives passed on CPU:
`target/debug/deps/eredu_backend_mlx-c35ae7dae8411ea1 backend::runtime::residency::manager::tests --nocapture --test-threads=1`
reported 28 passed and one existing ignored test in 0.11 s. This includes identical
keys with distinct source values under one eight-byte device budget, and changed
state retained through successful and failed equation completion. This run
predates the full prediction-module integration; its evidence is limited to the
native residency primitives.


After native prediction modules were connected to the target residency manager,
the focused CPU prediction suite passed **7 tests in 6.73 s**. It covers V3 and
DSpark internal phases, bounded capture, public V3/Inkling/Qwen activation
admission, failure behavior, and auxiliary quantization reporting. Runtime's
complete library suite passes **510 tests in 0.29 s**, including the new shared
versus sequential auxiliary sizing contract. The parallel paging matrix is still
being validated; a target-only unit-count assertion was updated to include the
new prediction owners in the common report.

```sh
cargo test -j1 -p eredu-backend-mlx --no-default-features --lib prediction_components:: -- --nocapture --test-threads=1
cargo test -j1 -p eredu-runtime --lib
```

Gemma's full-precision native matrix passed the first 27 placements, then stopped
at the eight-rank resident case because the global capture work exceeded the
fixture's four-GiB host allowance. Distributed accounting prices every producer
on every receiver and conservatively reserves parser storage at 128 bytes per
encoded byte. The eight-rank fixture work allowances now scale by four relative to
the four-rank allowances, including retained and encoded transport work. This is logical work, not physical memory allocation;
production admission limits remain enforced. The focused
`ring_gemma4_components_triple_axis` retry now passes all three remaining
placements in **243.92 s**, completing the **30 full-precision native placements**.
Each checks image input, component masks, effective queries, coordinated edits,
restoration and controlled replay. Full quantized, independent-bank and GGUF
matrices are still separate open coverage.

```sh
cargo test -j1 -p eredu-backend-mlx --no-default-features --lib ring_gemma4_components_triple_axis -- --ignored --nocapture --test-threads=1
```


The corresponding **6 Metal prediction tests pass in 6.27 s** after module loans,
shared residency registration and persistent overlay publication were integrated.
They cover V3/DSpark internal phases, bounded collection and public speculative
activation workflows for V3, Inkling and Qwen. These focused GPU tests complement
the seven CPU tests; the paged distributed matrix remains under validation.

```sh
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib prediction_components:: -- --ignored --nocapture --test-threads=1
```


Qwen3-Next prediction capture, effective queries, coordinated edits, restoration
and replay now pass all **9 native CPU Ring placements in 50.26 s** after
prediction modules were moved into the target's shared residency manager:
TP, PP and combined TP/PP, each resident, host-layerwise and disk-streamed.
Host and disk admission reports include the separately prepared prediction unit
and shared fusion owner in the common weight ledger. Unlike earlier results,
these placements exercise paging of the prediction modules themselves.

```sh
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib ring_qwen_prediction_components_next_matrix -- --ignored --nocapture --test-threads=1
```

DSpark passes all **21 native CPU Ring placements in 538.05 s** with actual
prediction paging: TP, PP, EP, TP/PP, TP/EP, PP/EP and TP/PP/EP, each with
resident, host-layerwise and disk-streamed weights. The same run covers phase
captures and masks, effective parameter queries, coordinated edits, restoration
and controlled replay. Shared DSpark input/readout modules and sequential
prediction units use the target's common residency ledger. This run precedes
the additional idle-device-eviction assertions and cold grouping metadata;
those changes need their own verification.

```sh
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib ring_deepseek_v4_dspark_components_and_parameters_matrix -- --ignored --nocapture --test-threads=1
```

Cold auxiliary grouping passes **511 runtime tests in 0.28 s**, including
shared-plus-sequential sizing and rejection of inconsistent physical owner
roles. Architecture library checks pass **607 tests with one existing ignore
in 0.40 s**. With those declarations checked against actual constructed
prediction modules, the focused native suites pass **7 CPU tests in 8.55 s**
and **6 Metal tests in 4.43 s**. All **273 existing numerical tests pass in
352.24 s** with these declarations. The added rank-accounting test initially
omitted the real embedding copies required on later pipeline stages; after
correcting that expectation it passes in **0.11 s**. It compares reported
prediction bytes with constructed module tensor geometry across TP, PP, EP
and eight-rank combined placement, including the additional pinned embeddings.
The additional idle-owner checks pass all **9 dense-target Inkling CPU Ring
placements in 168.55 s**: TP, PP and TP/PP with all three weight policies.
The common report confirms zero prediction loans while idle and no retained
device copies for host/disk owners, before and after effective queries,
coordinated edits, restoration and replay.

```sh
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib ring_inkling_prediction_components_dense_target_matrix -- --ignored --nocapture --test-threads=1
```

Nemotron-H dense-target prediction paging passes the same **9 CPU Ring
placements in 61.35 s**, including its multi-unit prediction schedule, idle
eviction, effective queries, coordinated edits, restoration and controlled
replay. These runs include cold module-group declarations and their construction
checks; earlier routed and quantized results still need paging follow-up.

```sh
cargo test -j1 -p eredu-backend-mlx --no-default-features --features metal --lib ring_nemotron_prediction_components_dense_target_matrix -- --ignored --nocapture --test-threads=1
```

```sh
cargo test -j1 -p eredu-architectures --test reference_numeric -- --test-threads=1
cargo test -j1 -p eredu-architectures --test reference_numeric prediction_rank_resources_count_tensor_shards_and_pipeline_expert_replicas -- --nocapture --test-threads=1
```


Qwen3-VL GGUF passes **3 native TP cases in 15.42 s**: dense, routed with
resident banks, and routed with an independent parameter bank. Each uses the
primary GGUF plus mandatory projector companion and exercises component
capture, effective queries, edits, restoration and replay. Temporal patch
slices are joined by the architecture's exact recipe, and the harness admits
each payload store once. Effective model-type comparisons resolve official
format aliases to their architecture family. The full residency/parallel GGUF
matrices remain open.

```sh
cargo test -j1 -p eredu-backend-mlx --no-default-features --lib ring_qwen_vl_gguf_components_tensor_parallel -- --ignored --nocapture --test-threads=1
```


Gemma 4 GGUF passes **3 native tensor-parallel cases in 93.38 s**: dense,
sparse with resident banks, and sparse with independent expert caching. Each
loads the primary text GGUF and its image-projector companion, then exercises
component captures and masks, effective queries, coordinated edits, restoration
and controlled replay. Admission now maps primary tensors into the family's
parameter tree and maps layer/routing scalars to their actual parameter slots.
A nonzero neutral GGUF regression checks all dense and sparse physical outputs
against the family schema and verifies embedding/per-layer/decoder ownership;
it passes in 0.08 s. The architecture library passes **608 tests with one existing
ignore in 2.16 s**. Full GGUF residency/parallel and independent-bank matrices
remain under verification.

```sh
cargo test -j1 -p eredu-architectures --lib gemma_gguf_admission_preserves_family_embedding_and_unit_ownership -- --nocapture
cargo test -j1 -p eredu-backend-mlx --no-default-features --lib ring_gemma4_gguf_components_tensor_parallel -- --ignored --nocapture --test-threads=1
```

### Gemma GGUF complete ordinary-bank native matrix

All 30 placements pass with the corrected family/scalar name mapping
(`/tmp/eredu-gemma-gguf-components-matrix1.log`, CPU build 102, 1154.23 s).
The matrix covers nine dense and 21 sparse TP/PP/EP combinations across resident,
host-layerwise and disk-streamed execution, including eight-rank TP+PP+EP.
Nonzero GGUF text weights plus the vision companion exercise shared-KV boundaries,
component captures and masks, effective queries/projections, coordinated overlays,
restoration and controlled replay. The independent-bank GGUF matrix and the
remaining full quantized matrices are still separate outstanding validation.

### Qwen3-VL GGUF complete ordinary-bank native matrix

All 30 placements pass with the admitted GGUF primary and vision-companion
sources (`/tmp/eredu-qwen-vl-gguf-components-matrix1.log`, Metal/image/audio build
107, 295.47 s). The nine dense and 21 routed placements cover resident,
host-layerwise and disk-streamed execution, every applicable TP/PP/EP combination,
component captures/masks, exact effective queries/projections, coordinated
parameter overlays, restoration and controlled replay. The separate independent-bank
GGUF matrix also passes all 21 placements in 232.64 s
(`/tmp/eredu-qwen-vl-gguf-components-independent-matrix1.log`, same build).

### Inkling routed prediction paging and public delivery checks

All 21 routed-target Inkling prediction placements pass with actual auxiliary
module paging, idle device eviction/loan checks, source-derived effective queries,
coordinated edits, restoration and replay
(`/tmp/eredu-inkling-routed-prediction-paging-matrix1.log`, Metal build 105,
649.57 s). Combined with the nine dense-target placements, the full 30-case F32
prediction matrix now exercises actual bounded extension residency. Affine and
MXFP4 paging verification remains separate.

The new shared token-step delivery path also passes public native CPU and Metal
capture-failure/replay tests (two tests, 3.05 s,
`/tmp/eredu-step-delivery-native-public-failures1.log`) and ordinary/controlled
component-mask parity across resident, host and disk execution (two tests, 1.82 s,
`/tmp/eredu-step-delivery-native-public-components1.log`). These use the public
facade binary `native_execution_control-e4be14240c79f783`. Direct public multi-rank
facade verification passes all six dense TP2/PP2 and MOVA TP2/PP2/EP2 placements
across resident, host and disk execution (54.32 s,
`/tmp/eredu-step-delivery-public-ring-matrix1.log`). Both greedy and seeded sampling
cover one-rank callback cancellation, first-token encoded-record failure, typed
peer rejection, cumulative readiness charges across reset and a corrected retry
matching the baseline. Later lifecycle/terminal record cases are a separate
extension under verification.

### Public lifecycle/terminal delivery and speculative visitor readiness

Controlled lifecycle and terminal publication now share the existing preparation
agreement after token commitment. Ordinary terminal records agree their result
without rewriting the established stop reason. All 73 portable conformance tests
pass (0.79 s), with 182 library tests (8.79 s, two existing ignores) and 19 portable
facade tests (7.27 s, one existing ignore), recorded in
`/tmp/eredu-boundary-delivery-portable-full1.log`. The mock now counts decode as
well as prefill, so the no-additional-forward assertions cover both.

The expanded public native Ring matrix passes all six placements in 74.44 s
(`/tmp/eredu-boundary-delivery-public-ring-matrix2.log`, public CLI build 2).
Dense TP2/PP2 and MOVA TP2/PP2/EP2 each cover resident, host and disk execution
with both greedy and seeded sampling. A single rank cancels from the first token
or the following lifecycle record; every rank retains the same one-token prefix.
One-rank budget failures at first-token, lifecycle and terminal publication retain
the original encoded-record error locally and typed stage/rank rejection on peers.
All corrected retries reproduce the baseline, and readiness usage survives reset.

The new neutral speculative-executor readiness contract passes all 75 facade
conformance tests (0.80 s,
`/tmp/eredu-speculative-visitor-readiness-facade2.log`). Additional cases reject
peer lane/scheduler readiness and collector installation before prefill and retain
the original local collector failure. The native no-default-feature check passes
in 26.30 s (`/tmp/eredu-speculative-visitor-readiness-check1.log`). Native visitor
failure verification remains pending at this point.

After the speculative readiness contract was added, all 608 architecture library
tests pass (one existing ignore, 0.39 s), all 252 core tests pass (6.36 s), and all
511 runtime tests pass (0.24 s), in
`/tmp/eredu-speculative-visitor-readiness-neutral-full1.log`. The first native
visitor test reached the expected local policy and peer rejection, but its
assertion incorrectly searched only for a leaf error hidden by a transparent
driver variant. The assertion now inspects that typed variant; the full native
matrix is rerun before acceptance.

The corrected native visitor rejection matrix passes **all eight placements in
2.53 s** (`/tmp/eredu-speculative-visitor-readiness-ring-failures2.log`, CPU build
110). Both ordinary and controlled visitors reject one rank's invalid scheduler
capacity across resident TP2, host TP2, disk PP2 and resident TP2/PP2. Every rank
returns before publication; the owner retains its typed scheduler policy, peers
retain the request-stage rank-zero rejection, accounting remains cumulative, and
the failed broad native operation remains fenced. The native build passed in
56.39 s. These results complete startup agreement verification, not the separate
post-action speculative delivery/cancellation audit.

### Qwen 3.5 routed recurrent components

The native text Qwen 3.5 MoE component matrix passes all 21 TP/PP/EP and
resident/host/disk placements (209.03 s,
`/tmp/eredu-qwen35-hybrid-moe-components-matrix1.log`, frozen Metal build 105).
It exercises recurrent-channel and dense/shared/routed component masks, effective
parameter queries/projections, coordinated overlays, restoration and replay. This
extends the earlier 48 accepted Qwen hybrid placements to 69; the separate
conditional-MoE 21-case matrix remains pending.

### Qwen Next-MoE actual prediction paging

All 21 routed-target Qwen Next-MoE placements pass actual extension paging,
idle ownership checks, component masks/captures, effective parameter queries,
coordinated overlays and controlled replay (475.39 s,
`/tmp/eredu-qwen-next-moe-prediction-paging-matrix1.log`, CPU build 110).
This also verifies successful ordinary/controlled startup with the new executor
visitor readiness contract. It precedes the subsequent scheduler-fact consensus
integration, whose acceptance is tracked separately.

Inkling affine prediction paging passes its first 27 placements, then the
three-axis resident case reaches the whole-case test watchdog (1158.80 s total,
`/tmp/eredu-inkling-affine-prediction-paging-matrix1.log`). Every rank completes the
mask trials and reaches the 1336-exchange parameter-query phase. Prediction modes
were accidentally excluded from the existing component-test watchdog multiplier;
they now receive that same bound, with per-collective deadlines unchanged. The
remaining three placements require a focused rerun before full acceptance.

### Gemma GGUF independent banks and complete Qwen recurrent placements

Gemma's GGUF independent-bank component matrix passes **all 21 placements in
1330.62 s** (`/tmp/eredu-gemma-gguf-components-independent-matrix1.log`, frozen
Metal/image/audio build 107), including all eight-rank TP/PP/EP residencies,
effective queries, coordinated edits, restoration and controlled replay.

Qwen 3.5 conditional MoE passes its remaining **21 placements in 543.23 s**
(`/tmp/eredu-qwen35-conditional-moe-components-matrix1.log`, same build). Together
with the earlier 69 placements, all 90 recurrent-component placements pass.
These are family capability and numerical results; the newer common scheduler
consensus and delivery integration has separate native acceptance still pending.

### Coordinated speculative scheduling and delivery

The shared scheduler now coordinates per-lane cancellation, completion, deadline
and optimistic eligibility before selecting an action. Runtime protocol tests use
two real host participants and bounded transport faults: all nine preparation
protocol tests pass (`/tmp/eredu-speculative-scheduler-consensus-tests3.log`).
The expanded core/runtime run passes 252/514 tests
(`/tmp/eredu-speculative-scheduler-consensus-neutral1.log`).

Core publication and pending-cancellation changes additionally pass all 252 tests
in 6.28 s (`/tmp/eredu-speculative-coordinated-publication-core3.log`). These retain
payloads through rollback and successful completion through publication, including
observation, publication and restoration failures. The facade passes all 81
conformance tests in 0.81 s (`/tmp/eredu-speculative-coordinated-behavior3.log`):
per-lane peer cancellation before drafting and during verification, delayed peer
completion without repeated native work, rejected candidate identity, unaffected
batch-lane completion, publication cancellation, bounded step-record failure and
caller-owned failure before further model work. Original typed failures remain
observable. The final native distributed checks recorded below validate these
new boundaries separately from the earlier family matrices.

Reproduce the portable coverage with:

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-core --lib
CARGO_INCREMENTAL=0 cargo test -p eredu-runtime --lib
CARGO_INCREMENTAL=0 cargo test -p eredu --no-default-features --test backend_conformance
```

The first native coordinated-control suite passes **all 20 placements in 11.47 s**
(`/tmp/eredu-speculative-coordinated-ring-delivery1.log`, Metal build 111,
`eredu_backend_mlx-f1a0f0df05777822`). Five cases each run on resident TP2, host
TP2, disk PP2 and resident TP2/PP2: rank-zero cancellation after prefill and during
pending verification; first-record budget failure; early controlling-closure
failure; and terminal closure failure. Cancellation preserves the same one-token
prefix on every rank and permits safe synchronization. Errors retain their typed
owner cause and delivery-stage rank-zero rejection on peers; the failed native
run stays fenced and readiness usage remains cumulative. This binary predates
the subsequent per-proposal sampling-failure guard, whose validation is separate.

Per-proposal result agreement now stops a peer's next native draft forward when
local sampling, capture or prefix validation fails. Execution observers, proposal
state creation and verification checkpoints also agree before subsequent native
work. The core/runtime run passes 252/514 tests (6.74/0.23 s,
`/tmp/eredu-speculative-draft-phase-neutral1.log`). The expanded facade run passes
all 82 tests (0.88 s, `/tmp/eredu-speculative-draft-phase-facade3.log`). Its three-token
draft fixture injects owner/peer sampling failures after the first proposal and
requires exactly one proposal forward in ordinary and controlled execution.

That test also detected a concrete error omitted from the public source chain by
a transparent driver variant. The driver now preserves that backend error as its
source without changing its display text. Native sampling-capture conversions
likewise preserve the original portable capture error. The final 24-placement
native run recorded below includes the additional sampling-budget case.

The focused Inkling affine rerun passes **all three eight-rank placements in
296.60 s** (`/tmp/eredu-inkling-affine-prediction-paging-three-axes1.log`, frozen
Metal build 111). Together with the earlier 27 accepted cases, all 30 affine
placements now pass actual prediction-extension paging, idle ownership and cold
source sizing, masks/captures, effective queries/projections, coordinated edits,
restoration and replay. The final three cases also exercise the new common
scheduler/publication protocol; they predate the additional per-proposal guard.
The whole-case watchdog fix changes no per-collective deadline.

Nemotron's routed-target F32 actual-paging matrix passes **all 21 placements in
303.71 s** (`/tmp/eredu-nemotron-routed-prediction-paging-matrix1.log`, frozen Metal
build 111). With the earlier nine dense-target cases, all 30 F32 Nemotron
prediction placements pass actual extension paging, cold-source/idle-loan checks,
component experiments, effective queries, coordinated overlays and replay.
MXFP4 actual-paging results are recorded separately below.

The completed per-proposal and error-source implementation passes **all 24 native
control-delivery placements in 7.64 s**
(`/tmp/eredu-speculative-draft-phase-ring-delivery3.log`, Metal build 113,
`eredu_backend_mlx-f1a0f0df05777822`). The sixth case exhausts only rank zero's
raw-logit capture allowance during draft sampling after successful prefill.
Its original typed capture limit remains in the public cause chain; peers receive
the delivery-stage rank-zero rejection before verification, and the failed native
run remains fenced. The other five cancellation/record/caller cases also pass on
the final implementation. The ordinary and controlled visitor startup matrix
additionally passes all eight placements in 3.52 s
(`/tmp/eredu-speculative-draft-phase-ring-visitor3.log`). Build 113 passed in 54.61 s.

After the completed coordination/error-source changes, the consolidated neutral
run passes **608 architecture tests** (one existing ignore, 0.39 s), **252 core
tests** (6.65 s) and **514 runtime tests** (0.23 s), recorded in
`/tmp/eredu-speculative-coordinated-neutral-final1.log`.

Nemotron MXFP4 actual prediction paging passes its **nine dense-target placements
in 106.50 s** (`/tmp/eredu-nemotron-mxfp4-dense-prediction-paging-matrix1.log`,
frozen Metal build 113). The routed-target 21-case follow-up is recorded below.

Nemotron MXFP4 routed-target actual paging passes **all 21 placements in 316.77 s**
(`/tmp/eredu-nemotron-mxfp4-routed-prediction-paging-matrix1.log`, frozen Metal
build 113). Together with the nine dense-target cases, all 30 MXFP4 placements
pass actual prediction paging, idle-loan checks, component experiments, parameter
queries, overlays, restoration and replay.

Qwen3.5 dense actual prediction paging passes **all nine placements in 68.46 s**
(`/tmp/eredu-qwen35-prediction-paging-matrix1.log`, frozen Metal build 113).
TP, PP and combined TP/PP each pass resident, host and disk execution with
component experiments, effective queries, coordinated edits, restoration, replay
and idle ownership checks. Routed and conditional follow-up remains separate.

Packed Qwen3-VL primary/projector admission now retains the exact GGUF formats
before constructing parameter topology. Shared requirement discovery also
expands encoded block widths to logical matrix dimensions and rejects partial
blocks. The architecture suite passes **610 tests** (one existing ignore, 0.42 s)
in `/tmp/eredu-qwen-vl-packed-geometry-neutral2.log`, including Q8_0/IQ4_NL text
and DeepStack geometry regressions. The new native packed-format matrix is still
pending; these neutral results do not establish its acceptance.

Inkling MXFP4 actual prediction paging passes **all 30 placements in 1168.91 s**
(`/tmp/eredu-inkling-mxfp4-prediction-paging-matrix1.log`, CPU build 112). Dense
and routed targets cover resident, host and disk execution across applicable
TP/PP/EP combinations, with idle ownership checks, component experiments,
effective queries, coordinated edits, restoration and replay.

Qwen3.5 routed actual prediction paging passes **all 21 placements in 473.40 s**
(`/tmp/eredu-qwen35-moe-prediction-paging-matrix1.log`, frozen Metal build 113).
Together with the nine dense placements, all 30 text-only Qwen3.5 placements
pass actual extension paging, component experiments, parameter queries, edits,
restoration, replay and idle ownership checks. Conditional coverage is separate.

Gemma affine component validation passes **all 30 placements in 2226.12 s**
(`/tmp/eredu-gemma-affine-components-matrix1.log`, frozen Metal/image/audio
build 107). Dense and sparse targets cover actual image ingress, shared KV,
resident/host/disk execution and every applicable TP/PP/EP combination, including
effective parameter queries, coordinated overlays, restoration and controlled
replay. MXFP4 and SafeTensors independent-bank follow-up remain separate.

The packed Qwen3-VL primary/projector focused suite passes **all three cases in
42.43 s** (`/tmp/eredu-qwen-vl-packed-projector-focused4.log`, Metal/image build
118): dense resident TP, routed disk PP and routed host TP/PP/EP with independent
expert banks. Both primary and mandatory DeepStack projector contain Q8_0 and
IQ4_NL matrices. This required retaining embedding formats through prepared
construction in addition to the cold admission and logical geometry fixes.
The full packed matrix results are recorded below.

DeepSeek V3 F32 actual prediction paging passes **all 21 placements in 381.79 s**
(`/tmp/eredu-v3-prediction-paging-matrix1.log`, CPU build 112). Resident, host and
disk execution cover all TP/PP/EP combinations with component experiments,
effective queries, coordinated target/prediction edits, restoration, replay and
idle ownership checks. Affine and MXFP4 follow-up remain separate.

The mixed Q8_0/IQ4_NL Qwen3-VL primary and DeepStack projector independent-bank
matrix passes **all 21 placements in 355.52 s**
(`/tmp/eredu-qwen-vl-packed-projector-independent-matrix1.log`, frozen build 118).
TP/PP/EP combinations each cover resident, host and disk execution with component
experiments, bounded effective queries, coordinated edits, restoration and replay.
The ordinary packed matrix result is recorded below.

Muse's projector patch recipe and cold logical geometry pass **611 architecture
tests** (one existing ignore, 0.44 s) in
`/tmp/eredu-muse-projector-patch-neutral3.log`. Nonzero encoded reads preserve
channel/spatial ordering through raw and canonical source names, while invalid
geometry and missing kernels fail admission. Native projector acceptance remains
pending its rerun.

The ordinary mixed Q8_0/IQ4_NL Qwen3-VL primary and DeepStack projector matrix
passes **all 30 placements in 450.25 s**
(`/tmp/eredu-qwen-vl-packed-projector-matrix1.log`, frozen build 118). With the
21 independent-bank placements, the packed format covers dense/routed targets,
resident/host/disk execution, TP/PP/EP combinations, image ingress, component
experiments, effective queries, coordinated overlays, restoration and replay.
These are CPU Ring results; distributed GPU validation is separate.

Qwen3.5 conditional actual prediction paging passes **all 30 placements in
673.85 s** (`/tmp/eredu-qwen35-conditional-prediction-paging-matrix1.log`, frozen
Metal build 113): nine dense and 21 routed placements with image/DeepStack input.
Together with Next/Next-MoE and text Qwen3.5, all 90 F32 Qwen prediction placements
now pass actual extension paging, component experiments, effective queries,
coordinated edits, restoration, replay and idle ownership checks. FP8 follow-up is
recorded separately.

The communicator-selected stream wrapper passes its native handle-lifetime test
in 0.16 s (`/tmp/eredu-native-communication-stream-wrapper2.log`). The singleton
fallback uses MLX's default device, so this Metal build requires native device
access; the sandboxed attempt could not obtain that stream. The implementation
also passes the Metal/image backend check in 21.11 s
(`/tmp/eredu-native-communication-stream-check2.log`). Distributed GPU acceptance
remains pending the rebuilt backend's explicit run.

DeepSeek V3 affine actual prediction paging passes **all 21 placements in
513.20 s** (`/tmp/eredu-v3-affine-prediction-paging-matrix1.log`, frozen CPU
build 112), with the same component, query, coordinated edit, restore, replay
and idle-ownership checks as F32. MXFP4 follow-up is recorded separately.

The first backend GPU transport retry still failed during manifest consensus
(`/tmp/eredu-qwen-vl-packed-projector-gpu-focused2.log`): that setup transport
called the native collective directly. It now uses the shared backend group
operation and retains that exact group in completion recovery. Distributed GPU
acceptance remains open until this correction is rebuilt and exercised.

The final portable boundary checks pass in
`/tmp/eredu-component-final-portable-checks1.log`: GGUF, checkpoint, core,
runtime and architectures; text without default features; NN and codec with all
features; media both without defaults and with all features. The same run passes
**82 backend-conformance tests** (0.87 s) and **19 portable-facade tests**
(7.53 s, one existing ignore). The backend without default features also checks
successfully in 41.80 s
(`/tmp/eredu-component-final-backend-portable-check1.log`). All commands use
`CARGO_INCREMENTAL=0` and `-j1`.

Muse's published-geometry packed projector retry passed construction and reached
parameter-overlay reference computation, but its whole-case test watchdog expired
at 180 seconds (`/tmp/eredu-muse-packed-projector-focused2.log`, total 204.60 s
including fixture preparation). A sampled worker was actively decoding packed
rows and computing their dot products, not waiting in a collective. The fixture
now declares a 1,800-second whole-case bound; production communication deadlines
are unchanged. This is not yet a passing native projector case.

DeepSeek V3 MXFP4 actual prediction paging passes **all 21 placements in
400.95 s** (`/tmp/eredu-v3-mxfp4-prediction-paging-matrix1.log`, frozen CPU
build 112). All 63 V3 F32/affine/MXFP4 placements now pass actual extension
paging, idle ownership, component experiments, queries, coordinated edits,
restoration and replay.

Qwen3.5 conditional MoE FP8 actual prediction paging passes **all 21 placements
in 612.20 s** (`/tmp/eredu-qwen35-fp8-conditional-prediction-paging-matrix1.log`,
frozen Metal build 113, CPU Ring execution). Combined with the 90 F32 placements,
this completes the recorded Qwen prediction paging follow-up matrix.

Native communication and recovery regression passes **50 tests in 5.18 s**
(`/tmp/eredu-native-transport-completion-regression1.log`, build 120), including
explicit local Ring tests, initial manifest mismatch, operation contracts,
singletons and completion quarantine. GPU Qwen retry 3 passed manifest agreement
but exposed the fixture's use of a GPU compute stream as its CPU weight-source
stream. The shared fixture setup and ordinary references now supply separate
CPU source streams when compute uses Metal. Native GPU acceptance remains open.

The neutral manifest/session protocol now retains the typed transport error at
both exchange stages, and the backend preserves that source through its public
neutral failure boundary. The runtime suite passes **515 tests in 0.27 s**
(`/tmp/eredu-manifest-error-source-neutral1.log`), including injected first- and
second-stage failures for both protocols. Native boundary verification follows
in the rebuilt backend.

The first complete distributed GPU component suite passes **three cases in
69.66 s** (`/tmp/eredu-qwen-vl-packed-projector-gpu-focused4.log`, Metal build
121): packed Qwen3-VL text and DeepStack projector with resident TP, streamed PP,
and host TP/PP/EP plus independent expert banks. All ranks use Metal device 0 on
one Apple M3 Ultra and communicate over local CPU Ring. This verifies separate
GPU compute and CPU transport; multiple physical GPUs and hosts remain hardware
validation gaps.

The same build passes **24 GPU coordinated control/delivery cases in 19.69 s**,
**eight GPU visitor-preparation cases in 4.95 s**, and the **FP8 prediction TP
component case in 45.26 s**. Logs are
`/tmp/eredu-speculative-control-delivery-gpu1.log`,
`/tmp/eredu-speculative-visitor-preparation-gpu1.log`, and
`/tmp/eredu-qwen-fp8-prediction-components-gpu1.log`.
The source-preserving communication suite passes **51 tests in 7.10 s** and
public distributed submission/recovery passes **six tests in 1.61 s**
(`/tmp/eredu-native-transport-source-regression2.log` and
`/tmp/eredu-native-public-transport-regression1.log`).

DeepSeek V4 sequential F32 actual prediction paging passes **all 21 placements
in 504.07 s** (`/tmp/eredu-v4-prediction-paging-matrix1.log`, frozen CPU build
112), including component experiments, queries, coordinated target/prediction
edits, restoration, replay and idle ownership.

Muse projector CPU retry 3 and GPU retry 1 both reached coordinated parameter
editing but rejected the fixture's two-GiB cumulative retention allowance
(`/tmp/eredu-muse-packed-projector-focused3.log`, 587.76 s, and
`/tmp/eredu-muse-packed-projector-gpu-focused1.log`, 49.26 s). This experiment
edits every selected projector matrix, so promotion costs follow full affected
matrix geometry. Its finite test allowance now adds 32 F32 matrix equivalents
for retained/copy work and eight for host work, using checked arithmetic and
preserving prior charges. The test additionally verifies the precise injected
post-publication rejection, so an unrelated budget rejection cannot stand in
for rollback coverage. Native reruns remain pending.

Kimi's tensor-parallel GPU component case passes in **24.83 s**
(`/tmp/eredu-kimi-target-components-gpu1.log`, frozen Metal build 121).
LFM2's GPU quantized component suite passes **all five cases in 37.50 s**
(`/tmp/eredu-lfm2-quantized-components-gpu1.log`): dense/routed affine and MXFP4,
plus an independent affine expert bank. These use the same physical Metal device
and local Ring transport described above.

The broader startup-source audit passes **74 neutral backend-independence tests
in 0.02 s** (`/tmp/eredu-partition-preparation-source-conformance1.log`) and the
Metal backend check in **16.12 s**
(`/tmp/eredu-partition-preparation-source-check1.log`). Architecture and mechanism
failures retain their typed causes through the factory handoff; failed handoffs
do not publish state. Native cause-chain regression is pending build 122.

Gemma MXFP4 component validation passes **all 30 placements in 1919.61 s**
(`/tmp/eredu-gemma-mxfp4-components-matrix1.log`, frozen Metal/image/audio
build 107, CPU Ring). Together with the F32 SafeTensors, F32 GGUF and affine
matrices, all four ordinary formats cover dense/sparse targets, image ingress,
shared KV, resident/host/disk execution, queries, coordinated edits and replay.
Independent-bank follow-up is recorded separately.

Additional distributed GPU cases pass on frozen Metal build 121: Gemma's three
dense/sparse/independent-bank TP cases in **141.25 s**; Inkling's four
dense/routed-target affine/MXFP4 prediction TP cases in **133.42 s**;
DeepSeek V3's three pipeline component cases in **17.02 s**; Nemotron MXFP4
prediction TP in **17.49 s**; and V4 sequential prediction PP in **19.49 s**.
Logs are `/tmp/eredu-gemma-components-gpu1.log`,
`/tmp/eredu-inkling-quantized-prediction-gpu1.log`,
`/tmp/eredu-v3-components-pipeline-gpu1.log`,
`/tmp/eredu-nemotron-mxfp4-prediction-gpu1.log`, and
`/tmp/eredu-v4-prediction-pipeline-gpu1.log`. These all use the same physical
Metal device 0 and local CPU Ring transport.

Build 122 passes in **2m 59s**
(`/tmp/eredu-preparation-muse-budget-build122.log`). Its expanded native
startup-source/transport suite passes **51 tests in 6.11 s**, and public
distributed submission/recovery passes **six tests in 1.63 s**
(`/tmp/eredu-native-startup-source-regression3.log` and
`/tmp/eredu-native-public-transport-regression2.log`). The original native cause
remains reachable through manifest, mechanism/factory, neural architecture/factory
and final replicated-session errors. These results do not replace the pending
Muse projector experiment with the updated budget.

Muse GPU retry 2 reached and verified the injected post-publication rollback,
then exhausted the cumulative host allowance on the positive retry
(`/tmp/eredu-muse-packed-projector-gpu-focused2.log`, 60.84 s). GGUF conversion
reserves eight F32 matrix equivalents per attempt; rejected preparations remain
charged. The harness now adds its finite geometry-based allowance above the
current ledger before each attempt, and gives the ordinary reference its own
allowance. Production costs and cumulative accounting are unchanged. The next
native retry is still required.

The final encoding audit adds two pending native suites. K2's existing full-tail
fixture now has a variant with exact F16 and BF16 scale companions on different
layers; its independent scalar oracle is unchanged because every scale is exactly
representable. Six tests cover ordinary and independently cached banks across
all seven TP/PP/EP combinations and three residencies (42 cases).

V4's new fixtures use published F32 or UE8M0 block-FP8 metadata for attention,
readout, expert and prediction matrices. Complete 128-wide TP blocks use hidden,
expert and vocabulary widths of 256, head/factor widths of 128, and deterministic
signed E4M3 values with nonuniform scales. Each queried FP8 row is checked against
an independent decoder of the actual source bytes, including fused expert-bank
coordinates. Sequential prediction and DSpark each have a 21-case matrix per
scale encoding, preceded by a four-case TP-focused suite. These tests are not yet
built or accepted; no additional support claim follows from adding the fixtures.

DeepSeek V4 sequential affine actual prediction paging passes **all 21 placements
in 849.87 s** (`/tmp/eredu-v4-affine-prediction-paging-matrix1.log`, frozen CPU
build 112). Resident, host and disk execution cover all TP/PP/EP combinations
with component experiments, effective queries, target/prediction edits, rollback,
restoration, replay and idle-loan checks.

The consolidated neutral run passes in
`/tmp/eredu-component-final-neutral2.log` (`CARGO_INCREMENTAL=0 cargo test -j1
-p eredu-core -p eredu-runtime -p eredu-nn -p eredu-evaluation
-p eredu-architectures`; build 4m 41s): **611 architecture library tests**,
**274 numerical tests**, **eight production-constructor conformance cases**,
10 K2 admission tests, nine realtime construction tests, five speculative
execution tests and 46 structural tests. Core passes **252**, runtime **515**,
backend independence **74**, exact binding placement three, NN **38** and
evaluation **21**. All documentation tests pass; two existing architecture/K2
tests remain explicitly ignored. The run includes the typed startup error
changes and Muse patch geometry; new native scale-format fixtures are separate.

DSpark affine actual prediction paging passes **all 21 placements in 1465.04 s**
(`/tmp/eredu-dspark-affine-prediction-paging-matrix1.log`, frozen CPU build 112).
The fused multi-depth prediction path covers resident/host/disk and every
TP/PP/EP combination with component experiments, effective queries, coordinated
edits, restoration, controlled replay and idle ownership checks.

DeepSeek V4 sequential MXFP4 actual prediction paging passes **all 21 placements
in 863.94 s** (`/tmp/eredu-v4-mxfp4-prediction-paging-gpu-matrix1.log`, frozen
Metal build 121). These use GPU compute on Metal device 0 with local CPU Ring
transport, and include idle ownership, component experiments, queries, coordinated
edits, restoration and controlled replay. Together with the separate CPU F32 and
affine results, this completes those three sequential-prediction paging formats.

Build 123 passes in **3m 00s** (`/tmp/eredu-muse-scale-format-build123.log`).
The first V4 FP8 focused run fails during cold preparation in **2.57 s**
(`/tmp/eredu-v4-fp8-scale-components-gpu-focused1.log`): the parallel declaration
incorrectly applies the FP8 default to the dense embedding, then treats its
block-scale rows as token rows. The correction retains dense embedding defaults
and derives vocabulary ownership from the declared primary weight. A related
schema audit keeps prediction hyper-head functions dense, matching their actual
operator and target-head semantics. Neutral and native reruns remain required.

The V4 vocabulary/embedding correction passes **612 architecture tests**
(one existing ignore, 0.42 s) in
`/tmp/eredu-v4-fp8-static-geometry-neutral1.log`. Both floating and UE8M0 scale
geometry retain token-row ownership independently of block-scale rows, and
prediction hyper-head source constraints now agree with the dense operator.
Native verification is pending the rebuilt adapter.

K2's mixed F16/BF16 scale-companion resident suite passes **all seven GPU
TP/PP/EP placements in 224.57 s**
(`/tmp/eredu-k2-non-f32-scales-resident-gpu1.log`, frozen build 123). The full
attention, dense-FFN and fused-expert tail fixture covers exact effective reads,
signed projections, component masks, coordinated edits, independent edited
checkpoints, restoration and controlled replay. Host/disk and independent-cache
follow-up remain separate.

DSpark MXFP4 actual prediction paging passes **all 21 GPU placements in
1254.93 s** (`/tmp/eredu-dspark-mxfp4-prediction-paging-gpu-matrix1.log`,
frozen build 121). F32, affine and MXFP4 now each have complete actual-paging
matrices for both sequential V4 prediction and DSpark. GPU cases use Metal
device 0 with local CPU Ring transport.

Build 124 passes in **2m 54s**. All **52 native distributed startup/completion
checks pass in 4.28 s** (`/tmp/eredu-native-startup-source-regression4.log`),
including boxed source-less policy and I/O errors through the portable boundary.
The V4 FP8 retry reaches lowering but rejects dense embedding/router sources
as FP8 in **1.55 s** (`/tmp/eredu-v4-fp8-scale-components-gpu-focused2.log`).
The family format resolver now preserves published dense defaults while honoring
explicit admitted transforms; native acceptance remains pending.

Gemma 4 full-precision SafeTensors independent-bank component validation passes
**all 21 GPU placements in 1558.15 s**
(`/tmp/eredu-gemma-independent-components-gpu-matrix1.log`, frozen build 122).
It includes effective queries, projections, coordinated overlays, restoration and
controlled replay with image input and shared KV. Affine independent banks remain
a separate running matrix.

The immutable parameter-index and V3 dense-format defaults pass **613 architecture
tests** (one existing ignore, 0.43 s) in
`/tmp/eredu-parameter-index-v3-format-neutral1.log`. Indexed and direct placement
projections agree on every tested rank and reserve identical per-query budgets.

Build 125 passes in **2m 47s**. V4 FP8 focused retry 3 fails in **1.58 s**
(`/tmp/eredu-v4-fp8-scale-components-gpu-focused3.log`): fused expert recipes
assemble weights but omit their published scale companions. The shared DeepSeek
recipe correction now assembles target and prediction companions using the same
expert ordering and gate/up concatenation. New neutral and native checks remain
required before accepting this source format.

DeepSeek companion assembly passes **614 architecture tests** (one existing
ignore, 0.39 s) in `/tmp/eredu-deepseek-encoded-companions-neutral1.log`. The
regression checks FP8/UE8M0 and MXFP4 packed companion geometry, expert ordering
and rejection of an incomplete scale family. Native validation remains pending.

Build 126 passes in **2m 45s**. V4 FP8 now completes construction, component
trials and exact parameter checks, but the focused suite fails after parameter
queries in **15.80 s**: a prediction readout differs by about 5.27e-4
(`-1.6312728` versus `-1.6317995`), outside its current relative tolerance.
This remains under investigation; no FP8 native matrix is accepted yet.
V3's first FP8 run reaches the same experiment stage but exceeds the fixture's
32 KiB device-cache allowance by 256 bytes. Its next run uses the existing
128 KiB wide-fixture allowance; production cache limits remain enforced.

Muse's published-geometry packed-projector focused run completes the **dense TP
resident case**, including masks, effective queries, coordinated edits and
restoration. It then fails when preparing the routed PP disk case
(`/tmp/eredu-muse-packed-projector-gpu-focused3.log`, suite elapsed 1096.92 s).
The routed preparation loop incorrectly counts vision layers as text expert
layers. The family loop now uses only its declared text depth; the remaining
projector matrix still requires verification.

K2's remaining mixed F16/BF16 scale-companion suites pass **all 35 GPU
placements in 921.89 s** (`/tmp/eredu-k2-non-f32-scales-remaining-gpu1.log`,
frozen build 123). Together with the seven ordinary resident cases, **all 42**
ordinary and independent-cache placements now pass across resident, host-layerwise
and disk execution. Exact source-value queries, signed projections, edited
references, rollback, restoration and controlled replay are included.

The Muse projector/text-layer correction passes **615 architecture tests**
(one existing ignore, 0.40 s) in
`/tmp/eredu-muse-gguf-expert-layer-neutral2.log`. The regression constructs one
vision plus one routed text unit and verifies exactly the projector and text
expert recipes, including rejection of a missing real expert source.

The V4 FP8 diagnostic rerun reproduces the mismatch during the first **active
overlay** comparison. Reservation rejection and publication rollback already
agree with baseline. Readout-input/output diagnostics are being added to
distinguish accumulation sensitivity from an operator or promotion mismatch;
the existing tolerance has not been increased.

Gemma 4 affine independent-bank validation passes **all 21 GPU placements in
2174.12 s** (`/tmp/eredu-gemma-affine-independent-components-gpu-matrix1.log`,
frozen build 122). This completes the ordinary four-format matrices (30 each)
and the full-precision SafeTensors, full-precision GGUF and affine independent-bank
matrices (21 each), alongside the earlier public CPU/Metal and neutral coverage.

### Published DeepSeek formats and Muse projector follow-up

The optimized Metal/Ring build completes in 12m46s. The Muse published-geometry
packed projector passes the full dense TP workflow again. Its routed PP case now
loads, exposing a fixture expectation that assigned all fifty vision layers to
the first stage. The selected construction instead binds twenty-six units per
stage. The fixture now derives unit, static-parameter and residency counts from
the retained neutral selection, as the Gemma fixture already did; rerun pending.

V3 published FP8 passes TP and PP with resident, host-layerwise and disk-streamed
weights. EP then rejects off-rank scale sources that the old expert catalog did
not enumerate. The complete-companion catalog correction is implemented and its
neutral checks pass; native EP and independent-bank reruns remain pending. Mixed
V4 FP8/MXFP4 additionally exposed a cold scale-layout mismatch. The shared family
format resolver now retains the expert encoding for target and prediction
placement; the architecture suite passes 616 tests with one existing ignore.

V4's active-overlay comparison still fails at the readout under the original
`2e-4 * max(1, abs(reference))` envelope. Maximum normalized-input difference is
`0.00019288063` (scaled `0.00012871089`); maximum output difference is
`0.00052666664` (scaled `0.00032275205`). Baseline, selected masks, effective
parameter queries, reservation rejection and publication rollback pass. The next
run verifies the promoted head independently against signed float64 dot products,
using complete bounded public queries checked against source bytes plus edits.
Each float32 multiplication is checked against `gamma(2n) * sum(abs(w*x))`; paired
readout comparisons account for the measured input effect and both rounding
bounds. Other activation tolerances remain unchanged. The native check now passes for the first edited prefix: the measured float64
input effect is `0.0005262494742055424`, and float32 arithmetic uses less than
0.00849 of its forward-error bound. Complete head queries agree exactly with
independently decoded source bytes plus the admitted edits. The second prefix
exposes a shared-SwiGLU discrepancy of `0.00025482`; independent gate/value
equation checks are under validation. The full FP8 matrix is still open.

Native Qwen regression verifies the parameter-index change: three tensor/pipeline
cases pass in 7.76s and streamed PP passes in 2.74s. The backend production check
passes in 10.42s after preserving typed session errors. Both CPU and Metal gather error-identity tests pass, as does the combined
prediction/recovery failure test. All 52 distributed startup/source regression
tests pass, including the ten explicitly ignored native cases. No recovery failure grants state-preserving retry
authority, and these diagnostics do not establish native completion.

```sh
CARGO_INCREMENTAL=0 cargo test --release -j1 -p eredu-backend-mlx \
  --no-default-features --features metal,image --lib --no-run
EREDU_TEST_RING_DEVICE=gpu target/release/deps/eredu_backend_mlx-21a6085cc132da23 \
  ring_muse_projector_gguf_components_focused --ignored --nocapture --test-threads=1
EREDU_TEST_RING_DEVICE=gpu target/release/deps/eredu_backend_mlx-21a6085cc132da23 \
  ring_deepseek_v3_fp8_components_matrix --ignored --nocapture --test-threads=1
CARGO_INCREMENTAL=0 cargo test -j1 -p eredu-architectures --lib
```

These are deterministic synthetic fixtures with native Metal computation and
local CPU Ring transport on one physical GPU, separate from the pinned released
checkpoint reference evidence above. The complete remaining matrices remain open.

The next debug native run additionally verifies complete gate/value matrices
against independently decoded source bytes and the admitted edits, and checks
SwiGLU outputs against a signed float64 equation. It propagates float32 dot-product
error through the declared activation/clamps, with a separately stated finite
elementary-function allowance. DSpark stream reconstruction similarly carries
rounding through its declared mixing coefficients and records head-input
quantization/normalization rounding as a separate term. These additions still
require native acceptance and regression against the existing F32/affine/MXFP4
fixtures; they do not justify declaring the outstanding numerical cases passed.

The latest architecture suite passes 616 tests, including exact off-rank expert
weight/scale source exclusion. Native V3 FP8 independent TP and mixed V4
sequential TP also pass. Their remaining placements remain open.

The next native run passes all three additional-format TP cases (59.52s):
V3 FP8, mixed V4 sequential prediction, and mixed DSpark. Independent shared
SwiGLU checks also pass, with maximum measured input propagation of
`0.0002834167040930424`. A later routed-unit comparison still exceeds the original
absolute envelope; equivalent independent fused gate/up checks are now being
validated. V3 expert-parallel resident and host-layerwise cases pass with the
complete scale-companion catalog; the full matrices are running.

The Muse larger-projector debug rerun passes dense TP, then exposes a vision
pipeline-cut geometry defect: the flattened `[8, 1536]` patch activation was
validated as `[1, 4, 1536]` text geometry. The family now declares separate patch
and projected-media wire schemas, checked cold patch bounds, and exact admitted
grid/placeholder extents. The neutral component fixture now cuts a three-layer
vision tower in PP, with different patch, media and text counts. This correction
still requires neutral and native acceptance.

V3 published FP8 now passes all 21 ordinary Metal/CPU-Ring placements in
503.06s (`ring_deepseek_v3_fp8_components_matrix`), including EP and combined
TP/PP/EP under all three residency policies. The source-companion correction
is therefore accepted on these native paths. Independent-bank validation is
still running.

The target stream-reconstruction regression exposed a separate support-report
gap: partitioned target readout inputs had execution hooks and declarations but
were omitted from component placement. Target placement now assigns that
read-only evidence to output invocations, as prediction placement already did.
The first three neutral Muse TP residency cases verify that ownership. Native
reconstruction regression is still pending on the corrected build.

The strengthened Muse fixture passes vision transport after the wire-geometry
correction, then reaches a decoder-resume gap on a rank that already has vision
context. Muse now accepts the primary-only decoder boundary into that retained
context without repeating assembly. Full neutral and native reruns remain open.

V3 FP8 independent banks also pass all 21 native placements in 586.81s
(`ring_deepseek_v3_fp8_independent_banks_matrix`). Together with the ordinary
matrix, all 42 published V3 FP8 cases pass. Each independent-bank case verifies
real bank requests and owned bytes alongside capture, masks, effective queries,
coordinated edits, rollback and replay.

Mixed V4 sequential prediction passes all 21 ordinary native placements in
719.66s (`ring_deepseek_v4_mtp_mixed_fp8_components_matrix`). This includes
published UE8M0 attention scales with MXFP4 experts, exact source decoding,
queries, coordinated overlays and controlled replay under all residency modes.

Muse's deeper neutral fixture next exposes static vision ownership: the merge
adapter and final projector must be resident on the last vision stage, while
patch/context construction requires input modules on each vision stage. The
family now declares these exact unit consumers for both topology and prepared
sources, using the existing shared-static residency accounting. The full rerun
is still pending; the earlier successful one-layer fixture does not establish
this deeper-tower case.

The strengthened three-vision-layer Muse neutral matrix now passes all 102
placements (18.86s), including tied/untied heads, dense/routed and independently
cached banks, every applicable TP/PP/EP combination and three residency policies.
It exercises distinct patch/media/text extents, continuation inside vision and
into the decoder, exact static-module consumers, and target projection-input
capture ownership. Native acceptance of the larger published projector remains
pending. Reproduce with:

```sh
CARGO_INCREMENTAL=0 cargo test -j1 -p eredu-architectures \
  --test reference_numeric \
  muse_components_follow_media_residency_and_all_parallel_bank_placements \
  -- --nocapture
```

Mixed DSpark also passes all 21 ordinary Metal/local-CPU-Ring placements
(`ring_deepseek_v4_dspark_mixed_fp8_components_matrix`): 986.95s.
This verifies the published attention/expert encodings, bounded prediction
residency, effective queries, overlays, rollback and replay across all seven
parallel combinations and three residency policies. Its signed stream/readout
reconstruction uses the propagated arithmetic bounds described above.

Mixed V4 independent banks pass all 21 GPU/local-Ring placements in 728.44s
(`ring_deepseek_v4_mixed_fp8_independent_banks_matrix`). Both ordinary and
independent-bank mixed V4 matrices are now accepted.

The corrected native build passes pure-FP8 sequential prediction with F32 and
UE8M0 scale companions, including the complete independently decoded edited
gate/up banks and signed F64 routed-unit checks. The corresponding independent
TP cases also pass. Full matrices are running. Pure-FP8 DSpark exposes a later
cancellation-sensitive stream-output comparison after a coordinated edit:
`0.1629057` versus `0.1626463`. A new check evaluates each declared stream
equation independently in both executions, then propagates measured input and
coefficient differences plus FP32 summation bounds. That check is not yet
accepted natively; the unchanged base comparisons and exact source queries
remain required.

CPU and Metal provider-failure regressions both pass (6.01s) after the partition
source-preservation changes. The architecture unit suite passes 616 tests with
one existing ignore; the full neutral numerical and distributed failure suites
are still running.

The final architecture pass accepts all 616 unit tests (one existing ignore)
and all 274 numerical tests (83.24s), including the strengthened Muse placement
fixture. All 12 distributed provider-failure tests pass in 195.01s. The
12-case V4/DSpark readout regression passes in 367.04s: full precision, affine
and MXFP4, each through TP resident and PP disk-streamed execution. This accepts
the target projection-input registration and the stream reconstruction changes
on those formats.

The larger Muse projector now passes routed PP disk-streamed native execution
on the corrected build. The focused combined TP/PP/EP independent-bank and
dense-TP cases, followed by the full 51-case matrix, remain open.

A subsequent failure audit found diagnostic-only communication wrappers still
dropping original causes. Source changes retain typed submission/completion
errors through the neutral communication and partition-session boundaries,
without changing poisoning, agreement or resource release. Their expanded
neutral/native failure tests remain pending; the passing runs above predate
this follow-up.

Mixed DSpark independent banks pass all 21 placements in 1086.74s
(`ring_deepseek_dspark_mixed_fp8_independent_banks_matrix`). Together, the
ordinary and independent-bank mixed V4/DSpark suites accept all 84 placements.

The communication-source correction passes all 515 runtime unit tests and all
74 neutral backend tests. The strengthened failure fixture traverses both real
submission and completion error paths, finds the original typed cause, and
proves a poisoned retry makes no additional backend call. Native follow-up is
still pending.

The larger Muse projector passes routed PP disk streaming, then its combined
TP/PP/EP independent-bank run fails at the first prefill with mismatched
collective order. Vision execution had created context while token lookups
remained deferred; inactive ranks inferred that ingress was already complete.
A neutral architecture hook now reports pending ingress from actual context:
Muse inspects unresolved token parts, while architectures with eager lookup
retain the default. An added neutral fixture executes vision and decoder ingress
separately to check this lifecycle. The correction still requires native proof;
the existing deadlines and resource-retention behavior are unchanged.

V4 sequential prediction with published F32-scale block FP8 now passes all
21 ordinary native placements in 797.76s
(`ring_deepseek_v4_mtp_fp8_float_components_matrix`). The full matrix includes
the independently decoded edited head, shared gate/value matrices and routed
gate/up banks, signed F64 arithmetic checks, exact effective queries, rollback
and controlled replay. The UE8M0 and independent-bank matrices are still running.

The remaining sequential ordinary UE8M0 matrix passes all 21 placements in
931.85s; F32-scale independent banks pass all 21 in 852.79s. Both include
source decoding, masks, effective queries, coordinated edits, rollback and replay.
The UE8M0 independent-bank matrix is still running.

The pending-ingress lifecycle test and the Muse neutral suite pass: nine tests
including all 102 placement cases (22.12s). Native communication checks pass
42 ordinary and ten explicit tests, plus all six public session/capture-hook
cases. The deadline test finds the original native exception through the new
source chain while retaining poisoned-retry and no-additional-native-call checks.

The new DSpark stream checks pass their independent equations but expose the
next cancellation-sensitive FFN contribution after an edit: `3.493431` versus
`3.4916306`. The next validation build additionally reconstructs that complete
write in signed F64 from independently decoded shared/routed down matrices,
captured units and exact route coefficients. Each execution must satisfy its
FP32 forward-error bound before measured propagation is used for comparison.
This follow-up is not yet accepted natively.

V4 sequential UE8M0 independent banks pass all 21 placements in 1009.51s
(`ring_deepseek_v4_fp8_ue8m0_independent_banks_matrix`). All 84 sequential
pure-FP8 placements are now accepted across F32/UE8M0 scales, ordinary and
independent banks, three residency policies and seven parallel combinations.
GPU ranks share one physical Metal device and use local CPU Ring transport.

The current typed-communication build passes all 12 distributed provider-failure
tests in 216.49s (`ring_provider_failure_`, Metal/local CPU Ring). Together with
the 52 completion/runtime and six public-session tests, this verifies original
causes, shared poisoning, bounded waits, safe retention and retry non-entry.

The DSpark F32-scale FP8 TP-resident case now passes the complete workflow,
including independent signed reconstruction of the combined routed/shared FFN
write, stream mixing, edited shared/routed SwiGLU units and the primary readout.
The source decoder checks complete effective matrices before applying the same
admitted additive edits in the F64 oracle. Each native write must match its own
equation within `gamma(3 * terms) * sum(abs(products))`; comparison between runs
then includes measured input/coefficient propagation. Normalized inputs retain the base comparison; edited units must also satisfy
their independent gate/value equations. Route identities, exact parameter
queries and replay remain required.
All four 21-case DSpark pure-FP8 matrices are now running; this first case does
not yet establish their complete acceptance.

The final architecture pass succeeds with 616 unit tests (one existing ignore)
and all 275 numerical tests (78.58s), including the pending-ingress lifecycle
fixture and all strengthened Muse placements. SHA-256 revalidation of the
original and widened SmolLM2/LFM2-350M files and all four LFM2-8B-A1B shards
matches the recorded provenance and original LFS digests. The public consumer
refresh remains pending.

The strengthened Muse packed-projector fixture passes its complete eight-rank
TP+PP+EP independent-bank host-layerwise workflow in 807.80s on the debug build
(`ring_muse_projector_gguf_components_independent_pipeline_focused`). This includes
vision/decoder continuation, component masks, effective queries/projections,
coordinated edits, rejection rollback, restoration and cached replay. It accepts
the pending-ingress correction natively. The fixture uses published projector
geometry with synthetic nonzero values; it is not a released-checkpoint accuracy
comparison. The full 30 ordinary and 21 independent-bank placements remain
pending on the optimized build.

Final feature verification passes the GGUF/checkpoint/core/runtime/architecture
checks, text without default features, NN/codec with all features, media with
and without features, and the MLX backend without default features. The portable
facade passes 19 tests (one existing ignore), and neutral backend conformance
passes all 82 tests. Both public example executables are rebuilding for the
released-checkpoint refresh.

The optimized final FP8 check passes all four TP-resident cases in 190.55s:
sequential V4 and fused DSpark, each with F32 and UE8M0 scale companions
(`ring_deepseek_v4_fp8_components_focused`). This verifies the final readout
evidence selection and independent edited-unit/FFN/stream arithmetic checks on
the optimized build. Full DSpark matrices remain in progress.

Both freshly rebuilt public dense examples pass their complete five-trial
reference workflow and nine association-case measurements. SmolLM2 compares
82,948 values per trial with maximum absolute error `7.6294e-5`; its maximum
signed score/margin reconstruction error is `1.5246e-5`. LFM2-350M compares
184,323 values per trial, with maximum absolute error `8.2016e-5` and maximum
reconstruction error `5.9283e-6`. The existing `2e-4` absolute/`3e-4` relative
reference tolerances are unchanged. Both verify backward queries, full-vocabulary
scores, coordinated edits and restoration. SmolLM2's keep-only trial changes
the winner; association effects remain illustrative measurements. Reproduce
using the pinned JSON references and the `component_reference_probe` commands
above, with the current `mlx,metal` build. The sparse controlled GPU refresh is
still running.

DSpark pure FP8 with UE8M0 scale companions passes all 21 ordinary Metal/local
CPU-Ring placements in 1534.79s
(`ring_deepseek_v4_dspark_fp8_ue8m0_components_matrix`). This completes all
residencies and seven parallel combinations with actual prediction paging,
independently decoded source values, signed edited FFN/stream/readout checks,
effective queries, coordinated edits, rollback and controlled replay. The other
three pure-FP8 DSpark matrices remain in their final placements.

DSpark pure FP8 with F32 scales passes all 21 ordinary placements in 1699.12s;
UE8M0 independent banks pass all 21 in 1578.77s. Together with UE8M0 ordinary
execution, 63 of the 84 pure-FP8 DSpark placements are accepted. F32 independent
banks remain on their last placement.

The refreshed sparse LFM2-8B-A1B controlled Metal consumer and independent
comparison both pass: 10,944,290 compared values, 150 write groups (921,600
components), and 20 predictions with 40 selected scores/differences (3,686,400
component contributions). Maximum write error is `0.00048828125`; signed
projection associativity error is `1.5019e-7`. Maximum score error is `0.05123`,
within 0.3652 of the declared BF16 output-rounding bound; the propagated
projection bound is at most `8.8844e-5`. The existing tolerances and one-BF16-ULP
logit condition are unchanged. No comparison failures remain.

The original reference environment no longer contained NumPy. This refresh used
the existing comparison environment with NumPy **2.4.6**; the independent
reference artifacts and all checkpoint hashes remain unchanged:

```sh
/tmp/eredu-nanbeige/venv/bin/python \
  eredu-evaluation/scripts/compare_component_sparse.py \
  /tmp/eredu-lfm2-sparse-validation/transformers-score-reconstruction.json \
  /tmp/eredu-lfm2-sparse-validation/native-scores-final135.json \
  --bf16-logit-ulps 1 --require-controlled \
  --require-write-reconstruction --require-score-reconstruction
```

The final F32-scale independent-bank DSpark matrix passes all 21 placements in
1719.71s (`ring_deepseek_dspark_fp8_float_independent_banks_matrix`). All 84
pure-FP8 DSpark placements are now accepted. Together with 84 pure-FP8 sequential
V4, 84 mixed V4/DSpark and 42 published V3 FP8 placements, the additional encoded
prediction and independent-bank matrices are complete. Every GPU rank uses the
same physical Metal device with local CPU Ring; multiple physical devices and
hosts remain hardware-validation gaps.

## Final Muse published-projector matrix

The optimized Metal/local CPU-Ring run passes all **21 independent-bank
placements in 1913.65s**: TP2, PP2, EP2, their three pairings and TP2/PP2/EP2,
each with resident, host-layerwise and disk-streamed weights. Every placement
executes component capture/masks, effective parameter queries/projections,
coordinated edits, rejected publication/removal rollback, restoration and cached
controlled replay. Independent banks use ordinary cache limits; this run is not
additional forced-eviction evidence.

The fixture has the published 50-layer, 1536-wide vision geometry, 8960 FFN
width, 16 heads and 14-pixel patches, with a bounded prepared patch grid and a
small two-layer decoder. Projector matrix encodings alternate deterministically
between Q8_0 and IQ4_NL. Tensor names seed deterministic nonzero fixture values
in `fixtures/muse_projector_gguf.rs`. This exercises real projector source
admission, mid-vision pipeline cuts, vision-to-decoder continuation and deferred
text ingress; synthetic values do not establish released Muse model accuracy.

All **30 ordinary-bank placements pass in 1957.01s** on the same executable:
nine dense and 21 sparse cases, with the same capture/mask/query/edit/rollback/
restoration/replay checks. Together, the two suites complete all **51 native
placements**. They ran sequentially to bound aggregate host memory. Exact commands:

```sh
CARGO_INCREMENTAL=0 cargo test --release -j1 -p eredu-backend-mlx \
  --no-default-features --features metal,image --lib --no-run
EREDU_TEST_RING_DEVICE=gpu target/release/deps/eredu_backend_mlx-21a6085cc132da23 \
  ring_muse_projector_gguf_components_independent_bank_matrix \
  --ignored --nocapture --test-threads=1
EREDU_TEST_RING_DEVICE=gpu target/release/deps/eredu_backend_mlx-21a6085cc132da23 \
  ring_muse_projector_gguf_components_matrix \
  --ignored --nocapture --test-threads=1
```

All GPU ranks use one physical Metal device with local CPU Ring transport;
these results do not validate multiple physical GPUs or hosts.

The final requirement audit is complete. Each required capability has a public
integration path, real implementation, behavioral coverage and documentation;
see the [completed checklist](component-analysis-checklist.md). Final crate and
feature checks, neutral/facade/native failure checks, all three refreshed released
consumers and the remaining encoded/media matrices pass. The declared hardware,
protocol, numerical and resource-accounting limitations remain explicit in the
integration guide. Illustrative association edits establish the API workflow,
not the paper's reported editing efficacy.

## Commit integration verification

Before publication, the implementation was rebased onto upstream
`df0af944` (candidate capture charges the logits row) and `f7eb8dc8`
(Qwen MLX checkpoint compatibility). The format-selection conflict retains
embedding formats through the shared matrix helper. The speculative capture
capability description now names phase-aware admission and retained distributed
producer/transport bindings rather than an obsolete single-rank restriction.

The native checkpoint suite exposed a selection-pushdown regression: independently
selecting three axes could return the correct scalar while retaining a larger
source read. A nonzero neutral fixture reproduced a four-element read where one
suffices. Independent axes now apply in source-axis order, collapse contiguous
source spans, and recurse only into the original smaller recipe. All six axis
orders preserve both the selected scalar and its one-element read bound; the
existing noncontiguous expert-row/column bounds still pass.

With this fix, all **99 checkpoint unit tests**, **620 architecture unit tests**
(one existing ignore), and **277 numerical conformance tests** pass. The numerical
run includes the new upstream Qwen MLX payload/recipe fixtures and the complete
component placement fixtures. Commands:

```sh
CARGO_INCREMENTAL=0 cargo test -j1 -p eredu-checkpoint --lib
CARGO_INCREMENTAL=0 cargo test -j1 -p eredu-architectures --lib --test reference_numeric
```

The final native build passes all **24 checkpoint-recipe tests** (0.13s), including
the original contiguous-span regression and dtype-preserving `SubtractOne`, and
all **17 capture tests** (0.61s), including explicit Metal cases and the upstream
candidate-row accounting check. The default-feature CLI check also passes
(18.70s). Exact commands:

```sh
CARGO_INCREMENTAL=0 cargo test -j1 -p eredu-backend-mlx \
  --no-default-features --features metal --lib --no-run
target/debug/deps/eredu_backend_mlx-f1a0f0df05777822 \
  backend::runtime::checkpoint::recipe::tests \
  --include-ignored --nocapture --test-threads=1
target/debug/deps/eredu_backend_mlx-f1a0f0df05777822 \
  composition::mlx::session::bounded_capture::tests \
  --include-ignored --nocapture --test-threads=1
CARGO_INCREMENTAL=0 cargo check -j1 -p eredu-cli
```
