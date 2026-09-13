# Component analysis implementation checklist

The LM Inspector infrastructure goal is complete for the applicable autoregressive
text workflows, including prepared media and embedded prediction. Each outcome
below has implementation, public integration, behavioral coverage and documentation.
The integration guide records exact loaded capabilities and protocol limitations.

- [x] Reassess Eredu and Inspector integration; read paper §§2–4, §6 and Appendix D.
- [x] Stable component topology, parameter joins, equations and actual loaded support.
- [x] Normalized-input, FFN-unit, attention-channel, residual and effective-value hooks.
- [x] Compact component selection and keep-only interventions at selected positions.
- [x] Bounded effective-parameter access and projections across the applicable matrix.
- [x] Full-vocabulary selected scores, competitors, log probabilities and ranks.
- [x] Exact token-ID trials, controlled parity, replay, snapshots and isolated branches.
- [x] Atomic reversible multi-parameter overlays, provenance and cache compatibility.
- [x] Complete resource accounting, failure recovery and cumulative trial budgets.
- [x] Compatible-family resident, streamed and parallel coverage.
- [x] Neutral, facade, native and released-checkpoint acceptance coverage.
- [x] Public headless example: attribution, ablation, backward queries and both edits.
- [x] Complete integration/support documentation and final requirement audit.

## Requirement audit

| Required outcome | Public path and behavioral evidence |
| --- | --- |
| Stable topology and equations | `inspect_architecture`, [component IDs and equations](../eredu-core/src/component.rs), and loaded capture/intervention discovery; architecture unit/numerical suites and family-native matrices |
| Actual component capture and causal masks | [Exact-token observed/intervened preparation](../eredu/src/api/observed.rs) and shared controlled drivers; [nonzero numerical fixtures](../eredu-architectures/tests/reference_numeric/components.rs) cover unit/channel masks, survivor recomputation and original/effective evidence |
| Effective reads and bounded projections | [`parameter_discovery`, `query_parameter`, `project_parameter`](../eredu/src/api/parameters.rs); independent packed-source decoding, selected geometry and reservation failures |
| Fixed-prefix scoring and repeatable trials | Exact-token replay, controlled snapshots/forks and full-vocabulary `TokenScores`; public reference consumers and neutral lifecycle conformance |
| Reversible coordinated edits | [Overlay admission, activation and removal](../eredu/src/api/parameters.rs); independent edited references, [peer transaction tests](../eredu-runtime/src/parameter_operations/coordination/tests.rs), cache-version rejection and restoration |
| Accounting and safe failure | Cumulative capture/query/edit budgets and typed outcomes; neutral admission/coordination tests, native completion tests and observed/unobserved provider failures |
| Released models and Inspector integration | [Public dense consumer](../eredu/examples/component_reference_probe.rs) and [controlled sparse consumer](../eredu/examples/component_sparse_probe.rs); refreshed pinned SmolLM2/LFM2 references, source hashes and the integration guide |
| Applicable execution matrix | Resident/host/disk and TP/PP/EP evidence below; all 51 larger Muse projector placements pass, completing the native acceptance matrix |


## Completed acceptance

- Final architecture verification: **620 unit and 277 numerical tests**. Portable
  facade: **19 passes**; neutral backend conformance: **82 passes**. Architecture
  and facade each retain one pre-existing ignored test. Required crate and feature
  boundary checks pass.
  Post-rebase verification also passes 99 checkpoint tests, 24 native recipe tests,
  17 native capture tests (including Metal), and the default-feature CLI check.
- Final communication/source validation: **515 runtime and 74 neutral backend
  tests**, plus native completion, public-session and observed/unobserved provider
  failures. Original causes, safe completion, poisoned retry non-entry and
  cumulative accounting are preserved.
- Resident, host-layerwise and disk-streamed execution; applicable TP/PP/EP
  combinations; target and embedded prediction; ordinary and independently
  cached experts. The [family matrix](component-analysis.md#execution-coverage)
  records exact formats and placement coverage.
- Published V3 FP8: **42 placements**. Mixed V4/DSpark, sequential pure FP8 and
  fused pure FP8: **84 placements each**, including F32/UE8M0 scale companions,
  actual prediction paging, source decoding, signed arithmetic checks and edits.
  Separate K2 native cases cover F16/BF16 scales and forced eviction.
- Final Muse published-geometry packed projector: **51 native placements**
  (30 ordinary in 1957.01s; 21 independent-bank in 1913.65s), plus **102 neutral
  placements**, including mid-vision cuts and deferred decoder ingress.
- Refreshed public SmolLM2-135M, LFM2-350M and controlled LFM2-8B-A1B consumers
  pass independent pinned-reference comparisons, reconstruction, causal masks,
  effective queries, coordinated edits and restoration. Source and derivative
  hashes match provenance. The sparse comparison checks **10,944,290 values**.
- The [validation record](component-validation.md#current-acceptance-summary)
  preserves exact commands, revisions, geometries, tolerances and detailed results.
  The guide and checklist's local links and documentation whitespace were checked.

## Scope and limitations

Distributed GPU validation uses local CPU Ring transport with all ranks on one
physical Metal device; multiple physical GPUs and hosts remain hardware-validation
gaps. Independent-cache matrices use ordinary limits unless explicitly marked as
forced-eviction cases. Synthetic published geometry supplements released-model
validation. Realtime codebook/frame experiments require their separate input/state
protocol; a text-token prefix does not specify those experiments.

Quantized edits promote only affected matrices or banks, with declared storage,
conversion and state costs. Budgets describe owned arrays and transfers rather
than physical allocator guarantees. Overlay changes invalidate incompatible state;
active overlays reject persistent prompt-cache operations. Initial speculative
trials use fresh exact-ID replay because speculative snapshots begin after prefill.

Inspector owns ranking, calibration, naming, graph/sufficient-set search and editing
efficacy. Historical uncaptured predictions require exact original token-ID replay;
tested output tokens are never forced. See the [integration guide](component-analysis.md)
for public API timing, score conventions, failure recovery and downstream steps.
