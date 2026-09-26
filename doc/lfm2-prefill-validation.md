# LFM2 completed chunked prefill

Ordinary replicated dense and routed LFM2/LFM2.5 text generation uses the shared
core prefill driver, including controlled generation. `PrefillChunkPolicy::Bounded`
selects the maximum positions per eligible invocation; its default remains 512.
`Unchunked` remains available for callers preferring a full pass. No device-memory
heuristic or allocator cap is added.

The attention cache and absolute positions advance across chunks. Each convolution
retains its last `kernel_size - 1` gated inputs, compacted through the general
neutral tensor contract. Ordinary prefill completes output and retained state
before publication. Cancellation is checked between completed chunks, with no
sampling or token commitment for intermediate prefixes. The first controlled step
still owns the entire prompt; pause/snapshot boundaries remain completed tokens.
Cancellation may leave partial prompt state and requires reset before a different
request. Native work already submitted is settled rather than preempted.

Shared dense and heterogeneous readout helpers select the last hidden position
before vocabulary projection. Both direct and routed text strategies honor that
selection. Prefixes currently compute one unused vocabulary row. Observed and
speculative paths retain their full-output contracts. Capture/intervention,
structured inputs, prediction extensions and distributed generation retain their
existing full-pass restrictions; these are implementation gaps with the reasons
recorded in `doc/execution-control.md`, not LFM2 architectural limitations.

## Reproducible checks

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --test reference_numeric lfm2_prefill --locked
CARGO_INCREMENTAL=0 cargo test -p eredu-runtime --test backend_independence --locked
CARGO_INCREMENTAL=0 cargo test -p eredu-core --lib prefill --locked
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --lib cold_prefill_chunking --locked
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --features metal --test chunked_prefill --locked -- --test-threads=1
CARGO_INCREMENTAL=0 cargo test -p eredu --no-default-features --features mlx,metal --test native_execution_control native_lfm2_workspace_forecasts --locked -- --ignored --test-threads=1
```

The scalar neutral fixture has nonzero weights, dense/routed feed-forward layers,
mixed, attention-only and convolution-only schedules, prompts of 1/2/3/4/11 tokens,
chunks of 1/2/3/4/32 tokens, and three cached decode steps. It compares complete
prompt logits, fixed history, layer positions and subsequent logits across resident
and unit-window traversal. The completion fixture verifies publication, rollback
and reuse after a completion failure without changing ordinary decode scheduling.
Core tests verify cancellation after a completed prefix without sampling/decode.

Native nonzero fixtures cover Llama/Mistral regressions plus dense/routed LFM2
SafeTensors and GGUF, and a convolution-only LFM2 schedule. Each covers resident,
host-layerwise and disk-streamed execution; SafeTensors also covers load-time affine
four-bit quantization. Prompts have 1/3/9 tokens,
chunks of 1/2/3/4/32, and three cached decode steps. Maximum absolute full/chunk
logit tolerance is `2e-4`; ordinary unchunked and controlled chunked greedy output
must match. Full observed prefill retains all vocabulary rows. Cold and loaded
chunk capabilities agree. The native convolution retention check compares histories
from 1,024 and 16,384 positions after synchronization, allowing at most 32 KiB of
retained storage, including native allocation overhead, in either case.

## Pinned released checkpoint

The independent reference uses `LiquidAI/LFM2-350M` revision
`f37d3f5c8c5484bc01dad379a595cf4c68c4e70e`. Its BF16 SafeTensors SHA-256 is
`387638dc889ff1a1395c3c2ab9605211e4c7e16f2d375361dd4e423b909a254e`, verified
against pinned Hugging Face LFS metadata and the downloaded bytes. The reference
exporter widens floating weights exactly to F32 outside the repository, records
both source and derived hashes, and runs PyTorch eager attention independently.
The derived SafeTensors SHA-256 is
`58315fed16065aaaa453bfb09313f5edf9a18fd4a761fbdc428b60ec0b7b7560`.

```sh
python3 -m venv /private/tmp/eredu-prefill-reference
/private/tmp/eredu-prefill-reference/bin/pip install torch==2.14.0 transformers==5.17.0
/private/tmp/eredu-prefill-reference/bin/python eredu-evaluation/scripts/download_component_fixture.py /private/tmp/eredu-prefill-lfm2 --model lfm2
/private/tmp/eredu-prefill-reference/bin/python eredu-evaluation/scripts/prefill_reference.py /private/tmp/eredu-prefill-lfm2/provenance.json /private/tmp/eredu-prefill-lfm2/reference.json --f32-checkpoint /private/tmp/eredu-prefill-lfm2/f32
EREDU_PREFILL_REFERENCE=/private/tmp/eredu-prefill-lfm2/reference.json CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --features metal --test chunked_prefill released_checkpoint --locked -- --ignored --nocapture --test-threads=1
```

The prompt is `[1,3,2,5,8,13,21,34,55]`; cached inputs are `[5,8,13]`.
Chunks 1/2/4/128 compare all vocabulary logits after prefill and every cached step
against the independent F32 reference with maximum absolute tolerance `2e-3`.
The same native harness compares four greedy outputs and additional active-memory
peaks for 128/1,024-token prompts with unchunked and 32-token prefill. These are
macOS Apple-silicon Metal measurements. iPhone peak-memory and cancellation-latency
measurements remain a device-validation gap; no wall-clock latency guarantee is
inferred from chunk size. Released BF16/quantized numerical tolerances are not
inferred from this F32 comparison.


## Recorded released result (2026-09-26)

The pinned F32 comparison passes on macOS Metal with PyTorch 2.14.0 and
Transformers 5.17.0. Maximum absolute errors over all four full-vocabulary
predictions are `0.00030612946`, `0.00015425682`, `0.00014543533` and
`0.0001308918` for chunks 1, 2, 4 and 128 respectively.

| Prompt positions | Unchunked additional active peak | 32-token additional active peak | Greedy output parity |
| --- | ---: | ---: | --- |
| 128 | 250,078,512 bytes | 76,160,052 bytes | identical four tokens |
| 1,024 | 399,556,888 bytes | 183,888,136 bytes | identical four tokens |

These include four generated tokens, use the existing allocator/retention defaults,
and exclude already-loaded model parameters by subtracting the pre-run active
baseline. They are allocator observations for this workload, not process-RSS limits
or predictions for other devices. The 1,024-token case reduces additional peak
active memory by approximately 54%. The released test completes in 4.60 seconds.


A separate load-time GGUF requantization probe failed during exact source-provenance
admission, before prefill, when applying affine four-bit transformation to the
synthetic F32 GGUF embedding. The prefill matrix therefore keeps GGUF's admitted
file encoding and exercises load-time affine quantization through SafeTensors.
This change does not repair or claim validation of GGUF requantization.


Final regression results: all 13 LFM2 neutral numerical tests pass, including
partition/residency/component regressions; all 75 runtime conformance tests pass;
the nine core prefill tests and cold selection test pass. The native CPU/Metal
matrix and retention tests pass (four ordinary tests, 23.49 seconds); the explicit
released reference passes separately. The public native LFM2 forecast/controlled
regression passes (0.99 seconds). Portable backend conformance and facade tests
pass (112 and 27 tests respectively; one existing opt-in test is ignored).

Formatting, boundary checks, production-library Clippy and focused native-test
Clippy pass. Broader all-target Clippy is blocked by eight existing
`cloned_ref_to_slice_refs` warnings in the unrelated
`eredu-runtime/src/memory_forecast/retention.rs` tests.
