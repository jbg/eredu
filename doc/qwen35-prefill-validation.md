# Qwen3.5 text prefill memory and cancellation

## Execution contract

Qwen3.5 and Qwen3-Next target-only causal text execution support bounded prefix
passes in resident, host-layerwise and disk-streamed execution. This includes
plain text through the conditional/composite adapter. Convolution history, FP32
recurrent matrices, attention K/V and absolute positions continue across passes.
Ordinary and controlled generation use the same prefix driver. Each nonfinal
prefix completes output and retained state without sampling; cancellation is
checked before another prefix is submitted.

The composite contract distinguishes plain-text prefixes from retained prepared
media requests. Structured media, capture, distributed execution and selected
prediction extensions retain their separately reported restrictions. Qwen3.5
support does not depend on Flash-Next (`qwen4_exp`). Ordinary unobserved readout
selects the final hidden position before vocabulary projection; observed and
prediction consumers retain full-output readout.

The MLX gated-delta mechanism evaluates both scan output and final recurrent
state during multi-token execution. This prevents upstream lazy graphs from
accumulating across recurrent layers. Single-token decode remains lazy. Native
completion does not cancel submitted work or release submission authority early.
Chunk size bounds submitted positions, not wall-clock latency or total KV memory.
Pico owns budget selection, approvals, memory-warning response and unloading.

## Pinned artifact and measurement scope

Validation uses `mlx-community/Qwen3.5-2B-4bit`, revision
`674aaa7240b91e8012fcad5d791b7dfe5ba90207`, at
`/Users/jbg/dev/pico/Build/ModelValidation/qwen35-2b-4bit`.
The local SafeTensors SHA-256 matches the publisher's LFS metadata for that revision:
`713fe7e5d3c3965f7106b0d0ee17615f7869c23c8d327996df8c1196fbcf07d5`.
The engine base is `62074945b334268bc9acc86f97a177d10308b5c1`, with the
implementation in this tree. Its parent is the reported replay revision
`bc1857a5aa01b012165dc1d8948a0bc72bba9655`.

The actual checkpoint has no MTP tensors. Converted-layout admission selects a
zero-depth prediction extension despite the copied configuration declaring one
MTP layer. Runtime prefix support confirms the target-only path. The selected
execution is a resident composite target; the text-only request skips vision
execution, although the loaded artifact includes vision weights.

Stored tensors comprise 786 BF16, 187 U32 packed tensors and 18 FP32 tensors.
Packed embedding lookup promotes activations to FP32; Q/K normalization and
recurrent accumulation also use FP32. The configuration's BF16 declaration alone
is therefore not a valid activation-workspace scalar width. Bounded runtime
summaries on the released model confirm FP32 for
`model.layers.0.mixer.qkv.projected`, `mixer.qkv.convolved`, `mixer.channels`
and `model.logits` (14-position diagnostic prompt). The isolated native scan
reports FP32 sequence output and FP32 state.

All following measurements use Mac Metal, default allocator policy, and fresh
processes. They are allocator active-byte high-water marks, not process footprint,
not enforced allocation limits, and not iPhone measurements. Loaded active bytes
are `1722149056`. The diagnostic resets the peak counter **after load and forecast**;
request peaks still include resident weights and cover prefill plus seven cached
decode steps. Eight greedy output tokens are requested. Successful reset and
synchronization return active bytes to `1722149056`.

| Input | Positions | Chunk | Request peak bytes | GiB |
| --- | ---: | ---: | ---: | ---: |
| Short text, no tools | 19 | 512 | 1945531895 | 1.81 |
| Frozen Pico tool prompt | 2739 | Unchunked | 4490471724 | 4.18 |
| Frozen Pico tool prompt | 2739 | 512 | 2399439304 | 2.23 |
| Frozen Pico tool prompt | 2739 | 128 | 2095669448 | 1.95 |
| Repeated/truncated frozen token IDs, synthetic stress input | 4096 | 512 | 2500894460 | 2.33 |

The frozen prompt comes from `/tmp/pico-eredu-update-replay.json`, combining its
shared messages with the first case. These diagnostics never execute a tool.
The unchunked, 512 and 128 runs have identical eight-token outputs. Independent
MLX-LM (`0.31.3`, MLX `0.32.2`, Transformers `5.17.0`) matches all eight tokens
on the exact 2739 input IDs, covering prefill and seven cached decode steps.
This is exact token agreement, not a claim of identical full logits.

The real Pico native bridge replay also completes the same frozen tool selection
and reset/drain on this implementation. Its requested/effective chunk is 512,
output limit 1024, full-pass reason absent, and additional-generation forecast is
`113685196` bytes to unknown (`insufficient_information`, incomplete). Its fresh
load-plus-single-generation allocator peak is `2399439304`; active bytes after
generation are `1810819264`, returning to `1722149056` after reset. Its 8 GiB
budget remains advisory. The log is `/tmp/qwen35-final-pico-replay.log`.

## Allocation attribution

A diagnostic ablation retains final-position readout but omits the native scan
completion boundary. It measures `21969715204` bytes unchunked, `5789559884`
bytes at chunk 512, `3115823692` at chunk 128, and `2568677460` at chunk 64.
Enabling the boundary yields the table above without changing the compared token
outputs. Thus upstream lazy-graph retention across recurrent invocations is a
measured major contributor. Final-position vocabulary projection alone does not
reduce the dominant unchunked peak in this case.

The native mechanism probe at 2739 positions, 16 heads, 128 key/value dimensions
and convolution width four measures:

- Scan: baseline `45252608`, peak `181370884`, sequence output `22437888`,
  final state `1048576` bytes. The Metal scan submits 32-position internal kernels
  for this length and retains their sequence outputs and dependent state versions.
- Depthwise convolution: baseline `67485696`, peak `134807552`, output
  `67313664` bytes. The probe evaluates inputs before resetting its peak.

These isolated peaks do not add up to a model peak: operator lifetimes overlap,
inputs can alias, and the full model has other retained roots. They distinguish
single-mechanism costs from the larger lazy graph. Attention/MLP intermediates,
normalization, projection conversions and convolution products are upstream
consumers affected by the evaluation boundary; there is no per-allocation trace
assigning an exact fraction of the model peak to each one. Quantized lookup
unpacks selected embedding rows and tied output uses quantized matrix multiplication;
there is no evidence here that full-vocabulary dequantization is the dominant cause.
Post-request allocator cache is approximately 256 MiB; it is distinct from active
bytes and from the historical 20.46 GiB peak.

## Forecasts

Loaded composite sessions report their actual state offset. Reset-state forecasts
retain selected architecture topology; advanced state still invalidates fresh-request
workspace projections. Cold and loaded contracts advertise plain-text prefix support
and final-position logits consistently with the selected execution.

Recurrent topology carries the actual projection specifications, convolution
channels/kernel, expanded heads and matrix dimensions. All projection-format and
promotion accounting consumers include it. Workspace estimates include the FP32
scan output and known projection/convolution contributions, while selected scan
scratch and state-version retention retain an unknown upper bound. No unknown
cost becomes zero or a likely-fit verdict. The frozen prompt's loaded forecast
reports effective chunk 512 (or 128 as requested), no full-pass fallback, and
`insufficient_information` with an unavailable working-memory upper bound.
The Mac measurements calibrate neither an iOS allocator bound nor process overhead.

## Cancellation and retry identity

A fresh-process chunk-512 run requests cancellation at approximately 107 ms after
entering generation and returns at 227 ms: approximately 120 ms after cancellation.
No token is emitted. Peak active allocation is `2241710536`; active bytes before
reset are `1755097280`, and reset/drain returns to `1722149056`. This exercises
native prefix work, unlike a 20 ms cancellation that stops during request preparation.
The result is one measured latency, not a deadline. Metal work already submitted
must finish safely before resources are released.

Repeated rendering of identical ordered messages and schemas produces identical
prompt bytes in Eredu. Reversing only object-key insertion order in the frozen tool
schemas changes Eredu's token count from 2739 to 2749. Twelve deterministic key-order
variants in the independent tokenizer range from 2738 to 2749. The array order,
messages and schema values remain the same. Pico's unsorted `JSONEncoder` output,
Eredu's order-preserving JSON representation and the template's `tojson` filter
explain how semantically identical schemas can yield different prompt bytes.
The historical journal contains no exact serialized requests or rendered prompts,
so this does **not** establish the specific cause of its 2740→2754 difference.
Capture/freeze serialized input or compare request/rendered-token fingerprints to
resolve a future retry; canonicalizing/freezing application input belongs in Pico.
The template renderer must preserve the requested serialization semantics.

## Historical phone evidence

The September 28 phone journal ends after an iOS memory warning and cancellation,
with no first semantic event or completed cleanup. It strongly indicates memory
pressure termination but has no matching OS jetsam/crash report. The September 29
`bc1857a5` Mac bridge replay succeeds with effective prefill 2739 and an incomplete
forecast; its `21969715204`-byte peak is the fresh allocator's **load plus single
generation lifetime counter**, not an isolated prefill or phone process measurement.
Its 8 GiB budget is advisory and memory risk was explicitly allowed.

The installed phone build predates these measurements. No fresh iPhone run is
recorded here; the earlier installation was blocked by the locked phone. A current
read-only `devicectl device info lockState` attempt returns CoreDeviceError 4016 (trusted
connectivity/power assertion unavailable), so current phone validation is unavailable.
Neither Mac success nor the reduced peaks establishes that the iPhone Air fits this request.

## Reproduction and focused coverage

Run native commands outside the sandbox. The shared Cargo cache is project-scoped:

```sh
export CARGO_BUILD_BUILD_DIR="$HOME/Library/Caches/cargo-build/eredu"
export QWEN35_CHECKPOINT=/Users/jbg/dev/pico/Build/ModelValidation/qwen35-2b-4bit
cargo build -p eredu --example prefill_memory_probe --features metal
python3 - <<'PYTHON'
import json
from pathlib import Path
replay = json.loads(Path("/tmp/pico-eredu-update-replay.json").read_text())
request = {"messages": replay["messages"] + replay["cases"][0]["messages"],
           "tools": replay["tools"], "output_tokens": 8}
Path("/tmp/qwen35-probe-tool.json").write_text(json.dumps(request))
PYTHON
# Optional positions repeats/truncates input IDs for synthetic stress cases.
target/debug/examples/prefill_memory_probe "$QWEN35_CHECKPOINT" /tmp/qwen35-probe-tool.json 512
# Cooperative cancellation, milliseconds after entering generation:
target/debug/examples/prefill_memory_probe "$QWEN35_CHECKPOINT" /tmp/qwen35-probe-tool.json 512 100
cargo run -p eredu-backend-mlx --features metal --example recurrent_workspace_probe -- 2739 16 128 128 4
target/debug/examples/prefill_memory_probe "$QWEN35_CHECKPOINT" /tmp/qwen35-probe-tool.json 128 > /tmp/qwen35-final-tool-128.json
python eredu-backend-mlx/validation/qwen35_mlx_reference.py /tmp/qwen35-final-tool-128.json --output /tmp/qwen35-reference.json

MLX_ENABLE_TF32=0 cargo test -p eredu-backend-mlx --features metal --test chunked_prefill qwen_hybrid -- --include-ignored --test-threads=1
cargo test -p eredu-architectures --test reference_numeric qwen_recurrent_channels_reconstruct_declared_recurrence_and_cached_masks
cargo test -p eredu-backend-mlx --features metal --test chunked_prefill composite_chunked_prefill_commits_whole_prompt_cache_identity
cargo test -p eredu-runtime --lib recurrent_workspace_retains_known_output_and_unknown_scan_tail
cargo test -p eredu-architectures --lib cold_prefill_chunking_combines_architecture_backend_and_selected_path
cargo test -p eredu-architectures --lib metadata_only_selection_forecasts_without_source_authority
```

Nonzero native fixtures cover text and composite adapters, resident/host/disk
execution and affine four-bit loading; lengths 1, 3 and 9; chunks 1, 2, 3, 4 and
32; full observed logits versus final-position readout; three cached decode steps;
and four-token ordinary/controlled parity. A paged composite cache test rejects
whole-request persistence during incomplete prefill, then saves/restores the completed
whole-input identity and compares the next decode. Maximum absolute logit tolerance is
`2e-4`. Strict FP32 Metal comparisons disable TF32: the default GEMM versus GEMV
selection gives a measured approximately `0.0023` difference on the small FP32
fixture. Released-checkpoint token comparisons use the default Metal configuration.
The neutral recurrence test compares arbitrary short prefixes and fixed-state
continuity against independently reconstructed convolution and recurrent equations.

Metadata-only SafeTensors inspection supports forecasting without filesystem
access. Its explicit metadata-only physical declaration rejects both direct source
opening and prepared-source construction with `ArtifactError::MetadataOnly`.
This regression and its focused test are separate from the phone memory incident.
