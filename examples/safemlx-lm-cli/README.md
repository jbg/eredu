# safemlx-lm CLI example

This workspace crate provides a small, script-friendly text-generation binary
using `safemlx-lm`. A model can be a local Hugging Face-style directory, a GGUF
file, or a Hugging Face identifier already present in the local cache.

```sh
cargo run --release -p safemlx-lm-cli -- \
  --model mlx-community/Qwen3-0.6B-4bit \
  "Write a Rust function that adds two integers."
```

Qwen2 and Qwen2.5 text checkpoints use the same command. Hugging Face
directories must declare `model_type: "qwen2"`; GGUF files must declare
`general.architecture: "qwen2"`. Standard Qwen2.5 instruct templates and
checkpoint EOS/stop tokens are applied by the shared chat and streaming path.
Qwen2 vision-language and MoE variants are rejected explicitly.

On a Linux system with the CUDA prerequisites installed, add the workspace
feature to build and run the same CLI on MLX's CUDA backend:

```sh
cargo run --release -p safemlx-lm-cli --features cuda -- \
  --model /path/to/model "Write a Rust function that adds two integers."
```

The Hugging Face form never downloads files. It scans the cache selected by
`HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, or `HF_HOME`. Model directories use
the cached `main` revision by default. GGUF selectors prefer `main`, then search
other cached snapshots when `main` does not contain the requested
quantization. Use `--revision` to limit selection to a cached ref or commit.
Target/draft resolution prefers a cached snapshot containing both artifacts.
Different owning commits produce a provenance warning, while loaded model
metadata and token-id vocabularies determine MTP compatibility.
`--require-same-revision` makes repository commit equality mandatory;
`--allow-mixed-revisions` acknowledges and suppresses the warning.

For a cached repository containing multiple GGUF files, append a
case-insensitive quantization selector to the model identifier. The full
quantization name and the llama.cpp-style alias are both accepted; for example,
`UD-Q4_K_M` can also be selected with `Q4_K_M` when no exact `Q4_K_M` file is
cached:

```sh
cargo run --release -p safemlx-lm-cli -- \
  --model unsloth/Qwen3-0.6B-GGUF:Q4_K_M \
  "Explain imaginary numbers."
```

Selection is limited to files already present in the cache.
For sharded GGUF checkpoints, the CLI resolves the first canonical shard and
the loader discovers the remaining shards.

Local model paths use the same interface:

```sh
cargo run --release -p safemlx-lm-cli -- \
  --model /path/to/model \
  --device gpu:0 \
  --temperature 0.7 --top-p 0.9 --max-tokens 512 \
  "Tell me a short story."
```

`--device` selects the main model execution device and accepts `cpu` or an
explicit zero-based GPU index such as `gpu:0` or `gpu:1`. It defaults to
`gpu:0`. CPU execution is substantially slower for most models. Some
model-specific custom kernels may remain unavailable on CPU; those
configurations return an error and are never silently moved to a GPU.

Mirostat V2 adaptively targets a desired surprise instead of using fixed
top-k, top-p, and min-p cutoffs. Set a nonzero temperature and optionally tune
the target (`tau`, in bits) and adaptation rate (`eta`):

```sh
cargo run --release -p safemlx-lm-cli -- \
  --model /path/to/model \
  --mirostat-v2 --temperature 1.0 \
  --mirostat-tau 5.0 --mirostat-eta 0.1 \
  "Tell me a strange but coherent short story."
```

Repetition, frequency, and presence penalties are applied before Mirostat
truncation. Mirostat V2 supports embedded and external MTP; its adaptive state
advances only for tokens committed from the target distribution.

Gemma 4 can use an explicit external assistant through the generalized MTP
engine. The target may be fully resident or use `--layerwise-host`; the
assistant is loaded independently and remains fully resident:

```sh
cargo run --release -p safemlx-lm-cli -- \
  --model /path/to/gemma4 \
  --draft-model /path/to/gemma4-assistant \
  --device gpu:0 --mtp-draft-device cpu \
  --mtp-draft-tokens 3 --temperature 0.7 \
  "Explain speculative decoding."
```

`--mtp-draft-device` accepts `target`, `cpu`, or `gpu:N` and defaults to
`target`. `target` reuses the main model's existing stream and therefore does
not perform same-request lookahead. An explicit device always creates a
distinct draft stream, even when it names the same physical device as
`--device`. For example, this runs target verification and drafting on
different streams of GPU 0:

```sh
cargo run --release -p safemlx-lm-cli -- \
  --model /path/to/gemma4 \
  --draft-model /path/to/gemma4-assistant \
  --device gpu:0 --mtp-draft-device gpu:0 \
  --mtp-draft-tokens 3 --verbose \
  "Explain speculative decoding."
```

To use two GPUs, select them independently:

```sh
cargo run --release -p safemlx-lm-cli -- \
  --model /path/to/gemma4 \
  --draft-model /path/to/gemma4-assistant \
  --device gpu:0 --mtp-draft-device gpu:1 \
  --mtp-draft-tokens 3 --verbose \
  "Explain speculative decoding."
```

The main model may also run on CPU. `--device cpu --mtp-draft-device target`
uses one CPU stream; an explicit `cpu` creates a second CPU stream, while
`--device cpu --mtp-draft-device gpu:0` verifies on CPU and drafts on GPU 0.

Same-GPU streams share model and state array storage; they do not physically
copy arrays between streams. Any different-device pairing copies only data
that must cross devices. In either split mode, after submitting a lazy
target-verification graph, the MTP scheduler continues one block ahead under
the assumption that every proposal will be accepted. It resolves the target
result only after that eligible draft work has been submitted.

A fully accepted block emits the conventional target bonus immediately. If the
bonus matches the first optimistic token, that token is consumed and the
remaining exact token/distribution pairs are reused; the shortened block is
extended from its promoted assistant frontier before verification. A mismatch
discards the whole branch and continues from the canonical bonus prefix.
Rejection or EOS also discards continuation work without promoting branch
cache, sampler, PRNG, history, output, callback, or canonical statistics state.
The default retains at most one target transaction and one lookahead block.
Use `--disable-mtp-lookahead` to remove same-request branch work entirely for
an equivalent canonical A/B run on the selected stream topology. This does not
disable ordinary MTP verification.

Two streams on one GPU are an opt-in experiment, not an assumed optimization:
target and assistant kernels can contend for the same compute and memory
bandwidth. Compare an explicit same-GPU `--mtp-draft-device gpu:N` run with and
without `--disable-mtp-lookahead`; adaptive lookahead disables future branches
whose retained proposal reuse does not cover discarded work. Verbose
diagnostics print the resolved main and draft placements plus
`mtp_stream_topology`, bonus match, reuse, and discard counters.

Target acceptance and draft sampling use disjoint deterministic PRNG
roots. Draft substreams are addressed by logical output position, making output
reproducible for a fixed request seed across scheduler interleavings, branch
promotion/discard, and temporary branch-slot unavailability. Target bonuses
remain outside the target cache until they lead the next verification, matching
the backend cache convention. Fully accepted bonus rounds do not truncate the
target KV cache; partial rollback preserves chunked KV backing capacity.

After four resolved branches, lookahead is disabled for that request when no
proposal has been reused or reused proposal tokens fall below discarded
proposal tokens. The rule is deterministic, excludes the consumed matching
bonus token, and only suppresses future optional draft work. Use
`--disable-mtp-adaptive-lookahead` to keep eligible lookahead enabled for
benchmarking.

Same-request lookahead currently requires an external Gemma assistant,
distinct target/draft streams, and a draft processor with exact clone/discard
semantics derived only from explicit history, immutable configuration, and the
supplied PRNG state. Mirostat V2 still uses scheduled, lossless MTP but waits
for target resolution because its adaptive state depends on committed target
probabilities. Native embedded predictors use the scheduler without optimistic
lookahead.
Any explicit `--mtp-draft-device cpu|gpu:N` requires `--draft-model`; embedded
MTP uses `--mtp-draft-device target`.

The assistant may be a safetensors directory or a GGUF file with
`general.architecture = "gemma4_assistant"` or the published
`"gemma4-assistant"` spelling. GGUF config is read from a
`safemlx.mtp.config` JSON metadata string or a sibling `config.json`.
Cached target and assistant GGUFs can use independent quantization selectors,
even when individual downloads placed them in separate commit snapshots. The
CLI distinguishes standardized `mtp-` sidecar filenames from target-model
GGUFs, so target and assistant files may use the same quantization:

```sh
cargo run --release -p safemlx-lm-cli -- \
  --model unsloth/gemma-4-26B-A4B-it-GGUF:Q4_K_M \
  --draft-model unsloth/gemma-4-26B-A4B-it-GGUF:Q8_0 \
  --mtp-draft-tokens 3 \
  "Explain speculative decoding."
```

Stochastic MTP uses lossless probability-ratio acceptance and supports the
same top-k, top-p, min-p, and repetition/frequency/presence policies as normal
generation. Under `--verbose`, the CLI reports proposal and acceptance counts
together with optimistic drafted/consumed/reused/discarded work, target bonus
matches/mismatches, adaptive-disable status, and cross-request draft
opportunities. The first
optimistic token matched by a target bonus counts as consumed rather than
reused or proposed.

DeepSeek-V3/R1, Inkling, Nemotron-H, Qwen3-Next, and Qwen3.5/3.6 safetensors
checkpoints with native MTP weights use those embedded weights automatically;
no `--draft-model` is needed. `--mtp-draft-tokens` is safely capped by the
checkpoint's executable prediction depth.

Dense checkpoints can be quantized while loading. For example, 4-bit affine
weights substantially reduce decode-time weight traffic and memory use:

```sh
cargo run --release -p safemlx-lm-cli -- \
  --model LiquidAI/LFM2.5-1.2B-Instruct \
  --quantize 4 \
  "Explain MLX in one paragraph."
```

The default quantization group size is 64 weights; change it with
`--quantization-group-size`. Load-time quantization is performed on every run,
so use a checkpoint already carrying matching quantization metadata when
startup time is important.

For a SafeTensors or registered GGUF family, select a bounded device window
through the same architecture-detecting loader:

```sh
cargo run --release -p safemlx-lm-cli -- \
  --model /path/to/model --layerwise-host \
  --device-layer-window 1 --mapped-shards 4 \
  --host-budget-bytes 24000000000 --device-budget-bytes 8000000000 \
  "Summarize bounded weight residency."
```

`--verbose` also prints logical current/peak host and device parameter bytes,
synchronous transfer counts, and backend-tagged shard/reader diagnostics. Apple
CPU and GPU tiers share unified physical memory, so these logical tiers do not
increase total capacity. Registered SafeTensors and unquantized F32/F16/BF16
GGUF sources can combine bounded load-time affine/MXFP4 conversion with
ordinary-layer residency. Checkpoint-native packed GGUF tensors load directly
and are never implicitly requantized or transcoded. KV-cache offload is not
supported by this path.

Stream dense layers from either backend with finite tier controls:

```sh
cargo run --release -p safemlx-lm-cli -- \
  --model /path/to/model.gguf --dense-disk-stream \
  --device-budget-bytes 8000000000 --host-budget-bytes 16000000000 \
  --dense-host-lookahead 2 \
  --dense-background-queue 2 "Explain bounded GGUF loading."
```

Supported MoE models can cache routed experts separately. This
includes DeepSeek-V3/R1, GPT-OSS, Inkling, Kimi Linear, LFM2, Nemotron-H, Qwen3,
Qwen3-Next, Qwen3-VL-MoE, and Qwen3.5-MoE:

```sh
cargo run --release -p safemlx-lm-cli -- \
  --model /path/to/sparse-model --expert-cache \
  --device-layer-window 1 --mapped-shards 4 \
  --expert-cache-device-budget-bytes 8000000000 \
  --expert-cache-host-budget-bytes 16000000000 \
  --expert-cache-scratch-bytes 2000000000 \
  --expert-cache-prefill-bank-bytes 1000000000 \
  --expert-cache-eviction lfu \
  "Explain sparse expert residency."
```

For GGUF, sparse expert caching is available for DeepSeek2, LFM2-MoE,
Nemotron-H-MoE, Qwen3-MoE, Qwen3-VL-MoE with its sibling mmproj,
Qwen3.5-MoE, and MoE Qwen3-Next. Other GGUF families still support the
host-layerwise and dense-stream modes listed above.

The ordinary device and host budgets govern non-expert layerwise weights; the
`--expert-cache-*` budgets govern hot and warm expert copies. A zero expert host
budget promotes misses directly from checkpoint storage. The scratch limit is
checked against each temporary compact bank and is separate from the device
cache budget. `--verbose` reports prefill and decode requests, hits, misses,
evictions, compact-bank bytes, and current expert occupancy separately.
Multi-token prefill is split before acquisition so each compact bank targets
`--expert-cache-prefill-bank-bytes` without exceeding the hard
`--expert-cache-scratch-bytes` limit. Decode remains a single routed bank.
Combine `--expert-cache` with `--dense-disk-stream` to stream non-expert units
while keeping expert-granular reuse.

Add `--expert-cache-benchmark` to run the real prompt through a cold prefill,
a repeated prefill with fresh attention state, and one decode using the repeated
prefill's state before normal generation begins. Each measurement reports its
own latency, route and coalescing counts, host/device hits, misses and evictions,
compact-bank bytes, and ending cache occupancy. The measurements are diagnostic
samples rather than performance guarantees; filesystem caching and routing
locality can substantially change later runs.

Route inspection and transfers are synchronous. Unified memory does not create
additional physical capacity, and useful disk-backed performance depends on
expert-routing locality. Mapped-shard and logical-transfer counters do not
measure exact physical disk I/O. Checkpoint-native packed formats are preserved.
Supported dense Qwen SafeTensors and independently cached MoE routes use
bounded load-time conversion; unsupported artifact/family combinations fail
explicitly.

When the positional prompt is omitted, the binary reads it from stdin. Generated
text is decoded and flushed to stdout incrementally, including when MTP is in
use. When generation reaches `--max-tokens`, `stop_reason: max_tokens` is written
to stderr even without `--verbose`; normal EOS termination stays silent.
This keeps stdout convenient to pipe or capture. `--verbose` reports every stop
reason (`eos`, `max_tokens`, or `generator_exhausted`) and writes model details,
separate load and generation times, time to first token, decode-only and overall
generated-token rates, total execution time, and MLX peak/current/cache
unified-memory statistics to stderr. It also prints explicit
diagnostics/content section markers so the two streams remain visually distinct
in a terminal. Generation time includes prompt prefill, and `token_rate` is
generated tokens divided by that generation time. `decode_token_rate` excludes
time to first token and the first generated token. The memory values cover
allocations managed by MLX, not total process resident memory or memory-mapped
files.

```sh
printf 'Summarize the purpose of MLX.' | \
  cargo run --release -q -p safemlx-lm-cli -- \
  --model /path/to/model > response.txt
```

Chat templates are applied automatically when supplied by the model. Pass
`--thinking on` or `--thinking off` to override thinking/reasoning when the
template exposes the standard `enable_thinking` switch:

```sh
cargo run --release -p safemlx-lm-cli -- \
  --model mlx-community/Qwen3-0.6B-4bit \
  --thinking off \
  "Answer in one sentence."
```

The default, `--thinking auto`, leaves the template's model-specific behavior
unchanged. An explicit mode fails with a clear error when the template does not
support the switch. Pass `--raw` to tokenize the prompt directly; raw prompts
cannot use an explicit thinking mode. Run with `--help` for all sampling and
repetition-penalty options.

For recognized Inkling templates, `--thinking on` selects the released
`reasoning_effort=high` behavior and `--thinking off` selects
`reasoning_effort=none`.

`--thinking on` also requires a recognized reasoning parser. Use
`--allow-unparsed-reasoning` only when raw reasoning wire content is acceptable.

For a structurally recognized chat protocol, ordinary chat uses semantic
parsing independently of native tool generation.
Reasoning channels are emitted as reasoning events rather than leaking their
wire-format markers into stdout; visible response text is streamed normally.
With `--verbose`, reasoning content streams immediately in a delimited stderr
section. The section closes before visible response text resumes on stdout, so
the two channels remain distinct and appear in model-production order without
exposing event chunk boundaries. An unregistered template remains usable
without tools through a templated text fallback, while `--raw` remains the
explicit no-template path.

Native tool calling accepts a JSON array of OpenAI-shaped function definitions.
It requires independently recognized tool rendering, output parsing, and
constraint capabilities; unsupported protocols fail instead of falling back to
unconstrained text:

```sh
cargo run --release -p safemlx-lm-cli -- \
  --model /path/to/model \
  --tools tools.json --tool-choice required \
  --max-parallel-tool-calls 2 \
  --stop '<caller-stop>' \
  "Look up the weather in Bogotá."
```

Visible text and canonical tool events are streamed to stdout. With `--verbose`,
reasoning content streams in event order in a delimited stderr section, and
every prepared-tool request reports `stop_reason` (`grammar_complete`,
`stop_sequence`, `eos`, or `max_tokens`).
External and embedded MTP automatically use the same semantic runtime and the
existing draft-placement and scheduler-lookahead options. `--tools` conflicts
with `--raw`; raw/unconstrained generation remains available intentionally for
plain completions.
