# SafeMLX

SafeMLX is an unofficial Rust workspace for
[MLX](https://github.com/ml-explore/mlx). It provides Rust bindings and
higher-level libraries for array computation, neural networks, local language
model inference, GGUF and SafeTensors checkpoints, and neural audio codecs.

The project is intended for Rust applications that want to use MLX directly,
or run supported local models without a Python runtime. Apple silicon with
Metal is the primary platform; the core bindings also maintain CPU and NVIDIA
CUDA builds for x86-64 Linux and Windows.

SafeMLX is independent of Apple and is not an official MLX project.

## What is included

- An idiomatic Rust API for MLX arrays, operations, transforms, neural-network
  modules, optimizers, quantization, streams, and distributed execution.
- High-level loading and generation for supported text, multimodal, and
  realtime speech model families.
- SafeTensors and bounded, streaming GGUF checkpoint readers.
- Composable ordinary-layer and routed-expert residency: complete layers may
  be fully resident, host-layerwise, or dense disk-streamed; independent expert
  caches compose with resident, host-layerwise, or streamed non-expert units.
- Schema-v4 prompt-cache persistence with exact ordered per-layer attention,
  compressed-MLA, recurrent/convolution, and multimodal prefix state, distinct
  windows, processed-prefix identity, and distributed layer identity.
- A native Mimi audio codec implementation for codec-token speech models.
- An example command-line text generator.

See [model and checkpoint support](doc/model-support.md) for current formats
and feature gates. The language-model crate's
[combined-topology migration ledger](safemlx-lm/README.md#authoritative-combined-topology-migration-ledger)
is the persistent family-by-family backlog and limitations source of truth.

## Workspace

| Path | Purpose |
| --- | --- |
| [`safemlx`](safemlx/) | Safe array, operation, neural-network, transform, optimizer, I/O, and distributed APIs |
| [`safemlx-sys`](safemlx-sys/) | Low-level bindings and native build integration for the vendored MLX C API |
| [`safemlx-gguf`](safemlx-gguf/) | Framework-independent, bounded pure-Rust GGUF reader, writer, and quantization converter |
| [`safemlx-lm`](safemlx-lm/) | Model loading, tokenization, generation, multimodal processing, parallelism, and weight residency |
| [`safemlx-lm-utils`](safemlx-lm-utils/) | Tokenizer and chat-template utilities |
| [`safemlx-codec`](safemlx-codec/) | Neural audio codec components, currently including Mimi |
| `safemlx-macros`, `safemlx-internal-macros` | Procedural macros used by the public crates |
| `safemlx-tests` | Workspace integration and compile-time tests; not published |
| [`examples/safemlx-lm-cli`](examples/safemlx-lm-cli/) | Example `safemlx-lm` text-generation binary; not published |

The `safemlx` package names distinguish these crates from the projects from
which parts of the workspace were derived.

## Getting started

The minimum supported Rust version is 1.89. Add the core crate to a project:

```toml
[dependencies]
safemlx = "0.1.3"
```

MLX records operations lazily. Create a stream for the target device, build the
array graph, and evaluate it before reading values on the host:

```rust
use safemlx::{array, Device, DeviceType, Stream};

let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
let left = array!([1.0, 2.0, 3.0]);
let right = array!([4.0, 5.0, 6.0]);
let sum = left.add(&right, &stream)?.into_evaluated()?;

assert_eq!(sum.as_slice::<f32>(), &[5.0, 7.0, 9.0]);
# Ok::<(), safemlx::error::Exception>(())
```

More examples are in the [`safemlx` crate README](safemlx/README.md). To try
local text generation with a supported model directory or GGUF file, use the
[`safemlx-lm` CLI example](examples/safemlx-lm-cli/README.md).
The architecture-dispatched language-model loader includes Qwen2/Qwen2.5 text
SafeTensors and `qwen2` GGUF checkpoints, plus Kimi Linear SafeTensors and
`kimi-linear` GGUF checkpoints. Qwen2 support includes biased GQA, configured
per-layer full/sliding attention with arbitrary GGUF Boolean layer patterns,
standard tokenizer/chat sidecars, and resident or bounded weight residency;
vision-language and MoE Qwen2 variants are rejected.
LFM2/LFM2-MoE uses the same generic validated `LayerSchedule<P>` container.
Each LFM2 policy independently records convolution versus full attention and
dense versus sparse-MoE feed-forward execution, so runtime and bounded expert
routes support arbitrary internal combinations without a dense-prefix fallback.
Qwen3.5/Qwen3-Next likewise normalize full self-attention and recurrent linear
attention into an ordered Qwen hybrid layer schedule before execution.
Qwen3-VL vision uses the generic schedule container with a
vision-specific policy. Every ordered entry records full versus spatial-window
attention and an optional exact DeepStack merger bank. Qwen3-VL normalizes to
full attention at every vision block; Qwen3.5 retains its configured
full/window topology. Resident, bounded, structural, and workspace paths query
only that schedule.
Nemotron-H completes the four-operator pilot with an ordered schedule covering
Mamba, full or sliding self-attention, dense MLP, and sparse MoE layers; the
same schedule drives cache identity, bounded recurrent/KV state, and stateless
feed-forward entries. Its heterogeneous live cache keeps bounded Mamba state
resident while paging full or sliding attention KV through the shared cache
residency manager, including tensor- and replicated expert-parallel execution.
GPT-OSS also uses `LayerSchedule<AttentionPolicy>` as its sole normalized
attention geometry. Explicit Hugging Face full/sliding layer lists and the
published alternating fallback normalize once, then drive resident, bounded,
paged, expert-parallel, fingerprint, and state-accounting paths without a raw
configuration fallback.
Inkling uses `LayerSchedule<architectures::inkling::model::LayerPolicy>` for
its two independent per-layer choices: exact full/sliding attention and
dense/sparse-MoE feed-forward execution. Hugging Face layer lists or indices
and GGUF Boolean attention patterns normalize once into that ordered schedule;
resident, bounded, paged-cache, structural, expert-parallel, fingerprint, and
memory-accounting paths all consume it without threshold-based fallbacks.
Kimi Linear uses the same generic schedule container with a Kimi-specific
policy carrying both `AttentionKind::{Kda, Mla}` and
`FeedForwardPolicy::{Dense, SparseMoe}`. Hugging Face layer lists and
dense/MoE prefix-frequency metadata, plus GGUF per-layer attention metadata,
normalize once before execution; resident, bounded, sparse-expert, structural,
cache, fingerprint, and memory-accounting paths consume only the ordered
schedule.
DeepSeek-V3/R1 normalizes its dense-versus-routed-MoE topology into
`LayerSchedule<architectures::deepseek_v3::model::LayerPolicy>`. Hugging Face
prefix/frequency fields and GGUF leading-dense metadata are source-only inputs;
resident, bounded, structural, tensor/pipeline/expert-parallel, cache-identity,
and fingerprint paths consume the ordered schedule. Internally supplied
schedules may use arbitrary dense/MoE ordering, while every layer continues to
use the model-wide compressed MLA cache geometry.
Gemma 4 text and assistant checkpoints normalize every decoder-layer choice
into `LayerSchedule<architectures::gemma4::model::LayerPolicy>`. Each entry owns
its attention mode and exact window, head/KV geometry, KV ownership/publication
and key-as-value topology, dense MLP width, and dense-only versus dense-plus-MoE
selection. Resident and bounded execution, multimodal masks, shared-KV routing,
assistant drafting, structural admission, architecture identity, cache
allocation, and memory reporting consume only that schedule; arbitrary Boolean
GGUF attention patterns and internally distinct windows or widths are supported.
Gemma 4 tensor parallelism derives text, vision, and audio execution geometry
from shared semantic partition plans. Uneven GQA, dense/routed intermediate,
vision patch/MLP, audio head/convolution, and modality ranges therefore flow
through layer construction and prompt-cache identity without equal-shard
reconstruction. Gemma pipeline execution uses dependency-safe contiguous text
stages and stage-zero vision/audio roots for multimodal SafeTensors or
text-plus-projector GGUF checkpoints. It relays per-layer residual inputs and
exact multimodal full/sliding masks as immutable auxiliary state. GGUF combines
the language file and sibling dense media projector in the same bounded-read
residency plan.
Shared-KV publisher/consumer groups are never split across stages.
Fully resident Gemma text stages can apply load-time affine or MXFP4
quantization through the same direct and derived binding plan used for
checkpoint validation; host-layerwise and dense disk-streamed stages require a
matching checkpoint-native packed encoding.
Qwen2/Qwen3/Qwen3-MoE and GPT-OSS use that same adapter-driven pipeline runtime
for SafeTensors and canonical `qwen2`/`qwen3`/`qwen3moe`/`gpt-oss` GGUF
checkpoints. Qwen stages preserve biased Q/K/V projections or Q/K
normalization, GQA, tied-head ownership, routed-expert semantics, and the exact
per-layer full/sliding schedule. GPT-OSS stages retain native MXFP4 experts and
their alternating or explicit attention schedule.
Fully resident Qwen stages support aligned affine or MXFP4 requantization;
GPT-OSS can MXFP4-quantize eligible dense matrices without transcoding its
expert banks. Host-layerwise and dense disk streaming require checkpoint-native
encodings.
Qwen3-MoE, Kimi Linear, Inkling text, GPT-OSS, LFM2-MoE, Nemotron-H-MoE, and text-only
Qwen3-Next/Qwen3.5-MoE additionally support arbitrary-geometry
TP+PP+EP through the same Cartesian topology and semantic layer plan for fully
resident, host-layerwise, or dense-streamed SafeTensors and their canonical
GGUF checkpoints. Their
independent expert caches compose with PP, TP+PP, PP+EP, and TP+PP+EP; without
EP, each stage owns the experts for its local layers. Nonexpert and shared
projections retain semantic TP shards while cached routed banks follow the
replicated cache geometry.
Qwen hybrid stages retain rank-local recurrent convolution/delta-rule state
alongside full-attention KV state through resident, host-layerwise, and
dense-streamed execution; independently cached experts use the same
SafeTensors or canonical GGUF semantic catalog.
LFM2 and LFM2-MoE use the same runtime for SafeTensors and canonical
`lfm2`/`lfm2moe` GGUF checkpoints. Their alternating causal-convolution and
full-attention layers materialize semantic state slots from the canonical
cache schedule: convolution histories remain fixed state, attention uses the
shared ordinary or paged KV contract, and both are atomically persisted and
restored without an LFM2-specific pipeline cache variant. Fully resident,
host-layerwise, and dense disk-streamed local layers share the bounded LFM2
binding plan. LFM2-MoE expert caches compose independently with PP, TP+PP,
PP+EP, and TP+PP+EP, while dense-only stages retain explicit zero-expert local
ownership.
Nemotron-H-MoE likewise composes TP+PP+EP for its mixed Mamba, dense, routed,
and attention schedule. Rank-local Mamba convolution/SSM state, attention KV,
resident or independently cached experts, and canonical `nemotron_h_moe` GGUF
all use the shared stage, cache, residency, and failure-consensus machinery.
Kimi Linear, Nemotron-H/Nemotron-H-MoE, Qwen3-Next/Qwen3.5, and text-only
Inkling stages use that same adapter-driven pipeline runtime for
SafeTensors and their canonical GGUF architectures. KDA, Mamba2, and linear
attention expose borrowed semantic state to the runtime; MLA and KV storage
remain the shared compressed/paged implementations. Inkling's KV and four
short-convolution histories exercise the combined `KeyValueWithFixedState`
layout without introducing a family cache variant. Fully resident,
host-layerwise, and dense disk-streamed stages share each family's bounded
binding plan.
The pipeline runtime stores one type-erased stage shell and derives ordinary,
paged, and persisted cache state from its canonical semantic layer schedule.
All decoder families use the same resident/non-resident layer executor and the
same architecture-adapter binding plan; family payloads provide only math,
auxiliary state, and identity. Static bindings are selected lazily by stage
role, so loading does not build a duplicate source stage or open unowned
checkpoint shards. Transport and residency do not dispatch on a closed
architecture enum.
Request/work identity, bounded fair queues, isolated program state,
cancellation, backpressure, exact cross-rank descriptor consensus, failure
poisoning, and telemetry live in one architecture-neutral distributed
scheduler. The decoder pipeline is a thin program adapter: each request state
contains its `PipelineCache`, and each work item describes a prefill or decode
transition. Moshi/PersonaPlex is a second adapter over the same scheduler: its
request state owns temporal/depth caches, delayed streams, samplers, and PRNG
state, while encoded and forced prompt frames are work items. The two programs
share lifecycle and queueing without conflating their execution semantics.
Realtime session handoff is bound to exact checkpoint-file content plus the
normalized execution and quantization identity, rather than model geometry.
The resident/layerwise runtime likewise uses a validated execution-group DAG
rather than a flat architecture-specific sequence. Gemma vision and audio are
independent ingress roots feeding text; Qwen-VL, multimodal Qwen3.5, and
Inkling declare vision-to-text dependencies. Media assembly happens at the
text-node boundary, and invalid, cyclic, disconnected, or unmerged graphs fail
before execution.
Llama and Mistral normalize Hugging Face and GGUF scalar metadata into
`LayerSchedule<AttentionPolicy>`; resident, bounded, paged, tensor,
pipeline, cache-identity, and memory paths then use the exact policy at each
layer. Internally mixed and distinct-window schedules are supported. Prompt-cache
schema v4 persists the exact ordered per-layer layout, including distinct
sliding windows and each layer's retained token interval.

## Platforms

| Platform | Current support |
| --- | --- |
| macOS on Apple silicon | Default Accelerate and Metal backend; the full workspace is tested in CI |
| Linux x86-64 | CPU builds are checked; CUDA 12/13 with cuDNN 9 is optional, with compile coverage and an opt-in GPU smoke workflow |
| Windows x86-64 | Native MSVC CPU builds and file-format tests; optional CUDA 12.9/13.0 compile and link coverage, with runtime GPU validation opt-in and experimental |
| iOS/iPadOS, tvOS, visionOS | `safemlx` and `safemlx-sys` cross-build for Apple silicon devices and simulators; applications must bundle the generated `mlx.metallib` |

Native prerequisites, CUDA architecture selection, NCCL, Windows DLL handling,
Apple deployment targets, and Xcode integration are documented in the
[`safemlx-sys` README](safemlx-sys/README.md).

## Development status

SafeMLX is under active development and its pre-1.0 APIs may still change.
Model support is architecture-specific rather than a promise that every
checkpoint using a related name will load; `safemlx-lm` exposes config-checking
APIs and returns explicit errors for unsupported configurations.

Normal development uses the latest stable Rust release. Update it before
working on the workspace:

```sh
rustup update stable
```

The committed lockfile makes CI and local compatibility checks reproducible.
To verify the language-model crates and their default features against the
minimum supported Rust version, install Rust 1.89.0 and run:

```sh
rustup toolchain install 1.89.0
cargo +1.89.0 check --locked -p safemlx-lm-utils -p safemlx-lm
```

The macOS CI suite runs the workspace tests as follows, keeping tests that
exercise concurrent MLX use on a single test thread:

```sh
cargo test --workspace -- \
  --skip cpu_stream_creation_is_concurrent_safe \
  --skip async_eval_cpu_streams_are_concurrent_safe
cargo test --workspace concurrent_safe -- --test-threads=1
```

The platform workflows in [`.github/workflows`](.github/workflows/) are the
reference for the build and test commands exercised on Linux, Windows, and
Apple cross-compilation targets.

## Provenance

The core bindings originated in `mlx-rs`; the language-model crates were
derived from the `mlx-lm` work in a later `mlx-rs` fork; and the vendored C API
originated in `mlx-c`. All have since been modified in this repository. See
[the provenance notes](doc/provenance.md) for source repositories, import
commits, and the GGUF conversion lineage.

## License

The SafeMLX crates are available under MIT or Apache-2.0 unless a crate or
vendored component states otherwise. See [`LICENSE-MIT`](LICENSE-MIT),
[`LICENSE-APACHE`](LICENSE-APACHE), and the metadata and notices shipped with
individual components.
