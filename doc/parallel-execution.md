# Parallel execution

`safemlx-lm` supports tensor parallelism (TP), pipeline parallelism (PP), and
expert parallelism (EP) through one Cartesian topology. Each process has exact
coordinates and subgroup membership for the axes that apply to its model.

Use the ordinary high-level loader only for a replicated topology. A
non-replicated request must use the selected family's tensor-parallel adapter,
the pipeline loader, or the expert-parallel loader. Unsupported combinations
fail preflight before checkpoint payloads are materialized.

## Topology matrix

Every listed combination supports prefill, cached decode, synchronized token
generation, and rank-local prompt-cache persistence. The concrete artifact,
quantization, and residency request must still pass `inspect_model`.

| Family | Axes | Executable combinations | Artifact notes |
| --- | --- | --- | --- |
| Llama / Mistral | TP, PP | TP, PP, TP+PP | SafeTensors and canonical GGUF |
| Qwen2 / Qwen3 dense | TP, PP | TP, PP, TP+PP | SafeTensors and canonical Qwen GGUF |
| Qwen3 MoE | TP, PP, EP | TP, PP, EP, TP+PP, TP+EP, PP+EP, TP+PP+EP | SafeTensors and `qwen3moe` GGUF |
| GPT-OSS | TP, PP, EP | all pure, pairwise, and triple-axis combinations | native MXFP4 SafeTensors and type-39 GGUF |
| Gemma 4 dense | TP, PP | TP, PP, TP+PP | SafeTensors and text-plus-projector GGUF |
| Gemma 4 MoE | TP, PP, EP | all pure, pairwise, and triple-axis combinations | media towers and typed multimodal ingress included |
| DeepSeek-V3 / R1 | TP, PP, EP | all pure, pairwise, and triple-axis combinations | SafeTensors and DeepSeek2 GGUF |
| Kimi Linear | TP, PP, EP | all pure, pairwise, and triple-axis combinations | SafeTensors and `kimi-linear` GGUF |
| LFM2 dense | TP, PP | TP, PP, TP+PP | SafeTensors and `lfm2` GGUF |
| LFM2-MoE | TP, PP, EP | all pure, pairwise, and triple-axis combinations | SafeTensors and `lfm2moe` GGUF |
| Nemotron-H dense | TP, PP | TP, PP, TP+PP | SafeTensors and `nemotron_h` GGUF |
| Nemotron-H-MoE | TP, PP, EP | all pure, pairwise, and triple-axis combinations | SafeTensors and `nemotron_h_moe` GGUF |
| Qwen3-Next / Qwen3.5 dense | TP, PP | TP, PP, TP+PP | SafeTensors and canonical GGUF; supported vision blocks receive PP owners |
| Qwen3-Next / Qwen3.5 MoE | TP, PP, EP | all pure, pairwise, and triple-axis combinations | resident or independently cached experts |
| Inkling | TP, PP, EP | all pure, pairwise, and triple-axis combinations | SafeTensors and text-plus-projector GGUF |
| Qwen3-VL dense | TP, PP | TP, PP, TP+PP | SafeTensors and GGUF with projector |
| Qwen3-VL-MoE | TP, PP, EP | all pure, pairwise, and triple-axis combinations | vision blocks and MoE text decoder are both placed |
| Muse-Glimmer | TP, PP | TP, PP, TP+PP | SafeTensors and canonical GGUF; EP does not apply |
| Moshi / PersonaPlex | TP | TP | realtime temporal/depth runtime; PP and EP do not apply |

“All pure, pairwise, and triple-axis combinations” means TP, PP, EP, TP+PP,
TP+EP, PP+EP, and TP+PP+EP.

## Tensor parallelism

Tensor-parallel adapters partition semantic parameter roles rather than slicing
every tensor by a uniform rule. Attention heads, GQA groups, embeddings,
intermediate widths, recurrent groups, experts, modality towers, and output
biases can have different ownership. Uneven supported partitions are recorded
in the rank-local cache and prompt-cache identity.

Applications should select the process-local device with
`distributed::device_for_local_rank`. A global rank is not a local GPU index;
launchers commonly restrict each process with `CUDA_VISIBLE_DEVICES`, making
the process-local index zero.

## Pipeline parallelism

Pipeline models use placed execution groups and explicit dependencies rather
than assuming that every architecture is a flat list of text layers. This lets
multimodal towers, projectors, shared-KV groups, hybrid recurrent blocks, and
decoder stages declare their real ownership and routing.

The distributed scheduler owns request identity, bounded fair queues,
per-request cache state, cancellation, descriptor consensus, and failure
publication. Different requests may occupy different stages concurrently;
tokens within one autoregressive request remain ordered by their sampled-token
dependency.

Stage transport and downstream execution are ordered with exact completion
events. Descriptor agreement occurs before point-to-point or collective payload
operations so mismatched batch, sequence, mask, route, or state metadata fails
without entering an incompatible communication order.

## Expert parallelism

EP partitions routed experts while non-expert weights follow the family's
replication or tensor-parallel plan. When EP is inactive, each pipeline stage
owns all experts for its local layers. When it is active, route exchange uses
compact variable-count all-to-all and every rank participates even when its
local route count is zero.

Fully resident expert ownership and independent rank-owned expert caches use
the same semantic catalog. Remote experts are not materialized. Shared experts
are combined exactly once after routed reduction.

## Residency and persistence

Parallel models can use fully resident, host-layerwise, or dense-streamed
non-expert weights where the family supports them. Registered MoE families can
compose those policies with independent expert caches. Checkpoint reads are
rank-selective, including GGUF payload selections.

Prompt caches are saved per rank beneath one shared root. Manifests contain
global layer identity, local tensor geometry, state ownership, and topology.
Changing TP, PP, or EP shape makes the cache incompatible by design.

## Failure model

Topology construction, artifact validation, rank ownership, and memory
admission happen before payload materialization. Runtime work uses exact
descriptor and disposition consensus. A request-local numerical or generation
failure does not poison unrelated requests unless ranks can no longer prove a
common collective order.

See [Cancellation and bounded execution](cancellation.md) for submitted-work
ownership and [Model loading, quantization, and memory](model-loading.md) for
distributed residency limits.
