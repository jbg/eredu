# Architecture and observation discovery

`eredu::api::inspect_architecture(path)` returns a serializable
`eredu_core::ArchitectureDescriptor` with no backend feature enabled. It reads
artifact headers and the admitted family plan. It does not create a device,
read tensor payloads, or initialize an accelerator.

For selected execution, use `eredu::api::inspect_local_model` or
`eredu_architectures::inspect_model` with the exact intended load request.
`ModelInspectionReport.architecture_descriptor` contains the same logical
graph, even when execution selection is rejected.
`ModelInspectionReport.observation_support` reports prefill and decode support
separately. Invalid/unrecognized artifacts can have no descriptor. Older
serialized reports deserialize with both discovery fields absent.

The descriptor has a `schema_version`, typed nodes and attributes, containment
through `parent`, and directed data-flow edges. Residual edges identify bypass
inputs; routed and shared experts form distinct branches. Attention head
sharing, receptive field, mechanism, recurrence, head dimensions, and positional
encoding are independent attributes of each layer. Consumers traverse these
types; they do not match family names or parse checkpoint keys.

Architecture descriptor schema **2** adds `layer_groups` and parameter sharing.
Observation catalogs and support reports retain schema **1**; their capture
selectors and admission version are unchanged. Older descriptors deserialize
with an empty `layer_groups` and absent `shared_with` declarations. Consumers
must treat that as unknown structure, rather than infer a single pass from the
number of decoder nodes.

Schema **3** adds component read input-projection chains. Each stage identifies
effective parameter rows, an optional bias and normalization, and the effective
observation consumed downstream. An empty chain means a direct read from the
normalized sublayer input. Check the descriptor version before interpreting these
equations; ignoring a latent normalization changes the meaning of a read. The
[component guide](component-analysis.md#topology-and-actual-support) describes
bounded parameter queries and current-position capture semantics.
Retained partition layouts derive stage observations from the projection's output
rows and the invocation's execution owner. The stage's effective and original
paths share that placement; shared checkpoint aliases do not move the invocation.
Loaded support still checks the selected executor's hook facts separately.

Each `ArchitectureLayerGroup` declares a `physical_layer_count`, ordered
`passes`, and `weight_sharing`. Each pass has a zero-based `index` and ordered
`executions`; an execution identifies its `physical_layer_index` and exact
graph `node_id`. Physical layer indices are local to the group. A pass visits
each physical layer once. Ordinary decoder stacks have one pass and
`weight_sharing: "none"`. Repeated stacks declare `"shared_across_passes"`.
Groups describe decoder stacks; media encoders and other opaque components
remain represented by their existing nodes and completeness declarations.

For the released Nanbeige 4.2 configuration, the decoder group contains **22
physical layers, 2 passes, and 44 executions**. LM Inspector can render
**“22 layers × 2 passes · shared weights”** from these fields, then expand
`passes[].executions[]` and resolve each `node_id` with `descriptor.node(...)`.
For example, physical layer `0` appears as `decoder.layers.0` in pass `0`
and `decoder.layers.22` in pass `1`. Their capture paths are respectively
`model.layers.0.input` / `.output` and `model.layers.22.input` / `.output`.
Use those advertised paths to select individual executions, including with
bounded `CapturePlan` selections. Support must still be joined by exact path.
For display, add one to pass and physical layer indices; do not change paths.

`ArchitectureNode.layer_index` retains its existing **logical execution**
ordinal (0–43 for Nanbeige), correcting the original documentation that called
it physical. Node IDs, logical parameter prefixes, graph edges, and observation
paths retain their identities. `ArchitectureParameterGroup.shared_with`
references another parameter-group ID when distinct logical groups reuse the
same weights. For example, `parameters:model.layers.22.self_attn` shares with
`parameters:model.layers.0.self_attn`. The inter-pass output normalization
shares with `parameters:model.norm`; it remains a distinct operation. Nodes
that already reference the same parameter group, such as tied embedding and
output weights, continue to do so. An absent `shared_with` means no additional
sharing declaration, not proof of independent weights.

Weight sharing describes architecture parameters. Each Nanbeige execution has
independent mutable KV state, and resident execution can materialize replicas
of the shared checkpoint weights. The group is neither a residency unit nor a
claim about native allocation sharing. Controlled and uninterrupted capture
use the same per-execution identities.

Three namespaces have different purposes:

| Field | Meaning |
| --- | --- |
| `ArchitectureNode.id` | Stable logical identity within this descriptor/schema |
| `ArchitectureParameterGroup.canonical_prefix` | Architecture-declared logical checkpoint module prefix; physical aliases and encoding companions remain in checkpoint schemas |
| `ObservationPoint.path` | Exact selector accepted by the existing observation API |

An operation with no observation paths is not instrumented by this catalog.
In particular, attention nodes do not imply that attention probabilities can be
captured. `completeness` distinguishes complete coverage from partial and
unsupported descriptions, with reasons. `None` in an attribute, tensor shape,
or byte estimate means unknown, never zero.

All currently cataloged values are tensors. Activations usually have
`[batch, sequence, hidden]` axes. Expert IDs, selected scores, and coefficients
have `[token_rows, selected_expert]`, where token rows flatten batch and
sequence. The semantic dtype is independent of checkpoint storage. MLX declares
F32 conversion for floating host observations; integer signedness is retained in
the returned `TensorObservationData`. Captures retain complete tensors, and byte
estimates remain unknown when the request shape or native dtype is unknown.
These statements describe the legacy `ObservationRequest` API. The admitted
`CapturePlan` API adds native slicing, previews, summaries, histograms, and raw
candidate-score summaries with explicit budgets. Selected support reports now
include `capture` transformation capabilities and conditions. See
[bounded capture and observed generation](bounded-capture.md).

Block inputs/outputs and logits are observed **before an intervention at that
exact path**. Earlier interventions may already have affected them. Routing
events remain read-only observations after expert dispatch. Separately declared
intervention targets support control before dispatch where the loaded session
advertises it; see [intervention plans](interventions.md).

The initial detailed projections cover Llama/Mistral, Qwen2/3, GPT-OSS, Qwen
hybrids, LFM2, Kimi Linear, Nemotron-H, and DeepSeek V3. Multimodal components
reuse existing Gemma, Inkling, and Muse component graphs; complex block
interiors, DeepSeek V4 hyper-connections, embedded prediction, and Moshi frame
execution have explicit partial descriptions. Their catalogs never fabricate
missing internal measurements. Conditional Qwen adapters currently expose block
and media boundaries, without normalized routing events. Media projector points
require media input during prefill and explicitly report decode as unsupported.
Partitioned support requires both retained invocation placement and callable hooks;
the logical graph remains independent of rank placement. All-dense V3 target
execution verifies both for TP, PP and combined TP/PP. Target-only sparse V3 TP now emits the same internal hooks, including additive
shared-write terms, and passes neutral TP/PP/residency conformance and native
TP/PP/EP acceptance for F32 SafeTensors/GGUF and load-time affine 4-bit/group-32.
Embedded prediction interiors remain unverified.

Architecture descriptor schema 4 includes component schema 3's `write_partition`:
`Complete` or `TensorParallelSum`. The latter declares full-hidden-width native
terms before an ordinary TP reduction; equal hidden coordinates across TP ranks
are distinct contributions. Placement requires complete TP groups and retains one
producer per TP coordinate across replicas. Partition evidence independently
specifies bounded host assembly arithmetic. The declaration alone does not enable
loaded hooks. Mixed V3 F32 SafeTensors/GGUF and load-time affine native TP/PP/EP
acceptance passes all 63 ordinary-residency cases.

Run the portable example:

```sh
cargo run -p eredu --no-default-features --example architecture_discovery -- /path/to/artifact
```

The example also contains a compiled generic `capture_advertised` helper using
an application's existing `ModelRuntime`. A complete native example inspects,
prints support, selects an advertised path, loads the model, and captures both
prefill and decode values:

```sh
cargo run -p eredu-backend-mlx --example capture_discovery -- /path/to/artifact 1 2 3
```

Provide valid token IDs for the artifact. This example explicitly chooses CPU
execution; native resources are created only after cold discovery.

LM Inspector should retain the inspected descriptor, render `nodes`/`edges`,
list `observations.points`, and join selected support by exact `path`. Offer
capture for `Supported` points, show conditional requirements, and display
unsupported/unverified reasons. Pass the selected path to
`ObservationSelector::Exact` and use the existing `inspect_prefill` /
`inspect_decode` calls. Do not use an empty selector list as “capture none”:
the existing observation API treats it as capture-all. Missing returned values
must remain missing, not become numerical zero. For ordinary instrumented text
generation, use `capture_discovery`, attach a capture declaration to
`PreparedChatRequest`, and run the session from `start_controlled_chat`; `CapturePlan::none()` and empty new-plan selections both
mean capture none. The complete [`observed_generate`](../eredu/examples/observed_generate.rs)
example includes prompt alignment, ordinary sampling, bounded capture and cancellation.

Mixed V3 independently cached expert variants pass all 63 CPU Ring TP/PP/EP and
residency cases for F32 SafeTensors/GGUF and affine 4-bit/group-32. Parameter
discovery reflects the effective source-preserving companion dtype and retained
generated destinations. An owned expert with no selected routes remains declared;
zero local cache activity does not mean its parameters or topology are absent.

Architecture descriptor schema 5 adds `component_scopes` (empty when absent in
older JSON), using component schema 4. Primary `components`, `routed_components`
and `component_readout` still describe target scores only. Each V3 prediction
depth has a separate scope joined to its execution group, decoder, routed/shared
components and its own readout head. Its residual starts with a linear projection
of separately normalized token embedding and supplied hidden state. Fusion input
column ranges and effective parameter names are declared; clients need no
checkpoint-name parsing. The supplied hidden source is the actual invocation
capture, not an assumption that every invocation consumes a preceding depth.

All prediction-scope observations, including generated projection inputs and
routed-unit values, require `PredictionExecution`. Selected target-only call paths
report them unsupported even when their tensor collectors are available. A full
declaration therefore does not advertise public speculative component support.
The shared readout equation preserves existing primary JSON fields; Rust struct
construction now places head fields in `ComponentReadoutEquation`.


Internal V3 speculative calls now carry explicit invocation phase, proposal depth
and physical sequence width through the shared architecture strategy. This does
not change loaded capture support: a typed hook fact alone cannot supply public
phase/frontier admission, cumulative collection or record delivery. The existing
`prediction_inspection` gate remains false in ordinary and cold adapters.

Descriptor schema 7 extends the separately scored component domains with a fused
prediction scope spanning ordered `execution_groups` and separately declared
`static_parameter_roles`. Component schema 6 adds direct residual sources with
explicit expansion and dynamic `score_writes` in vocabulary space. These joins
preserve DSpark's multiple decoder blocks and anchor-dependent Markov addition
without treating either block count as proposal capacity or dynamic scores as a
static bias. `speculative_invocations` declares context-cache roots that execute
without a component score equation. Descendants inherit the nearest explicit
invocation binding; ambiguous or unknown roots fail validation. Loaded support
still requires the corresponding selected executor and native collector facts.


Descriptor schema 8 carries component schema 7's optional `output_gate`. This is a
per-row scalar multiplier after the affine write and any output normalization.
It declares the gate's effective input, actual projection input, exact read
parameters, activation and original/effective observations. The gate is measured
again for every trial. Qwen shared FFN groups retain ordinary gated-unit identities
and add this separate post-projection relationship. Older groups omit the field.


Schema **9** adds `component_transforms`, with an empty default for older
serialized descriptors. Component schema **8** declares identity normalization
and exact tensor transformations between observed boundaries: constant or learned
scaling, normalization, and causal depthwise convolution. Each transform names its
owning node, actual input, original output, optional effective output and canonical
parameter identities. Node ownership locates the transform in a target or
prediction invocation. These records are equations, not extra additive residual
contributions and not loaded-operation capabilities. For causal kernels, the last
tap consumes the current input; earlier taps may consume retained state. The
ordinary capture API admits only the requested current positions. Inkling's
expanded transform declarations currently supplement its partial component graph;
complete scalar groups, prediction scopes and native component support remain work
in progress.

Descriptor schema 11/component schema 10 describes gated-delta recurrence with
separate projected key-head and recurrent value-head geometry, exact per-head or
per-key-channel decay, and normalization/gating before scalar output channels.
Qwen recurrent channels use `.mixer.channels`; Kimi channels use
`.attention.channels`. Both join the actual pre-projection tensor and effective
write columns. Fused causal Q/K/V transforms preserve their physical segmentation.
The L2 normalization kind uses additive epsilon under the square root of the
sum of squares, while RMS uses the mean. Captured channels already include these
stateful calculations; discovery does not imply historical state observations.

Descriptor schema 12/component schema 11 adds `scaled_softcap`, projected-head
`output_scale` (default one), and `token_embedding_normalization` before media
assembly. The existing `embedding_normalization` describes normalization of the
assembled tensor; consumers must keep those stages distinct. Muse declares its
per-channel sigmoid gate and four separate pre/post RMS normalizations, with
HF centered gains and GGUF full gains. Sparse postnorm applies after summing
the complete route-weighted write. Loaded support still depends on the selected
executor and collector mechanisms.
