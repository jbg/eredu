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
require media input during prefill and explicitly report decode as unsupported. Partitioned capture
ownership remains unverified in support reporting; the logical graph remains
independent of rank placement.

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
generation, use `capture_discovery`, `prepare_observed_chat`, and
`generate_observed_chat`; `CapturePlan::none()` and empty new-plan selections both
mean capture none. The complete [`observed_generate`](../eredu/examples/observed_generate.rs)
example includes prompt alignment, ordinary sampling, bounded capture and cancellation.
