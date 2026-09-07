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
events are read-only observations after expert dispatch; they do not expose a
pre-dispatch intervention opportunity.

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
