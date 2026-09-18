# Component validation

Component behavior is specified by the [analysis guide](component-analysis.md),
[execution matrix](component-analysis.md#execution-coverage) and
[component checklist](component-analysis-checklist.md). Discovery, original and
effective capture, parameter queries, projections and overlays have independent
capability and resource contracts. A family name alone does not prove support.

## Behavioral requirements

Fixtures use nonzero parameters and inputs, real callbacks and complete mutable
state. They compare ordinary and controlled execution, resident/Host/Disk,
applicable TP/PP/EP combinations, independently cached expert banks, capture
transforms, intervention order and reversible overlays. Tests check failed
publication rollback, cached-state rejection, sibling isolation and restoration.
An injected failure retains its original cause and cumulative reservations.

Effective parameters are compared with independent host decoders for the
selected F32, affine, MXFP4, block-FP8 and GGUF formats. Packed expert gate/value
halves and write banks retain their global placement. Encoded sources and
floating overlays use their actual local shapes and precision.

FP8 dynamically quantizes multiplication inputs. Effective weight columns alone
cannot reconstruct a quantized write from pre-quantization captures.
`LoadedParameter::input_transform`, `ComponentGroup::write_input` and
`ComponentReadout::projection_input` expose the selected transform and actual
inputs. A head correction is measured separately from residual contributions.
Unknown transform metadata does not imply identity.

## Reproduction

```sh
cargo test -p eredu-architectures --lib
cargo test -p eredu-architectures --test reference_numeric
cargo test -p eredu --no-default-features --test portable_facade
cargo test -p eredu --no-default-features --test backend_conformance
```

The numerical harness accepts Cargo-compatible filters; use the test names for
the relevant family, source format and partition boundary. Native fixture
modules under `eredu-backend-mlx/src/tests` exercise actual adapters, parameter
storage and collectives. Metadata checks alone do not prove native execution.
Build and hardware requirements are in [development](development.md) and
[bounded validation](bounded-inference-validation.md).

## Released models and hardware scope

Released comparisons use pinned SmolLM2-135M, LFM2-350M and LFM2-8B-A1B sources.
The dense component comparisons have maximum absolute errors `7.6294e-5` and
`8.2016e-5`; the sparse controlled comparison covers 10,944,290 values with its
BF16 reconstruction bounds. Pinned artifacts, commands, per-field tolerances
and full comparison procedures are retained in the runnable validation tools
under `validation/` and `eredu-backend-mlx/validation/`.

Synthetic published-geometry fixtures cover V3 FP8, V4/DSpark encodings and
Muse's packed projector. Such fixtures prove their equations and ownership
paths, not full released-checkpoint accuracy. Neutral and native fixture scopes
must remain distinct. Tests running multiple local ranks on one physical Metal
device with CPU Ring transport do not establish multiple physical GPUs or hosts.

[Prepared-chat validation](prepared-chat-validation.md) covers a separate pinned
Qwen text/image request. Realtime frame protocols have their own media/session
contracts and are outside the text-prefix experiment API. Managed admission,
component discovery and numerical correctness each require their own evidence.
